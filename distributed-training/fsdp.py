# fsdp_task4.py
# Task 4 — Fully Sharded Data Parallel (FSDP) — Student Template
#
# Quick sanity check (single run):
#   torchrun --standalone --nproc-per-node=4 fsdp.py \
#       --world_size 4 --global_batch_size 32 --mode ddp --quick_demo
#
# Assignment workflow:
#   - Implement ALL TODOs (measurement, multi-seed aggregation).
#   - Run Step 1 (DDP baseline), Step 2 (FSDP Full Shard),
#     then Step 3 comparison/discussion. Produce tables/plots for Deliverables.
#
# Metrics to report per run:
#   * throughput (samples/s)
#   * time_per_step (s)
#   * peak GPU memory per rank (bytes)
#
# Notes:
#   - Use mean ± std over 3 seeds across Steps 1–3 (same protocol as the pipeline task).
#   - Warm up 20 steps, then measure >= 100 steps (configurable via CLI).
#   - This task does NOT use pipeline micro-batches/chunks.

import argparse
import os
import time
import json
import math
import contextlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertLayer

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from hf_utils import generate_inputs_for_model, get_number_of_params


def set_seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def pretty_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if abs(n) < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def get_peak_bytes(device):
    return torch.cuda.max_memory_allocated(device)


def reset_peak(device):
    torch.cuda.reset_peak_memory_stats(device)


def ddp_model(model, device):
    model.to(device)
    return DDP(model, device_ids=[device.index], output_device=device.index)


def fsdp_full_shard_model(model, device, use_mixed_precision=True):
    """
    FSDP with FULL_SHARD (parameters, gradients, optimizer states fully sharded).
    """
    mp = MixedPrecision(
        param_dtype=torch.float16 if use_mixed_precision else None,
        reduce_dtype=torch.float16 if use_mixed_precision else None,
        buffer_dtype=torch.float16 if use_mixed_precision else None,
    )
    fsdp = FSDP(
        model.to(device),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=device,
    )
    return fsdp


def make_model():
    config = BertConfig()
    config.return_dict = False
    model = BertForMaskedLM(config)
    return model


def make_optimizer(model, lr=1e-4, weight_decay=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# ---------------------------
# Training / evaluation steps
# ---------------------------

@torch.no_grad()
def run_eval_step(model, inputs, autocast_enabled=True):
    # Throughput/time measurement without backward
    with autocast(enabled=autocast_enabled):
        out = model(**inputs)
    return out


def run_train_step(model, inputs, optimizer, scaler: GradScaler | None, autocast_enabled=True):
    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=autocast_enabled):
        out = model(**inputs)

        # Try to get a scalar loss from the model (if labels were provided)
        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]) and out[0].ndim == 0:
            loss = out[0]
        elif hasattr(out, "loss") and torch.is_tensor(out.loss) and out.loss.ndim == 0:
            loss = out.loss
        else:
            # We only have logits -> build fake labels and compute CE loss.
            logits = out.logits if hasattr(out, "logits") else out[0]  # [B, T, V]
            B, T, V = logits.shape
            device = logits.device

            # Fake labels: random token id in [0, V)
            labels = torch.randint(low=0, high=V, size=(B, T), device=device)  # [B, T]

            # Cross-entropy over time+batch
            loss = F.cross_entropy(
                logits.view(-1, V),     # [B*T, V]
                labels.view(-1),        # [B*T]
                reduction="mean"        # scalar
            )


    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return float(loss.detach().item())


# ---------------------------
# Warmup + measurement
# ---------------------------

def warmup_and_measure(args, model, device, global_batch_size, steps_warmup, steps_measure,
                       do_backward=True, autocast_enabled=True):
    """
    Generic warmup + measure. Returns:
      - throughput_samples_per_s: float
      - time_per_step_s: float
      - peak_bytes_all_ranks: List[int] gathered across ranks
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_batch_size = global_batch_size // world_size

    # Build a full-batch input
    inputs = generate_inputs_for_model(BertForMaskedLM, unwrap_model(model), "BertForMaskedLM",
                                       local_batch_size, device)

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # Warmup (not timed)
    reset_peak(device)
    for _ in range(steps_warmup):
        if do_backward:
            _ = run_train_step(model, inputs, optimizer, scaler, autocast_enabled=args.amp)
        else:
            _ = run_eval_step(model, inputs, autocast_enabled=args.amp)
    dist.barrier(); torch.cuda.synchronize(device)

    # Measure
    reset_peak(device)
    t0 = time.time()
    for _ in range(steps_measure):
        if do_backward:
            _ = run_train_step(model, inputs, optimizer, scaler, autocast_enabled=args.amp)
        else:
            _ = run_eval_step(model, inputs, autocast_enabled=args.amp)
    dist.barrier(); torch.cuda.synchronize(device)
    elapsed = time.time() - t0

    throughput = (global_batch_size * steps_measure) / elapsed
    time_per_step = elapsed / steps_measure
    peak_local = int(get_peak_bytes(device))

    # Gather peak memory from all ranks
    peak_all = [None for _ in range(world_size)]
    dist.all_gather_object(peak_all, peak_local)

    return float(throughput), float(time_per_step), peak_all


# ---------------------------
# Build model by mode
# ---------------------------

def build_model_by_mode(args, device):
    base = make_model()
    if args.mode == "ddp":
        return ddp_model(base, device)
    elif args.mode == "fsdp_full":
        return fsdp_full_shard_model(base, device, use_mixed_precision=args.amp)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


# ---------------------------
# One run for a given seed
# ---------------------------

def run_one_seed(args, device):
    """
    Returns at least:
      - 'seed', 'throughput_samples_per_s', 'time_per_step_s', 'mem_peak_per_rank_bytes'
      - (optional) 'max_gbs_fit'
    """
    rank = dist.get_rank()
    set_seed_all(args.seed)

    model = build_model_by_mode(args, device)

    thr, tps, peaks = warmup_and_measure(
        args=args,
        model=model,
        device=device,
        global_batch_size=args.global_batch_size,
        steps_warmup=args.steps_warmup,
        steps_measure=args.steps_measure,
        do_backward=True,
        autocast_enabled=args.amp,
    )

    out = {
        "seed": int(args.seed),
        "throughput_samples_per_s": float(thr),
        "time_per_step_s": float(tps),
        "mem_peak_per_rank_bytes": peaks,
    }

    return out


# ---------------------------
# Aggregation and saving
# ---------------------------

def aggregate_and_save(args, per_seed):
    """
    per_seed: List[dict]
    Compute mean/std for throughput and time_per_step.
    For memory peaks, compute per-rank mean/std.
    """
    import statistics as stats
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    thrs = [x["throughput_samples_per_s"] for x in per_seed]
    tps = [x["time_per_step_s"] for x in per_seed]

    agg = {
        "throughput_mean": float(stats.mean(thrs)),
        "throughput_std": float(stats.pstdev(thrs) if len(thrs) > 1 else 0.0),
        "time_per_step_mean": float(stats.mean(tps)),
        "time_per_step_std": float(stats.pstdev(tps) if len(tps) > 1 else 0.0),
    }

    mem_matrix = [x["mem_peak_per_rank_bytes"] for x in per_seed]  # shape [nseed][world_size]
    mem_mean = []
    mem_std = []
    for r in range(world_size):
        col = [row[r] for row in mem_matrix]
        col = [int(c) if c is not None else 0 for c in col]
        mem_mean.append(float(stats.mean(col)))
        mem_std.append(float(stats.pstdev(col) if len(col) > 1 else 0.0))
    agg["mem_peak_per_rank_mean_bytes"] = mem_mean
    agg["mem_peak_per_rank_std_bytes"] = mem_std

    if args.out and rank == 0:
        payload = {
            "spec": {
                "mode": args.mode,
                "world_size": args.world_size,
                "global_batch_size": args.global_batch_size,
                "steps_warmup": args.steps_warmup,
                "steps_measure": args.steps_measure,
                "amp": bool(args.amp),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seeds": args.seeds,
            },
            "per_seed": per_seed,
            "aggregate": agg,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Rank 0] Results saved to: {args.out}")

    if rank == 0:
        print("[Rank 0] Aggregate:", json.dumps(agg, indent=2))
    return agg


# ---------------------------
# Main
# ---------------------------

def run(args):
    rank = args.rank
    device = args.device

    # Print model config once
    if rank == 0:
        model = make_model()
        print(unwrap_model(model).config)
        print(f"Total params ≈ {get_number_of_params(model) // 10 ** 6}M")
        del model

    if args.quick_demo:
        # quick fwd/bwd to sanity check the selected mode
        model = build_model_by_mode(args, device)
        gbs = max(2, args.global_batch_size // 2)
        inputs = generate_inputs_for_model(BertForMaskedLM, unwrap_model(model), "BertForMaskedLM", gbs, device)
        opt = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)
        loss = run_train_step(model, inputs, opt, scaler, autocast_enabled=args.amp)
        dist.barrier()
        print(f"[Rank {rank}] quick_demo ok, loss={loss:.4f}")
        return

    # Multi-seed loop
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    per_seed = []
    for s in seeds:
        args.seed = s
        out = run_one_seed(args, device)
        per_seed.append(out)

    # Aggregate & save on rank 0
    aggregate_and_save(args, per_seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    p.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    p.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    p.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    p.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    p.add_argument('--global_batch_size', type=int, default=64)
    p.add_argument('--steps_warmup', type=int, default=20)
    p.add_argument('--steps_measure', type=int, default=100)

    p.add_argument('--mode', type=str, default="ddp",
                   choices=["ddp", "fsdp_full"],
                   help='Step 1: ddp; Step 2: fsdp_full')

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--amp', action='store_true', help="enable mixed precision (recommended for FSDP)")

    p.add_argument('--min_gbs', type=int, default=8)
    p.add_argument('--max_gbs_cap', type=int, default=8192)

    p.add_argument('--seeds', type=str, default="1,2,3")
    p.add_argument('--out', type=str, default="", help="optional JSON path to save results")
    p.add_argument('--quick_demo', action='store_true')

    args = p.parse_args()

    if args.cuda:
        dev_id = args.rank % max(1, torch.cuda.device_count())
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        raise RuntimeError("This task requires CUDA GPUs.")

    backend = "nccl"
    torch.cuda.set_device(args.rank)
    dist.init_process_group(backend=backend, rank=args.rank, world_size=args.world_size)

    try:
        run(args)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()