# pipeline.py
# Distributed Pipeline Parallelism — Student Template
#
# Quick sanity check (one step):
#   torchrun --standalone --nproc-per-node=4 pipeline.py  --world_size 4 --global_batch_size 8 --chunks 4 --quick_demo
   
#
# For the assignment:
#   - Implement ALL TODOs below (measurement, seeds aggregation, custom partitioning, optional stage timing).
#   - Then run your experiments for Steps 1–4 and produce tables/plots.
#
# Notes:
#   - Use mean ± std over 3 seeds across Steps 1–4 (same protocol as the pipeline task).
#   - Warm up 20 steps, then measure >= 100 steps (configurable via CLI).

import argparse
import os
import time
import json
import functools
import logging
import traceback
import sys

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint
from transformers import BertModel, BertConfig
import numpy as np

from hf_utils import generate_inputs_for_model, get_number_of_params


def log(message, level='info'):
    """Helper to log with the current rank prefix (if available)."""
    prefix = "[rank ?]"
    try:
        if dist.is_initialized():
            prefix = f"[rank {dist.get_rank()}]"
    except Exception:
        # dist not initialized yet
        pass

    logger = logging.getLogger()
    msg = f"{prefix} {message}"
    if level == 'info':
        logger.info(msg)
    elif level == 'warning':
        logger.warning(msg)
    elif level == 'error':
        logger.error(msg)
    else:
        logger.debug(msg)


def safe_schedule_step(schedule, *args, **kwargs):
    """Call schedule.step but catch exceptions, log stack, and try to clean up.

    This prevents silent rank crashes that leave other ranks hanging at barriers.
    """
    try:
        return schedule.step(*args, **kwargs)
    except Exception:
        log("Exception in schedule.step():\n" + traceback.format_exc(), level='error')
        # Try to notify/clean up the process group so other ranks don't hang.
        try:
            if dist.is_initialized():
                # Prefer abort if available (forces all ranks to exit quickly).
                if hasattr(dist, 'abort'):
                    log("Calling dist.abort() to unblock other ranks", level='error')
                    dist.abort()
                else:
                    log("Destroying process group after exception", level='error')
                    dist.destroy_process_group()
        except Exception:
            log("Failed to clean up process group after exception:\n" + traceback.format_exc(), level='error')
        # Re-raise so the current process still fails visibly.
        raise


# ---------------------------
# Split spec helpers
# ---------------------------

def make_split_spec_even(model: BertModel, world_size: int):
    """
    Build split points at 'encoder.layer.{k}' to roughly balance blocks per rank.

    """
    n_layers = model.config.num_hidden_layers
    per = (n_layers + world_size - 1) // world_size  # ceil
    if per == 0:
        raise ValueError(f"world_size ({world_size}) cannot exceed num_hidden_layers ({n_layers})")
    return {
        f"encoder.layer.{i * per}": SplitPoint.BEGINNING    
        for i in range(1, world_size)
    }

def parse_pattern(pattern: str) -> list[int]:
    parts = pattern.split('-')
    sizes = []

    for part in parts:
        try:
            size = int(part)
            sizes.append(size)
        except ValueError:
            raise ValueError(f"Invalid partition pattern: {pattern}")
        
    return sizes

def make_split_spec_custom(model: BertModel, pattern: str, world_size: int):
    """
    TODO (Step 4): parse a custom pattern like "6-2-2-2" and produce split points.
      - Validate: number of parts == world_size
      - Validate: sum(parts) == num_hidden_layers
      - Compute cumulative sums c1, c2, c3,... and place SplitPoint.BEGINNING
        at 'encoder.layer.{c1}', 'encoder.layer.{c2}', ...
    Return a dict[str, SplitPoint].
    """
    parsed_pattern = parse_pattern(pattern)
    if len(parsed_pattern) != world_size:
        raise ValueError(f"Number of parts in pattern ({len(parsed_pattern)}) does not match world_size ({world_size})")
    
    n_layers = model.config.num_hidden_layers
    if sum(parsed_pattern) != n_layers:
        raise ValueError(f"Sum of parts in pattern ({sum(parsed_pattern)}) does not equal num_hidden_layers ({n_layers})")
    
    split_sec = {}
    cumulative_sum = 0
    for i in range(len(parsed_pattern) - 1):
        cumulative_sum += parsed_pattern[i]
        split_sec[f"encoder.layer.{cumulative_sum}"] = SplitPoint.BEGINNING

    return split_sec

# ---------------------------
# Optional: simple per-stage timing (hint)
# ---------------------------

def wrap_stage_forward_for_timing(module: torch.nn.Module, device: torch.device):
    """
    OPTIONAL (Step 4): wrap stage forward to collect avg forward time per step.
    Use CUDA events; store accumulator in module._acc_ms.
    """
    # >>> OPTIONAL: YOUR CODE HERE <<<
    # Hints:
    #   start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    #   with torch.cuda.device(device): start.record(); out = orig(*args, **kw); end.record()
    #   torch.cuda.synchronize(device); module._acc_ms += start.elapsed_time(end)
    module._acc_ms = 0.0
    module._num_calls = 0

    orig_forward = module.forward

    @functools.wraps(orig_forward)
    def timed_forward(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.device(device):
            start.record()
            out = orig_forward(*args, **kwargs)
            end.record()

        torch.cuda.synchronize(device)
        elapsed_ms = start.elapsed_time(end)
        module._acc_ms += elapsed_ms
        module._num_calls += 1

        return out
    
    module.forward = timed_forward


# ---------------------------
# Warmup + measurement loop
# ---------------------------

def warmup_and_measure(schedule: ScheduleGPipe,
                       model: BertModel,
                       device: torch.device,
                       global_batch_size: int,
                       chunks: int,
                       steps_warmup: int,
                       steps_measure: int):
    """
    TODO (Step 1, Step 2, Step 3): implement full measurement.

    Requirements:
      - Assert global_batch_size % chunks == 0 (integer microbatch size).
      - Warm up for `steps_warmup` steps (do not time).
      - Measure for `steps_measure` steps:
          * Generate inputs each step on rank 0's device
          * Rank 0: schedule.step(**inputs); other ranks: schedule.step()
          * Use dist.barrier() + torch.cuda.synchronize() before/after timing
      - Throughput (samples/s) = (global_batch_size * steps_measure) / elapsed_seconds
      - Peak memory per rank via torch.cuda.max_memory_allocated(rank)
      - Return: (throughput: float, peaks_bytes: list[int])

    Hints:
      - Reset memory stats before timing on each rank: torch.cuda.reset_peak_memory_stats(rank)
      - Use time.time() wall clock for elapsed_seconds
    """
    # >>> YOUR CODE HERE <<<
    assert global_batch_size % chunks == 0, "global_batch_size must be divisible by chunks"
    # microbatch_size = global_batch_size // chunks
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Warmup
    for _ in range(steps_warmup):
        if rank == 0:
            # Generate FULL (global) inputs on rank 0 so ScheduleGPipe can
            # split into `chunks` microbatches internally.
            inputs = generate_inputs_for_model(BertModel, model, "BertModel",
                                               global_batch_size, device)
            safe_schedule_step(schedule, **inputs)
        else:
            safe_schedule_step(schedule)

    dist.barrier()
    torch.cuda.synchronize(device)

    print(f"[Rank {rank}] Starting measurement...")
    # Reset peak memory stats on the correct CUDA device (if available)
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    dist.barrier()
    torch.cuda.synchronize(device)

    start_time = time.time()
    for step_idx in range(steps_measure):
        if rank == 0:
            # Generate inputs on rank 0
            inputs = generate_inputs_for_model(BertModel, model, "BertModel",
                                               global_batch_size, device)
            if step_idx == 0:
                # Log shapes once at start of measurement
                try:
                    shapes = {k: tuple(v.shape) for k, v in inputs.items()}
                except Exception:
                    shapes = {k: str(type(v)) for k, v in inputs.items()}
                log(f"Measure step 0 inputs shapes: {shapes}; requested chunks={chunks}")
            safe_schedule_step(schedule, **inputs)
        else:
            safe_schedule_step(schedule)

    dist.barrier()
    torch.cuda.synchronize(device)
    
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    total_samples = global_batch_size * steps_measure
    throughput = total_samples / elapsed_seconds
    
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    
    peaks_bytes = [0] * world_size
    if torch.cuda.is_available():
        peak_tensor = torch.tensor([peak_memory_bytes], dtype=torch.long, device=device)
        gathered = [torch.zeros_like(peak_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, peak_tensor)
        
        peaks_bytes = [int(t.item()) for t in gathered]
    
    if rank == 0:
        print(f"[Rank {rank}] Measurement complete!")
        print(f"  Elapsed time: {elapsed_seconds:.2f} seconds")
        print(f"  Total samples: {total_samples}")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Peak memory per rank (MB): {[p / 1024**2 for p in peaks_bytes]}")
    
    return float(throughput), peaks_bytes

# ---------------------------
# One run for a given seed
# ---------------------------

def run_one_seed(args, pipe_ir, model):
    """
    TODO: If you want per-stage forward timing (Step 4), wrap your stage module first.

    Return a dict with at least:
      - 'seed': int
      - 'throughput_samples_per_s': float
      - 'mem_peak_per_rank_bytes': List[int]
      - (optional) 'stage_forward_time_ms_per_rank': List[float or None]
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log(f"run_one_seed: starting (seed={getattr(args, 'seed', None)})")
    # Build runtime schedule
    stage = pipe_ir.build_stage(args.rank, device=args.device)
    log("Built stage module for this rank")
    schedule = ScheduleGPipe(stage, args.chunks)

    # OPTIONAL: per-stage forward timing
    stage_module = pipe_ir.get_stage_module(rank)
    wrap_stage_forward_for_timing(stage_module, args.device)
    log("Wrapped stage forward for timing (if CUDA enabled)")

    log("Starting warmup and measurement")
    thr, peaks = warmup_and_measure(
        schedule=schedule,
        model=model,
        device=args.device,
        global_batch_size=args.global_batch_size,
        chunks=args.chunks,
        steps_warmup=args.steps_warmup,
        steps_measure=args.steps_measure,
    )

    log(f"Finished warmup_and_measure: throughput={thr:.2f} samples/s")

    # OPTIONAL: stage timing average per step if you wrapped forward
    stage_ms_local = getattr(stage_module, "_acc_ms", None)
    if stage_ms_local is not None:
        stage_ms_local = float(stage_ms_local) / args.steps_measure
    else:
        stage_ms_local = None

    # Gather per-rank peaks to rank 0
    # (Hint: use dist.all_gather_object to collect Python ints from all ranks)
    # >>> YOUR CODE HERE <<<  (replace the fake single-rank list below)
    peaks_all = peaks if isinstance(peaks, list) else [peaks]

    world_size = dist.get_world_size()
    gathered_stage_times = [None] * world_size
    dist.all_gather_object(gathered_stage_times, stage_ms_local)

    out = {
        "seed": args.seed,
        "throughput_samples_per_s": float(thr),
        "mem_peak_per_rank_bytes": peaks_all,
        "stage_forward_time_ms_per_rank": gathered_stage_times
    }
    return out

def run_multi_seed(args, pipe_ir, model):
    rank = dist.get_rank()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    if rank == 0:
        log(f"\n{'='*60}")
        log(f"Running {len(seeds)} seeds: {seeds}")
        log(f"{'='*60}")

    all_results = []

    for seed in seeds:
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Update seed in args
        args.seed = seed
        
        if rank == 0:
            log(f"\nRunning seed {seed}...")
        
        # Run one seed
        result = run_one_seed(args, pipe_ir, model)
        all_results.append(result)
        
        if rank == 0:
            log(f"  Throughput: {result['throughput_samples_per_s']:.2f} samples/s")
            log(f"  Peak memory (MB): {[p/1024**2 for p in result['mem_peak_per_rank_bytes']]}")
            if result['stage_forward_time_ms_per_rank'][rank] is not None:
                log(f"  Stage forward times (ms): {result['stage_forward_time_ms_per_rank']}")

    if rank == 0:
        throughputs = [r['throughput_samples_per_s'] for r in all_results]
        peak_memories = np.array([r['mem_peak_per_rank_bytes'] for r in all_results])
        
        # Aggregate stage times if available
        stage_times = []
        for r in all_results:
            stage_times.append(r['stage_forward_time_ms_per_rank'])
        stage_times = np.array(stage_times, dtype=float)  # Shape: (num_seeds, world_size)
        
        aggregated = {
            'num_runs': len(seeds),
            'seeds': seeds,
            'throughput_mean': float(np.mean(throughputs)),
            'throughput_std': float(np.std(throughputs)),
            'throughput_all': throughputs,
            'peak_memory_mean_per_rank_bytes': np.mean(peak_memories, axis=0).tolist(),
            'peak_memory_std_per_rank_bytes': np.std(peak_memories, axis=0).tolist(),
            'peak_memory_mean_per_rank_mb': (np.mean(peak_memories, axis=0) / 1024**2).tolist(),
            'peak_memory_std_per_rank_mb': (np.std(peak_memories, axis=0) / 1024**2).tolist(),
        }
        
        # Add stage timing statistics if available
        if not np.all(np.isnan(stage_times)):
            aggregated['stage_time_mean_per_rank_ms'] = np.nanmean(stage_times, axis=0).tolist()
            aggregated['stage_time_std_per_rank_ms'] = np.nanstd(stage_times, axis=0).tolist()
        
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS ({len(seeds)} seeds)")
        print(f"{'='*60}")
        print(f"Throughput: {aggregated['throughput_mean']:.2f} ± {aggregated['throughput_std']:.2f} samples/s")
        print(f"Peak Memory per rank (MB):")
        for i, (mean, std) in enumerate(zip(aggregated['peak_memory_mean_per_rank_mb'], 
                                            aggregated['peak_memory_std_per_rank_mb'])):
            print(f"  Rank {i}: {mean:.2f} ± {std:.2f} MB")
        
        if 'stage_time_mean_per_rank_ms' in aggregated:
            print(f"Stage Forward Time per rank (ms):")
            for i, (mean, std) in enumerate(zip(aggregated['stage_time_mean_per_rank_ms'],
                                                aggregated['stage_time_std_per_rank_ms'])):
                print(f"  Rank {i}: {mean:.2f} ± {std:.2f} ms")
        
        # Save to JSON if output file specified
        if hasattr(args, 'out') and args.out:
            output_data = {
                'config': {
                    'world_size': args.world_size,
                    'global_batch_size': args.global_batch_size,
                    'chunks': args.chunks,
                    'partition': args.partition,
                    'steps_warmup': args.steps_warmup,
                    'steps_measure': args.steps_measure,
                },
                'per_seed_results': all_results,
                'aggregated': aggregated,
            }
            
            with open(args.out, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nResults saved to: {args.out}")
    
    return all_results

# ---------------------------
# Main
# ---------------------------

def run(args):
    rank = args.rank
    world_size = args.world_size

    config = BertConfig()
    config.return_dict = False

    # Create model
    model = BertModel(config)
    model.to(args.device)
    model.eval()

    if rank == 0:
        print(model.config)
        print(f"Total params ≈ {get_number_of_params(model) // 10 ** 6}M")

    # Example microbatch
    assert args.global_batch_size % args.chunks == 0, "global_batch_size must be divisible by chunks"
    example_mb = generate_inputs_for_model(BertModel, model, "BertModel",
                                           args.global_batch_size // args.chunks, args.device)

    # Split points
    if args.partition == "even":
        split_spec = make_split_spec_even(model, world_size)
    else:
        split_spec = make_split_spec_custom(model, args.partition, world_size)

    if rank == 0:
        print("Split points:", list(split_spec.keys()))

    # Build pipeline IR
    pipe_ir = pipeline(
        model,
        mb_args=(),
        mb_kwargs=example_mb,
        split_spec=split_spec,
    )
    assert pipe_ir.num_stages == world_size, f"nstages={pipe_ir.num_stages} != world_size={world_size}"

    # Quick test (one step)
    if args.quick_demo:
        stage = pipe_ir.build_stage(args.rank, device=args.device)
        schedule = ScheduleGPipe(stage, args.chunks)
        full_inputs = generate_inputs_for_model(BertModel, model, "BertModel",
                                                args.global_batch_size, args.device)
        if rank == 0:
            safe_schedule_step(schedule, **full_inputs)
        else:
            _ = safe_schedule_step(schedule)
        dist.barrier()
        print(f"[Rank {rank}] quick_demo complete")
        return

    # ===== Multi-seed measurements (Step 1 / Step 2 / Step 3 / Step 4) =====
    # TODO: loop over args.seeds, set torch.manual_seed/torch.cuda.manual_seed_all,
    #       call run_one_seed each time, aggregate mean ± std on rank 0,
    #       and optionally save JSON to args.out for plotting.
    #
    # Hints:
    #   - Parse seeds from comma-separated string (e.g., "1,2,3").
    #   - On rank 0, compute mean/std for throughput; for memory, compute mean/std per rank.
    #   - If args.out is set, write a dict containing spec + per_seed + agg results.
    #
    # >>> YOUR CODE HERE <<<
    results = run_multi_seed(args, pipe_ir, model)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Experiment complete!")
        print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    p.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    p.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    p.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    p.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    p.add_argument('--global_batch_size', type=int, default=64)
    p.add_argument('--chunks', type=int, default=4)
    p.add_argument('--steps_warmup', type=int, default=20)
    p.add_argument('--steps_measure', type=int, default=100)
    p.add_argument('--partition', type=str, default="even",
                   help='either "even" or a custom pattern like "3-1-2-2" (Step 4)')
    p.add_argument('--seeds', type=str, default="1,2,3",
                   help='comma-separated list, e.g., "1,2,3"')
    p.add_argument('--out', type=str, default="",
                   help="optional JSON path to save results (for tables/plots)")
    p.add_argument('--quick_demo', action='store_true', help="run a single untimed step to sanity-check PiPPy")

    args = p.parse_args()

    # Configure root logger to print timestamps and level. Include rank later
    # in each message using the `log()` helper above.
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S')
    handler.setFormatter(fmt)
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)

    if args.cuda:
        local_rank = int(os.getenv("LOCAL_RANK", args.rank % max(1, torch.cuda.device_count())))
        dev_id = local_rank
        args.device = torch.device(f"cuda:{dev_id}")
        torch.cuda.set_device(dev_id)
    else:
        args.device = torch.device("cpu")

    backend = "nccl" if args.cuda else "gloo"
    # Do not pass `device_id` here for compatibility; setting the CUDA device
    # above is sufficient and avoids NCCL guessing warnings/hangs.
    dist.init_process_group(backend=backend, rank=args.rank, world_size=args.world_size)

    try:
        run(args)
    except Exception:
        # Log full traceback and attempt a clean shutdown of the process group
        log("Unhandled exception in run():\n" + traceback.format_exc(), level='error')
        try:
            if dist.is_initialized():
                if hasattr(dist, 'abort'):
                    log("Calling dist.abort() after unhandled exception", level='error')
                    dist.abort()
                else:
                    log("Destroying process group after unhandled exception", level='error')
                    dist.destroy_process_group()
        except Exception:
            log("Error cleaning up process group after unhandled exception:\n" + traceback.format_exc(), level='error')
        # Ensure process exits with non-zero code
        sys.exit(1)
    finally:
        # Only barrier/destroy if the process group is still active
        try:
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            # Log and ignore cleanup errors
            log("Error during final cleanup:\n" + traceback.format_exc(), level='warning')


if __name__ == "__main__":
    main()