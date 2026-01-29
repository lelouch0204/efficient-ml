for mode in "ddp" "fsdp_full"; do
  echo ">>> Running Step 3: chunks=4 (4 GPUs, even split)"
  torchrun --standalone --nproc-per-node=4 fsdp.py \
    --world_size 4 --global_batch_size 64 \
    --seeds 1,2,3 --mode ${mode} \
    --steps_warmup 20 --steps_measure 100 \
    --out runs_task4/results_${mode}.json
done

# echo ">>> Running Step 3: chunks=4 (4 GPUs, even split)"
#   torchrun --standalone --nproc-per-node=4 pipeline.py \
#     --world_size 4 --global_batch_size 64 --chunks 4 \
#     --partition "6-2-2-2" --seeds 1,2,3 \
#     --steps_warmup 20 --steps_measure 100 \
#     --out runs/results_step1_gpu4_chunks_4_uneven.json