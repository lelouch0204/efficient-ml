# for num_gpus in 2 3 4; do
#   echo ">>> Running Step 3: chunks=4 (${num_gpus} GPUs, even split)"
#   torchrun --standalone --nproc-per-node=${num_gpus} pipeline.py \
#     --world_size ${num_gpus} --global_batch_size 64 --chunks 4 \
#     --partition even --seeds 1,2,3 \
#     --steps_warmup 20 --steps_measure 100 \
#     --out runs_l40/results_step1_gpu${num_gpus}_chunks_4.json
# done

echo ">>> Running Step 3: chunks=4 (4 GPUs, even split)"
  torchrun --standalone --nproc-per-node=4 pipeline.py \
    --world_size 4 --global_batch_size 64 --chunks 4 \
    --partition "6-2-2-2" --seeds 1,2,3 \
    --steps_warmup 20 --steps_measure 100 \
    --out runs/results_step1_gpu4_chunks_4_uneven.json