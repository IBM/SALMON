# We use 8 x 6 = 48 V100-32GB GPUs
# On AiMOS cluster [https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/]
# salloc --nodes 8 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/generate_oasst1_response1.sh

set -e
set -x

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export MODEL_DIR="/your/model/dir"
export DATA_DIR="/your/data/dir"
export OMP_NUM_THREADS=6
export GPUS_PER_NODE=6
export NUM_NODES=2
export MASTER_PORT=9901

LOCAL_NODE_RANK=$((SLURM_PROCID % NUM_NODES))
GROUP_RANK=$((SLURM_PROCID / NUM_NODES))
GROUP_SIZE=$((SLURM_NNODES / NUM_NODES))
SYNC_NODE_RANK=$((GROUP_RANK * NUM_NODES))

# MASTER_ADDR should be SYNC_NODE_RANK-th node in $(scontrol show hostnames $SLURM_JOB_NODELIST)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)

echo "$MASTER_ADDR, $GROUP_RANK / $GROUP_SIZE: $LOCAL_NODE_RANK"

torchrun --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $LOCAL_NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    batch_generation.py \
    --ckpt_dir $MODEL_DIR/dromedary-2-70b-sft-12shard \
    --tokenizer_path $MODEL_DIR/tokenizer.model \
    --generate_max_len 768 \
    --max_seq_len 768 \
    --max_shared_seq_len 640 \
    --max_batch_size 64 \
    --group_rank $GROUP_RANK \
    --group_size $GROUP_SIZE \
    --input_file "$DATA_DIR/oasst1_prompts.json" \
    --output_file "$DATA_DIR/oasst1_dromedary2_sft_response1.json" \
    --meta_prompt_file "../prompts/synthetic_inference_prompts/dromedary_inference_prompt.txt" \
    --temperature 0.7 \
    --top_p 1.0 \
    --seed 43
