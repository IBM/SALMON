# We use 1 x 8 = 8 A100-80GB GPUs
# salloc --nodes 1 --time 24:00:00 --gres=gpu:80g:8 srun bash scripts/generate_synthetic_preference.sh

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/your/model/dir"
export DATA_DIR="/your/data/dir"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    synthetic_preference.py \
    --model_name "$MODEL_DIR/llama-2-70b-hf" \
    --adapters_name "$MODEL_DIR/dromedary-2-70b-sft-qlora/adapter_model" \
    --preferece_prompt "../prompts/synthetic_preference_prompt.txt" \
    --rm_principles "../prompts/principles/principle_collection_rm.json" \
    --response_pattern "$DATA_DIR/oasst1_dromedary2_sft_response*.json" \
    --output_file "$DATA_DIR/oasst1_dromedary2_sft_preference.json"
