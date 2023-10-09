# We use 6 x 8 = 48 A100-80GB GPUs
# salloc --nodes 6 --time 24:00:00 --gres=gpu:80g:8 srun bash scripts/train_ppo_model_70b_qlora_salmon.sh

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/your/model/dir"
export DATA_DIR="/your/data/dir"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1 # Not sure if this is needed

LEARNING_RATE=2e-5
KL_COEF=0.01
EPOCH=2
ROLLOUT_BATCH_SIZE=576
STEP_BATCH_SZIE=288
ROLLOUT_PER_DEVICE_BATCH_SIZE=6
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=3
STEP_PER_DEVICE_BATCH_SIZE=6

JOB_ID=29400
NUM_NODES=6
NUM_TRAINERS=8

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n 1)

HOST_NODE_ADDR="$MASTER_ADDR:29400"

# SALMON-SPECIFIC HYPER-PARAMETERS
MAX_PRINCIPLES=3
UNFINISHED_PENALTY=-6.0
LENGTH_BONUS=5.0
LENGTH_BONUS_UPPER_BOUND=0.8
SAMPLING_TEMPARATURE=0.7

cd ..

torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_TRAINERS \
    --rdzv-id=$JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$HOST_NODE_ADDR \
    finetune_qlora_ppo.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --step_batch_size $STEP_BATCH_SZIE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --policy_model_name_or_path "$MODEL_DIR/dromedary-2-70b-sft-qlora/adapter_model" \
    --reward_model_name_or_path "$MODEL_DIR/llama-2-70b-qlora-rm-pmp-sft/adapter_model" \
    --learning_rate $LEARNING_RATE \
    --init_value_with_reward True \
    --warmup_steps 5 \
    --dataset_path "$DATA_DIR/salmon_merged_prompts.json" \
    --train_splits "train" \
    --policy_meta_prompt_pattern "../prompts/dromedary_inference_prompt.txt" \
    --reward_meta_prompt_pattern "../prompts/salmon_reward_model_prompt_v1.txt" \
    --principle_collection_path "../prompts/principles/principle_collection_ppo.json" \
    --max_principles $MAX_PRINCIPLES \
    --stop_token "\n\n### User" \
    --output_dir "$NEW_MODEL_DIR/dromedary-2-70b-ppo-qlora-salmon" \
    --total_epochs $EPOCH \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 100000 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --penalty_reward_value $UNFINISHED_REWARD_PENALTY \
    --length_bonus_score $LENGTH_BONUS \
    --length_bonus_upper_bound $LENGTH_BONUS_UPPER_BOUND \
    --relative_stop_token_penalty True \
    --penalize_no_stop_token True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --kl_coef $KL_COEF \
    --max_grad_norm 1.0 \
    --whitening_async_stats "full_batch" \
    --clean_tokens_after_eos True \
    --whiten_rewards False \
    --query_len 256 \
    --response_len 1024 \
    --model_max_length 1664 \
    --enable_reasoning_principles True \
    --enable_redteaming_principles True \
    --temperature $SAMPLING_TEMPARATURE \
    --noptepochs 2
