# We use 1 x 8 = 8 A100-80GB GPUs

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/your/model/dir"
export DATA_DIR="/your/data/dir"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

NUM_EPOCHS=1
LEARNING_RATE=5e-5
BATCH_SIZE=4
GRAD_ACCUMULATION=2

cd ..

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_qlora_rm.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path "$MODEL_DIR/llama-2-70b-hf" \
    --learning_rate $LEARNING_RATE \
    --model_max_length 1024 \
    --query_len 256 \
    --response_len 640 \
    --dataset_path "$DATA_DIR/pmp_data.json" \
    --meta_prompt_pattern "../prompts/pmp_reward_model_prompt.txt" \
    --double_quant True \
    --quant_type "nf4" \
    --bits 4 \
    --lora_r 64 \
    --output_dir "$MODEL_DIR/llama-2-70b-qlora-rm-pmp" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --weight_decay 0.0 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True
