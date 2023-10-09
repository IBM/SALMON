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
LEARNING_RATE=3e-5
BATCH_SIZE=4
GRAD_ACCUMULATION=2

PMP_CKPT=llama-2-70b-qlora-rm-pmp/checkpoint-xxxx

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
    --model_max_length 1280 \
    --query_len 128 \
    --response_len 768 \
    --dataset_path "$DATA_DIR/oasst1_dromedary2_sft_aggregated_preference.json" \
    --principle_collection_path "../principles/principle_collection_rm.json" \
    --meta_prompt_pattern "../prompts/salmon_reward_model_prompt_v0.txt" \
    --double_quant True \
    --quant_type "nf4" \
    --bits 4 \
    --lora_r 64 \
    --output_dir "$MODEL_DIR/llama-2-70b-qlora-rm-pmp-sft" \
    --resume_dir "$MODEL_DIR/$PMP_CKPT" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 40 \
    --save_total_limit 20 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False
