#!/bin/bash
MODEL_TYPE=llama3-8b
NOW=$(date -d now '+%Y%m%d-%H%M%S')

PRETRAIN_DIR=$MODEL_TYPE-pretrain
OUTPUT_DIR=lora-$MODEL_TYPE-$NOW

mkdir -p ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR

cd ./src
deepspeed \
    --num_gpus $MLP_WORKER_GPU \
    --num_nodes $MLP_WORKER_NUM \
    --hostfile $MLP_MPI_HOSTFILE \
    src/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 \
    --unfreeze_vision_tower False \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path ./pre-model/llama3-v-1_0 \
    --model_type $MODEL_TYPE \
    --version llama \
    --data_path ./data/train_EM_VLM4AD_llama.json \
    --vision_tower ./pre-model/llama3-v-1_0/siglip-so400m-patch14-384 \
    --image_folder /tmp/algorithm/src \
    --mm_projector_type mlp2x_gelu \
    --mm_interactor_type ca_interactor \
    --tune_mm_mlp_adapter True \
    --is_multiple_images True \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /tmp/model/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 /tmp/model/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt