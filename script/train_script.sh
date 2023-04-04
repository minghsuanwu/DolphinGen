#! /bin/bash

seg_max_length=512
MODEL_DIR=./pretraining/chatglm-6b
DATA_PATH=./data/zh_seed_tasks.json
OUTPUT_DIR=./result_chatglm

accelerate launch --config_file accelerate_config/example_config.yaml run.py \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--learning_rate 5e-5 \
	--max_length ${seg_max_length} \
	--model_name_or_path ${MODEL_DIR} \
	--dataset_path ${DATA_PATH} \
	--output_dir ${OUTPUT_DIR} \
	--num_warmup_steps 0 \
	--num_train_epochs 1 \
	--steps_to_log 50 \
	--use_lora \
	--lora_r 8 \
	--lora_alpha 32 \
	--lora_dropout 0.1 \
	--cuda_devices 0 \
	--seed 0 \
	--max_ckpts_to_keep 3 \
	--save_final

