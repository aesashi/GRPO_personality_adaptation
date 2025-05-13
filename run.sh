#!/bin/bash

# Configuration variables - adjust these as needed
MODEL_NAME="MKJ-TOE/Qwen2.5-7B-Instruct-addsptoken-v1.0"
DATA_PATH="grpo_train.jsonl" 
OUTPUT_DIR="output2" 

# Training parameters
MAX_SEQ_LENGTH=2048
LORA_RANK=64
LEARNING_RATE=5e-6
NUM_TRAIN_EPOCHS=1
MAX_STEPS=1241
BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
NUM_GENERATIONS=8
GPU_MEMORY_UTIL=0.9


# Run the training script
python grpo_train.py \
  --model_name $MODEL_NAME \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --max_seq_length $MAX_SEQ_LENGTH \
  --lora_rank $LORA_RANK \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --max_steps $MAX_STEPS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --num_generations $NUM_GENERATIONS \
  --gpu_memory_utilization $GPU_MEMORY_UTIL
