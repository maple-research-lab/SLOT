# NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL_NAME=DeepSeek-R1-Distill-Llama-70B
export lm_local="./lm_head/${MODEL_NAME}_lm_head.pt"
export tensor_parallel_size_my=8
export data_parallel_size=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL=deepseek-ai/$MODEL_NAME
OUTPUT_DIR=data/evals/$MODEL
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$tensor_parallel_size_my,data_parallel_size=$data_parallel_size,max_model_length=32768,max_num_batched_tokens=32768,gpu_memory_utilization=0.90,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

TASK="aime24" # "math_500", "gpqa:diamond"
export CHOT_STEPS=5
export CHOT_LR=0.1
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR > $TASK-$MODEL_NAME-$CHOT_STEPS-$CHOT_LR.log 2>&1 
