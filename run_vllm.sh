export VLLM_WORKER_MULTIPROC_METHOD=spawn 
MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B ## refer to table.2 for more models
export tensor_parallel_size_my=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/$MODEL_NAME
OUTPUT_DIR=data/evals/$MODEL
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$tensor_parallel_size_my,data_parallel_size=1,max_model_length=32768,gpu_memory_utilization=0.90,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

TASK="aime24" # "aime24", "math_500", "gpqa:diamond"

## baseline, the result should be 56.6
export CHOT_STEPS=0
export CHOT_LR=0.1
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR > $TASK-$MODEL_NAME-$CHOT_STEPS-$CHOT_LR.log 2>&1 

## with slot, the result should be 66.6 (+10%)
export CHOT_STEPS=1
export CHOT_LR=0.1
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR > $TASK-$MODEL_NAME-$CHOT_STEPS-$CHOT_LR.log 2>&1 
