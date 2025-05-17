

# export HF_ENDPOINT=https://hf-mirror.com ## if you have no vpn
export HF_HOME=/ssdwork/huyang/r1  ## set to yours huggingface home

export model_path=/ssdwork/huyang/r1/simple_GRPO_debug/simple_grpo_v1/models/Qwen2.5-7B
# export model_path=Qwen/Qwen2.5-7B ## or your local path to Qwen2.5-7B


## Baseline
export times=0; python eval_only_slot.py \
    --model_path $model_path 



## SLOT with iters=5
export times=5; export lr=0.1; python eval_only_slot.py \
    --model_path $model_path 


