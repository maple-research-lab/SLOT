import os

from transformers import AutoTokenizer

import torch
import re
from datasets import load_dataset
import random
import argparse

from modeling_qwen2_glot import Qwen2ForCausalLM

def reward_correct(item, answer):
    from math_verify import parse, verify, ExprExtractionConfig
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0:
        return -1.0
    lastnum = nums[-1]
    
    ans_parsed = None
    ground_truth_parsed = None

    try:
        ans_parsed = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    except Exception as e:
        return -1.0

    try:
        ground_truth_parsed = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    except Exception as e:
        return -1.0
    
    if ans_parsed is None or ground_truth_parsed is None:
        return -1.0

    verification_result = verify(ans_parsed, ground_truth_parsed)
    result_score = 1.0 if verification_result else -1.0
    return result_score

def reward_format(item, answer):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    match_obj = re.match(pattern, answer, re.DOTALL) 
    result_score = 1.25 if match_obj else -1.0
    return result_score

def evaluate_model(model, tokenizer, eval_samples=None, split="test", generation_params=None, seed=42, log_file="evaluation_log.txt"):
    """Evaluates the model's performance on the GSM8K dataset."""
    print("Starting model evaluation...")
    model.eval()    
    random.seed(seed)
    
    # Load the evaluation dataset
    eval_dataset = load_dataset("openai/gsm8k", "main", split=split)
    eval_QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} 
                for x,y in zip(eval_dataset['question'], eval_dataset['answer'])]
    
    # Randomly select samples for evaluation if specified
    if eval_samples is not None and len(eval_QAs) > eval_samples:
        eval_QAs = random.sample(eval_QAs, eval_samples)
    
    # Print the actual number of samples being evaluated
    print(f"Evaluating {len(eval_QAs)} samples")
    
    # Append evaluation info to the log
    with open(log_file, "a") as f:
        f.write(f"Number of evaluation samples: {len(eval_QAs)}\n\n")
    
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
    
    correct = 0
    format_correct = 0
    total = len(eval_QAs)
    
    for i, qa in enumerate(eval_QAs):
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{total} samples")
            
        prompt = qa['Q']
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        
        os.environ["prompt_only"] = "True" # Ensure this env var is handled correctly if needed elsewhere
        outputs = model.generate(
            **inputs,
            **generation_params,
        )
        
        completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Check format and correctness
        format_score = reward_format(qa, completion)
        correct_score = reward_correct(qa, completion)
        
        is_format_correct = format_score > 0
        is_answer_correct = correct_score > 0
        
        if is_format_correct:
            format_correct += 1
        if is_answer_correct:
            correct += 1
            
        # Log sample information
        with open(log_file, "a") as f:
            f.write(f"Sample {i+1}:\n")
            f.write(f"Question: {qa['Q']}\n")
            f.write(f"Model Response: {completion}\n")
            f.write(f"Correct Answer: {qa['A']}\n")
            f.write(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}\n\n")
            
        # Print detailed information for every sample
        print(f"\n--- Sample {i+1} ---")
        print("Question:", qa['Q'])
        print("Model Response:", completion)
        print("Correct Answer:", qa['A'])
        print(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}")
    
    accuracy = correct / total if total > 0 else 0
    format_accuracy = format_correct / total if total > 0 else 0
    
    print(f"\nEvaluation Results (Samples: {total}):")
    print(f"Answer Accuracy: {accuracy:.4f}")
    print(f"Format Accuracy: {format_accuracy:.4f}")
    
    # Log overall results
    with open(log_file, "a") as f:
        f.write(f"Evaluation Results (Samples: {total}):\n")
        f.write(f"Answer Accuracy: {accuracy:.4f}\n")
        f.write(f"Format Accuracy: {format_accuracy:.4f}\n")
    
    return accuracy, format_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/ssdwork/huyang/r1/simple_GRPO_debug/slot_gsm8k/models/Qwen2.5-7B", help="Path to the model")
    parser.add_argument("--eval_samples", type=int, default=None, help="Number of samples to evaluate, None for full evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Dataset split to evaluate on")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consistent evaluation samples")
    args = parser.parse_args()
    # args.eval_samples = 30
    
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Ensure same model loading parameters as training if applicable
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa" # Use 'flash_attention_2' if available and preferred
    ).to("cuda") # Consider adding device management if multiple GPUs
    
    # Set generation parameters
    generation_params = {
        "do_sample": False,
        "temperature": args.temperature if args.do_sample else None,
        "max_new_tokens": 512 # Added a sensible default, adjust if needed
    }

    # Get environment variables (consider passing as args instead for clarity)
    times = os.environ.get("times", "3")
    lr = os.environ.get("lr", "0.001")
    
    # Create log directory and file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_times_{times}_lr_{lr}.txt")
    
    # Log basic information
    with open(log_file, "w") as f: # Use 'w' to overwrite for a new run
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Times (from env): {times}\n")
        f.write(f"LR (from env): {lr}\n")
        f.write(f"Eval Samples: {'All' if args.eval_samples is None else args.eval_samples}\n")
        f.write(f"Dataset Split: {args.split}\n")
        f.write(f"Do Sample: {args.do_sample}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Seed: {args.seed}\n\n")
    
    # Call evaluate_model, passing the log file path
    accuracy, format_accuracy = evaluate_model(
        model, 
        tokenizer, 
        eval_samples=args.eval_samples, 
        split=args.split, 
        generation_params=generation_params, 
        seed=args.seed,
        log_file=log_file # Pass log file path
    )
    
    # Log final results (already done inside evaluate_model, but can add a summary here if needed)
    print(f"Evaluation complete. Results logged to {log_file}")

if __name__ == "__main__":
    main()
