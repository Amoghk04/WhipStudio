#!/usr/bin/env python3
"""
Improved GRPO training script for WhipStudio ML Debug Environment.

This script trains Qwen2.5-1.5B-Coder (or similar) to debug broken PyTorch scripts
using Group Relative Policy Optimization (GRPO) with the WhipStudio environment
as the reward oracle.

Improvements over basic train_grpo.py:
1. Memory-efficient training with 4-bit quantization
2. LoRA fine-tuning for reduced VRAM usage
3. Curriculum learning (easier tasks first)
4. Gradient checkpointing for large contexts
5. Checkpoint saving with best model tracking
6. Early stopping based on validation scores
7. Wandb/TensorBoard logging support

Requirements:
    pip install trl>=0.15.0 transformers>=4.46.0 datasets torch httpx
    pip install accelerate peft bitsandbytes wandb

Usage:
    # Basic training
    python improved_agent.py \
        --env_url https://your-space.hf.space \
        --output_dir ./whipstudio-debugger

    # Memory-efficient training (8GB VRAM)
    python improved_agent.py \
        --env_url https://your-space.hf.space \
        --use_4bit \
        --use_lora \
        --gradient_checkpointing \
        --output_dir ./whipstudio-debugger-lora

    # Full training with wandb logging
    python improved_agent.py \
        --env_url https://your-space.hf.space \
        --use_wandb \
        --wandb_project whipstudio \
        --num_iterations 100 \
        --output_dir ./whipstudio-debugger
"""

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# TRL imports
try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    raise ImportError("Please install trl>=0.15.0: pip install trl")

# PEFT imports (optional)
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs.
Return ONLY the complete corrected Python code. No markdown, no backticks, no explanation.
The script must print metrics in the format specified by the task description.
Keep all torch.manual_seed() calls intact.
Wrap metrics in ##METRICS_START## and ##METRICS_END## markers."""

# Task ordering by difficulty for curriculum learning
TASK_DIFFICULTY = {
    "task1": 1,  # Easy: broken loop
    "task4": 2,  # Medium: wrong loss
    "task5": 2,  # Medium: frozen backbone
    "task2": 3,  # Medium: NaN loss (tricky)
    "task3": 4,  # Hard: OOM + leakage
}

ALL_TASKS = list(TASK_DIFFICULTY.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Environment Client
# ══════════════════════════════════════════════════════════════════════════════

class WhipStudioEnv:
    """Client for the WhipStudio RL environment."""

    def __init__(self, env_url: str, timeout: float = 180.0):
        self.env_url = env_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout, connect=15.0)
        self._task_cache: dict[str, dict] = {}

    def reset(self, task_id: str) -> dict:
        """Reset environment and return observation."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.env_url}/reset", json={"task_id": task_id})
            resp.raise_for_status()
            data = resp.json()
            obs = data.get("observation", data)
            self._task_cache[task_id] = obs
            return obs

    def step(self, fixed_code: str, attempt: int = 1) -> dict:
        """Submit a fix and return the full step result."""
        payload = {
            "action": {
                "fixed_code": fixed_code,
                "attempt_number": attempt,
            }
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.env_url}/step", json=payload)
            resp.raise_for_status()
            return resp.json()

    def get_task_obs(self, task_id: str) -> dict:
        """Get cached observation or reset to obtain it."""
        if task_id not in self._task_cache:
            self.reset(task_id)
        return self._task_cache[task_id]

    def health_check(self) -> bool:
        """Verify the environment is reachable."""
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                resp = client.get(f"{self.env_url}/health")
                return resp.status_code == 200
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Prompt Utilities
# ══════════════════════════════════════════════════════════════════════════════

def build_user_prompt(task_description: str, buggy_code: str) -> str:
    """Build the user prompt for the model."""
    return f"Task: {task_description}\n\nBuggy code:\n{buggy_code}"


def format_chat(tokenizer: Any, user_prompt: str) -> str:
    """Format as a chat message and return the full text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_code_from_response(response: str) -> str:
    """Extract Python code from model response, stripping markdown if present."""
    text = response.strip()
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Reward Function
# ══════════════════════════════════════════════════════════════════════════════

def create_reward_function(env: WhipStudioEnv, verbose: bool = True):
    """
    Create a reward function compatible with TRL's GRPOTrainer.
    
    Includes reward shaping:
    - Bonus for valid Python syntax
    - Bonus for including required output markers
    - Environment reward from grader
    """

    def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
        """Compute rewards for a batch of completions."""
        rewards = []
        task_ids = kwargs.get("task_id", ["task1"] * len(completions))

        for i, completion in enumerate(completions):
            task_id = task_ids[i] if i < len(task_ids) else "task1"

            try:
                # Extract assistant's response
                if isinstance(completion, list):
                    text = ""
                    for msg in completion:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            text = msg.get("content", "")
                            break
                    if not text and completion:
                        text = str(completion[-1].get("content", ""))
                elif isinstance(completion, str):
                    text = completion
                else:
                    text = str(completion)

                fixed_code = extract_code_from_response(text)
                
                # Reward shaping: syntax check
                syntax_bonus = 0.0
                try:
                    compile(fixed_code, "<string>", "exec")
                    syntax_bonus = 0.05
                except SyntaxError:
                    pass
                
                # Reward shaping: output markers present
                marker_bonus = 0.0
                if "LOSSES:" in fixed_code or "##METRICS" in fixed_code:
                    marker_bonus = 0.02

                if not fixed_code.strip():
                    rewards.append(0.0)
                    continue

                # Get environment reward
                env.reset(task_id)
                result = env.step(fixed_code, attempt=1)
                env_reward = float(result.get("reward", 0.0) or 0.0)
                
                # Total reward (capped at 1.0)
                total_reward = min(1.0, env_reward + syntax_bonus + marker_bonus)
                rewards.append(total_reward)

                if verbose:
                    print(f"  [reward] task={task_id} env={env_reward:.3f} syntax={syntax_bonus:.2f} total={total_reward:.3f}")

            except Exception as e:
                if verbose:
                    print(f"  [reward] ERROR task={task_id}: {e}")
                rewards.append(0.0)

        return rewards

    return reward_fn


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Generation with Curriculum
# ══════════════════════════════════════════════════════════════════════════════

def generate_curriculum_dataset(
    env: WhipStudioEnv,
    tokenizer: Any,
    samples_per_task: int = 10,
    curriculum_stage: int = 0,  # 0 = all tasks, 1 = easier tasks weighted, etc.
) -> Dataset:
    """
    Generate a dataset with curriculum-based sampling.
    
    Args:
        env: WhipStudio environment client
        tokenizer: Model tokenizer
        samples_per_task: Base samples per task
        curriculum_stage: 0=uniform, higher=bias toward easier tasks
    """
    records = []
    
    # Compute task weights based on curriculum stage
    task_weights = {}
    for task_id, difficulty in TASK_DIFFICULTY.items():
        if curriculum_stage == 0:
            weight = 1.0
        else:
            # Higher curriculum_stage = more weight on easier tasks
            weight = max(0.2, 1.0 - (difficulty - 1) * 0.2 * curriculum_stage)
        task_weights[task_id] = weight
    
    # Normalize weights
    total_weight = sum(task_weights.values())
    task_weights = {k: v / total_weight for k, v in task_weights.items()}

    for task_id in ALL_TASKS:
        print(f"  Fetching observation for {task_id} (weight={task_weights[task_id]:.2f})...")
        obs = env.reset(task_id)
        user_prompt = build_user_prompt(
            task_description=obs.get("task_description", ""),
            buggy_code=obs.get("buggy_code", ""),
        )
        formatted = format_chat(tokenizer, user_prompt)
        
        # Number of samples proportional to weight
        n_samples = max(1, int(samples_per_task * task_weights[task_id] * len(ALL_TASKS)))
        
        for _ in range(n_samples):
            records.append({
                "prompt": formatted,
                "task_id": task_id,
            })

    random.shuffle(records)
    return Dataset.from_list(records)


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading Utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
    use_8bit: bool = False,
    gradient_checkpointing: bool = False,
):
    """Load model with optional quantization and gradient checkpointing."""
    
    print(f"Loading model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config
    quantization_config = None
    if use_4bit:
        if not PEFT_AVAILABLE:
            raise ImportError("4-bit quantization requires peft and bitsandbytes")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit quantization")
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("  Using 8-bit quantization")
    
    # Model kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if not (use_4bit or use_8bit) else None,
        "device_map": "auto",
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Prepare for k-bit training if quantized
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {param_count / 1e6:.1f}M, Trainable: {trainable / 1e6:.1f}M")
    
    return model, tokenizer


def apply_lora(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list[str]] = None,
):
    """Apply LoRA adapters to the model."""
    if not PEFT_AVAILABLE:
        raise ImportError("LoRA requires peft: pip install peft")
    
    if target_modules is None:
        # Default targets for Qwen2 and similar architectures
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA applied: r={lora_r}, trainable params: {trainable / 1e6:.2f}M")
    
    return model, lora_config


# ══════════════════════════════════════════════════════════════════════════════
# Validation & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    tokenizer,
    env: WhipStudioEnv,
    task_ids: list[str] = None,
    max_new_tokens: int = 2048,
) -> dict[str, float]:
    """Evaluate model on tasks and return scores."""
    if task_ids is None:
        task_ids = ALL_TASKS
    
    model.eval()
    scores = {}
    
    for task_id in task_ids:
        obs = env.reset(task_id)
        user_prompt = build_user_prompt(obs["task_description"], obs["buggy_code"])
        formatted = format_chat(tokenizer, user_prompt)
        
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        fixed_code = extract_code_from_response(response)
        
        env.reset(task_id)
        result = env.step(fixed_code, attempt=1)
        reward = float(result.get("reward", 0.0) or 0.0)
        scores[task_id] = reward
        print(f"    {task_id}: {reward:.4f}")
    
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Improved GRPO training for WhipStudio")
    
    # Environment
    parser.add_argument("--env_url", type=str, required=True,
                        help="URL of the WhipStudio HF Space")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./whipstudio-debugger",
                        help="Directory to save the trained model")
    
    # Quantization & Memory
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    
    # Training
    parser.add_argument("--num_iterations", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--group_size", type=int, default=4,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--samples_per_task", type=int, default=10,
                        help="Base samples per task in dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens to generate per completion")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    
    # Curriculum
    parser.add_argument("--curriculum_stages", type=int, default=3,
                        help="Number of curriculum stages (0 = no curriculum)")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="whipstudio",
                        help="W&B project name")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    
    # Hub
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push trained model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID on HF Hub")
    
    args = parser.parse_args()

    # ── Verify environment ──
    print(f"\n{'=' * 60}")
    print("WhipStudio Improved GRPO Training")
    print(f"{'=' * 60}")
    print(f"Environment: {args.env_url}")
    
    env = WhipStudioEnv(args.env_url)
    if not env.health_check():
        raise ConnectionError(f"Cannot reach WhipStudio at {args.env_url}")
    print("Environment is reachable ✓")

    # ── Initialize wandb ──
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed, skipping logging")
            args.use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"grpo-{args.model_name.split('/')[-1]}",
            )

    # ── Load model ──
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # ── Apply LoRA ──
    peft_config = None
    if args.use_lora:
        model, peft_config = apply_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )

    # ── Create output directory ──
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ── Training with curriculum ──
    best_avg_score = 0.0
    epochs_without_improvement = 0
    
    n_stages = max(1, args.curriculum_stages)
    epochs_per_stage = args.num_iterations // n_stages
    
    for stage in range(n_stages):
        print(f"\n{'=' * 60}")
        print(f"Curriculum Stage {stage + 1}/{n_stages}")
        print(f"{'=' * 60}")
        
        # Generate dataset for this curriculum stage
        dataset = generate_curriculum_dataset(
            env, tokenizer,
            samples_per_task=args.samples_per_task,
            curriculum_stage=stage,
        )
        print(f"Dataset: {len(dataset)} samples")
        
        # Create reward function
        reward_fn = create_reward_function(env, verbose=True)
        
        # Configure GRPO
        grpo_config = GRPOConfig(
            output_dir=str(output_path / f"stage_{stage}"),
            num_train_epochs=epochs_per_stage,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_completion_length=args.max_new_tokens,
            num_generations=args.group_size,
            logging_steps=1,
            save_steps=epochs_per_stage,
            save_total_limit=2,
            bf16=True,
            report_to="wandb" if args.use_wandb else "none",
            beta=args.beta,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
            peft_config=peft_config if stage == 0 else None,  # Only apply peft on first stage
        )
        
        # Train
        print(f"\nTraining stage {stage + 1}...")
        train_result = trainer.train()
        print(f"  Stage {stage + 1} complete: {train_result.global_step} steps")
        
        # Evaluate
        print("\nEvaluating...")
        scores = evaluate_model(model, tokenizer, env)
        avg_score = sum(scores.values()) / len(scores)
        print(f"  Average score: {avg_score:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "stage": stage + 1,
                "avg_score": avg_score,
                **{f"score/{k}": v for k, v in scores.items()},
            })
        
        # Track best model
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            epochs_without_improvement = 0
            # Save best model
            best_path = output_path / "best"
            trainer.save_model(str(best_path))
            tokenizer.save_pretrained(str(best_path))
            print(f"  New best model saved (score={avg_score:.4f})")
        else:
            epochs_without_improvement += epochs_per_stage
        
        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    # ── Final save ──
    final_path = output_path / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nFinal model saved to {final_path}")

    # ── Push to hub ──
    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing to Hub as {args.hub_model_id}...")
        trainer.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print("Pushed to Hub ✓")

    # ── Final evaluation ──
    print(f"\n{'=' * 60}")
    print("Final Evaluation on All Tasks")
    print(f"{'=' * 60}")
    final_scores = evaluate_model(model, tokenizer, env)
    final_avg = sum(final_scores.values()) / len(final_scores)
    print(f"\nFinal average score: {final_avg:.4f}")
    print(f"Best average score during training: {best_avg_score:.4f}")

    if args.use_wandb:
        wandb.log({"final_avg_score": final_avg})
        wandb.finish()

    # ── Save training summary ──
    summary = {
        "model_name": args.model_name,
        "final_avg_score": final_avg,
        "best_avg_score": best_avg_score,
        "final_scores": final_scores,
        "curriculum_stages": n_stages,
        "use_lora": args.use_lora,
        "use_4bit": args.use_4bit,
    }
    with open(output_path / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete! ✓")


if __name__ == "__main__":
    main()
