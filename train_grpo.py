"""
GRPO training script for WhipStudio ML Debug environment.

Trains Qwen2.5-1.5B-Coder to debug broken PyTorch scripts
by interacting with the WhipStudio HF Space as a reward oracle.

Requirements:
    pip install trl>=0.15.0 transformers>=4.46.0 datasets torch httpx accelerate peft bitsandbytes

Usage:
    python train_grpo.py \
        --env_url https://your-space.hf.space \
        --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --output_dir ./whipstudio-debugger-1.5b \
        --num_iterations 50 \
        --group_size 4 \
        --learning_rate 1e-5
"""

import argparse
import json
import os
import random
import re
import time
from typing import Any

import httpx
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Constants ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs.
Return ONLY the complete corrected Python code. No markdown, no backticks, no explanation.
The script must print metrics in the format specified by the task description.
Keep all torch.manual_seed() calls intact."""

TASK_IDS = ["task1", "task2", "task3", "task4", "task5"]


# ── Environment interaction ────────────────────────────────────────────────

class WhipStudioEnv:
    """Client for the WhipStudio RL environment running on HF Spaces."""

    def __init__(self, env_url: str, timeout: float = 120.0):
        self.env_url = env_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout, connect=15.0)
        # Cache task observations so we don't reset before every reward call
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


# ── Prompt construction ────────────────────────────────────────────────────

def build_user_prompt(task_description: str, buggy_code: str) -> str:
    """Build the user prompt for the model."""
    return f"Task: {task_description}\n\nBuggy code:\n{buggy_code}"


def format_chat(tokenizer: Any, user_prompt: str) -> str:
    """Format as a chat message and return the full text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Code extraction ───────────────────────────────────────────────────────

def extract_code_from_response(response: str) -> str:
    """Extract Python code from model response, stripping markdown if present."""
    text = response.strip()
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


# ── Reward function ───────────────────────────────────────────────────────

def create_reward_function(env: WhipStudioEnv):
    """
    Create a reward function compatible with TRL's GRPOTrainer.

    TRL calls this with a list of completions (and optionally prompts).
    We submit each completion to the WhipStudio environment and return
    the reward scores.
    """

    def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            completions: List of conversation-style completions from GRPOTrainer.
                         Each element is a list of message dicts.
            **kwargs: May contain 'task_id' list matching each completion.

        Returns:
            List of float rewards in [0.0, 1.0].
        """
        rewards = []
        task_ids = kwargs.get("task_id", ["task1"] * len(completions))

        for i, completion in enumerate(completions):
            task_id = task_ids[i] if i < len(task_ids) else "task1"

            try:
                # Extract the assistant's response text from the completion
                if isinstance(completion, list):
                    # conversation format: [{"role": "assistant", "content": "..."}]
                    text = ""
                    for msg in completion:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            text = msg.get("content", "")
                            break
                    if not text:
                        text = str(completion[-1].get("content", "")) if completion else ""
                elif isinstance(completion, str):
                    text = completion
                else:
                    text = str(completion)

                fixed_code = extract_code_from_response(text)
                if not fixed_code.strip():
                    rewards.append(0.0)
                    continue

                # Reset env for this task and submit the code
                env.reset(task_id)
                result = env.step(fixed_code, attempt=1)
                reward = float(result.get("reward", 0.0) or 0.0)
                rewards.append(reward)

                print(f"  [reward] task={task_id} reward={reward:.4f} len={len(fixed_code)}")

            except Exception as e:
                print(f"  [reward] ERROR task={task_id}: {e}")
                rewards.append(0.0)

        return rewards

    return reward_fn


# ── Dataset generation ─────────────────────────────────────────────────────

def generate_training_dataset(env: WhipStudioEnv, tokenizer: Any, samples_per_task: int = 10) -> Dataset:
    """
    Generate a dataset of prompts by resetting the environment for each task.

    Since the tasks are deterministic (fixed seeds), we repeat each task
    multiple times — the model will generate different completions each time
    (that's the exploration in GRPO).

    Each record has:
    - prompt: The formatted chat prompt (ready for tokenization)
    - task_id: Which task this corresponds to (used by reward function)
    """
    records = []

    for task_id in TASK_IDS:
        print(f"  Fetching observation for {task_id}...")
        obs = env.reset(task_id)
        user_prompt = build_user_prompt(
            task_description=obs.get("task_description", ""),
            buggy_code=obs.get("buggy_code", ""),
        )
        formatted = format_chat(tokenizer, user_prompt)

        for _ in range(samples_per_task):
            records.append({
                "prompt": formatted,
                "task_id": task_id,
            })

    random.shuffle(records)
    return Dataset.from_list(records)


# ── Main training loop ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training for WhipStudio ML Debugger")
    parser.add_argument("--env_url", type=str, required=True,
                        help="URL of the WhipStudio HF Space (e.g., https://your-space.hf.space)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./whipstudio-debugger-1.5b",
                        help="Directory to save the trained model")
    parser.add_argument("--num_iterations", type=int, default=50,
                        help="Number of GRPO training epochs")
    parser.add_argument("--group_size", type=int, default=4,
                        help="Number of completions per prompt for GRPO (G in the paper)")
    parser.add_argument("--samples_per_task", type=int, default=10,
                        help="Number of prompt copies per task in the dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens the model can generate per completion")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient for GRPO")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the trained model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID on HF Hub (e.g., your-username/whipstudio-debugger)")
    parser.add_argument("--use_peft", action="store_true",
                        help="Use LoRA (PEFT) for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (only used with --use_peft)")
    args = parser.parse_args()

    # ── Verify environment ──
    print(f"Connecting to WhipStudio environment at {args.env_url}")
    env = WhipStudioEnv(args.env_url)
    if not env.health_check():
        raise ConnectionError(f"Cannot reach WhipStudio environment at {args.env_url}")
    print("Environment is reachable")

    # ── Load model and tokenizer ──
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({param_count / 1e6:.0f}M params)")

    # ── Optional LoRA ──
    peft_config = None
    if args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"Using LoRA with rank={args.lora_r}")

    # ── Generate training dataset ──
    print("Generating training prompts...")
    dataset = generate_training_dataset(env, tokenizer, samples_per_task=args.samples_per_task)
    print(f"Dataset: {len(dataset)} samples across {len(TASK_IDS)} tasks")

    # ── Create reward function ──
    reward_fn = create_reward_function(env)

    # ── Configure GRPO ──
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_iterations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_completion_length=args.max_new_tokens,
        num_generations=args.group_size,
        logging_steps=1,
        save_steps=10,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        beta=args.beta,
        remove_unused_columns=False,
    )

    # ── Initialize trainer ──
    print("Starting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )

    # ── Train ──
    train_result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Final loss:  {train_result.training_loss:.4f}")

    # ── Save ──
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing to Hub as {args.hub_model_id}...")
        trainer.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print("Pushed to Hub")

    # ── Quick evaluation on all tasks ──
    print("\n--- Post-training evaluation ---")
    model.eval()
    for task_id in TASK_IDS:
        obs = env.reset(task_id)
        user_prompt = build_user_prompt(obs["task_description"], obs["buggy_code"])
        formatted = format_chat(tokenizer, user_prompt)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
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
        print(f"  {task_id}: reward={reward:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
