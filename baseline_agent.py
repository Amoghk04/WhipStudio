import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

import httpx

SYSTEM_PROMPT = """
You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs in it.
Rules:
- Return ONLY the complete corrected Python code, nothing else.
- No markdown, no backticks, no explanation text.
- The script must print losses in format: LOSSES:[v1, v2, ...]
- For task3, also print: VAL_ACCS:[v1,...] and FINAL_LOSS:X.XX
- Keep all torch.manual_seed() calls intact.
""".strip()


SUPPORTED_MODEL_IDS = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def get_model(model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    from smolagents import InferenceClientModel

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Set HF_TOKEN to run /baseline with InferenceClientModel."
        )

    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(
            f"Unsupported model_id '{model_id}'. Supported options: {SUPPORTED_MODEL_IDS}"
        )

    return InferenceClientModel(
        model_id=model_id,
        token=hf_token,
    )


def _generate_fixed_code(model, prompt: str) -> str:
    def _extract_text(response) -> str:
        if isinstance(response, str):
            return response

        if hasattr(response, "content"):
            content = getattr(response, "content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, str):
                        chunks.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if text:
                            chunks.append(str(text))
                if chunks:
                    return "\n".join(chunks)

        if isinstance(response, dict):
            text = response.get("content") or response.get("text")
            if isinstance(text, str):
                return text

        return str(response)

    if hasattr(model, "generate"):
        generate = getattr(model, "generate")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            return _extract_text(generate(messages=messages))
        except TypeError:
            return _extract_text(generate(messages))

    if callable(model):
        try:
            return _extract_text(model(prompt, system_prompt=SYSTEM_PROMPT))
        except TypeError:
            return _extract_text(model(prompt))

    raise AttributeError("Model does not support callable() or generate() inference APIs")


async def run_single_task(
    task_id: str,
    env_url: str = "http://localhost:7860",
    model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
) -> float:
    """Backwards-compatible wrapper that returns just the score."""
    result = await run_single_task_detailed(task_id, env_url, model_id)
    return result["score"]


async def run_single_task_detailed(
    task_id: str,
    env_url: str = "http://localhost:7860",
    model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
) -> dict:
    """Run the baseline agent on a single task. Returns detailed results."""
    model = get_model(model_id)
    timeout = httpx.Timeout(900.0, connect=10.0)

    attempts_log = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        reset_resp = await client.post(f"{env_url}/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        obs = reset_resp.json().get("observation", reset_resp.json())

        best_reward = 0.0
        best_code = ""
        best_output = ""

        for attempt in range(1, 4):
            prompt = f"""
Task: {obs.get('task_description', '')}
Buggy code:
{obs.get('buggy_code', '')}
Previous execution output (if any):
{obs.get('error_log', 'None')}
Previous score: {obs.get('last_reward', 0.0)}
""".strip()

            fixed_code = _generate_fixed_code(model, prompt)
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python", 1)[1].split("```", 1)[0].strip()
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```", 1)[1].split("```", 1)[0].strip()

            step_payload = {
                "action": {
                    "fixed_code": fixed_code,
                    "attempt_number": attempt,
                }
            }
            step_resp = await client.post(f"{env_url}/step", json=step_payload)
            if step_resp.status_code == 422:
                step_resp = await client.post(
                    f"{env_url}/step",
                    json={
                        "fixed_code": fixed_code,
                        "attempt_number": attempt,
                    },
                )
            step_resp.raise_for_status()
            result = step_resp.json()

            reward = float(result.get("reward", 0.0) or 0.0)
            obs = result.get("observation", obs)
            output_log = obs.get("error_log", "") if isinstance(obs, dict) else ""

            attempts_log.append({
                "attempt": attempt,
                "code": fixed_code,
                "output": output_log[:3000],
                "reward": reward,
            })

            if reward > best_reward:
                best_reward = reward
                best_code = fixed_code
                best_output = output_log

            if result.get("done") or reward >= 0.95:
                break

    return {
        "score": best_reward,
        "fixed_code": best_code,
        "output": best_output[:3000],
        "attempts": attempts_log,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:7860")
    args = parser.parse_args()

    async def main():
        scores = {}
        for tid in ["task1", "task2", "task3"]:
            try:
                s = await asyncio.wait_for(run_single_task(tid, args.env_url), timeout=95.0)
            except TimeoutError:
                s = 0.0
            scores[tid] = round(s, 4)
            print(f"{tid}: {s:.4f}")
        print(f"Average: {sum(scores.values()) / 3:.4f}")

    asyncio.run(main())
