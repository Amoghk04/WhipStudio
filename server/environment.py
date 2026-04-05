import re
import math
import time
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MLDebugAction, MLDebugObservation
    from .sandbox import execute_code
    from .tasks import task1_broken_loop, task2_nan_loss, task3_oom_leakage, task4_wrong_loss, task5_frozen_backbone, task6_io_mismatch
    from .tasks.graders import parse_losses, parse_val_accs, score_task
except ImportError:
    from models import MLDebugAction, MLDebugObservation
    from server.sandbox import execute_code
    from server.tasks import task1_broken_loop, task2_nan_loss, task3_oom_leakage, task4_wrong_loss, task5_frozen_backbone, task6_io_mismatch
    from server.tasks.graders import parse_losses, parse_val_accs, score_task

TASKS = {
    "task1": task1_broken_loop,
    "task2": task2_nan_loss,
    "task3": task3_oom_leakage,
    "task4": task4_wrong_loss,
    "task5": task5_frozen_backbone,
    "task6": task6_io_mismatch,
}


class MLDebugEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = "task1"
        self._best_reward = 0.0
        self._trajectory: list[dict] = []

    def reset(self, task_id: str = "task1", **kwargs) -> MLDebugObservation:  # type: ignore[override]
        if task_id not in TASKS:
            task_id = "task1"

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = task_id
        self._best_reward = 0.0
        self._trajectory = []

        task = TASKS[self._task_id]
        return MLDebugObservation(
            task_id=self._task_id,
            task_description=task.TASK_DESCRIPTION.strip(),
            buggy_code=task.BUGGY_CODE.strip(),
            error_log="",
            last_reward=0.0,
            metrics={},
            done=False,
            reward=0.0,
        )

    def step(self, action: MLDebugAction) -> MLDebugObservation:  # type: ignore[override]
        self._state.step_count += 1

        if not action.fixed_code or not action.fixed_code.strip():
            done = True
            metrics = {
                "exit_code": -1,
                "elapsed_seconds": 0.0,
                "timed_out": False,
                "step": self._state.step_count,
                "best_reward_so_far": self._best_reward,
                "error": "empty code submitted",
            }
            task = TASKS[self._task_id]
            return MLDebugObservation(
                task_id=self._task_id,
                task_description=task.TASK_DESCRIPTION.strip(),
                buggy_code=task.BUGGY_CODE.strip(),
                error_log="empty code submitted",
                last_reward=0.0,
                metrics=metrics,
                done=done,
                reward=0.0,
            )

        run_result1 = execute_code(action.fixed_code)
        reward1, breakdown1 = score_task(self._task_id, run_result1)

        consistency_flag = False
        reward_variance = 0.0
        final_reward = reward1
        final_breakdown = breakdown1
        run_result = run_result1

        if reward1 > 0.5:
            run_result2 = execute_code(action.fixed_code)
            reward2, breakdown2 = score_task(self._task_id, run_result2)
            reward_variance = abs(reward1 - reward2)
            if reward_variance > 0.15:
                consistency_flag = True
                final_reward = min(reward1, reward2)
                if reward2 < reward1:
                    final_breakdown = breakdown2
                    run_result = run_result2
            else:
                consistency_flag = False
                final_reward = (reward1 + reward2) / 2.0
        else:
            consistency_flag = False
            final_reward = reward1

        self._best_reward = max(self._best_reward, final_reward)
        done = self._state.step_count >= 3 or final_reward >= 0.95

        losses = parse_losses(run_result.stdout)
        val_accs = parse_val_accs(run_result.stdout)
        final_loss = None
        if losses:
            final_loss = losses[-1]
        else:
            match = re.search(r"FINAL_LOSS:([-\d.]+)", run_result.stdout)
            if match:
                final_loss = float(match.group(1))

        metrics = {
            "exit_code": run_result.exit_code,
            "elapsed_seconds": run_result.elapsed_seconds,
            "timed_out": run_result.timed_out,
            "step": self._state.step_count,
            "best_reward_so_far": self._best_reward,
            "final_loss": final_loss,
            "nan_count": sum(1 for x in losses if math.isnan(x) or math.isinf(x)) if losses else 0,
            "val_acc": val_accs[-1] if val_accs else None,
            "consistency_flag": consistency_flag,
            "reward_variance": round(reward_variance, 4),
            "reward_breakdown": final_breakdown,
        }

        task = TASKS[self._task_id]

        self._trajectory.append({
            "step": self._state.step_count,
            "reward": final_reward,
            "best_reward": self._best_reward,
            "metrics": metrics,
            "done": done,
            "timestamp": time.time(),
        })

        return MLDebugObservation(
            task_id=self._task_id,
            task_description=task.TASK_DESCRIPTION.strip(),
            buggy_code=task.BUGGY_CODE.strip(),
            error_log=(run_result.stdout + "\n" + run_result.stderr).strip()[:2000],
            last_reward=final_reward,
            metrics=metrics,
            done=done,
            reward=final_reward,
        )

    @property
    def trajectory(self) -> list[dict]:
        return list(self._trajectory)

    @property
    def state(self) -> State:
        return self._state
