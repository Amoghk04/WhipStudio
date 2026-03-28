from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MLDebugAction, MLDebugObservation


class MLDebugEnv(EnvClient[MLDebugAction, MLDebugObservation, State]):
    def _step_payload(self, action: MLDebugAction) -> Dict:
        return {
            "fixed_code": action.fixed_code,
            "explanation": action.explanation,
            "attempt_number": action.attempt_number,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MLDebugObservation]:
        obs_data = payload.get("observation", {})
        observation = MLDebugObservation(
            task_id=obs_data.get("task_id", "task1"),
            task_description=obs_data.get("task_description", ""),
            buggy_code=obs_data.get("buggy_code", ""),
            error_log=obs_data.get("error_log", ""),
            last_reward=obs_data.get("last_reward", 0.0),
            metrics=obs_data.get("metrics", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
