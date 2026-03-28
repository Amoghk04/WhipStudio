from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MLDebugAction(Action):
    """Agent submits a corrected training script."""

    fixed_code: str = Field(
        ...,
        description="The corrected Python training script. Must be complete runnable code.",
    )
    explanation: str = Field(
        default="",
        description="Optional: agent's explanation of bugs found (not scored, for logging)",
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Which attempt this is. Max 3 per episode.",
    )


class MLDebugObservation(Observation):
    """What the agent sees on reset() and after each step()."""

    task_id: str = Field(..., description="task1 | task2 | task3")
    task_description: str = Field(..., description="Plain English task instructions")
    buggy_code: str = Field(..., description="The broken training script")
    error_log: str = Field(
        default="",
        description="stdout+stderr from the previous attempt. Empty on first step.",
    )
    last_reward: float = Field(
        default=0.0,
        description="Reward from previous attempt. 0.0 on first step.",
    )
    metrics: dict = Field(
        default_factory=dict,
        description="Structured: {final_loss, nan_count, val_acc, timed_out, exit_code}",
    )
