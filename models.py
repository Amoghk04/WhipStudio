from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

# ── Action Types ───────────────────────────────────────────────────────────

class MLDebugAction(Action):
    """
    Polymorphic action model. The agent specifies `action_type` to choose
    which tool to invoke. Default is `submit_fix` for backward compatibility.
    """
    
    action_type: Literal[
        "submit_fix",
        "execute_snippet",
        "inspect_tensor",
        "run_training_probe",
        "get_variable_state",
        "inspect_diff",
    ] = Field(
        default="submit_fix",
        description="Which tool to invoke. Default: submit_fix",
    )
    
    # ── Session tracking ──
    episode_id: str = Field(
        default="",
        description="Episode ID returned from reset(). Required for session tracking.",
    )
    
    # ── submit_fix fields (legacy, backward compat) ──
    fixed_code: str = Field(
        default="",
        description="For submit_fix: The complete fixed Python script.",
    )
    explanation: str = Field(
        default="",
        description="For submit_fix: Optional explanation of bugs found.",
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        le=10,
        description="For submit_fix: Which attempt this is.",
    )
    
    # ── execute_snippet fields ──
    code: str = Field(
        default="",
        description="For execute_snippet/run_training_probe: Python code to run.",
    )
    
    # ── inspect_tensor fields ──
    setup_code: str = Field(
        default="",
        description="For inspect_tensor/get_variable_state: Setup code (imports, definitions).",
    )
    target_expression: str = Field(
        default="",
        description="For inspect_tensor: The expression to inspect (e.g., 'model.weight.grad').",
    )
    
    # ── run_training_probe fields ──
    steps: int = Field(
        default=5,
        ge=1,
        le=10,
        description="For run_training_probe: Number of training steps (max 10).",
    )
    
    # ── get_variable_state fields ──
    expressions: list[str] = Field(
        default_factory=list,
        description="For get_variable_state: List of expressions to evaluate (max 10).",
    )
    
    # ── inspect_diff fields ──
    proposed_code: str = Field(
        default="",
        description="For inspect_diff: The proposed fix to diff against original.",
    )


# ── Observation Types ──────────────────────────────────────────────────────

class VariableResult(Observation):
    """Result of evaluating a single variable expression."""
    repr_str: str = Field(default="", description="String representation of the value")
    type_str: str = Field(default="", description="Type name of the value")
    value: Optional[float | int | bool | str] = Field(default=None, description="Scalar value if applicable")
    shape: Optional[list[int]] = Field(default=None, description="Shape for tensors/arrays")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")


class MLDebugObservation(Observation):
    """
    Polymorphic observation returned by step().
    Fields populated depend on which action_type was invoked.
    """

    # ── Common fields ──
    action_type: str = Field(default="submit_fix", description="Which action produced this observation")
    turn: int = Field(default=0, description="Current turn number in this episode")
    episode_done: bool = Field(default=False, description="Whether the episode is complete")
    reward: Optional[float] = Field(default=None, description="Reward (None for tools, grader score for submit_fix)")
    error: Optional[str] = Field(default=None, description="Error message if action failed")
    episode_id: str = Field(default="", description="Episode ID for session tracking")
    
    # ── Task context (always present after reset) ──
    task_id: str = Field(default="", description="task1 | task2 | ... | task6")
    task_description: str = Field(default="", description="Plain English task instructions")
    buggy_code: str = Field(default="", description="The broken training script")
    
    # ── execute_snippet / run_training_probe output ──
    stdout: str = Field(default="", description="Captured stdout from code execution")
    stderr: str = Field(default="", description="Captured stderr from code execution")
    exit_code: int = Field(default=0, description="Exit code from code execution")
    timed_out: bool = Field(default=False, description="Whether execution timed out")
    
    # ── inspect_tensor output ──
    shape: Optional[list[int]] = Field(default=None, description="Tensor shape")
    dtype: Optional[str] = Field(default=None, description="Tensor dtype")
    requires_grad: Optional[bool] = Field(default=None, description="Whether tensor requires grad")
    grad_is_none: Optional[bool] = Field(default=None, description="Whether .grad is None")
    min_val: Optional[float] = Field(default=None, description="Tensor min value")
    max_val: Optional[float] = Field(default=None, description="Tensor max value")
    mean_val: Optional[float] = Field(default=None, description="Tensor mean value")
    is_nan: Optional[bool] = Field(default=None, description="Whether tensor contains NaN")
    is_inf: Optional[bool] = Field(default=None, description="Whether tensor contains Inf")
    
    # ── run_training_probe output ──
    losses: list[float] = Field(default_factory=list, description="Loss values per training step")
    grad_norms: dict[str, float] = Field(default_factory=dict, description="Layer name -> gradient norm")
    optimizer_param_count: Optional[int] = Field(default=None, description="Number of optimizer parameters")
    final_loss: Optional[float] = Field(default=None, description="Final loss value")
    loss_is_nan: bool = Field(default=False, description="Whether final loss is NaN")
    loss_is_inf: bool = Field(default=False, description="Whether final loss is Inf")
    
    # ── get_variable_state output ──
    results: dict[str, dict] = Field(
        default_factory=dict,
        description="Expression -> VariableResult dict for get_variable_state",
    )
    
    # ── inspect_diff output ──
    diff: str = Field(default="", description="Unified diff between original and proposed code")
    lines_changed: int = Field(default=0, description="Number of lines changed")
    additions: int = Field(default=0, description="Number of lines added")
    deletions: int = Field(default=0, description="Number of lines deleted")
    
    # ── submit_fix output ──
    success: bool = Field(default=False, description="Whether reward >= 0.7")
    grader_details: dict = Field(default_factory=dict, description="Detailed grader breakdown")
    turns_used: int = Field(default=0, description="How many tool calls before this submission")
    
    # ── Legacy fields for backward compat ──
    error_log: str = Field(default="", description="Legacy: stdout+stderr combined")
    last_reward: float = Field(default=0.0, description="Legacy: same as reward")
    metrics: dict = Field(default_factory=dict, description="Legacy: structured metrics")
    done: bool = Field(default=False, description="Legacy: same as episode_done")
