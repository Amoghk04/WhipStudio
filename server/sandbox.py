import os
import subprocess
import sys
import tempfile
import time

from .tasks.graders import RunResult

TIMEOUT_SECONDS = 30
MAX_OUTPUT_BYTES = 8192
BANNED_PATTERNS = [
    "os.system",
    "subprocess.",
    "shutil.rmtree",
    "open(",
    "__import__",
    "exec(",
    "socket.",
    "urllib.",
    "requests.",
]
SAFE_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
    "HOME": "/tmp",
    "PYTHONPATH": os.pathsep.join(sys.path),
    "PYTHONDONTWRITEBYTECODE": "1",
}


def strip_markdown_code(code: str) -> str:
    if "```python" in code:
        return code.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in code:
        return code.split("```", 1)[1].split("```", 1)[0].strip()
    return code.strip()


def execute_code(code: str) -> RunResult:
    """Execute agent-submitted code in an isolated subprocess."""
    cleaned_code = strip_markdown_code(code)

    for pattern in BANNED_PATTERNS:
        if pattern in cleaned_code:
            return RunResult(
                exit_code=-1,
                stdout="",
                stderr=f'Execution blocked: banned pattern "{pattern}" detected.',
                elapsed_seconds=0.0,
                timed_out=False,
                fixed_code=cleaned_code,
            )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as temp_file:
        temp_file.write(cleaned_code)
        tmp_path = temp_file.name

    start = time.time()
    try:
        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=SAFE_ENV,
            cwd="/tmp",
        )
        return RunResult(
            exit_code=proc.returncode,
            stdout=proc.stdout[:MAX_OUTPUT_BYTES],
            stderr=proc.stderr[:2048],
            elapsed_seconds=round(time.time() - start, 2),
            timed_out=False,
            fixed_code=cleaned_code,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            exit_code=-1,
            stdout="",
            stderr="Execution timed out after 30 seconds.",
            elapsed_seconds=TIMEOUT_SECONDS,
            timed_out=True,
            fixed_code=cleaned_code,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
