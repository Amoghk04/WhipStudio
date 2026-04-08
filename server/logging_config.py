"""
WhipStudio Logging Configuration

Provides structured logging for the WhipStudio environment.
"""

import logging
import os
import sys
import time
from functools import wraps
from typing import Optional

# ── Logger Setup ───────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure logging for WhipStudio.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        json_format: Use JSON format for structured logging
    
    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("whipstudio")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Format
    if json_format:
        fmt = '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "whipstudio") -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


# ── Metrics Tracking ───────────────────────────────────────────────────────

class Metrics:
    """
    Simple metrics tracker for WhipStudio.
    Thread-safe counters for tracking usage.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self._counters = {
            "total_resets": 0,
            "total_steps": 0,
            "total_tool_calls": 0,
            "total_submissions": 0,
        }
        self._task_counts = {}
        self._tool_counts = {}
        self._rewards = []
    
    def reset(self):
        """Reset all metrics."""
        self.__init__()
    
    def increment(self, key: str, amount: int = 1):
        """Increment a counter."""
        if key not in self._counters:
            self._counters[key] = 0
        self._counters[key] += amount
    
    def record_reset(self, task_id: str):
        """Record a reset event."""
        self.increment("total_resets")
        if task_id not in self._task_counts:
            self._task_counts[task_id] = 0
        self._task_counts[task_id] += 1
    
    def record_step(self, action_type: str, reward: float = 0.0):
        """Record a step event."""
        self.increment("total_steps")
        
        if action_type not in self._tool_counts:
            self._tool_counts[action_type] = 0
        self._tool_counts[action_type] += 1
        
        if action_type == "submit_fix":
            self.increment("total_submissions")
            self._rewards.append(reward)
        else:
            self.increment("total_tool_calls")
    
    def get_metrics(self) -> dict:
        """Get all metrics as a dictionary."""
        uptime = time.time() - self.start_time
        avg_reward = sum(self._rewards) / len(self._rewards) if self._rewards else 0.0
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_resets": self._counters["total_resets"],
            "total_steps": self._counters["total_steps"],
            "total_tool_calls": self._counters["total_tool_calls"],
            "total_submissions": self._counters["total_submissions"],
            "avg_reward": round(avg_reward, 4),
            "task_distribution": dict(self._task_counts),
            "tool_usage": dict(self._tool_counts),
        }


# ── Decorators ─────────────────────────────────────────────────────────────

def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger()
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                log.debug(f"{func.__name__} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                log.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator


def log_request(logger: Optional[logging.Logger] = None):
    """Decorator to log API request handling."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            log = logger or get_logger()
            start = time.time()
            func_name = func.__name__
            
            log.info(f"Request: {func_name}")
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start
                log.info(f"Response: {func_name} ({elapsed:.3f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start
                log.error(f"Error: {func_name} ({elapsed:.3f}s) - {e}")
                raise
        return wrapper
    return decorator


# ── Global Metrics Instance ────────────────────────────────────────────────

# Singleton metrics instance
_metrics = None


def get_metrics() -> Metrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


# ── Initialize Logging ─────────────────────────────────────────────────────

# Auto-configure logging on import
_log_level = os.environ.get("LOG_LEVEL", "INFO")
_log_file = os.environ.get("LOG_FILE", None)
_json_format = os.environ.get("LOG_JSON", "false").lower() == "true"

setup_logging(level=_log_level, log_file=_log_file, json_format=_json_format)
