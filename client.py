"""
WhipStudio OpenEnv Client

This module provides an EnvClient wrapper for interacting with the WhipStudio
ML debugging environment via HTTP.
"""

from openenv import EnvClient

from models import MLDebugAction, MLDebugObservation


def create_client(base_url: str = "http://localhost:7860") -> EnvClient[MLDebugAction, MLDebugObservation]:
    """
    Create an EnvClient for the WhipStudio environment.
    
    Args:
        base_url: The base URL of the WhipStudio server.
        
    Returns:
        An EnvClient configured for MLDebugAction/MLDebugObservation.
        
    Example:
        >>> client = create_client("http://localhost:7860")
        >>> obs = client.reset(task_id="task1")
        >>> print(obs.buggy_code)
        >>> 
        >>> # Use tools to debug
        >>> obs = client.step(MLDebugAction(
        ...     action_type="execute_snippet",
        ...     code="print('hello')",
        ...     episode_id=obs.episode_id,
        ... ))
        >>> print(f"Turn {obs.turn}: {obs.stdout}")
        >>>
        >>> # Submit a fix
        >>> obs = client.step(MLDebugAction(
        ...     action_type="submit_fix",
        ...     fixed_code="...",
        ...     episode_id=obs.episode_id,
        ... ))
        >>> print(f"Reward: {obs.reward}")
    """
    return EnvClient[MLDebugAction, MLDebugObservation](
        base_url=base_url,
        action_cls=MLDebugAction,
        observation_cls=MLDebugObservation,
    )


# Convenience alias
Client = create_client


if __name__ == "__main__":
    # Quick test
    import sys
    
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    print(f"Connecting to {url}...")
    
    client = create_client(url)
    
    # Test reset
    obs = client.reset(task_id="task1")
    print(f"✓ Reset successful - Episode: {obs.episode_id[:8]}...")
    print(f"  Task: {obs.task_id}")
    print(f"  Turn: {obs.turn}")
    
    # Test a tool call
    obs = client.step(MLDebugAction(
        action_type="execute_snippet",
        code="print('Hello from client!')",
        episode_id=obs.episode_id,
    ))
    print(f"✓ Step successful - Turn: {obs.turn}")
    print(f"  stdout: {obs.stdout.strip()}")
    
    print("\n✅ Client working correctly!")
