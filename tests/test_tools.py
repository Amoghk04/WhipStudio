"""
Tool execution tests for WhipStudio.

Tests the debugging tools: execute_snippet, inspect_tensor, 
run_training_probe, get_variable_state, inspect_diff.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSecurityChecks:
    """Test security validation for code execution."""
    
    def test_banned_import_socket(self):
        """Socket import should be rejected."""
        from server.environment import check_code_security
        
        code = "import socket\ns = socket.socket()"
        is_safe, error = check_code_security(code)
        assert not is_safe
        assert "socket" in error.lower()
    
    def test_banned_import_requests(self):
        """Requests import should be rejected."""
        from server.environment import check_code_security
        
        code = "import requests\nrequests.get('http://evil.com')"
        is_safe, error = check_code_security(code)
        assert not is_safe
        assert "requests" in error.lower()
    
    def test_banned_import_subprocess(self):
        """Subprocess import should be rejected."""
        from server.environment import check_code_security
        
        code = "import subprocess\nsubprocess.run(['ls'])"
        is_safe, error = check_code_security(code)
        assert not is_safe
        assert "subprocess" in error.lower()
    
    def test_allowed_imports(self):
        """Standard ML imports should be allowed."""
        from server.environment import check_code_security
        
        code = """
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import math
import json
"""
        is_safe, error = check_code_security(code)
        assert is_safe, f"Should be safe: {error}"
    
    def test_file_write_outside_tmp(self):
        """File writes outside /tmp should be rejected."""
        from server.environment import check_code_security
        
        code = "open('/etc/passwd', 'w').write('hacked')"
        is_safe, error = check_code_security(code)
        assert not is_safe
        assert "tmp" in error.lower() or "file" in error.lower()
    
    def test_file_write_in_tmp_allowed(self):
        """File writes in /tmp should be allowed."""
        from server.environment import check_code_security
        
        code = "open('/tmp/test.txt', 'w').write('ok')"
        is_safe, error = check_code_security(code)
        assert is_safe, f"Should be safe: {error}"


class TestToolDefinitions:
    """Test that tool definitions are complete."""
    
    def test_all_tools_defined(self):
        """All 6 tools should be defined."""
        from server.environment import TOOL_DEFINITIONS
        
        expected_tools = {
            "execute_snippet",
            "inspect_tensor", 
            "run_training_probe",
            "get_variable_state",
            "inspect_diff",
            "submit_fix"
        }
        
        # TOOL_DEFINITIONS is a list of dicts
        defined_tools = {t["name"] for t in TOOL_DEFINITIONS}
        assert expected_tools == defined_tools
    
    def test_tool_definitions_have_required_fields(self):
        """Each tool definition should have name, description, action_fields."""
        from server.environment import TOOL_DEFINITIONS
        
        for tool_def in TOOL_DEFINITIONS:
            assert "name" in tool_def, f"Tool missing name"
            assert "description" in tool_def, f"{tool_def.get('name')} missing description"
            assert "action_fields" in tool_def, f"{tool_def.get('name')} missing action_fields"


class TestActionParsing:
    """Test that actions are parsed correctly."""
    
    def test_submit_fix_action(self):
        """submit_fix action should parse correctly."""
        from models import MLDebugAction
        
        action = MLDebugAction(
            action_type="submit_fix",
            fixed_code="import torch\nprint('hello')"
        )
        assert action.action_type == "submit_fix"
        assert "import torch" in action.fixed_code
    
    def test_execute_snippet_action(self):
        """execute_snippet action should parse correctly."""
        from models import MLDebugAction
        
        action = MLDebugAction(
            action_type="execute_snippet",
            code="print('test')"
        )
        assert action.action_type == "execute_snippet"
        assert action.code == "print('test')"
    
    def test_inspect_tensor_action(self):
        """inspect_tensor action should parse correctly."""
        from models import MLDebugAction
        
        action = MLDebugAction(
            action_type="inspect_tensor",
            setup_code="import torch; t = torch.randn(3, 4)",
            target_expression="t.shape"
        )
        assert action.action_type == "inspect_tensor"
        assert action.target_expression == "t.shape"
    
    def test_get_variable_state_action(self):
        """get_variable_state action should parse correctly."""
        from models import MLDebugAction
        
        action = MLDebugAction(
            action_type="get_variable_state",
            setup_code="x = 1",
            expressions=["x", "x + 1"]
        )
        assert action.action_type == "get_variable_state"
        assert len(action.expressions) == 2
    
    def test_run_training_probe_action(self):
        """run_training_probe action should parse correctly."""
        from models import MLDebugAction
        
        action = MLDebugAction(
            action_type="run_training_probe",
            code="# training code",
            steps=5
        )
        assert action.action_type == "run_training_probe"
        assert action.steps == 5
    
    def test_inspect_diff_action(self):
        """inspect_diff action should parse correctly."""
        from models import MLDebugAction
        
        action = MLDebugAction(
            action_type="inspect_diff",
            proposed_code="# fixed code"
        )
        assert action.action_type == "inspect_diff"


class TestObservationModel:
    """Test observation model fields."""
    
    def test_observation_has_all_fields(self):
        """Observation should have fields for all tools."""
        from models import MLDebugObservation
        
        obs = MLDebugObservation()
        
        # Common fields
        assert hasattr(obs, "turn")
        assert hasattr(obs, "episode_done")
        assert hasattr(obs, "task_id")
        
        # execute_snippet fields
        assert hasattr(obs, "stdout")
        assert hasattr(obs, "stderr")
        assert hasattr(obs, "exit_code")
        assert hasattr(obs, "timed_out")
        
        # inspect_tensor fields
        assert hasattr(obs, "shape")
        assert hasattr(obs, "dtype")
        assert hasattr(obs, "requires_grad")
        assert hasattr(obs, "grad_is_none")
        assert hasattr(obs, "min_val")
        assert hasattr(obs, "max_val")
        assert hasattr(obs, "mean_val")
        assert hasattr(obs, "is_nan")
        assert hasattr(obs, "is_inf")
        
        # run_training_probe fields
        assert hasattr(obs, "losses")
        assert hasattr(obs, "grad_norms")
        assert hasattr(obs, "optimizer_param_count")
        assert hasattr(obs, "final_loss")
        assert hasattr(obs, "loss_is_nan")
        assert hasattr(obs, "loss_is_inf")
        
        # get_variable_state fields
        assert hasattr(obs, "results")
        
        # inspect_diff fields
        assert hasattr(obs, "diff")
        assert hasattr(obs, "lines_changed")
        assert hasattr(obs, "additions")
        assert hasattr(obs, "deletions")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
