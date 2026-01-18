"""
Pytest configuration and fixtures for AgentForge tests.

I've set up fixtures here that are shared across all test files.
This makes it easy to test without needing real API keys or network access.

Author: Prof. Shahab Anbarjafari
"""

import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_llm():
    """
    A mock LLM backend for testing.
    
    This returns predictable responses without needing an API key.
    """
    from agentforge.integrations.base import LLMIntegrator
    
    class MockLLM(LLMIntegrator):
        def __init__(self):
            self.model = "mock-model"
            self.temperature = 0.7
            self.max_tokens = 100
            self.options = {}
            self.call_count = 0
            self.last_prompt = None
        
        @property
        def default_model(self) -> str:
            return "mock-model"
        
        def generate(self, prompt: str, **kwargs) -> str:
            self.call_count += 1
            self.last_prompt = prompt
            return f"Mock response to: {prompt[:50]}..."
    
    return MockLLM()


@pytest.fixture
def sample_skill():
    """A simple test skill."""
    from agentforge.core import Skill
    
    class TestSkill(Skill):
        name = "test_skill"
        description = "A skill for testing"
        requires_llm = False
        
        def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "processed": True,
                "input_keys": list(input_data.keys()),
                "message": "Test skill executed",
            }
    
    return TestSkill()


@pytest.fixture
def sample_llm_skill():
    """A test skill that requires an LLM."""
    from agentforge.core import Skill
    
    class LLMTestSkill(Skill):
        name = "llm_test_skill"
        description = "A skill that uses LLM"
        requires_llm = True
        
        def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            if not self._llm:
                raise RuntimeError("LLM not configured")
            
            response = self._llm.generate("Test prompt")
            return {
                "generated": response,
                "used_llm": True,
            }
    
    return LLMTestSkill()


@pytest.fixture
def sample_data():
    """Sample data for testing data analysis."""
    return [
        {"name": "Alice", "score": 85, "department": "Engineering"},
        {"name": "Bob", "score": 92, "department": "Sales"},
        {"name": "Charlie", "score": 78, "department": "Engineering"},
        {"name": "Diana", "score": 95, "department": "Marketing"},
        {"name": "Eve", "score": 88, "department": "Engineering"},
    ]


@pytest.fixture
def sample_html():
    """Sample HTML for testing web scraping."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Welcome</h1>
        <article>
            <p>This is the first paragraph.</p>
            <p>This is the second paragraph.</p>
        </article>
        <a href="/link1">Link 1</a>
        <a href="/link2">Link 2</a>
    </body>
    </html>
    """


@pytest.fixture
def sample_config():
    """Sample agent configuration."""
    return {
        "name": "test_agent",
        "version": "1.0",
        "llm": {
            "backend": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        },
        "skills": [
            "web_scraper",
            {"skill": "content_generation", "template": "summarize"},
        ],
        "options": {
            "continue_on_error": False,
        },
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create a temporary config file for testing."""
    import yaml
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    
    return config_path


@pytest.fixture
def mock_requests_get(sample_html):
    """Mock requests.get for web scraping tests."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        yield mock_get

