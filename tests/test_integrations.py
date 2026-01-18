"""
Tests for LLM integrations.

I'm testing the base LLM class and the mock behavior here.
Real API tests would require actual API keys.

Author: Prof. Shahab Anbarjafari
"""

import pytest
from unittest.mock import MagicMock, patch
import os


class TestLLMIntegrator:
    """Tests for the base LLMIntegrator class."""
    
    def test_mock_llm_generate(self, mock_llm):
        """Test that mock LLM generates responses."""
        response = mock_llm.generate("Hello world")
        
        assert "Mock response" in response
        assert mock_llm.call_count == 1
    
    def test_mock_llm_tracks_prompts(self, mock_llm):
        """Test that mock LLM tracks prompts."""
        mock_llm.generate("First prompt")
        mock_llm.generate("Second prompt")
        
        assert mock_llm.call_count == 2
        assert mock_llm.last_prompt == "Second prompt"
    
    def test_llm_generate_with_template(self, mock_llm):
        """Test template-based generation."""
        # Mock the template method
        mock_llm.generate_with_template = MagicMock(return_value="Summary here")
        
        result = mock_llm.generate_with_template("summarize", text="Some text")
        
        mock_llm.generate_with_template.assert_called_once()
    
    def test_llm_count_tokens(self, mock_llm):
        """Test token counting."""
        count = mock_llm.count_tokens("Hello world, this is a test")
        
        # Rough estimate: ~4 chars per token
        assert count > 0
        assert count < 20  # Should be around 7


class TestOpenAIBackend:
    """Tests for the OpenAI backend."""
    
    def test_openai_requires_api_key(self):
        """Test that OpenAI backend requires API key."""
        from agentforge.integrations import OpenAIBackend
        
        # Ensure no API key is set
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        
        try:
            with pytest.raises(ValueError, match="API key not found"):
                OpenAIBackend()
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
    
    def test_openai_with_api_key(self):
        """Test OpenAI backend creation with API key."""
        from agentforge.integrations import OpenAIBackend
        
        # Create with explicit key
        backend = OpenAIBackend(api_key="test-key")
        
        assert backend.model == "gpt-4o-mini"  # default
        assert backend.api_key == "test-key"
    
    def test_openai_custom_model(self):
        """Test OpenAI backend with custom model."""
        from agentforge.integrations import OpenAIBackend
        
        backend = OpenAIBackend(api_key="test-key", model="gpt-4o")
        
        assert backend.model == "gpt-4o"


class TestGroqBackend:
    """Tests for the Groq backend."""
    
    def test_groq_requires_api_key(self):
        """Test that Groq backend requires API key."""
        from agentforge.integrations import GroqBackend
        
        old_key = os.environ.pop("GROQ_API_KEY", None)
        
        try:
            with pytest.raises(ValueError, match="API key not found"):
                GroqBackend()
        finally:
            if old_key:
                os.environ["GROQ_API_KEY"] = old_key
    
    def test_groq_with_api_key(self):
        """Test Groq backend creation with API key."""
        from agentforge.integrations import GroqBackend
        
        backend = GroqBackend(api_key="test-key")
        
        assert backend.model == "llama-3.1-70b-versatile"
        assert backend.api_key == "test-key"
    
    def test_groq_available_models(self):
        """Test listing available Groq models."""
        from agentforge.integrations import GroqBackend
        
        backend = GroqBackend(api_key="test-key")
        models = backend.list_models()
        
        assert "llama-3.1-70b-versatile" in models


class TestHuggingFaceBackend:
    """Tests for the HuggingFace backend."""
    
    def test_huggingface_creation(self):
        """Test HuggingFace backend creation."""
        from agentforge.integrations import HuggingFaceBackend
        
        # Should work without API key (for public models)
        backend = HuggingFaceBackend()
        
        assert backend.model == "mistralai/Mistral-7B-Instruct-v0.2"
    
    def test_huggingface_custom_model(self):
        """Test HuggingFace with custom model."""
        from agentforge.integrations import HuggingFaceBackend
        
        backend = HuggingFaceBackend(model="meta-llama/Llama-2-7b-chat-hf")
        
        assert backend.model == "meta-llama/Llama-2-7b-chat-hf"
    
    def test_huggingface_quantization_options(self):
        """Test quantization options."""
        from agentforge.integrations import HuggingFaceBackend
        
        backend = HuggingFaceBackend(load_in_4bit=True)
        
        assert backend.load_in_4bit is True

