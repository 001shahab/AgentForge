"""
LLM backend integrations for AgentForge.

I've designed these backends to be swappable - you can start with OpenAI for
quick prototyping and switch to a local HuggingFace model for production
without changing your agent code.

Available backends:
    - OpenAIBackend: Use OpenAI's GPT models (requires API key)
    - GroqBackend: Use Groq's fast inference API (requires API key)
    - HuggingFaceBackend: Run models locally with HuggingFace Transformers
"""

from agentforge.integrations.base import LLMIntegrator
from agentforge.integrations.openai_backend import OpenAIBackend
from agentforge.integrations.groq_backend import GroqBackend
from agentforge.integrations.huggingface import HuggingFaceBackend

__all__ = [
    "LLMIntegrator",
    "OpenAIBackend",
    "GroqBackend",
    "HuggingFaceBackend",
]

