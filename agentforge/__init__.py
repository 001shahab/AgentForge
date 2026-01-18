"""
AgentForge - A lightweight framework for modular AI agents.

I built this library to make it easy for anyone to create AI agents that can
chain together different tasks like web scraping, data analysis, and content
generation. Whether you're using local models or cloud APIs, AgentForge has
you covered.

Author: Prof. Shahab Anbarjafari
Organization: 3S Holding OÜ, Tartu, Estonia
Email: shb@3sholding.com

Example:
    >>> from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
    >>> from agentforge.integrations import OpenAIBackend
    >>>
    >>> # Create an agent that scrapes and summarizes
    >>> agent = Agent(
    ...     skills=[WebScraperSkill(), ContentGenerationSkill()],
    ...     llm=OpenAIBackend()
    ... )
    >>> result = agent.run({"url": "https://example.com"})
"""

__version__ = "0.1.0"
__author__ = "Prof. Shahab Anbarjafari"
__email__ = "shb@3sholding.com"
__license__ = "MIT"

# I'm importing the main classes here so users can do simple imports
from agentforge.core import Agent, Skill, SkillRegistry

# Let's also expose the built-in skills for convenience
from agentforge.skills import (
    WebScraperSkill,
    DataAnalysisSkill,
    ContentGenerationSkill,
)

# And the LLM backends
from agentforge.integrations import (
    LLMIntegrator,
    OpenAIBackend,
    GroqBackend,
    HuggingFaceBackend,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core classes
    "Agent",
    "Skill",
    "SkillRegistry",
    # Skills
    "WebScraperSkill",
    "DataAnalysisSkill",
    "ContentGenerationSkill",
    # LLM Backends
    "LLMIntegrator",
    "OpenAIBackend",
    "GroqBackend",
    "HuggingFaceBackend",
]

