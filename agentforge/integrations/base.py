"""
Base class for LLM integrations.

I've created this abstract base class so all LLM backends have a consistent
interface. This makes it easy to swap between OpenAI, Groq, or local models
without changing your agent code.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentforge.integrations")


class LLMIntegrator(ABC):
    """
    Abstract base class for LLM backends.
    
    All LLM integrations should inherit from this class and implement
    the generate() method. I've also included some common functionality
    like prompt templates that all backends can use.
    
    Attributes:
        model: The model identifier to use
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens to generate
    """
    
    # Common prompt templates that work well across different LLMs
    PROMPT_TEMPLATES = {
        "summarize": (
            "Please summarize the following text concisely:\n\n"
            "{text}\n\n"
            "Summary:"
        ),
        "analyze": (
            "Analyze the following data and provide key insights:\n\n"
            "{text}\n\n"
            "Analysis:"
        ),
        "chain_of_thought": (
            "Let's think through this step by step.\n\n"
            "Question: {question}\n\n"
            "Step-by-step reasoning:"
        ),
        "extract": (
            "Extract the following information from the text:\n"
            "Fields to extract: {fields}\n\n"
            "Text:\n{text}\n\n"
            "Extracted information (JSON format):"
        ),
        "classify": (
            "Classify the following text into one of these categories: {categories}\n\n"
            "Text: {text}\n\n"
            "Category:"
        ),
        "rewrite": (
            "Rewrite the following text to be more {style}:\n\n"
            "{text}\n\n"
            "Rewritten text:"
        ),
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize the LLM backend.
        
        Args:
            model: Model identifier (uses a default if not specified)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific options
        """
        self.model = model or self.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.options = kwargs
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """The default model to use if none is specified."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        This is the core method that all backends must implement. It takes
        a prompt string and returns the generated text.
        
        Args:
            prompt: The input prompt for the LLM
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
        """
        pass
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """
        Async version of generate.
        
        By default, I'm just calling the sync version. Override this if
        your backend has native async support.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, **kwargs))
    
    def generate_with_template(
        self,
        template_name: str,
        **template_vars
    ) -> str:
        """
        Generate text using a predefined template.
        
        This is super handy for common tasks like summarization. Just pick
        a template and fill in the variables.
        
        Args:
            template_name: Name of the template to use
            **template_vars: Variables to fill into the template
            
        Returns:
            The generated text
            
        Example:
            >>> llm.generate_with_template("summarize", text="Long article here...")
        """
        if template_name not in self.PROMPT_TEMPLATES:
            available = ", ".join(self.PROMPT_TEMPLATES.keys())
            raise ValueError(
                f"Unknown template: '{template_name}'. Available: {available}"
            )
        
        template = self.PROMPT_TEMPLATES[template_name]
        prompt = template.format(**template_vars)
        
        return self.generate(prompt)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Have a multi-turn conversation.
        
        This method handles the chat format that most LLMs expect. Override
        this if your backend has special chat handling.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional parameters
            
        Returns:
            The assistant's response
        """
        # Default implementation: concatenate messages into a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        return self.generate(prompt, **kwargs)
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        This is a rough estimate using a simple heuristic. Override this
        if your backend has a proper tokenizer.
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model='{self.model}')>"

