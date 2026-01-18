"""
Groq API integration for AgentForge.

Groq offers blazing-fast inference for open-source models like Llama and
Mixtral. I've designed this backend to work just like the OpenAI one,
so you can easily switch between them.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from agentforge.integrations.base import LLMIntegrator

logger = logging.getLogger("agentforge.integrations.groq")


class GroqBackend(LLMIntegrator):
    """
    Groq API backend for fast inference.
    
    Groq specializes in fast inference for open-source models. If you need
    speed, this is a great option. They support Llama, Mixtral, and other
    popular models.
    
    Environment Variables:
        GROQ_API_KEY: Your Groq API key
        
    Example:
        >>> from agentforge.integrations import GroqBackend
        >>> llm = GroqBackend(model="llama-3.1-70b-versatile")
        >>> response = llm.generate("Explain quantum computing")
        >>> print(response)
    """
    
    # Available models on Groq (as of my last update)
    AVAILABLE_MODELS = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Groq backend.
        
        Args:
            model: Model to use (default: llama-3.1-70b-versatile)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: API key (falls back to GROQ_API_KEY env var)
            **kwargs: Additional options
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please set the GROQ_API_KEY "
                "environment variable or pass api_key to the constructor."
            )
        
        self._client = None
    
    @property
    def default_model(self) -> str:
        return "llama-3.1-70b-versatile"
    
    @property
    def client(self):
        """Lazy-load the Groq client."""
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "Groq library not installed. Please install it with:\n"
                    "pip install groq\n"
                    "or: pip install agentforge[groq]"
                )
            
            self._client = Groq(api_key=self.api_key)
        
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Groq's API.
        
        The API is very similar to OpenAI's, which makes it easy to switch
        between them if you need to.
        
        Args:
            prompt: The input prompt
            **kwargs: Override generation parameters
            
        Returns:
            The generated text
        """
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        model = kwargs.pop("model", self.model)
        
        logger.debug(f"Generating with Groq {model}")
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        result = response.choices[0].message.content
        logger.debug(f"Generated {len(result)} characters")
        
        return result
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """
        Async version using Groq's async client.
        """
        try:
            from groq import AsyncGroq
        except ImportError:
            return await super().generate_async(prompt, **kwargs)
        
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        model = kwargs.pop("model", self.model)
        
        async_client = AsyncGroq(api_key=self.api_key)
        
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Have a multi-turn conversation.
        
        Works just like the OpenAI backend's chat method.
        """
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        model = kwargs.pop("model", self.model)
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def list_models(self) -> List[str]:
        """Get a list of available models on Groq."""
        return self.AVAILABLE_MODELS.copy()

