"""
OpenAI API integration for AgentForge.

This backend connects to OpenAI's API for text generation. I've designed it
to be simple to use - just set your OPENAI_API_KEY environment variable
and you're good to go.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from agentforge.integrations.base import LLMIntegrator

logger = logging.getLogger("agentforge.integrations.openai")


class OpenAIBackend(LLMIntegrator):
    """
    OpenAI API backend for text generation.
    
    Uses OpenAI's chat completions API, which works with GPT-4, GPT-3.5,
    and other models. You'll need an API key from OpenAI to use this.
    
    Environment Variables:
        OPENAI_API_KEY: Your OpenAI API key
        OPENAI_ORG_ID: (Optional) Your organization ID
        OPENAI_BASE_URL: (Optional) Custom API base URL
        
    Example:
        >>> from agentforge.integrations import OpenAIBackend
        >>> llm = OpenAIBackend(model="gpt-4o-mini")
        >>> response = llm.generate("Tell me a joke")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI backend.
        
        Args:
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: API key (falls back to OPENAI_API_KEY env var)
            organization: Organization ID (falls back to OPENAI_ORG_ID env var)
            base_url: Custom API base URL (for proxies or compatible APIs)
            **kwargs: Additional options passed to the API
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        # I'm getting the API key from the environment if not provided directly
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY "
                "environment variable or pass api_key to the constructor."
            )
        
        self.organization = organization or os.environ.get("OPENAI_ORG_ID")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        
        # I'm lazy-loading the client to avoid import errors if openai isn't installed
        self._client = None
    
    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"
    
    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI library not installed. Please install it with:\n"
                    "pip install openai\n"
                    "or: pip install agentforge[openai]"
                )
            
            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )
        
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's API.
        
        I'm using the chat completions endpoint here because it's more
        flexible and works with all the modern models.
        
        Args:
            prompt: The input prompt
            **kwargs: Override any generation parameters
            
        Returns:
            The generated text
        """
        # Allow kwargs to override instance settings
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        model = kwargs.pop("model", self.model)
        
        logger.debug(f"Generating with OpenAI {model}")
        
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
        Async version using OpenAI's async client.
        
        OpenAI's Python library has native async support, so I'm using
        that here for better performance.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            # Fall back to sync-in-executor if async client not available
            return await super().generate_async(prompt, **kwargs)
        
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        model = kwargs.pop("model", self.model)
        
        async_client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url,
        )
        
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
        
        This method handles the chat format properly for OpenAI's API,
        which expects a list of messages with role and content.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            The assistant's response
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
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken if available.
        
        This gives an accurate token count for OpenAI models.
        """
        try:
            import tiktoken
            
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except ImportError:
            # Fall back to rough estimate
            return super().count_tokens(text)

