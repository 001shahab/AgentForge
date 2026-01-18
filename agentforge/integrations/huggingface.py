"""
HuggingFace Transformers integration for AgentForge.

This backend runs models locally using the Transformers library. It's great
for privacy-sensitive applications or when you want to avoid API costs.
Just be aware that you'll need a decent GPU for larger models.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agentforge.integrations.base import LLMIntegrator

logger = logging.getLogger("agentforge.integrations.huggingface")


class HuggingFaceBackend(LLMIntegrator):
    """
    Local HuggingFace Transformers backend.
    
    Run open-source models like Llama, Mistral, or Phi locally on your
    machine. This is perfect for development, privacy, or cost savings.
    
    Note: Larger models require significant GPU memory. I recommend:
    - 7B models: 16GB+ VRAM
    - 13B models: 24GB+ VRAM
    - 70B models: Multiple GPUs or quantization
    
    Environment Variables:
        HF_TOKEN: (Optional) HuggingFace token for gated models
        CUDA_VISIBLE_DEVICES: Control which GPUs to use
        
    Example:
        >>> from agentforge.integrations import HuggingFaceBackend
        >>> llm = HuggingFaceBackend(model="mistralai/Mistral-7B-Instruct-v0.2")
        >>> response = llm.generate("What is the capital of France?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        **kwargs
    ):
        """
        Initialize the HuggingFace backend.
        
        Args:
            model: Model identifier from HuggingFace Hub
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device to use ('cuda', 'cpu', or 'auto')
            torch_dtype: Data type ('float16', 'bfloat16', 'float32')
            load_in_8bit: Use 8-bit quantization (saves memory)
            load_in_4bit: Use 4-bit quantization (saves more memory)
            trust_remote_code: Allow custom model code (be careful!)
            **kwargs: Additional options for the model
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        
        # I'm using lazy loading here because model loading is slow
        self._pipeline = None
        self._tokenizer = None
    
    @property
    def default_model(self) -> str:
        return "mistralai/Mistral-7B-Instruct-v0.2"
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @property
    def pipeline(self):
        """Lazy-load the text generation pipeline."""
        if self._pipeline is None:
            self._load_model()
        return self._pipeline
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise ImportError(
                "Transformers library not installed. Please install with:\n"
                "pip install transformers torch\n"
                "or: pip install agentforge[huggingface]"
            )
        
        logger.info(f"Loading model: {self.model}")
        logger.info(f"Device: {self.device}")
        
        # Determine the torch dtype
        if self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.torch_dtype == "float32":
            dtype = torch.float32
        else:
            # Auto-select based on device
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Get HF token if available
        token = os.environ.get("HF_TOKEN")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
            token=token,
        )
        
        # Set up model loading kwargs
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "token": token,
        }
        
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        elif self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = dtype
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            **model_kwargs
        )
        
        # Create pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
        )
        
        logger.info(f"Model loaded successfully")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the local model.
        
        I'm formatting the prompt using the model's chat template if
        available, which usually gives better results.
        
        Args:
            prompt: The input prompt
            **kwargs: Override generation parameters
            
        Returns:
            The generated text
        """
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        
        logger.debug(f"Generating with local model {self.model}")
        
        # Format the prompt using chat template if available
        formatted_prompt = self._format_prompt(prompt)
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self.pipeline.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Add top_p if temperature > 0
        if temperature > 0:
            gen_kwargs["top_p"] = kwargs.get("top_p", 0.9)
        
        # Generate
        outputs = self.pipeline(
            formatted_prompt,
            **gen_kwargs
        )
        
        # Extract the generated text (without the prompt)
        full_text = outputs[0]["generated_text"]
        
        # Try to extract just the response
        if full_text.startswith(formatted_prompt):
            result = full_text[len(formatted_prompt):].strip()
        else:
            result = full_text.strip()
        
        logger.debug(f"Generated {len(result)} characters")
        
        return result
    
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt using the model's chat template."""
        try:
            # Try to use the chat template
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception:
            # Fall back to simple formatting if no chat template
            return f"User: {prompt}\n\nAssistant:"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Have a multi-turn conversation.
        
        Uses the model's chat template for proper formatting.
        """
        try:
            # Use chat template if available
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fall back to simple formatting
            parts = []
            for msg in messages:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            formatted = "\n\n".join(parts) + "\n\nAssistant:"
        
        return self.generate(formatted, **kwargs)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using the model's tokenizer.
        
        This gives an accurate count since we're using the actual tokenizer.
        """
        if self._tokenizer is None:
            # Load tokenizer if not already loaded
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            except Exception:
                return super().count_tokens(text)
        
        tokens = self._tokenizer.encode(text)
        return len(tokens)
    
    def unload_model(self) -> None:
        """
        Unload the model to free up memory.
        
        Call this when you're done generating and want to reclaim GPU memory.
        """
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Try to free GPU memory
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Model unloaded")

