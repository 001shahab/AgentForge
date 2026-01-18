"""
Image generation plugin for AgentForge.

This plugin generates images using Stable Diffusion models through the
diffusers library. It's great for creating illustrations, thumbnails,
or any visual content your agents might need.

Author: Prof. Shahab Anbarjafari

Note: Requires diffusers and torch. Install with: pip install agentforge[image]
      You'll also need a decent GPU (8GB+ VRAM recommended).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

from agentforge.core import Skill

logger = logging.getLogger("agentforge.plugins.image_gen")


class ImageGenerationPlugin(Skill):
    """
    Generate images using Stable Diffusion.
    
    This plugin creates images from text prompts. It uses the diffusers
    library to run Stable Diffusion models locally.
    
    Input:
        - prompt: Text description of the image to generate
        - negative_prompt: Things to avoid in the image
        - width: Image width (default: 512)
        - height: Image height (default: 512)
        - num_steps: Number of inference steps (default: 50)
        - guidance_scale: CFG scale (default: 7.5)
        - seed: Random seed for reproducibility
        - save_path: Path to save the image (optional)
        
    Output:
        - image_base64: Base64-encoded PNG image
        - save_path: Where the image was saved (if requested)
        - prompt: The prompt used
        - seed: The seed used
        
    Example:
        >>> gen = ImageGenerationPlugin(model="stabilityai/stable-diffusion-2-1")
        >>> result = gen.execute({
        ...     "prompt": "A serene mountain landscape at sunset",
        ...     "width": 768,
        ...     "height": 512
        ... })
        >>> # Save the base64 image or use result["save_path"]
    """
    
    name = "image_generation"
    description = "Generate images from text prompts using Stable Diffusion"
    requires_llm = False
    
    DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1-base"
    
    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "float16",
        enable_attention_slicing: bool = True,
        **kwargs
    ):
        """
        Initialize the image generator.
        
        Args:
            model: HuggingFace model ID (default: SD 2.1 base)
            device: Device to run on ('cuda', 'cpu', 'mps')
            torch_dtype: Data type ('float16', 'float32')
            enable_attention_slicing: Reduce memory usage (slower but works on smaller GPUs)
            **kwargs: Additional options
        """
        super().__init__(**kwargs)
        self.model_id = model or self.DEFAULT_MODEL
        self.device = device
        self.torch_dtype = torch_dtype
        self.enable_attention_slicing = enable_attention_slicing
        self._pipeline = None
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.device:
            return self.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    @property
    def pipeline(self):
        """Lazy-load the diffusion pipeline."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline
    
    def _load_pipeline(self) -> None:
        """Load the Stable Diffusion pipeline."""
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError(
                "diffusers and/or torch not installed. Please install with:\n"
                "pip install diffusers torch\n"
                "or: pip install agentforge[image]"
            )
        
        device = self._get_device()
        logger.info(f"Loading image model: {self.model_id}")
        logger.info(f"Device: {device}")
        
        # Determine dtype
        if self.torch_dtype == "float16" and device != "cpu":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Get HF token if available
        token = os.environ.get("HF_TOKEN")
        
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            token=token,
        )
        
        self._pipeline = self._pipeline.to(device)
        
        if self.enable_attention_slicing:
            self._pipeline.enable_attention_slicing()
        
        logger.info("Image model loaded successfully")
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.
        
        Args:
            input_data: Dictionary with prompt and generation options
            
        Returns:
            Dictionary with generated image
        """
        prompt = input_data.get("prompt")
        if not prompt:
            raise ValueError("No prompt provided. Please specify 'prompt'.")
        
        negative_prompt = input_data.get("negative_prompt", "")
        width = input_data.get("width", 512)
        height = input_data.get("height", 512)
        num_steps = input_data.get("num_steps", 50)
        guidance_scale = input_data.get("guidance_scale", 7.5)
        seed = input_data.get("seed")
        save_path = input_data.get("save_path")
        
        logger.info(f"Generating image: {prompt[:50]}...")
        
        # Set up generator for reproducibility
        import torch
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._get_device()).manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=self._get_device()).manual_seed(seed)
        
        # Generate the image
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        image = result.images[0]
        
        # Convert to base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        output = {
            "image_base64": image_base64,
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
        }
        
        # Save if path provided
        if save_path:
            image.save(save_path)
            output["save_path"] = save_path
            logger.info(f"Image saved to: {save_path}")
        
        logger.info("Image generated successfully")
        
        return output
    
    def generate_multiple(
        self,
        prompt: str,
        count: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple images from the same prompt.
        
        This is useful for getting variations to choose from.
        
        Args:
            prompt: Text prompt
            count: Number of images to generate
            **kwargs: Additional generation options
            
        Returns:
            List of generation results
        """
        results = []
        
        for i in range(count):
            logger.info(f"Generating image {i + 1}/{count}")
            result = self.execute({"prompt": prompt, **kwargs})
            results.append(result)
        
        return results
    
    def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            
            try:
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("Image model unloaded")

