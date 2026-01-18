"""
Content generation skill for AgentForge.

This skill uses an LLM to generate text content. I've designed it to handle
common tasks like summarization, rewriting, and creative writing. It's the
skill you'll use when you need AI-powered text generation in your agents.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agentforge.core import Skill

logger = logging.getLogger("agentforge.skills.generation")


class ContentGenerationSkill(Skill):
    """
    Generate text content using an LLM.
    
    This skill handles various text generation tasks. It can summarize,
    rewrite, expand, or generate new content based on your prompts.
    
    Input:
        - prompt: Direct prompt for the LLM (if not using templates)
        - template: Name of a prompt template to use
        - text: Text for templates that need it (summarize, rewrite, etc.)
        - content: Alternative to 'text' (for chaining with scraper)
        - style: Writing style for rewrite template
        - question: Question for chain-of-thought template
        
    Output:
        - generated: The generated text
        - template_used: Which template was used (if any)
        - model: Which model generated the response
        
    Example:
        >>> gen = ContentGenerationSkill()
        >>> gen.set_llm(OpenAIBackend())
        >>> result = gen.execute({
        ...     "template": "summarize",
        ...     "text": "Long article content here..."
        ... })
        >>> print(result["generated"])
    """
    
    name = "content_generation"
    description = "Generate text content using an LLM (summarize, rewrite, create)"
    requires_llm = True
    
    # Default generation parameters
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1024
    
    def __init__(
        self,
        default_template: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ):
        """
        Initialize the content generation skill.
        
        Args:
            default_template: Template to use if none specified
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options
        """
        super().__init__(**kwargs)
        self.default_template = default_template
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content based on the input.
        
        Args:
            input_data: Dictionary with prompt or template configuration
            
        Returns:
            Dictionary with generated content
        """
        if not self._llm:
            raise RuntimeError(
                "No LLM configured. Please attach an LLM backend to use "
                "the content generation skill."
            )
        
        # Get generation parameters
        temperature = input_data.get("temperature", self.temperature)
        max_tokens = input_data.get("max_tokens", self.max_tokens)
        
        # Determine what to generate
        prompt = input_data.get("prompt")
        template = input_data.get("template", self.default_template)
        
        result = {
            "model": getattr(self._llm, "model", "unknown"),
        }
        
        if prompt:
            # Direct prompt - just generate
            logger.debug("Generating from direct prompt")
            generated = self._llm.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result["generated"] = generated
            result["template_used"] = None
            
        elif template:
            # Use a template
            logger.debug(f"Generating from template: {template}")
            generated = self._generate_from_template(input_data, template)
            result["generated"] = generated
            result["template_used"] = template
            
        else:
            # Maybe we got text/content from a previous skill?
            text = input_data.get("text") or input_data.get("content")
            if text:
                # Default to summarization
                logger.debug("No prompt or template, defaulting to summarize")
                generated = self._llm.generate_with_template("summarize", text=text)
                result["generated"] = generated
                result["template_used"] = "summarize"
            else:
                raise ValueError(
                    "No prompt, template, or text provided. Please provide one of: "
                    "'prompt' (direct text), 'template' (template name), or "
                    "'text'/'content' (text to process)"
                )
        
        logger.info(f"Generated {len(result['generated'])} characters")
        
        return result
    
    def _generate_from_template(self, input_data: Dict[str, Any], template: str) -> str:
        """Generate content using a specific template."""
        # Get text from various possible sources
        text = input_data.get("text") or input_data.get("content", "")
        
        if template == "summarize":
            if not text:
                raise ValueError("Template 'summarize' requires 'text' or 'content'")
            return self._llm.generate_with_template("summarize", text=text)
        
        elif template == "analyze":
            if not text:
                raise ValueError("Template 'analyze' requires 'text' or 'content'")
            return self._llm.generate_with_template("analyze", text=text)
        
        elif template == "chain_of_thought":
            question = input_data.get("question", text)
            if not question:
                raise ValueError("Template 'chain_of_thought' requires 'question'")
            return self._llm.generate_with_template("chain_of_thought", question=question)
        
        elif template == "rewrite":
            style = input_data.get("style", "clear and professional")
            if not text:
                raise ValueError("Template 'rewrite' requires 'text' or 'content'")
            return self._llm.generate_with_template("rewrite", text=text, style=style)
        
        elif template == "classify":
            categories = input_data.get("categories", [])
            if not text:
                raise ValueError("Template 'classify' requires 'text' or 'content'")
            if not categories:
                raise ValueError("Template 'classify' requires 'categories'")
            return self._llm.generate_with_template(
                "classify",
                text=text,
                categories=", ".join(categories)
            )
        
        elif template == "extract":
            fields = input_data.get("fields", [])
            if not text:
                raise ValueError("Template 'extract' requires 'text' or 'content'")
            if not fields:
                raise ValueError("Template 'extract' requires 'fields'")
            return self._llm.generate_with_template(
                "extract",
                text=text,
                fields=", ".join(fields)
            )
        
        else:
            # Try to use it as a custom template
            logger.warning(f"Unknown template '{template}', treating as custom prompt")
            prompt = template.format(**input_data)
            return self._llm.generate(prompt)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate that we have something to generate from."""
        has_prompt = "prompt" in input_data
        has_template = "template" in input_data or self.default_template
        has_text = "text" in input_data or "content" in input_data
        
        return has_prompt or has_template or has_text


class ChatSkill(Skill):
    """
    Have a conversation with an LLM.
    
    Unlike ContentGenerationSkill which is for one-shot generation, this
    skill maintains conversation context for multi-turn dialogues.
    
    Input:
        - message: The user's message
        - history: (Optional) Previous conversation history
        - system_prompt: (Optional) System prompt to set behavior
        
    Output:
        - response: The assistant's response
        - history: Updated conversation history
    """
    
    name = "chat"
    description = "Have a multi-turn conversation with an LLM"
    requires_llm = True
    
    def __init__(self, system_prompt: Optional[str] = None, **kwargs):
        """
        Initialize the chat skill.
        
        Args:
            system_prompt: Default system prompt for conversations
            **kwargs: Additional options
        """
        super().__init__(**kwargs)
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant. Be concise and accurate."
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chat message and generate a response."""
        if not self._llm:
            raise RuntimeError("No LLM configured for chat")
        
        message = input_data.get("message", "")
        if not message:
            return {"response": "", "error": "No message provided"}
        
        # Get or initialize history
        history = input_data.get("history", [])
        system_prompt = input_data.get("system_prompt", self.system_prompt)
        
        # Build messages list
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history
        messages.extend(history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = self._llm.chat(messages)
        
        # Update history
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        
        return {
            "response": response,
            "history": new_history,
        }


class TranslationSkill(Skill):
    """
    Translate text between languages using an LLM.
    
    Input:
        - text: Text to translate
        - source_language: (Optional) Source language (auto-detected if not specified)
        - target_language: Target language (required)
        
    Output:
        - translated: Translated text
        - source_language: Detected or specified source language
        - target_language: Target language
    """
    
    name = "translation"
    description = "Translate text between languages"
    requires_llm = True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate text."""
        if not self._llm:
            raise RuntimeError("No LLM configured for translation")
        
        text = input_data.get("text") or input_data.get("content", "")
        target_language = input_data.get("target_language", "English")
        source_language = input_data.get("source_language", "auto-detected")
        
        if not text:
            return {"translated": "", "error": "No text provided"}
        
        prompt = f"""Translate the following text to {target_language}.
Only output the translation, nothing else.

Text to translate:
{text}

Translation:"""
        
        translated = self._llm.generate(prompt, temperature=0.3)
        
        return {
            "translated": translated.strip(),
            "source_language": source_language,
            "target_language": target_language,
        }

