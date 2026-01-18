"""
Voice synthesis plugin for AgentForge.

This plugin converts text to speech using Google's gTTS or other TTS engines.
It's great for creating audio content, accessibility features, or
voice-enabled applications.

Author: Prof. Shahab Anbarjafari

Note: Requires gtts. Install with: pip install agentforge[voice]
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentforge.core import Skill

logger = logging.getLogger("agentforge.plugins.voice")


class VoiceSynthesisPlugin(Skill):
    """
    Convert text to speech.
    
    This plugin generates audio from text using text-to-speech engines.
    By default it uses Google TTS (gTTS), which is free and supports
    many languages.
    
    Input:
        - text: Text to convert to speech
        - content: Alternative to 'text' (for chaining)
        - language: Language code (default: 'en')
        - slow: Speak slowly (default: False)
        - save_path: Path to save the audio file
        
    Output:
        - audio_base64: Base64-encoded MP3 audio
        - save_path: Where the audio was saved (if requested)
        - duration_estimate: Estimated duration in seconds
        - language: Language used
        
    Example:
        >>> voice = VoiceSynthesisPlugin()
        >>> result = voice.execute({
        ...     "text": "Hello, welcome to AgentForge!",
        ...     "save_path": "greeting.mp3"
        ... })
        >>> # Audio saved to greeting.mp3
    """
    
    name = "voice_synthesis"
    description = "Convert text to speech audio"
    requires_llm = False
    
    # Supported languages (subset of gTTS supported languages)
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh-CN": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
        "ar": "Arabic",
        "hi": "Hindi",
        "nl": "Dutch",
        "pl": "Polish",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "th": "Thai",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "no": "Norwegian",
        "el": "Greek",
        "he": "Hebrew",
        "id": "Indonesian",
        "ms": "Malay",
        "cs": "Czech",
        "ro": "Romanian",
        "hu": "Hungarian",
        "uk": "Ukrainian",
    }
    
    def __init__(
        self,
        engine: str = "gtts",
        default_language: str = "en",
        **kwargs
    ):
        """
        Initialize the voice synthesis plugin.
        
        Args:
            engine: TTS engine to use ('gtts' is currently supported)
            default_language: Default language for synthesis
            **kwargs: Additional options
        """
        super().__init__(**kwargs)
        self.engine = engine
        self.default_language = default_language
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert text to speech.
        
        Args:
            input_data: Dictionary with text and options
            
        Returns:
            Dictionary with audio data
        """
        # Get text from various sources
        text = input_data.get("text") or input_data.get("content") or input_data.get("generated")
        
        if not text:
            raise ValueError("No text provided. Please specify 'text' or 'content'.")
        
        language = input_data.get("language", self.default_language)
        slow = input_data.get("slow", False)
        save_path = input_data.get("save_path")
        
        logger.info(f"Synthesizing speech: {text[:50]}... (lang={language})")
        
        if self.engine == "gtts":
            audio_data = self._synthesize_gtts(text, language, slow)
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine}")
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        
        # Estimate duration (rough: ~150 words per minute)
        word_count = len(text.split())
        duration_estimate = word_count / 150 * 60  # seconds
        if slow:
            duration_estimate *= 1.5
        
        result = {
            "audio_base64": audio_base64,
            "language": language,
            "word_count": word_count,
            "duration_estimate": round(duration_estimate, 1),
            "format": "mp3",
        }
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "wb") as f:
                f.write(audio_data)
            
            result["save_path"] = str(save_path)
            logger.info(f"Audio saved to: {save_path}")
        
        return result
    
    def _synthesize_gtts(self, text: str, language: str, slow: bool) -> bytes:
        """Synthesize speech using gTTS."""
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError(
                "gTTS not installed. Please install with:\n"
                "pip install gtts\n"
                "or: pip install agentforge[voice]"
            )
        
        # Split long text into chunks (gTTS has limits)
        max_chars = 5000
        chunks = self._split_text(text, max_chars)
        
        # Generate audio for each chunk
        audio_parts = []
        for chunk in chunks:
            tts = gTTS(text=chunk, lang=language, slow=slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            audio_parts.append(buf.read())
        
        # Combine if multiple chunks
        if len(audio_parts) == 1:
            return audio_parts[0]
        else:
            # Simple concatenation (works for MP3)
            return b"".join(audio_parts)
    
    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks at sentence boundaries."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences
        sentences = text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def list_languages(self) -> Dict[str, str]:
        """Get a dictionary of supported language codes and names."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of execute.
        
        Note: gTTS makes network requests, so this runs in an executor
        to avoid blocking.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data)


class AudioTranscriptionPlugin(Skill):
    """
    Transcribe audio to text.
    
    This is a placeholder for future audio transcription capabilities.
    You could integrate with Whisper or other ASR services here.
    
    Note: Not yet implemented. This is here to show the plugin pattern.
    """
    
    name = "audio_transcription"
    description = "Transcribe audio to text (coming soon)"
    requires_llm = False
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio to text."""
        raise NotImplementedError(
            "Audio transcription is not yet implemented. "
            "Consider using OpenAI Whisper or similar services."
        )

