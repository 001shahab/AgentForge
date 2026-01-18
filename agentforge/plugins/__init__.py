"""
Extension plugins for AgentForge.

These are optional capabilities that extend what your agents can do.
Each plugin is designed to work as a skill, so you can drop them into
any agent workflow.

Available plugins:
    - RSSMonitorPlugin: Monitor RSS feeds for new content
    - ImageGenerationPlugin: Generate images with Stable Diffusion
    - VoiceSynthesisPlugin: Convert text to speech

Note: Some plugins require additional dependencies. Check the docs or
install with: pip install agentforge[rss,image,voice]
"""

from agentforge.plugins.rss import RSSMonitorPlugin
from agentforge.plugins.image_gen import ImageGenerationPlugin
from agentforge.plugins.voice import VoiceSynthesisPlugin

__all__ = [
    "RSSMonitorPlugin",
    "ImageGenerationPlugin",
    "VoiceSynthesisPlugin",
]

