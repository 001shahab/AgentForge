"""
Built-in skills for AgentForge.

I've included some of the most common automation tasks as ready-to-use skills.
You can use these directly or subclass them to customize the behavior.

Available skills:
    - WebScraperSkill: Fetch and parse web pages
    - DataAnalysisSkill: Analyze data with pandas
    - ContentGenerationSkill: Generate text using LLMs
"""

from agentforge.skills.scraping import WebScraperSkill
from agentforge.skills.analysis import DataAnalysisSkill
from agentforge.skills.generation import ContentGenerationSkill

__all__ = [
    "WebScraperSkill",
    "DataAnalysisSkill",
    "ContentGenerationSkill",
]

