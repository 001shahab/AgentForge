"""
Core module for AgentForge.

This is where the magic happens. I've designed the Agent and Skill classes
to be as flexible as possible while staying simple to use. The key idea is
that skills are modular building blocks, and agents chain them together.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union

from tenacity import retry, stop_after_attempt, wait_exponential

# I'm setting up logging here so users can see what's happening
logger = logging.getLogger("agentforge")


class Skill(ABC):
    """
    Base class for all skills in AgentForge.
    
    A skill is a modular unit of work - it takes some input, does something
    useful, and returns output. I've made it abstract so you're forced to
    implement the execute() method, which is where your actual logic goes.
    
    Attributes:
        name: Human-readable name for this skill
        description: What this skill does (used for documentation and LLM context)
        requires_llm: Whether this skill needs an LLM backend to function
    
    Example:
        >>> class MyCustomSkill(Skill):
        ...     name = "my_skill"
        ...     description = "Does something amazing"
        ...     
        ...     def execute(self, input_data: dict) -> dict:
        ...         # Your logic here
        ...         return {"result": "amazing!"}
    """
    
    name: str = "base_skill"
    description: str = "Base skill class"
    requires_llm: bool = False
    
    def __init__(self, **kwargs):
        """
        Initialize the skill with optional configuration.
        
        I'm accepting **kwargs here so subclasses can easily add their own
        parameters without having to override __init__ every time.
        """
        self.config = kwargs
        self._llm = None
    
    def set_llm(self, llm: "LLMIntegrator") -> None:
        """Attach an LLM backend to this skill."""
        self._llm = llm
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the skill's main logic.
        
        This is the method you need to implement in your subclass. It should
        take a dictionary of inputs and return a dictionary of outputs.
        
        Args:
            input_data: Dictionary containing input data for the skill
            
        Returns:
            Dictionary containing the skill's output
            
        Raises:
            NotImplementedError: If you forget to implement this in your subclass
        """
        raise NotImplementedError("You need to implement execute() in your skill")
    
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of execute.
        
        By default, I'm just wrapping the sync execute() in an executor.
        Override this if your skill can do true async operations.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate that the input data has everything we need.
        
        Override this to add custom validation logic. Returns True by default.
        """
        return True
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class SkillRegistry:
    """
    A registry for discovering and instantiating skills.
    
    I've created this so you can register custom skills and then reference
    them by name in YAML configs. It's a simple pattern but makes the
    configuration-driven approach much cleaner.
    
    Example:
        >>> registry = SkillRegistry()
        >>> registry.register("my_skill", MyCustomSkill)
        >>> skill = registry.get("my_skill")
    """
    
    _instance: Optional["SkillRegistry"] = None
    
    def __new__(cls) -> "SkillRegistry":
        # I'm using a singleton pattern here so all parts of the app
        # share the same registry
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills = {}
            cls._instance._register_builtins()
        return cls._instance
    
    def _register_builtins(self) -> None:
        """Register the built-in skills that come with AgentForge."""
        # I'm importing here to avoid circular imports
        from agentforge.skills.scraping import WebScraperSkill
        from agentforge.skills.analysis import DataAnalysisSkill
        from agentforge.skills.generation import ContentGenerationSkill
        
        self._skills["web_scraper"] = WebScraperSkill
        self._skills["data_analysis"] = DataAnalysisSkill
        self._skills["content_generation"] = ContentGenerationSkill
        
        # Also register the plugins if they're available
        try:
            from agentforge.plugins.rss import RSSMonitorPlugin
            self._skills["rss_monitor"] = RSSMonitorPlugin
        except ImportError:
            pass  # feedparser not installed
        
        try:
            from agentforge.plugins.image_gen import ImageGenerationPlugin
            self._skills["image_generation"] = ImageGenerationPlugin
        except ImportError:
            pass  # diffusers not installed
        
        try:
            from agentforge.plugins.voice import VoiceSynthesisPlugin
            self._skills["voice_synthesis"] = VoiceSynthesisPlugin
        except ImportError:
            pass  # gtts not installed
    
    def register(self, name: str, skill_class: Type[Skill]) -> None:
        """
        Register a new skill class.
        
        Args:
            name: The name to register the skill under
            skill_class: The skill class (not instance) to register
        """
        if not issubclass(skill_class, Skill):
            raise TypeError(f"Expected a Skill subclass, got {type(skill_class)}")
        self._skills[name] = skill_class
        logger.debug(f"Registered skill: {name}")
    
    def get(self, name: str, **kwargs) -> Skill:
        """
        Get an instance of a registered skill.
        
        Args:
            name: The name of the skill to instantiate
            **kwargs: Arguments to pass to the skill's constructor
            
        Returns:
            An instance of the requested skill
            
        Raises:
            KeyError: If no skill with that name is registered
        """
        if name not in self._skills:
            available = ", ".join(self._skills.keys())
            raise KeyError(f"Unknown skill: '{name}'. Available: {available}")
        return self._skills[name](**kwargs)
    
    def list_skills(self) -> List[str]:
        """Get a list of all registered skill names."""
        return list(self._skills.keys())
    
    def get_skill_info(self, name: str) -> Dict[str, str]:
        """Get information about a registered skill."""
        skill_class = self._skills.get(name)
        if skill_class is None:
            raise KeyError(f"Unknown skill: '{name}'")
        return {
            "name": skill_class.name,
            "description": skill_class.description,
            "requires_llm": skill_class.requires_llm,
        }


class Agent:
    """
    The main agent class that chains skills together.
    
    I've designed this to be the central orchestrator of your automation
    workflow. You give it a list of skills and optionally an LLM backend,
    and it handles running them in sequence (or parallel if you want).
    
    The output of each skill becomes the input to the next one, which makes
    it easy to build complex pipelines from simple building blocks.
    
    Attributes:
        skills: List of skills to run in order
        llm: Optional LLM backend for skills that need it
        config: Configuration dictionary or path to YAML file
        
    Example:
        >>> from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
        >>> from agentforge.integrations import OpenAIBackend
        >>> 
        >>> agent = Agent(
        ...     skills=[WebScraperSkill(), ContentGenerationSkill()],
        ...     llm=OpenAIBackend()
        ... )
        >>> result = agent.run({"url": "https://news.example.com"})
        >>> print(result["summary"])
    """
    
    def __init__(
        self,
        skills: Optional[List[Skill]] = None,
        llm: Optional["LLMIntegrator"] = None,
        config: Optional[Union[Dict[str, Any], str]] = None,
        name: str = "agent",
        max_retries: int = 3,
        continue_on_error: bool = False,
    ):
        """
        Initialize the agent.
        
        Args:
            skills: List of Skill instances to chain together
            llm: LLM backend for skills that need text generation
            config: Configuration dict or path to YAML file
            name: A name for this agent (useful for logging)
            max_retries: How many times to retry failed operations
            continue_on_error: If True, skip failed skills instead of stopping
        """
        self.max_retries = max_retries
        self.continue_on_error = continue_on_error
        self._llm = llm
        
        # If config is provided, we'll load skills from there
        if config is not None:
            from agentforge.config import load_config
            self._config = load_config(config) if isinstance(config, str) else config
            self.skills = self._build_skills_from_config()
            # Use name from config if not explicitly provided
            self.name = self._config.get("name", name)
        else:
            self._config = {}
            self.skills = skills or []
            self.name = name
        
        # Attach the LLM to any skills that need it
        self._attach_llm_to_skills()
        
        logger.info(f"Agent '{self.name}' initialized with {len(self.skills)} skills")
    
    def _build_skills_from_config(self) -> List[Skill]:
        """Build skills from the configuration."""
        registry = SkillRegistry()
        skills = []
        
        for skill_config in self._config.get("skills", []):
            if isinstance(skill_config, str):
                # Simple case: just the skill name
                skills.append(registry.get(skill_config))
            elif isinstance(skill_config, dict):
                # Complex case: skill name with parameters
                skill_name = skill_config.pop("skill", skill_config.pop("name", None))
                if skill_name is None:
                    raise ValueError("Skill config must have 'skill' or 'name' key")
                skills.append(registry.get(skill_name, **skill_config))
        
        return skills
    
    def _attach_llm_to_skills(self) -> None:
        """Give the LLM backend to skills that need it."""
        if self._llm is None:
            return
        
        for skill in self.skills:
            if skill.requires_llm:
                skill.set_llm(self._llm)
    
    @property
    def llm(self) -> Optional["LLMIntegrator"]:
        """Get the LLM backend."""
        return self._llm
    
    @llm.setter
    def llm(self, value: "LLMIntegrator") -> None:
        """Set the LLM backend and update all skills."""
        self._llm = value
        self._attach_llm_to_skills()
    
    def add_skill(self, skill: Skill) -> "Agent":
        """
        Add a skill to the chain.
        
        I'm returning self here so you can chain these calls:
        agent.add_skill(skill1).add_skill(skill2)
        """
        self.skills.append(skill)
        if skill.requires_llm and self._llm is not None:
            skill.set_llm(self._llm)
        return self
    
    def run(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent synchronously.
        
        This executes each skill in order, passing the output of one
        as the input to the next. The final output contains everything
        accumulated along the way.
        
        Args:
            initial_input: The starting data for the first skill
            
        Returns:
            The accumulated output from all skills
        """
        logger.info(f"Agent '{self.name}' starting run with {len(self.skills)} skills")
        
        current_data = initial_input.copy()
        
        for i, skill in enumerate(self.skills):
            logger.debug(f"Running skill {i + 1}/{len(self.skills)}: {skill.name}")
            
            try:
                # Validate input
                if not skill.validate_input(current_data):
                    raise ValueError(f"Input validation failed for skill '{skill.name}'")
                
                # Execute with retry logic
                result = self._execute_with_retry(skill, current_data)
                
                # Merge results into current data
                current_data.update(result)
                
            except Exception as e:
                logger.error(f"Skill '{skill.name}' failed: {e}")
                if self.continue_on_error:
                    current_data["_errors"] = current_data.get("_errors", [])
                    current_data["_errors"].append({
                        "skill": skill.name,
                        "error": str(e)
                    })
                else:
                    raise
        
        logger.info(f"Agent '{self.name}' completed run")
        return current_data
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _execute_with_retry(self, skill: Skill, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a skill with retry logic."""
        return skill.execute(data)
    
    async def run_async(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent asynchronously.
        
        Same as run(), but uses async execution for better performance
        when dealing with I/O-bound operations.
        """
        logger.info(f"Agent '{self.name}' starting async run")
        
        current_data = initial_input.copy()
        
        for i, skill in enumerate(self.skills):
            logger.debug(f"Running skill {i + 1}/{len(self.skills)}: {skill.name}")
            
            try:
                if not skill.validate_input(current_data):
                    raise ValueError(f"Input validation failed for skill '{skill.name}'")
                
                result = await skill.execute_async(current_data)
                current_data.update(result)
                
            except Exception as e:
                logger.error(f"Skill '{skill.name}' failed: {e}")
                if self.continue_on_error:
                    current_data["_errors"] = current_data.get("_errors", [])
                    current_data["_errors"].append({
                        "skill": skill.name,
                        "error": str(e)
                    })
                else:
                    raise
        
        logger.info(f"Agent '{self.name}' completed async run")
        return current_data
    
    async def run_parallel(
        self,
        initial_input: Dict[str, Any],
        skill_groups: Optional[List[List[int]]] = None
    ) -> Dict[str, Any]:
        """
        Run skills with some parallelism.
        
        By default, all skills run in sequence. But if you provide skill_groups,
        skills within each group run in parallel. This is useful when you have
        independent tasks that don't depend on each other.
        
        Args:
            initial_input: Starting data
            skill_groups: List of lists, where each inner list contains indices
                         of skills that can run in parallel. If None, each skill
                         is its own group (sequential execution).
        
        Example:
            # Skills 0 and 1 run in parallel, then skill 2
            agent.run_parallel(data, skill_groups=[[0, 1], [2]])
        """
        if skill_groups is None:
            # Default to sequential
            skill_groups = [[i] for i in range(len(self.skills))]
        
        current_data = initial_input.copy()
        
        for group in skill_groups:
            # Run all skills in this group in parallel
            tasks = [
                self.skills[i].execute_async(current_data)
                for i in group
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge all results
            for result in results:
                if isinstance(result, Exception):
                    if not self.continue_on_error:
                        raise result
                    current_data["_errors"] = current_data.get("_errors", [])
                    current_data["_errors"].append({"error": str(result)})
                else:
                    current_data.update(result)
        
        return current_data
    
    def __repr__(self) -> str:
        skill_names = [s.name for s in self.skills]
        return f"<Agent(name='{self.name}', skills={skill_names})>"


# For type hints without circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agentforge.integrations.base import LLMIntegrator

