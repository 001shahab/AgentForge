"""
Configuration loading utilities for AgentForge.

I wanted to make it super easy to configure agents using YAML files, so this
module handles all the parsing and validation. You can define your entire
agent workflow in a simple YAML file and load it with one function call.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load an agent configuration from a YAML file.
    
    I've designed the config format to be intuitive and flexible. You can
    define skills, LLM settings, and various options all in one file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is malformed
        
    Example YAML:
        ```yaml
        name: my_agent
        
        llm:
          backend: openai
          model: gpt-4
          
        skills:
          - web_scraper
          - skill: content_generation
            prompt_template: summarize
        ```
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # I'm doing some basic validation here
    if config is None:
        config = {}
    
    # Resolve any environment variable references
    config = _resolve_env_vars(config)
    
    return config


def _resolve_env_vars(obj: Any) -> Any:
    """
    Recursively resolve environment variable references in config.
    
    I'm supporting a simple ${VAR_NAME} syntax, which makes it easy to
    keep secrets out of your config files.
    """
    if isinstance(obj, str):
        # Check for ${VAR} pattern
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            value = os.environ.get(var_name)
            if value is None:
                # I'm not raising an error here because the var might be optional
                return obj
            return value
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    else:
        return obj


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save a configuration to a YAML file.
    
    This is handy when you've built up a config programmatically and want
    to persist it for later use.
    
    Args:
        config: Configuration dictionary to save
        config_path: Where to save the YAML file
    """
    config_path = Path(config_path)
    
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_template_config(
    name: str = "my_agent",
    skills: Optional[List[str]] = None,
    llm_backend: str = "openai",
) -> Dict[str, Any]:
    """
    Create a template configuration dictionary.
    
    I've included this to make it easy to get started - just call this
    function and you'll have a sensible starting point.
    
    Args:
        name: Name for the agent
        skills: List of skill names to include
        llm_backend: Which LLM backend to use
        
    Returns:
        A template configuration dictionary
    """
    if skills is None:
        skills = ["web_scraper", "content_generation"]
    
    return {
        "name": name,
        "version": "1.0",
        "description": "My AgentForge agent",
        
        "llm": {
            "backend": llm_backend,
            "model": _get_default_model(llm_backend),
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        
        "skills": skills,
        
        "options": {
            "continue_on_error": False,
            "max_retries": 3,
            "logging_level": "INFO",
        },
    }


def _get_default_model(backend: str) -> str:
    """Get a sensible default model for each backend."""
    defaults = {
        "openai": "gpt-4o-mini",
        "groq": "llama-3.1-70b-versatile",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
    }
    return defaults.get(backend, "gpt-4o-mini")


class ConfigValidator:
    """
    Validates agent configurations.
    
    I'm using this to catch configuration errors early, before you try
    to actually run the agent. It checks for common mistakes like missing
    required fields or invalid skill names.
    """
    
    REQUIRED_FIELDS = ["name"]
    VALID_BACKENDS = ["openai", "groq", "huggingface"]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if the config is valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        self._check_required_fields()
        self._check_llm_config()
        self._check_skills_config()
        
        return len(self.errors) == 0
    
    def _check_required_fields(self) -> None:
        """Check that all required fields are present."""
        for field in self.REQUIRED_FIELDS:
            if field not in self.config:
                self.errors.append(f"Missing required field: '{field}'")
    
    def _check_llm_config(self) -> None:
        """Validate the LLM configuration."""
        llm_config = self.config.get("llm", {})
        
        if not llm_config:
            self.warnings.append(
                "No LLM configuration found. Skills requiring LLM won't work."
            )
            return
        
        backend = llm_config.get("backend")
        if backend and backend not in self.VALID_BACKENDS:
            self.errors.append(
                f"Invalid LLM backend: '{backend}'. "
                f"Valid options: {self.VALID_BACKENDS}"
            )
    
    def _check_skills_config(self) -> None:
        """Validate the skills configuration."""
        skills = self.config.get("skills", [])
        
        if not skills:
            self.warnings.append("No skills defined. Agent won't do anything.")
            return
        
        # We could check if skills are registered, but that requires
        # importing the registry which might have side effects
        for skill in skills:
            if isinstance(skill, dict) and "skill" not in skill and "name" not in skill:
                self.errors.append(
                    f"Skill configuration must have 'skill' or 'name' key: {skill}"
                )
    
    def get_report(self) -> str:
        """Get a human-readable validation report."""
        lines = []
        
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")
        
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        if not self.errors and not self.warnings:
            lines.append("✓ Configuration is valid!")
        
        return "\n".join(lines)

