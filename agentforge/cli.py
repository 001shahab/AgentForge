"""
Command-line interface for AgentForge.

I've built this CLI to make it easy to run agents from the terminal.
You can initialize new projects, run agent configs, and explore
available skills without writing any code.

Author: Prof. Shahab Anbarjafari

Usage:
    agentforge init my_agent
    agentforge run config.yaml
    agentforge list-skills
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("agentforge.cli")


@click.group()
@click.version_option(version="0.1.0", prog_name="AgentForge")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool):
    """
    AgentForge - Build modular AI agents with ease.
    
    A lightweight framework for creating AI agents that chain together
    tasks like web scraping, data analysis, and content generation.
    
    Designed by Prof. Shahab Anbarjafari, 3S Holding OÜ.
    """
    if debug:
        logging.getLogger("agentforge").setLevel(logging.DEBUG)


@main.command()
@click.argument("name")
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--template", "-t", default="basic", 
              type=click.Choice(["basic", "scraper", "analyzer", "full"]),
              help="Project template to use")
def init(name: str, output: str, template: str):
    """
    Initialize a new AgentForge project.
    
    Creates a new directory with a template configuration file
    and example scripts to get you started quickly.
    
    Examples:
    
        agentforge init my_agent
        
        agentforge init news_bot --template scraper
        
        agentforge init my_project -o ./projects
    """
    from agentforge.config import create_template_config, save_config
    
    output_dir = Path(output) / name
    
    if output_dir.exists():
        click.echo(f"Error: Directory '{output_dir}' already exists.", err=True)
        raise SystemExit(1)
    
    click.echo(f"Creating new AgentForge project: {name}")
    
    # Create directory structure
    output_dir.mkdir(parents=True)
    (output_dir / "outputs").mkdir()
    
    # Create config based on template
    if template == "basic":
        config = create_template_config(name=name, skills=["content_generation"])
    elif template == "scraper":
        config = create_template_config(
            name=name,
            skills=["web_scraper", "content_generation"]
        )
    elif template == "analyzer":
        config = create_template_config(
            name=name,
            skills=["data_analysis", "content_generation"]
        )
    else:  # full
        config = create_template_config(
            name=name,
            skills=["web_scraper", "data_analysis", "content_generation"]
        )
    
    # Save config
    config_path = output_dir / "config.yaml"
    save_config(config, config_path)
    
    # Create a simple runner script
    runner_script = f'''#!/usr/bin/env python
"""
Runner script for {name} agent.

This is a simple example showing how to run your agent programmatically.
Modify this to suit your needs!
"""

from agentforge import Agent
from agentforge.integrations import OpenAIBackend  # or GroqBackend, HuggingFaceBackend

def main():
    # Load the agent from config
    agent = Agent(
        config="config.yaml",
        llm=OpenAIBackend()  # Make sure OPENAI_API_KEY is set
    )
    
    # Run the agent
    result = agent.run({{"input": "Hello, AgentForge!"}})
    
    print("Agent output:")
    print(result)

if __name__ == "__main__":
    main()
'''
    
    (output_dir / "run.py").write_text(runner_script)
    
    # Create .env.example
    env_example = """# Environment variables for AgentForge
# Copy this to .env and fill in your values

# OpenAI API key (for OpenAI backend)
OPENAI_API_KEY=your_openai_api_key_here

# Groq API key (for Groq backend)
GROQ_API_KEY=your_groq_api_key_here

# HuggingFace token (for gated models)
HF_TOKEN=your_huggingface_token_here
"""
    (output_dir / ".env.example").write_text(env_example)
    
    click.echo(f"\n✓ Project created at: {output_dir}")
    click.echo("\nNext steps:")
    click.echo(f"  1. cd {output_dir}")
    click.echo("  2. Copy .env.example to .env and add your API keys")
    click.echo("  3. Edit config.yaml to customize your agent")
    click.echo("  4. Run: python run.py")
    click.echo("\nOr use the CLI: agentforge run config.yaml")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--input", "-i", "input_data", help="JSON input data for the agent")
@click.option("--input-file", "-f", type=click.Path(exists=True),
              help="Path to JSON file with input data")
@click.option("--output", "-o", type=click.Path(), help="Save output to file")
@click.option("--llm", "-l", default="openai",
              type=click.Choice(["openai", "groq", "huggingface"]),
              help="LLM backend to use")
@click.option("--model", "-m", help="Specific model to use")
def run(
    config_path: str,
    input_data: Optional[str],
    input_file: Optional[str],
    output: Optional[str],
    llm: str,
    model: Optional[str],
):
    """
    Run an agent from a configuration file.
    
    Loads the agent configuration from a YAML file and executes it
    with the provided input data.
    
    Examples:
    
        agentforge run config.yaml
        
        agentforge run config.yaml -i '{"url": "https://example.com"}'
        
        agentforge run config.yaml -f input.json -o result.json
        
        agentforge run config.yaml --llm groq --model llama-3.1-70b
    """
    from agentforge import Agent
    from agentforge.config import load_config, ConfigValidator
    
    click.echo(f"Loading configuration: {config_path}")
    
    # Load and validate config
    config = load_config(config_path)
    
    validator = ConfigValidator(config)
    if not validator.validate():
        click.echo("Configuration validation failed:", err=True)
        click.echo(validator.get_report(), err=True)
        raise SystemExit(1)
    
    # Parse input data
    if input_file:
        with open(input_file, "r") as f:
            agent_input = json.load(f)
    elif input_data:
        agent_input = json.loads(input_data)
    else:
        agent_input = {}
    
    # Create LLM backend
    llm_backend = _create_llm_backend(llm, model)
    
    click.echo(f"Using LLM: {llm} ({llm_backend.model})")
    
    # Create and run agent
    agent = Agent(config=config, llm=llm_backend)
    
    click.echo(f"Running agent: {agent.name}")
    click.echo(f"Skills: {[s.name for s in agent.skills]}")
    
    try:
        result = agent.run(agent_input)
        
        # Output results
        if output:
            with open(output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\n✓ Results saved to: {output}")
        else:
            click.echo("\n--- Results ---")
            click.echo(json.dumps(result, indent=2, default=str))
        
        click.echo("\n✓ Agent completed successfully")
        
    except Exception as e:
        click.echo(f"\n✗ Agent failed: {e}", err=True)
        logger.exception("Agent execution failed")
        raise SystemExit(1)


@main.command("list-skills")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed info")
def list_skills(verbose: bool):
    """
    List all available skills.
    
    Shows the built-in skills and any registered plugins.
    Use --verbose to see descriptions and requirements.
    
    Examples:
    
        agentforge list-skills
        
        agentforge list-skills --verbose
    """
    from agentforge.core import SkillRegistry
    
    registry = SkillRegistry()
    skills = registry.list_skills()
    
    click.echo("Available Skills:")
    click.echo("-" * 40)
    
    for skill_name in sorted(skills):
        if verbose:
            try:
                info = registry.get_skill_info(skill_name)
                click.echo(f"\n{skill_name}")
                click.echo(f"  Description: {info['description']}")
                click.echo(f"  Requires LLM: {info['requires_llm']}")
            except Exception as e:
                click.echo(f"\n{skill_name}")
                click.echo(f"  (Error loading info: {e})")
        else:
            click.echo(f"  • {skill_name}")
    
    click.echo(f"\nTotal: {len(skills)} skills")


@main.command("validate")
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str):
    """
    Validate an agent configuration file.
    
    Checks the YAML syntax and validates the configuration
    structure without running the agent.
    
    Examples:
    
        agentforge validate config.yaml
    """
    from agentforge.config import load_config, ConfigValidator
    
    click.echo(f"Validating: {config_path}")
    
    try:
        config = load_config(config_path)
    except Exception as e:
        click.echo(f"✗ Failed to load config: {e}", err=True)
        raise SystemExit(1)
    
    validator = ConfigValidator(config)
    is_valid = validator.validate()
    
    click.echo(validator.get_report())
    
    if is_valid:
        click.echo("\n✓ Configuration is valid")
    else:
        raise SystemExit(1)


@main.command("gui")
@click.option("--port", "-p", default=8501, help="Port for the Streamlit app")
@click.option("--host", "-h", default="localhost", help="Host to bind to")
def gui(port: int, host: str):
    """
    Launch the AgentForge GUI.
    
    Opens a Streamlit-based web interface for building and
    running agents interactively.
    
    Examples:
    
        agentforge gui
        
        agentforge gui --port 8080
    """
    import subprocess
    
    gui_path = Path(__file__).parent / "gui.py"
    
    click.echo(f"Launching AgentForge GUI at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(gui_path),
            "--server.port", str(port),
            "--server.address", host,
        ])
    except KeyboardInterrupt:
        click.echo("\nGUI stopped")
    except FileNotFoundError:
        click.echo(
            "Streamlit not installed. Please install with:\n"
            "pip install streamlit\n"
            "or: pip install agentforge[gui]",
            err=True
        )
        raise SystemExit(1)


def _create_llm_backend(backend_name: str, model: Optional[str]):
    """Create an LLM backend based on the name."""
    if backend_name == "openai":
        from agentforge.integrations import OpenAIBackend
        return OpenAIBackend(model=model)
    elif backend_name == "groq":
        from agentforge.integrations import GroqBackend
        return GroqBackend(model=model)
    elif backend_name == "huggingface":
        from agentforge.integrations import HuggingFaceBackend
        return HuggingFaceBackend(model=model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend_name}")


if __name__ == "__main__":
    main()

