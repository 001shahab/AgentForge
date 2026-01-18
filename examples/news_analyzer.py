#!/usr/bin/env python
"""
News Analyzer Example for AgentForge.

This example shows how to build an agent that:
1. Scrapes news articles from a website
2. Summarizes the content using an LLM
3. Analyzes trends and generates insights

I've designed this as a practical example of chaining multiple skills
together to create something useful.

Author: Prof. Shahab Anbarjafari
3S Holding OÜ, Tartu, Estonia

Usage:
    python news_analyzer.py
    
    Or with a specific URL:
    python news_analyzer.py --url https://news.ycombinator.com
"""

import argparse
import json
import logging
import os
from datetime import datetime

# Set up logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the news analyzer."""
    parser = argparse.ArgumentParser(description="Analyze news with AI")
    parser.add_argument(
        "--url",
        default="https://news.ycombinator.com",
        help="URL to analyze (default: Hacker News)"
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "groq", "huggingface"],
        default="openai",
        help="LLM backend to use"
    )
    parser.add_argument(
        "--model",
        help="Specific model to use"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    args = parser.parse_args()
    
    # Check for API key
    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your_key_here")
        return 1
    
    if args.backend == "groq" and not os.environ.get("GROQ_API_KEY"):
        print("Error: Please set the GROQ_API_KEY environment variable")
        print("Example: export GROQ_API_KEY=your_key_here")
        return 1
    
    print("=" * 60)
    print("AgentForge News Analyzer")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"Backend: {args.backend}")
    print()
    
    # Import AgentForge components
    from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
    from agentforge.integrations import OpenAIBackend, GroqBackend, HuggingFaceBackend
    
    # Create the LLM backend
    if args.backend == "openai":
        llm = OpenAIBackend(model=args.model)
    elif args.backend == "groq":
        llm = GroqBackend(model=args.model)
    else:
        llm = HuggingFaceBackend(model=args.model)
    
    print(f"Using model: {llm.model}")
    print()
    
    # Create skills
    scraper = WebScraperSkill()
    generator = ContentGenerationSkill()
    
    # Create the agent
    agent = Agent(
        name="news_analyzer",
        skills=[scraper, generator],
        llm=llm,
    )
    
    print("Running agent...")
    print()
    
    # Run the agent
    try:
        result = agent.run({
            "url": args.url,
            "template": "analyze",  # Use the analysis template
        })
        
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print()
        
        # Print the scraped title
        print(f"Page Title: {result.get('title', 'Unknown')}")
        print()
        
        # Print the content summary
        print("Content Preview (first 500 chars):")
        print("-" * 40)
        content = result.get("content", "")
        print(content[:500] + "..." if len(content) > 500 else content)
        print()
        
        # Print the AI analysis
        print("AI Analysis:")
        print("-" * 40)
        print(result.get("generated", "No analysis generated"))
        print()
        
        # Save to file if requested
        if args.output:
            result["timestamp"] = datetime.now().isoformat()
            result["source_url"] = args.url
            result["model"] = llm.model
            
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Results saved to: {args.output}")
        
        print()
        print("✓ Analysis complete!")
        return 0
        
    except Exception as e:
        logger.exception("Analysis failed")
        print(f"Error: {e}")
        return 1


def run_from_config():
    """
    Alternative: Run the same agent from a YAML config file.
    
    This shows how you can use configuration files instead of
    building the agent in Python code.
    """
    from agentforge import Agent
    from agentforge.integrations import OpenAIBackend
    
    # Load agent from config
    agent = Agent(
        config="news_analyzer.yaml",
        llm=OpenAIBackend()
    )
    
    # Run with input
    result = agent.run({
        "url": "https://news.ycombinator.com"
    })
    
    return result


if __name__ == "__main__":
    exit(main())

