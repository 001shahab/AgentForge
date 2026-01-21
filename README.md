# AgentForge

[![PyPI version](https://badge.fury.io/py/agentforge.svg)](https://badge.fury.io/py/agentforge)
[![Tests](https://github.com/001shahab/AgentForge/workflows/CI/badge.svg)](https://github.com/001shahab/AgentForge/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**A lightweight framework for building modular AI agents.**

I created AgentForge because I wanted a simple, clean way to chain together different AI-powered tasks without the complexity of larger frameworks. Whether you're scraping websites, analyzing data, or generating content, AgentForge makes it easy to build agents that do the heavy lifting for you.

Paper: https://arxiv.org/abs/2601.13383

---

## ⚖️ Ownership & Licensing

**© 2024 3S Holding OÜ. All Rights Reserved.**

| | |
|---|---|
| **Creator** | Prof. Shahab Anbarjafari |
| **Organization** | 3S Holding OÜ, Tartu, Estonia |
| **Contact** | 📧 shb@3sholding.com |

This project is released under the **MIT License** for open-source use. 

**🏢 For commercial/business licensing, enterprise support, or partnership inquiries, please contact us directly at shb@3sholding.com**

---

## What is AgentForge?

AgentForge is a Python library that lets you create AI agents by chaining together "skills" - modular components that each do one thing well. Think of it like building with LEGO blocks: you pick the skills you need, snap them together, and your agent handles the rest.

**Key features:**

- 🔗 **Modular Skills** - Web scraping, data analysis, content generation, and more
- 🤖 **Multiple LLM Backends** - OpenAI, Groq, or local models via HuggingFace
- 📝 **YAML Configuration** - Define agents in simple config files
- 🖥️ **CLI & GUI** - Run agents from the terminal or a web interface
- ⚡ **Async Support** - Built for performance with async operations
- 🔌 **Extensible** - Easy to create your own custom skills

## Who Is This For?

| 🔬 **Researchers** | 📊 **Marketers** | 💻 **Developers** | 🌟 **Everyone** |
|:---:|:---:|:---:|:---:|
| Auto-summarize papers | Monitor competitors | Automate workflows | AI for daily tasks |
| Compare multiple sources | Track product announcements | Generate documentation | Summarize articles |
| Extract key insights | Analyze messaging | Review code | Get quick answers |

**👉 See [EXAMPLES.md](EXAMPLES.md) for ready-to-run code for each use case!**

## Quick Start

### Installation

```bash
# Basic installation
pip install agentforge

# With OpenAI support
pip install agentforge[openai]

# With all optional dependencies
pip install agentforge[all]
```

### Your First Agent

Here's a simple agent that scrapes a webpage and summarizes it:

```python
from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

# Set up your API key
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Create an agent with two skills
agent = Agent(
    skills=[
        WebScraperSkill(),
        ContentGenerationSkill(default_template="summarize")
    ],
    llm=OpenAIBackend()
)

# Run it!
result = agent.run({"url": "https://news.ycombinator.com"})
print(result["generated"])
```

That's it! The agent scrapes the page, then uses the LLM to summarize what it found.

## Using YAML Configuration

Don't want to write Python? Define your agent in YAML:

```yaml
# my_agent.yaml
name: news_summarizer
description: Scrapes news and creates a summary

llm:
  backend: openai
  model: gpt-4o-mini

skills:
  - web_scraper
  - skill: content_generation
    template: summarize
```

Then run it from the command line:

```bash
agentforge run my_agent.yaml -i '{"url": "https://news.ycombinator.com"}'
```

## Available Skills

AgentForge comes with several built-in skills:

| Skill | Description | Requires LLM |
|-------|-------------|--------------|
| `web_scraper` | Fetch and parse web pages | No |
| `data_analysis` | Analyze data with pandas | No |
| `content_generation` | Generate text with templates | Yes |
| `rss_monitor` | Monitor RSS feeds | No |
| `image_generation` | Create images with Stable Diffusion | No |
| `voice_synthesis` | Text-to-speech conversion | No |

## LLM Backends

Choose the backend that works best for you:

### OpenAI (Cloud)
```python
from agentforge.integrations import OpenAIBackend

llm = OpenAIBackend(model="gpt-4o-mini")
```

### Groq (Fast Cloud Inference)
```python
from agentforge.integrations import GroqBackend

llm = GroqBackend(model="llama-3.1-70b-versatile")
```

### HuggingFace (Local)
```python
from agentforge.integrations import HuggingFaceBackend

llm = HuggingFaceBackend(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    load_in_4bit=True  # For lower memory usage
)
```

## CLI Commands

AgentForge includes a command-line interface:

```bash
# Initialize a new project
agentforge init my_project

# Run an agent from config
agentforge run config.yaml

# List available skills
agentforge list-skills --verbose

# Validate a config file
agentforge validate config.yaml

# Launch the web GUI
agentforge gui
```

## Creating Custom Skills

Building your own skill is straightforward:

```python
from agentforge.core import Skill, SkillRegistry

class MyCustomSkill(Skill):
    name = "my_skill"
    description = "Does something awesome"
    requires_llm = False
    
    def execute(self, input_data):
        # Your logic here
        text = input_data.get("text", "")
        return {
            "processed_text": text.upper(),
            "word_count": len(text.split())
        }

# Register it so you can use it in YAML configs
registry = SkillRegistry()
registry.register("my_skill", MyCustomSkill)
```

## Examples

**📚 [EXAMPLES.md](EXAMPLES.md)** - Detailed, copy-paste examples for different use cases:
- 🔬 Researchers: Summarize papers, compare sources
- 📊 Marketers: Monitor competitors, track announcements
- 💻 Developers: Generate docs, review code, analyze APIs
- 🌟 Everyone: Summarize articles, create digests, plan trips

**`examples/` folder** - Complete runnable scripts:
- **news_analyzer.py** - Scrape news sites and generate AI analysis
- **stock_monitor.py** - Monitor stocks and get alerts
- **quickstart.ipynb** - Interactive Jupyter notebook tutorial

## Environment Variables

AgentForge uses environment variables for API keys (never hardcode them!):

```bash
export OPENAI_API_KEY=your_openai_key
export GROQ_API_KEY=your_groq_key
export HF_TOKEN=your_huggingface_token  # For gated models
```

## Contributing

I'd love your help making AgentForge better! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest tests/ -v`)
5. Submit a pull request

Please make sure to:
- Follow PEP8 style (we use Black for formatting)
- Add tests for new features
- Update documentation as needed

## Development Setup

```bash
# Clone the repo
git clone https://github.com/001shahab/AgentForge.git
cd agentforge

# Install with Poetry
poetry install

# Run tests
poetry run pytest tests/ -v

# Format code
poetry run black agentforge tests

# Lint
poetry run ruff check agentforge tests
```

## Roadmap

Here's what I'm planning for future releases:

- [ ] More built-in skills (email, database, etc.)
- [ ] Agent memory and state persistence
- [ ] Multi-agent collaboration
- [ ] Better error recovery and retry logic
- [ ] Plugin marketplace

## License & Legal

**© 2024 3S Holding OÜ, Tartu, Estonia. All Rights Reserved.**

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

### Open Source Use ✅
You are free to use, modify, and distribute this software under the MIT License terms.

### Commercial & Enterprise Use 🏢
For commercial licensing, custom development, enterprise support, or partnership opportunities:

📧 **Contact:** Prof. Shahab Anbarjafari — shb@3sholding.com

## Acknowledgments

This project wouldn't be possible without the amazing open-source community. Special thanks to the teams behind:

- [OpenAI](https://openai.com/) for their API
- [Groq](https://groq.com/) for blazing-fast inference
- [HuggingFace](https://huggingface.co/) for democratizing AI
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for web parsing
- [Pandas](https://pandas.pydata.org/) for data analysis
- [Streamlit](https://streamlit.io/) for the GUI framework

---

## Contact

| Purpose | Contact |
|---------|---------|
| 🐛 **Bug Reports & Features** | [GitHub Issues](https://github.com/001shahab/AgentForge/issues) |
| 💬 **General Questions** | shb@3sholding.com |
| 🏢 **Business & Commercial** | shb@3sholding.com |
| 🤝 **Partnerships** | shb@3sholding.com |
| ✅ **Partnerships** | https://arxiv.org/abs/2601.13383 |

---

**Made with ❤️ in Estonia by [3S Holding OÜ](https://3sholding.com)**

Happy building! 🚀
