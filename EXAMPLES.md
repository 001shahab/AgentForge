# AgentForge Examples

Real-world examples for different use cases. Each example is self-contained and ready to run!

**© 2024 3S Holding OÜ, Tartu, Estonia**

| | |
|---|---|
| **Creator** | Prof. Shahab Anbarjafari |
| **Contact** | shb@3sholding.com |
| **Commercial Use** | For business licensing, contact shb@3sholding.com |

---

## Quick Setup (Everyone Needs This First!)

Before running any example, install AgentForge and set your API key:

```bash
# Install AgentForge with OpenAI support
pip install agentforge[openai]

# Set your OpenAI API key (get one at https://platform.openai.com/api-keys)
export OPENAI_API_KEY="your-api-key-here"
```

Don't have an OpenAI key? You can use Groq for free:
```bash
pip install agentforge[groq]
export GROQ_API_KEY="your-groq-key"  # Get at https://console.groq.com
```

---

## 🔬 For Researchers: Summarize Research Papers

**Goal:** Automatically fetch a research paper abstract and get a simplified explanation.

### Example 1: Summarize an ArXiv Paper

```python
"""
Research Paper Summarizer
Fetches a paper from ArXiv and explains it in simple terms.
"""

from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

# Create the agent
agent = Agent(
    name="paper_summarizer",
    skills=[
        WebScraperSkill(),
        ContentGenerationSkill()
    ],
    llm=OpenAIBackend(model="gpt-4o-mini")
)

# Example: Summarize a machine learning paper
result = agent.run({
    "url": "https://arxiv.org/abs/2301.00234",  # Any ArXiv paper URL
    "prompt": """Based on the content above, please:
    1. What is the main problem this paper solves?
    2. What is the key idea or method?
    3. What are the main results?
    
    Explain it simply, as if to a smart undergraduate student."""
})

print("=" * 50)
print("PAPER SUMMARY")
print("=" * 50)
print(result["generated"])
```

### Example 2: Compare Multiple Papers

```python
"""
Compare multiple research papers on a topic.
"""

from agentforge.skills import WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

# Set up components
scraper = WebScraperSkill()
llm = OpenAIBackend()

# Fetch multiple papers
papers = [
    "https://arxiv.org/abs/2301.00234",
    "https://arxiv.org/abs/2302.00234",
]

all_content = []
for url in papers:
    result = scraper.execute({"url": url})
    all_content.append(f"Paper from {url}:\n{result['content'][:2000]}")

# Compare them
combined = "\n\n---\n\n".join(all_content)
comparison = llm.generate(f"""
Compare these research papers:

{combined}

Create a table comparing:
- Main contribution
- Methodology
- Key results
- Limitations
""")

print(comparison)
```

---

## 📊 For Marketers: Competitor Monitoring

**Goal:** Automatically monitor competitor websites and get actionable insights.

### Example 1: Analyze a Competitor's Homepage

```python
"""
Competitor Website Analyzer
Scrapes a competitor's website and extracts key marketing insights.
"""

from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

# Create the agent
agent = Agent(
    name="competitor_analyzer",
    skills=[
        WebScraperSkill(),
        ContentGenerationSkill()
    ],
    llm=OpenAIBackend()
)

# Analyze a competitor
result = agent.run({
    "url": "https://stripe.com",  # Replace with your competitor's URL
    "prompt": """Analyze this company's website and provide:
    
    1. **Value Proposition**: What's their main selling point?
    2. **Target Audience**: Who are they trying to reach?
    3. **Key Features**: What features do they highlight?
    4. **Pricing Strategy**: Any pricing info visible?
    5. **Tone & Messaging**: How do they communicate?
    
    Format as a brief marketing report."""
})

print("=" * 50)
print("COMPETITOR ANALYSIS REPORT")
print("=" * 50)
print(result["generated"])
```

### Example 2: Monitor Multiple Competitors

```python
"""
Monitor multiple competitors and create a comparison report.
"""

from agentforge.skills import WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

scraper = WebScraperSkill()
generator = ContentGenerationSkill()
generator.set_llm(OpenAIBackend())

# Your competitors
competitors = [
    {"name": "Competitor A", "url": "https://stripe.com"},
    {"name": "Competitor B", "url": "https://square.com"},
]

# Gather data
reports = []
for comp in competitors:
    print(f"Analyzing {comp['name']}...")
    data = scraper.execute({"url": comp["url"]})
    reports.append({
        "name": comp["name"],
        "title": data.get("title", "Unknown"),
        "content": data.get("content", "")[:1500]
    })

# Generate comparison
comparison_prompt = "Compare these competitors:\n\n"
for r in reports:
    comparison_prompt += f"## {r['name']}\n{r['content']}\n\n"
    
comparison_prompt += """
Create a competitive analysis table with:
- Company Name
- Main Product/Service
- Unique Selling Point
- Apparent Target Market
- Strengths (based on messaging)
"""

result = generator.execute({"prompt": comparison_prompt})
print("\n" + "=" * 50)
print("COMPETITIVE ANALYSIS")
print("=" * 50)
print(result["generated"])
```

### Example 3: Product Announcement Monitor

```python
"""
Monitor a company's news/blog for new announcements.
"""

from agentforge.plugins import RSSMonitorPlugin
from agentforge.skills import ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

# Note: Install RSS support first: pip install agentforge[rss]

# Monitor RSS feeds
rss = RSSMonitorPlugin()
generator = ContentGenerationSkill()
generator.set_llm(OpenAIBackend())

# Get latest news
news = rss.execute({
    "feeds": [
        "https://blog.hubspot.com/rss.xml",  # Example marketing blog
    ],
    "max_entries": 5
})

# Summarize the latest
entries_text = "\n\n".join([
    f"Title: {e['title']}\nSummary: {e['summary']}"
    for e in news["entries"]
])

result = generator.execute({
    "prompt": f"""Here are the latest blog posts from a competitor:

{entries_text}

Summarize the key themes and any product announcements. 
What can we learn from their content strategy?"""
})

print("COMPETITOR NEWS SUMMARY")
print(result["generated"])
```

---

## 💻 For Developers: Automate Repetitive Tasks

**Goal:** Save time by automating common development workflows.

### Example 1: Documentation Generator

```python
"""
Generate documentation from code descriptions.
"""

from agentforge.skills import ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

generator = ContentGenerationSkill()
generator.set_llm(OpenAIBackend())

# Your function to document
code = '''
def calculate_discount(price, discount_percent, is_member=False):
    base_discount = price * (discount_percent / 100)
    if is_member:
        base_discount *= 1.1  # Members get 10% extra
    return max(0, price - base_discount)
'''

result = generator.execute({
    "prompt": f"""Generate comprehensive documentation for this Python function:

```python
{code}
```

Include:
1. Docstring (Google style)
2. Parameter descriptions
3. Return value description
4. Usage examples
5. Edge cases to consider"""
})

print(result["generated"])
```

### Example 2: Code Review Assistant

```python
"""
Get AI-powered code review suggestions.
"""

from agentforge.integrations import OpenAIBackend

llm = OpenAIBackend()

# Your code to review
code = '''
def get_user_data(user_id):
    import requests
    response = requests.get(f"http://api.example.com/users/{user_id}")
    data = response.json()
    return data["name"], data["email"], data["age"]
'''

review = llm.generate(f"""
Review this Python code for:
1. Security issues
2. Error handling
3. Code style
4. Performance
5. Suggestions for improvement

```python
{code}
```

Be specific and provide fixed code where needed.
""")

print("CODE REVIEW")
print("=" * 50)
print(review)
```

### Example 3: API Response Analyzer

```python
"""
Fetch an API and analyze the response structure.
"""

from agentforge.skills import WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

scraper = WebScraperSkill()
generator = ContentGenerationSkill()
generator.set_llm(OpenAIBackend())

# Fetch a public API (this one returns JSON)
api_result = scraper.execute({
    "url": "https://api.github.com/repos/python/cpython"
})

result = generator.execute({
    "text": api_result["content"][:3000],
    "prompt": """Analyze this API response and create:
    1. A summary of what data is available
    2. TypeScript interface definitions for the response
    3. Example code to fetch and use this data
    """
})

print(result["generated"])
```

---

## 🌟 For Everyone: Simple Everyday Tasks

**Goal:** Make AI work for you in daily tasks.

### Example 1: Article Summarizer (The Simplest Example!)

```python
"""
The simplest AgentForge example.
Give it any article URL, get a summary.
"""

from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

# Create agent in 3 lines
agent = Agent(
    skills=[WebScraperSkill(), ContentGenerationSkill()],
    llm=OpenAIBackend()
)

# Run it!
result = agent.run({
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "template": "summarize"
})

print("SUMMARY:")
print(result["generated"])
```

### Example 2: Recipe Finder and Simplifier

```python
"""
Find a recipe online and get a simplified version.
"""

from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

agent = Agent(
    skills=[WebScraperSkill(), ContentGenerationSkill()],
    llm=OpenAIBackend()
)

result = agent.run({
    "url": "https://www.allrecipes.com/recipe/10813/best-chocolate-chip-cookies/",
    "prompt": """From this recipe, create a simplified version:
    
    1. List only the essential ingredients (skip optional ones)
    2. Reduce the steps to 5 or fewer
    3. Add a "Quick Tip" for beginners
    
    Format it nicely with emojis!"""
})

print("🍪 SIMPLIFIED RECIPE")
print(result["generated"])
```

### Example 3: News Digest

```python
"""
Get a quick digest of today's top stories.
"""

from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

agent = Agent(
    name="news_digest",
    skills=[WebScraperSkill(), ContentGenerationSkill()],
    llm=OpenAIBackend()
)

result = agent.run({
    "url": "https://news.ycombinator.com",
    "prompt": """From this page, identify the top 5 stories and for each:
    
    1. Give a one-line summary
    2. Rate its importance (⭐ to ⭐⭐⭐⭐⭐)
    3. Who should care about this?
    
    Keep it brief and scannable!"""
})

print("📰 TODAY'S NEWS DIGEST")
print("=" * 50)
print(result["generated"])
```

### Example 4: Travel Planner Helper

```python
"""
Get travel information from a destination page.
"""

from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
from agentforge.integrations import OpenAIBackend

agent = Agent(
    skills=[WebScraperSkill(), ContentGenerationSkill()],
    llm=OpenAIBackend()
)

result = agent.run({
    "url": "https://en.wikipedia.org/wiki/Tallinn",  # Any city
    "prompt": """Based on this information, create a quick travel guide:
    
    🌍 **Quick Facts** (population, language, currency)
    ☀️ **Best Time to Visit**
    🏛️ **Top 3 Must-See Places**
    🍽️ **Local Food to Try**
    💡 **Insider Tip**
    
    Keep it fun and useful for a first-time visitor!"""
})

print("✈️ TRAVEL MINI-GUIDE")
print(result["generated"])
```

---

## 🚀 Running These Examples

1. **Copy any example** into a file (e.g., `my_example.py`)
2. **Make sure you've set up** your API key (see top of this file)
3. **Run it:**
   ```bash
   python my_example.py
   ```

## 💡 Tips for Success

- **Start simple:** Try Example 1 from "For Everyone" first
- **Customize prompts:** The `prompt` field is where the magic happens - tweak it!
- **Handle errors:** Add try/except for production use
- **Rate limits:** If using free APIs, add delays between requests

## ❓ Need Help?

- Check the [README.md](README.md) for full documentation
- Look at `examples/` folder for more complete examples
- Contact: shb@3sholding.com

---

*Happy automating!* 🤖

