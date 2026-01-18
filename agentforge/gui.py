"""
Streamlit GUI for AgentForge.

I've built this web interface to make AgentForge accessible to everyone,
even if you're not comfortable with code. You can build agents visually,
configure skills, and see results in real-time.

Author: Prof. Shahab Anbarjafari

Run with: streamlit run gui.py
Or: agentforge gui
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Streamlit not installed. Please install with:\n"
        "pip install streamlit\n"
        "or: pip install agentforge[gui]"
    )


def main():
    """Main entry point for the Streamlit app."""
    
    # Page config
    st.set_page_config(
        page_title="AgentForge",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for a cleaner look
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .skill-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .success-box {
            background: #d4edda;
            border-radius: 8px;
            padding: 1rem;
        }
        .error-box {
            background: #f8d7da;
            border-radius: 8px;
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🤖 AgentForge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Build modular AI agents with ease</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # LLM Backend selection
        llm_backend = st.selectbox(
            "LLM Backend",
            ["OpenAI", "Groq", "HuggingFace (Local)"],
            help="Choose which AI model provider to use"
        )
        
        # Model selection based on backend
        if llm_backend == "OpenAI":
            model = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            )
            api_key_env = "OPENAI_API_KEY"
        elif llm_backend == "Groq":
            model = st.selectbox(
                "Model",
                ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            )
            api_key_env = "GROQ_API_KEY"
        else:
            model = st.text_input(
                "Model ID",
                value="mistralai/Mistral-7B-Instruct-v0.2",
                help="HuggingFace model identifier"
            )
            api_key_env = "HF_TOKEN"
        
        # API Key input
        api_key = st.text_input(
            f"API Key ({api_key_env})",
            type="password",
            value=os.environ.get(api_key_env, ""),
            help="Your API key (stored only in this session)"
        )
        
        st.divider()
        
        # Generation parameters
        st.subheader("Generation Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 4000, 1024, 100)
        
        st.divider()
        
        # About section
        st.markdown("""
        **AgentForge** v0.1.0
        
        Designed by Prof. Shahab Anbarjafari
        
        3S Holding OÜ, Tartu, Estonia
        
        [GitHub](https://github.com/3sholding/agentforge) | 
        [Documentation](https://agentforge.readthedocs.io)
        """)
    
    # Main content area - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Quick Run", 
        "🏗️ Agent Builder", 
        "📊 Data Analysis",
        "🌐 Web Scraper"
    ])
    
    # Tab 1: Quick Run
    with tab1:
        st.header("Quick Run")
        st.markdown("Generate content with a simple prompt.")
        
        prompt = st.text_area(
            "Your Prompt",
            height=150,
            placeholder="Enter your prompt here...\n\nExample: Summarize the key benefits of renewable energy in 3 bullet points."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            run_button = st.button("🚀 Generate", type="primary", use_container_width=True)
        
        if run_button and prompt:
            if not api_key:
                st.error(f"Please enter your {api_key_env} in the sidebar.")
            else:
                with st.spinner("Generating..."):
                    try:
                        result = _run_generation(
                            prompt, llm_backend, model, api_key,
                            temperature, max_tokens
                        )
                        st.success("Generation complete!")
                        st.markdown("### Result")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Tab 2: Agent Builder
    with tab2:
        st.header("Agent Builder")
        st.markdown("Build a custom agent by chaining skills together.")
        
        # Skill selection
        available_skills = _get_available_skills()
        
        selected_skills = st.multiselect(
            "Select Skills (in order)",
            available_skills,
            default=["web_scraper", "content_generation"],
            help="Skills will be executed in the order selected"
        )
        
        if selected_skills:
            st.markdown("### Skill Configuration")
            
            skill_configs = {}
            for skill_name in selected_skills:
                with st.expander(f"⚙️ {skill_name}", expanded=True):
                    config = _get_skill_config_ui(skill_name)
                    skill_configs[skill_name] = config
            
            # Agent input
            st.markdown("### Agent Input")
            agent_input = st.text_area(
                "Input Data (JSON)",
                value='{\n  "url": "https://example.com"\n}',
                height=100,
            )
            
            if st.button("🏃 Run Agent", type="primary"):
                if not api_key:
                    st.error(f"Please enter your {api_key_env} in the sidebar.")
                else:
                    try:
                        input_data = json.loads(agent_input)
                        with st.spinner("Running agent..."):
                            result = _run_agent(
                                selected_skills, skill_configs, input_data,
                                llm_backend, model, api_key, temperature, max_tokens
                            )
                        st.success("Agent completed!")
                        st.json(result)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in input data")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Tab 3: Data Analysis
    with tab3:
        st.header("Data Analysis")
        st.markdown("Upload data and analyze it with AI assistance.")
        
        upload_type = st.radio(
            "Data Source",
            ["Upload CSV", "Paste Data", "Sample Data"],
            horizontal=True
        )
        
        df = None
        
        if upload_type == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file:
                import pandas as pd
                df = pd.read_csv(uploaded_file)
        
        elif upload_type == "Paste Data":
            csv_text = st.text_area(
                "Paste CSV data",
                height=150,
                placeholder="name,value\nAlice,100\nBob,85\nCharlie,92"
            )
            if csv_text:
                import pandas as pd
                import io
                df = pd.read_csv(io.StringIO(csv_text))
        
        else:  # Sample Data
            import pandas as pd
            df = pd.DataFrame({
                "Name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "Department": ["Sales", "Engineering", "Sales", "Marketing", "Engineering"],
                "Score": [85, 92, 78, 95, 88],
                "Years": [3, 5, 2, 4, 3],
            })
            st.info("Using sample employee data")
        
        if df is not None:
            st.markdown("### Data Preview")
            st.dataframe(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Show Statistics"):
                    st.markdown("### Statistics")
                    st.dataframe(df.describe())
            
            with col2:
                if st.button("🤖 AI Analysis") and api_key:
                    with st.spinner("Analyzing..."):
                        analysis = _analyze_data_with_ai(
                            df, llm_backend, model, api_key,
                            temperature, max_tokens
                        )
                        st.markdown("### AI Analysis")
                        st.markdown(analysis)
    
    # Tab 4: Web Scraper
    with tab4:
        st.header("Web Scraper")
        st.markdown("Scrape web pages and optionally summarize the content.")
        
        url = st.text_input("URL to Scrape", placeholder="https://example.com")
        selector = st.text_input(
            "CSS Selector (optional)",
            placeholder="article p, .content",
            help="Leave empty to get all text content"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            extract_links = st.checkbox("Extract Links")
        with col2:
            summarize = st.checkbox("Summarize Content")
        
        if st.button("🌐 Scrape", type="primary") and url:
            with st.spinner("Scraping..."):
                try:
                    from agentforge.skills import WebScraperSkill
                    
                    scraper = WebScraperSkill()
                    result = scraper.execute({
                        "url": url,
                        "selector": selector or None,
                        "extract_links": extract_links,
                    })
                    
                    st.success(f"Scraped: {result.get('title', 'Unknown')}")
                    
                    st.markdown("### Content")
                    content = result.get("content", "")
                    st.text_area("Scraped Content", content, height=300)
                    
                    if summarize and api_key and content:
                        with st.spinner("Summarizing..."):
                            summary = _run_generation(
                                f"Please summarize this content concisely:\n\n{content[:4000]}",
                                llm_backend, model, api_key, temperature, max_tokens
                            )
                            st.markdown("### Summary")
                            st.markdown(summary)
                    
                    if extract_links and result.get("links"):
                        st.markdown("### Links")
                        for link in result["links"][:20]:
                            st.markdown(f"- [{link['text'][:50]}]({link['url']})")
                            
                except Exception as e:
                    st.error(f"Scraping failed: {e}")


def _get_available_skills() -> List[str]:
    """Get list of available skills."""
    try:
        from agentforge.core import SkillRegistry
        return SkillRegistry().list_skills()
    except Exception:
        return ["web_scraper", "data_analysis", "content_generation"]


def _get_skill_config_ui(skill_name: str) -> Dict[str, Any]:
    """Render configuration UI for a skill and return the config."""
    config = {}
    
    if skill_name == "web_scraper":
        config["selector"] = st.text_input(
            "CSS Selector", key=f"{skill_name}_selector",
            help="Optional CSS selector to target specific elements"
        )
    
    elif skill_name == "content_generation":
        config["template"] = st.selectbox(
            "Template",
            ["summarize", "analyze", "rewrite", "custom"],
            key=f"{skill_name}_template"
        )
        if config["template"] == "rewrite":
            config["style"] = st.text_input(
                "Style", value="professional and clear",
                key=f"{skill_name}_style"
            )
    
    elif skill_name == "data_analysis":
        config["operations"] = st.multiselect(
            "Operations",
            ["describe", "info", "unique", "missing", "correlation"],
            default=["describe"],
            key=f"{skill_name}_ops"
        )
    
    return config


def _run_generation(
    prompt: str,
    backend: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Run a simple generation."""
    llm = _create_llm(backend, model, api_key)
    return llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)


def _run_agent(
    skills: List[str],
    skill_configs: Dict[str, Dict],
    input_data: Dict[str, Any],
    backend: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Run an agent with the specified skills."""
    from agentforge import Agent
    from agentforge.core import SkillRegistry
    
    registry = SkillRegistry()
    llm = _create_llm(backend, model, api_key)
    
    skill_instances = []
    for skill_name in skills:
        config = skill_configs.get(skill_name, {})
        skill = registry.get(skill_name, **config)
        skill_instances.append(skill)
    
    agent = Agent(skills=skill_instances, llm=llm)
    return agent.run(input_data)


def _analyze_data_with_ai(
    df,
    backend: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Analyze a DataFrame with AI."""
    llm = _create_llm(backend, model, api_key)
    
    # Create a summary of the data
    summary = f"""DataFrame with {len(df)} rows and {len(df.columns)} columns.

Columns: {', '.join(df.columns)}

Sample data:
{df.head().to_string()}

Statistics:
{df.describe().to_string()}
"""
    
    prompt = f"""Analyze this dataset and provide key insights:

{summary}

Please provide:
1. Overview of the data
2. Key patterns or trends
3. Notable observations
4. Suggestions for further analysis
"""
    
    return llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)


def _create_llm(backend: str, model: str, api_key: str):
    """Create an LLM backend."""
    if "OpenAI" in backend:
        os.environ["OPENAI_API_KEY"] = api_key
        from agentforge.integrations import OpenAIBackend
        return OpenAIBackend(model=model)
    elif "Groq" in backend:
        os.environ["GROQ_API_KEY"] = api_key
        from agentforge.integrations import GroqBackend
        return GroqBackend(model=model)
    else:
        os.environ["HF_TOKEN"] = api_key
        from agentforge.integrations import HuggingFaceBackend
        return HuggingFaceBackend(model=model)


if __name__ == "__main__":
    main()

