"""
Tests for the core AgentForge module.

I'm testing the Agent and Skill classes here, which are the foundation
of the entire framework.

Author: Prof. Shahab Anbarjafari
"""

import pytest
from typing import Any, Dict


class TestSkill:
    """Tests for the Skill base class."""
    
    def test_skill_creation(self, sample_skill):
        """Test that we can create a skill."""
        assert sample_skill.name == "test_skill"
        assert sample_skill.description == "A skill for testing"
        assert sample_skill.requires_llm is False
    
    def test_skill_execute(self, sample_skill):
        """Test skill execution."""
        result = sample_skill.execute({"key": "value"})
        
        assert result["processed"] is True
        assert "key" in result["input_keys"]
        assert result["message"] == "Test skill executed"
    
    def test_skill_with_config(self):
        """Test skill with custom configuration."""
        from agentforge.core import Skill
        
        class ConfigurableSkill(Skill):
            name = "configurable"
            
            def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"config": self.config}
        
        skill = ConfigurableSkill(option1="value1", option2=42)
        result = skill.execute({})
        
        assert result["config"]["option1"] == "value1"
        assert result["config"]["option2"] == 42
    
    def test_skill_set_llm(self, sample_llm_skill, mock_llm):
        """Test attaching an LLM to a skill."""
        sample_llm_skill.set_llm(mock_llm)
        
        result = sample_llm_skill.execute({})
        
        assert result["used_llm"] is True
        assert "Mock response" in result["generated"]
    
    def test_skill_without_llm_raises(self, sample_llm_skill):
        """Test that a skill requiring LLM raises without one."""
        with pytest.raises(RuntimeError, match="LLM not configured"):
            sample_llm_skill.execute({})


class TestSkillRegistry:
    """Tests for the SkillRegistry."""
    
    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        from agentforge.core import SkillRegistry
        
        registry1 = SkillRegistry()
        registry2 = SkillRegistry()
        
        assert registry1 is registry2
    
    def test_list_builtin_skills(self):
        """Test listing built-in skills."""
        from agentforge.core import SkillRegistry
        
        registry = SkillRegistry()
        skills = registry.list_skills()
        
        # Should have at least the core skills
        assert "web_scraper" in skills
        assert "data_analysis" in skills
        assert "content_generation" in skills
    
    def test_get_skill(self):
        """Test getting a skill from the registry."""
        from agentforge.core import SkillRegistry
        
        registry = SkillRegistry()
        skill = registry.get("web_scraper")
        
        assert skill.name == "web_scraper"
    
    def test_get_unknown_skill_raises(self):
        """Test that getting an unknown skill raises."""
        from agentforge.core import SkillRegistry
        
        registry = SkillRegistry()
        
        with pytest.raises(KeyError, match="Unknown skill"):
            registry.get("nonexistent_skill")
    
    def test_register_custom_skill(self, sample_skill):
        """Test registering a custom skill."""
        from agentforge.core import SkillRegistry, Skill
        
        registry = SkillRegistry()
        
        # Create a custom skill class
        class CustomSkill(Skill):
            name = "custom"
            
            def execute(self, input_data):
                return {"custom": True}
        
        registry.register("custom", CustomSkill)
        
        skill = registry.get("custom")
        assert skill.name == "custom"


class TestAgent:
    """Tests for the Agent class."""
    
    def test_agent_creation(self, sample_skill):
        """Test creating an agent with skills."""
        from agentforge import Agent
        
        agent = Agent(
            skills=[sample_skill],
            name="test_agent"
        )
        
        assert agent.name == "test_agent"
        assert len(agent.skills) == 1
        assert agent.skills[0].name == "test_skill"
    
    def test_agent_run(self, sample_skill):
        """Test running an agent."""
        from agentforge import Agent
        
        agent = Agent(skills=[sample_skill], name="test")
        result = agent.run({"input": "test"})
        
        assert result["processed"] is True
        assert "input" in result["input_keys"]
    
    def test_agent_chain_skills(self):
        """Test that skills are chained properly."""
        from agentforge import Agent
        from agentforge.core import Skill
        
        # Create skills that add to the data
        class AddOneSkill(Skill):
            name = "add_one"
            
            def execute(self, input_data):
                return {"value": input_data.get("value", 0) + 1}
        
        class AddTwoSkill(Skill):
            name = "add_two"
            
            def execute(self, input_data):
                return {"value": input_data.get("value", 0) + 2}
        
        agent = Agent(skills=[AddOneSkill(), AddTwoSkill()])
        result = agent.run({"value": 0})
        
        # 0 + 1 = 1, then 1 + 2 = 3
        assert result["value"] == 3
    
    def test_agent_with_llm(self, sample_llm_skill, mock_llm):
        """Test agent with LLM backend."""
        from agentforge import Agent
        
        agent = Agent(
            skills=[sample_llm_skill],
            llm=mock_llm,
            name="llm_agent"
        )
        
        result = agent.run({})
        
        assert result["used_llm"] is True
        assert mock_llm.call_count == 1
    
    def test_agent_continue_on_error(self, sample_skill):
        """Test agent continues on error when configured."""
        from agentforge import Agent
        from agentforge.core import Skill
        
        class FailingSkill(Skill):
            name = "failing"
            
            def execute(self, input_data):
                raise ValueError("Intentional failure")
        
        agent = Agent(
            skills=[FailingSkill(), sample_skill],
            continue_on_error=True
        )
        
        result = agent.run({})
        
        # Should have recorded the error and continued
        assert "_errors" in result
        assert result["processed"] is True  # From the second skill
    
    def test_agent_add_skill(self, sample_skill):
        """Test adding skills to an agent."""
        from agentforge import Agent
        
        agent = Agent(name="test")
        agent.add_skill(sample_skill)
        
        assert len(agent.skills) == 1
        
        # Test method chaining
        agent.add_skill(sample_skill).add_skill(sample_skill)
        assert len(agent.skills) == 3
    
    def test_agent_from_config(self, sample_config, mock_llm):
        """Test creating an agent from a config dict."""
        from agentforge import Agent
        
        # Simplified config without cloud LLM
        config = {
            "name": "config_agent",
            "skills": ["web_scraper"],
        }
        
        agent = Agent(config=config, llm=mock_llm)
        
        assert agent.name == "config_agent"
        assert len(agent.skills) == 1


class TestAgentAsync:
    """Tests for async agent operations."""
    
    @pytest.mark.asyncio
    async def test_agent_run_async(self, sample_skill):
        """Test async agent execution."""
        from agentforge import Agent
        
        agent = Agent(skills=[sample_skill], name="async_test")
        result = await agent.run_async({"input": "test"})
        
        assert result["processed"] is True
    
    @pytest.mark.asyncio
    async def test_skill_execute_async(self, sample_skill):
        """Test async skill execution."""
        result = await sample_skill.execute_async({"key": "value"})
        
        assert result["processed"] is True

