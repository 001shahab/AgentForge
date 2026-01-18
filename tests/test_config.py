"""
Tests for configuration loading.

I'm testing the YAML config loading and validation here.

Author: Prof. Shahab Anbarjafari
"""

import pytest
import os
from pathlib import Path


class TestConfigLoading:
    """Tests for config loading functions."""
    
    def test_load_config(self, temp_config_file):
        """Test loading a config file."""
        from agentforge.config import load_config
        
        config = load_config(temp_config_file)
        
        assert config["name"] == "test_agent"
        assert config["llm"]["backend"] == "openai"
    
    def test_load_config_file_not_found(self):
        """Test loading a nonexistent file raises."""
        from agentforge.config import load_config
        
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
    
    def test_save_config(self, tmp_path):
        """Test saving a config file."""
        from agentforge.config import save_config, load_config
        
        config = {"name": "saved_agent", "skills": ["web_scraper"]}
        config_path = tmp_path / "saved.yaml"
        
        save_config(config, config_path)
        
        assert config_path.exists()
        
        loaded = load_config(config_path)
        assert loaded["name"] == "saved_agent"
    
    def test_create_template_config(self):
        """Test creating a template config."""
        from agentforge.config import create_template_config
        
        config = create_template_config(name="my_agent")
        
        assert config["name"] == "my_agent"
        assert "skills" in config
        assert "llm" in config
    
    def test_env_var_resolution(self, tmp_path):
        """Test environment variable resolution in config."""
        from agentforge.config import load_config
        import yaml
        
        # Set an env var
        os.environ["TEST_API_KEY"] = "secret123"
        
        # Create config with env var reference
        config_path = tmp_path / "env_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"api_key": "${TEST_API_KEY}"}, f)
        
        config = load_config(config_path)
        
        assert config["api_key"] == "secret123"
        
        # Clean up
        del os.environ["TEST_API_KEY"]


class TestConfigValidation:
    """Tests for config validation."""
    
    def test_valid_config(self, sample_config):
        """Test validating a valid config."""
        from agentforge.config import ConfigValidator
        
        validator = ConfigValidator(sample_config)
        is_valid = validator.validate()
        
        assert is_valid is True
        assert len(validator.errors) == 0
    
    def test_missing_name(self):
        """Test validation fails without name."""
        from agentforge.config import ConfigValidator
        
        config = {"skills": ["web_scraper"]}
        validator = ConfigValidator(config)
        is_valid = validator.validate()
        
        assert is_valid is False
        assert any("name" in e for e in validator.errors)
    
    def test_invalid_backend(self):
        """Test validation fails with invalid backend."""
        from agentforge.config import ConfigValidator
        
        config = {
            "name": "test",
            "llm": {"backend": "invalid_backend"}
        }
        validator = ConfigValidator(config)
        is_valid = validator.validate()
        
        assert is_valid is False
        assert any("backend" in e.lower() for e in validator.errors)
    
    def test_warning_no_skills(self):
        """Test warning when no skills defined."""
        from agentforge.config import ConfigValidator
        
        config = {"name": "test"}
        validator = ConfigValidator(config)
        validator.validate()
        
        assert any("skills" in w.lower() for w in validator.warnings)
    
    def test_get_report(self, sample_config):
        """Test getting validation report."""
        from agentforge.config import ConfigValidator
        
        validator = ConfigValidator(sample_config)
        validator.validate()
        
        report = validator.get_report()
        
        assert "valid" in report.lower()

