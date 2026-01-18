"""
Tests for built-in skills.

I'm testing the web scraper, data analysis, and content generation
skills here.

Author: Prof. Shahab Anbarjafari
"""

import pytest
from unittest.mock import MagicMock, patch


class TestWebScraperSkill:
    """Tests for the WebScraperSkill."""
    
    def test_scraper_creation(self):
        """Test creating a scraper skill."""
        from agentforge.skills import WebScraperSkill
        
        scraper = WebScraperSkill()
        
        assert scraper.name == "web_scraper"
        assert scraper.requires_llm is False
    
    def test_scraper_execute(self, mock_requests_get, sample_html):
        """Test scraping a page."""
        from agentforge.skills import WebScraperSkill
        
        scraper = WebScraperSkill()
        result = scraper.execute({"url": "https://example.com"})
        
        assert result["url"] == "https://example.com"
        assert result["status_code"] == 200
        assert result["title"] == "Test Page"
        assert "Welcome" in result["content"]
    
    def test_scraper_with_selector(self, mock_requests_get):
        """Test scraping with CSS selector."""
        from agentforge.skills import WebScraperSkill
        
        scraper = WebScraperSkill()
        result = scraper.execute({
            "url": "https://example.com",
            "selector": "article p"
        })
        
        assert "first paragraph" in result["content"]
        assert result["element_count"] == 2
    
    def test_scraper_extract_links(self, mock_requests_get):
        """Test extracting links."""
        from agentforge.skills import WebScraperSkill
        
        scraper = WebScraperSkill()
        result = scraper.execute({
            "url": "https://example.com",
            "extract_links": True
        })
        
        assert "links" in result
        assert len(result["links"]) >= 2
    
    def test_scraper_validate_input(self):
        """Test input validation."""
        from agentforge.skills import WebScraperSkill
        
        scraper = WebScraperSkill()
        
        # Valid URL
        assert scraper.validate_input({"url": "https://example.com"}) is True
        
        # Missing URL
        assert scraper.validate_input({}) is False
        
        # Invalid URL
        assert scraper.validate_input({"url": "not-a-url"}) is False
    
    def test_scraper_no_url_raises(self):
        """Test that missing URL raises an error."""
        from agentforge.skills import WebScraperSkill
        
        scraper = WebScraperSkill()
        
        with pytest.raises(ValueError, match="No URL provided"):
            scraper.execute({})


class TestDataAnalysisSkill:
    """Tests for the DataAnalysisSkill."""
    
    def test_analyzer_creation(self):
        """Test creating a data analysis skill."""
        from agentforge.skills import DataAnalysisSkill
        
        analyzer = DataAnalysisSkill()
        
        assert analyzer.name == "data_analysis"
        assert analyzer.requires_llm is False
    
    def test_analyzer_with_data(self, sample_data):
        """Test analyzing data."""
        from agentforge.skills import DataAnalysisSkill
        
        analyzer = DataAnalysisSkill()
        result = analyzer.execute({
            "data": sample_data,
            "operations": ["describe"]
        })
        
        assert "result" in result
        assert result["row_count"] == 5
        assert "statistics" in result
    
    def test_analyzer_group_by(self, sample_data):
        """Test group by operation."""
        from agentforge.skills import DataAnalysisSkill
        
        analyzer = DataAnalysisSkill()
        result = analyzer.execute({
            "data": sample_data,
            "group_by": "department"
        })
        
        # Should have grouped counts
        assert result["row_count"] == 3  # 3 departments
    
    def test_analyzer_with_aggregation(self, sample_data):
        """Test group by with aggregation."""
        from agentforge.skills import DataAnalysisSkill
        
        analyzer = DataAnalysisSkill()
        result = analyzer.execute({
            "data": sample_data,
            "group_by": "department",
            "aggregate": {"score": "mean"}
        })
        
        assert result["row_count"] == 3
    
    def test_analyzer_sort(self, sample_data):
        """Test sorting."""
        from agentforge.skills import DataAnalysisSkill
        
        analyzer = DataAnalysisSkill()
        result = analyzer.execute({
            "data": sample_data,
            "sort_by": "score",
            "ascending": False
        })
        
        # Diana should be first (score 95)
        assert result["result"][0]["name"] == "Diana"
    
    def test_analyzer_filter(self, sample_data):
        """Test filtering."""
        from agentforge.skills import DataAnalysisSkill
        
        analyzer = DataAnalysisSkill()
        result = analyzer.execute({
            "data": sample_data,
            "filter": "score > 85"
        })
        
        # Should filter to 3 rows
        assert result["row_count"] == 3
    
    def test_analyzer_limit(self, sample_data):
        """Test limiting results."""
        from agentforge.skills import DataAnalysisSkill
        
        analyzer = DataAnalysisSkill()
        result = analyzer.execute({
            "data": sample_data,
            "limit": 2
        })
        
        assert result["row_count"] == 2


class TestContentGenerationSkill:
    """Tests for the ContentGenerationSkill."""
    
    def test_generator_creation(self):
        """Test creating a content generation skill."""
        from agentforge.skills import ContentGenerationSkill
        
        gen = ContentGenerationSkill()
        
        assert gen.name == "content_generation"
        assert gen.requires_llm is True
    
    def test_generator_with_prompt(self, mock_llm):
        """Test generation with direct prompt."""
        from agentforge.skills import ContentGenerationSkill
        
        gen = ContentGenerationSkill()
        gen.set_llm(mock_llm)
        
        result = gen.execute({"prompt": "Test prompt"})
        
        assert "generated" in result
        assert mock_llm.call_count == 1
    
    def test_generator_with_template(self, mock_llm):
        """Test generation with template."""
        from agentforge.skills import ContentGenerationSkill
        
        gen = ContentGenerationSkill()
        gen.set_llm(mock_llm)
        
        result = gen.execute({
            "template": "summarize",
            "text": "This is some text to summarize."
        })
        
        assert result["template_used"] == "summarize"
    
    def test_generator_default_template(self, mock_llm):
        """Test generation with default template."""
        from agentforge.skills import ContentGenerationSkill
        
        gen = ContentGenerationSkill(default_template="analyze")
        gen.set_llm(mock_llm)
        
        result = gen.execute({"text": "Some data to analyze."})
        
        assert result["template_used"] == "analyze"
    
    def test_generator_without_llm_raises(self):
        """Test that generation without LLM raises."""
        from agentforge.skills import ContentGenerationSkill
        
        gen = ContentGenerationSkill()
        
        with pytest.raises(RuntimeError, match="No LLM configured"):
            gen.execute({"prompt": "Test"})
    
    def test_generator_validate_input(self, mock_llm):
        """Test input validation."""
        from agentforge.skills import ContentGenerationSkill
        
        gen = ContentGenerationSkill()
        gen.set_llm(mock_llm)
        
        # Has prompt
        assert gen.validate_input({"prompt": "test"}) is True
        
        # Has text (will use default behavior)
        assert gen.validate_input({"text": "test"}) is True
        
        # Empty - invalid
        assert gen.validate_input({}) is False

