"""
Web scraping skill for AgentForge.

I've built this skill to make web scraping easy and reliable. It handles
the common tasks like fetching pages, parsing HTML, and extracting content.
You can use CSS selectors to grab exactly what you need.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from agentforge.core import Skill

logger = logging.getLogger("agentforge.skills.scraping")


class WebScraperSkill(Skill):
    """
    Fetch and parse web pages.
    
    This skill downloads a web page and extracts content from it. You can
    use CSS selectors to target specific elements, or just grab all the
    text content.
    
    Input:
        - url: The URL to scrape
        - selector: (Optional) CSS selector to extract specific elements
        - extract_links: (Optional) Whether to extract links
        - extract_images: (Optional) Whether to extract image URLs
        
    Output:
        - content: The extracted text content
        - html: The raw HTML (if requested)
        - links: List of links (if extract_links=True)
        - images: List of image URLs (if extract_images=True)
        - title: Page title
        
    Example:
        >>> scraper = WebScraperSkill()
        >>> result = scraper.execute({
        ...     "url": "https://example.com",
        ...     "selector": "article p"
        ... })
        >>> print(result["content"])
    """
    
    name = "web_scraper"
    description = "Fetch and parse web pages to extract content"
    requires_llm = False
    
    # Default headers to look like a real browser
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    def __init__(
        self,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        **kwargs
    ):
        """
        Initialize the web scraper.
        
        Args:
            timeout: Request timeout in seconds
            headers: Custom headers to send with requests
            verify_ssl: Whether to verify SSL certificates
            **kwargs: Additional options
        """
        super().__init__(**kwargs)
        self.timeout = timeout
        self.headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        self.verify_ssl = verify_ssl
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch and parse a web page.
        
        Args:
            input_data: Dictionary containing at minimum a 'url' key
            
        Returns:
            Dictionary with extracted content
        """
        url = input_data.get("url")
        if not url:
            raise ValueError("No URL provided. Please include 'url' in input_data.")
        
        selector = input_data.get("selector")
        extract_links = input_data.get("extract_links", False)
        extract_images = input_data.get("extract_images", False)
        include_html = input_data.get("include_html", False)
        
        logger.info(f"Scraping: {url}")
        
        # Fetch the page
        response = self._fetch(url)
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements (they just add noise)
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        
        result = {
            "url": url,
            "status_code": response.status_code,
            "title": self._get_title(soup),
        }
        
        # Extract content based on selector or get all text
        if selector:
            elements = soup.select(selector)
            content_parts = [elem.get_text(strip=True) for elem in elements]
            result["content"] = "\n\n".join(content_parts)
            result["element_count"] = len(elements)
        else:
            result["content"] = self._get_clean_text(soup)
        
        # Optionally include raw HTML
        if include_html:
            if selector:
                elements = soup.select(selector)
                result["html"] = "\n".join(str(elem) for elem in elements)
            else:
                result["html"] = str(soup)
        
        # Extract links if requested
        if extract_links:
            result["links"] = self._extract_links(soup, url)
        
        # Extract images if requested
        if extract_images:
            result["images"] = self._extract_images(soup, url)
        
        logger.info(f"Scraped {len(result['content'])} characters from {url}")
        
        return result
    
    def _fetch(self, url: str) -> requests.Response:
        """Fetch a URL and return the response."""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def _get_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title."""
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Try og:title as fallback
        og_title = soup.find("meta", property="og:title")
        if og_title:
            return og_title.get("content", "")
        
        # Try h1 as last resort
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        
        return ""
    
    def _get_clean_text(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from the page.
        
        I'm doing some extra cleanup here to remove excessive whitespace
        and make the text more readable.
        """
        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Collapse multiple newlines
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = []
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Make absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Skip javascript: and mailto: links
            if absolute_url.startswith(("javascript:", "mailto:", "#")):
                continue
            
            links.append({
                "url": absolute_url,
                "text": a.get_text(strip=True),
            })
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all images from the page."""
        images = []
        
        for img in soup.find_all("img", src=True):
            src = img["src"]
            # Make absolute URL
            absolute_url = urljoin(base_url, src)
            
            # Skip data URLs (they're usually tiny icons)
            if absolute_url.startswith("data:"):
                continue
            
            images.append({
                "url": absolute_url,
                "alt": img.get("alt", ""),
            })
        
        return images
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate that we have a URL."""
        url = input_data.get("url", "")
        if not url:
            return False
        
        # Basic URL validation
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


class MultiPageScraperSkill(Skill):
    """
    Scrape multiple pages in sequence.
    
    This is useful when you need to follow pagination or scrape a list
    of URLs. It uses WebScraperSkill under the hood.
    
    Input:
        - urls: List of URLs to scrape
        - selector: (Optional) CSS selector to use for all pages
        - max_pages: (Optional) Maximum number of pages to scrape
        
    Output:
        - pages: List of results from each page
        - total_content: Combined content from all pages
    """
    
    name = "multi_page_scraper"
    description = "Scrape multiple pages and combine the results"
    requires_llm = False
    
    def __init__(self, delay: float = 1.0, **kwargs):
        """
        Initialize the multi-page scraper.
        
        Args:
            delay: Delay between requests in seconds (be nice to servers!)
            **kwargs: Options passed to WebScraperSkill
        """
        super().__init__(**kwargs)
        self.delay = delay
        self.scraper = WebScraperSkill(**kwargs)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape multiple pages."""
        import time
        
        urls = input_data.get("urls", [])
        if not urls:
            raise ValueError("No URLs provided. Please include 'urls' list.")
        
        max_pages = input_data.get("max_pages", len(urls))
        selector = input_data.get("selector")
        
        pages = []
        all_content = []
        
        for i, url in enumerate(urls[:max_pages]):
            logger.info(f"Scraping page {i + 1}/{min(len(urls), max_pages)}: {url}")
            
            page_input = {"url": url}
            if selector:
                page_input["selector"] = selector
            
            try:
                result = self.scraper.execute(page_input)
                pages.append(result)
                all_content.append(result.get("content", ""))
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                pages.append({"url": url, "error": str(e)})
            
            # Be nice to the server
            if i < len(urls) - 1:
                time.sleep(self.delay)
        
        return {
            "pages": pages,
            "total_content": "\n\n---\n\n".join(all_content),
            "pages_scraped": len(pages),
            "pages_failed": sum(1 for p in pages if "error" in p),
        }

