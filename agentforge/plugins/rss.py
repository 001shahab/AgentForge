"""
RSS feed monitoring plugin for AgentForge.

I built this plugin so you can easily monitor RSS feeds and extract new
content. It's perfect for news monitoring, blog aggregation, or any task
that needs to track updates from multiple sources.

Author: Prof. Shahab Anbarjafari

Note: Requires feedparser. Install with: pip install agentforge[rss]
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentforge.core import Skill

logger = logging.getLogger("agentforge.plugins.rss")


class RSSMonitorPlugin(Skill):
    """
    Monitor RSS feeds for new content.
    
    This plugin fetches RSS/Atom feeds and extracts the entries. You can
    filter by date, keywords, or use it to get the latest posts from
    your favorite blogs and news sites.
    
    Input:
        - feed_url: URL of the RSS/Atom feed
        - feeds: List of feed URLs (alternative to feed_url)
        - max_entries: Maximum entries to return per feed
        - since: Only return entries newer than this datetime
        - keywords: Filter entries containing these keywords
        
    Output:
        - entries: List of feed entries with title, link, summary, etc.
        - feed_title: Title of the feed
        - entry_count: Number of entries returned
        
    Example:
        >>> rss = RSSMonitorPlugin()
        >>> result = rss.execute({
        ...     "feed_url": "https://news.ycombinator.com/rss",
        ...     "max_entries": 10
        ... })
        >>> for entry in result["entries"]:
        ...     print(entry["title"])
    """
    
    name = "rss_monitor"
    description = "Monitor RSS/Atom feeds for new content"
    requires_llm = False
    
    def __init__(self, cache_entries: bool = True, **kwargs):
        """
        Initialize the RSS monitor.
        
        Args:
            cache_entries: Whether to cache seen entries (for deduplication)
            **kwargs: Additional options
        """
        super().__init__(**kwargs)
        self.cache_entries = cache_entries
        self._seen_entries: set = set()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch and parse RSS feeds.
        
        Args:
            input_data: Dictionary with feed URL(s) and options
            
        Returns:
            Dictionary with parsed entries
        """
        try:
            import feedparser
        except ImportError:
            raise ImportError(
                "feedparser not installed. Please install with:\n"
                "pip install feedparser\n"
                "or: pip install agentforge[rss]"
            )
        
        # Get feed URL(s)
        feed_url = input_data.get("feed_url")
        feeds = input_data.get("feeds", [])
        
        if feed_url:
            feeds = [feed_url] + feeds
        
        if not feeds:
            raise ValueError("No feed URL provided. Please specify 'feed_url' or 'feeds'.")
        
        max_entries = input_data.get("max_entries", 20)
        since = input_data.get("since")
        keywords = input_data.get("keywords", [])
        only_new = input_data.get("only_new", False)
        
        if isinstance(since, str):
            since = datetime.fromisoformat(since)
        
        all_entries = []
        feed_info = []
        
        for url in feeds:
            logger.info(f"Fetching feed: {url}")
            
            try:
                parsed = feedparser.parse(url)
                
                feed_info.append({
                    "url": url,
                    "title": parsed.feed.get("title", "Unknown"),
                    "link": parsed.feed.get("link", ""),
                })
                
                for entry in parsed.entries[:max_entries]:
                    processed = self._process_entry(entry, url)
                    
                    # Apply filters
                    if since and processed.get("published_dt"):
                        if processed["published_dt"] < since:
                            continue
                    
                    if keywords:
                        if not self._matches_keywords(processed, keywords):
                            continue
                    
                    if only_new and self.cache_entries:
                        entry_hash = self._hash_entry(processed)
                        if entry_hash in self._seen_entries:
                            continue
                        self._seen_entries.add(entry_hash)
                    
                    all_entries.append(processed)
                    
            except Exception as e:
                logger.error(f"Failed to fetch feed {url}: {e}")
                feed_info.append({
                    "url": url,
                    "error": str(e),
                })
        
        # Sort by date (newest first)
        all_entries.sort(
            key=lambda x: x.get("published_dt") or datetime.min,
            reverse=True
        )
        
        # Limit total entries
        all_entries = all_entries[:max_entries]
        
        return {
            "entries": all_entries,
            "entry_count": len(all_entries),
            "feeds": feed_info,
            "feed_count": len(feeds),
        }
    
    def _process_entry(self, entry: Any, feed_url: str) -> Dict[str, Any]:
        """Process a single feed entry."""
        # Parse the published date
        published_dt = None
        published_str = entry.get("published") or entry.get("updated")
        
        if published_str:
            try:
                import time
                from email.utils import parsedate_to_datetime
                
                # Try standard email date format first
                try:
                    published_dt = parsedate_to_datetime(published_str)
                except Exception:
                    # Try ISO format
                    published_dt = datetime.fromisoformat(
                        published_str.replace("Z", "+00:00")
                    )
            except Exception:
                pass
        
        # Get summary/content
        summary = ""
        if "summary" in entry:
            summary = entry.summary
        elif "content" in entry and entry.content:
            summary = entry.content[0].get("value", "")
        
        # Clean up HTML in summary
        if summary:
            from bs4 import BeautifulSoup
            summary = BeautifulSoup(summary, "html.parser").get_text(strip=True)
        
        return {
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "summary": summary[:500] if summary else "",  # Truncate long summaries
            "published": published_str or "",
            "published_dt": published_dt,
            "author": entry.get("author", ""),
            "feed_url": feed_url,
            "id": entry.get("id") or entry.get("link", ""),
        }
    
    def _matches_keywords(self, entry: Dict[str, Any], keywords: List[str]) -> bool:
        """Check if an entry matches any of the keywords."""
        text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
        return any(kw.lower() in text for kw in keywords)
    
    def _hash_entry(self, entry: Dict[str, Any]) -> str:
        """Generate a hash for deduplication."""
        unique_str = f"{entry.get('id', '')}{entry.get('title', '')}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the seen entries cache."""
        self._seen_entries.clear()
        logger.info("Entry cache cleared")

