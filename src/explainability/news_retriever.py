"""
File: src/explainability/news_retriever.py
Role: Real-Time External Context Engine
Description: Queries the Tavily Search API for real-time localized disruptions (weather,
monsoon flooding, transport strikes, supply shocks) targeting specific APMC mandi clusters.
Dependencies: tavily-python, dotenv
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

class NewsRetriever:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key) if self.api_key else None

    def search_disruptions(self, target_mandi, neighbor_mandis=None, commodity="Tomato", query_context="weather flood harvest arrival"):
        """
        Executes a targeted search for regional agricultural disruptions.
        
        Args:
            target_mandi: Name of the primary APMC market (e.g., 'Pune').
            neighbor_mandis: List of dicts with top influential neighbors.
            commodity: Commodity name (e.g., 'Tomato').
            query_context: Context keywords to probe (weather, floods, road strikes).
            
        Returns:
            list of dicts containing title, content summary, and source url.
        """
        if not self.client:
            return [{"title": "No Search Client", "content": "TAVILY_API_KEY not configured.", "url": ""}]
            
        # Build query focusing on the market cluster
        cluster_names = [target_mandi]
        if neighbor_mandis:
            for nbr in neighbor_mandis[:2]:
                cluster_names.append(nbr.get("mandi", ""))
                
        cluster_str = " ".join(cluster_names)
        query = f"{cluster_str} {commodity} APMC mandi {query_context}"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_raw_content=False
            )
            
            snippets = []
            for item in response.get("results", []):
                snippets.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", "")
                })
            return snippets
        except Exception as e:
            return [{"title": "Search Error", "content": f"Unable to fetch web context: {e}", "url": ""}]
