"""
File: src/explainability/advisory_generator.py
Role: Grounded Neuro-Symbolic Synthesizer
Description: Combines mathematical GNN feature/spatial attribution with live Tavily news context,
using Groq's Qwen3.8-27b model to generate concise, hallucination-resistant farmer advisories.
Dependencies: groq, dotenv
"""

import os
from dotenv import load_dotenv
import groq

load_dotenv()

class AdvisoryGenerator:
    def __init__(self, api_key=None, model="qwen/qwen3.8-27b"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.client = groq.Groq(api_key=self.api_key) if self.api_key else None

    def generate_advisory(self, attribution, news_snippets=None, historical_trend="Historically, harvest peaks in this season trigger a price crash."):
        """
        Synthesizes GNN attribution and external news into an actionable advisory.
        
        Args:
            attribution: Dict from STGNNExplainer containing prediction, neighbors, and salient days.
            news_snippets: List of dicts from NewsRetriever with live search evidence.
            historical_trend: Baseline expectation for this commodity and season.
            
        Returns:
            str: Grounded 2-3 sentence natural-language explanation.
        """
        if not self.client:
            return "Advisory unavailable: GROQ_API_KEY is not configured."

        # Format mathematical findings
        target = attribution.get("target_mandi", "Unknown")
        pred_class = attribution.get("predicted_class", "Stable")
        conf = attribution.get("confidence", 0.0) * 100
        
        neighbors_str = ", ".join([
            f"{n['mandi']} (attention weight: {n['attention_weight']}, distance: {n['distance_km']} km)"
            for n in attribution.get("top_influential_neighbors", [])
        ]) or "No strong spatial neighbor pull"
        
        salient_days_str = ", ".join([
            f"Day t{d['day_lag']} ({d['saliency_percent']}% influence)"
            for d in attribution.get("most_salient_days", [])
        ]) or "Gradual uniform momentum"

        # Format news evidence
        news_str = ""
        if news_snippets:
            for i, item in enumerate(news_snippets[:2]):
                if item.get("content"):
                    news_str += f"- [{item.get('title')}]: {item.get('content')[:200]}...\n"
        if not news_str.strip():
            news_str = "No verified external news disruption found. Rely strictly on spatial supply-demand metrics."

        prompt = f"""You are an expert Agricultural Market Analyst advising Indian farmers and mandi traders.
Synthesize the following mathematical neural network data and real-world search evidence into a clear, grounded explanation.

[NEURAL NETWORK FORECAST]
- Target Mandi: {target}
- Forecasted State: {pred_class} (Confidence: {conf:.1f}%)
- Historical Baseline Expectation: {historical_trend}
- Spatial Influences (GATv2 Attention): {neighbors_str}
- Temporal Momentum (Salient Lag Days): {salient_days_str}

[VERIFIED EXTERNAL NEWS CONTEXT (TAVILY)]
{news_str}

[RULES]
1. Never hallucinate fake government policies, strikes, or floods. Only reference news events if supported by the external context above.
2. If the forecast diverges from the historical baseline, explain WHY using the neighbor attention or verified weather/road disruption.
3. Keep the output strictly to 2 or 3 concise, actionable sentences suitable for an agricultural advisory SMS or WhatsApp bulletin.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise, evidence-grounded agricultural market economist. Be concise and factual."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2, # Low temperature for factual precision
                max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating advisory via Groq: {e}"
