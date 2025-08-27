"""
Advanced example showing step() and final() markers with structured outputs.
"""

from dataclasses import dataclass
from typing import List
from functai import magic, step, final


@dataclass
class Analysis:
    sentiment: str
    confidence: float
    keywords: List[str]


@magic(adapter="json")
def analyze_text(text: str) -> Analysis:
    """Analyze text for sentiment and extract key information."""
    
    # Step 1: Extract sentiment
    sentiment: str = step(desc="Determine if positive, negative, or neutral")
    
    # Step 2: Calculate confidence
    confidence: float = step(desc="Confidence score between 0 and 1")
    
    # Step 3: Extract keywords
    keywords: List[str] = step(desc="List of important keywords")
    
    # Final output combines all steps
    result: Analysis = final(desc="Complete text analysis")
    
    return result


if __name__ == "__main__":
    # Example usage
    text = "This new AI library is absolutely fantastic! It makes working with LLMs so much easier."
    
    analysis = analyze_text(text)
    print(f"Text: {text}")
    print(f"Sentiment: {analysis.sentiment}")
    print(f"Confidence: {analysis.confidence}")
    print(f"Keywords: {', '.join(analysis.keywords)}")