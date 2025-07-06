import re
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(text: Union[str, float, None]) -> str:
    """
    Clean and normalize text by removing special characters and extra spaces
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text in lowercase
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def extract_duration_hours(duration_str: Union[str, float, None]) -> float:
    """
    Extract duration in hours from various string formats
    
    Args:
        duration_str: String representation of duration
        
    Returns:
        Duration in hours as a float
    """
    if pd.isna(duration_str):
        return 5.0
    
    duration_str = str(duration_str).lower()
    
    # Look for hours pattern
    hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|jam)', duration_str)
    if hours_match:
        return float(hours_match.group(1))
    
    # Look for minutes pattern and convert to hours
    minutes_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|menit)', duration_str)
    if minutes_match:
        return float(minutes_match.group(1)) / 60
    
    # Look for days pattern and convert to hours (assuming 8 hours per day)
    days_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:days?|hari)', duration_str)
    if days_match:
        return float(days_match.group(1)) * 8
    
    # Look for weeks pattern and convert to hours (assuming 40 hours per week)
    weeks_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:weeks?|minggu)', duration_str)
    if weeks_match:
        return float(weeks_match.group(1)) * 40
    
    # If it's just a number
    numbers = re.findall(r'\d+(?:\.\d+)?', duration_str)
    if numbers:
        return float(numbers[0])
    
    # Default value if no pattern is found
    return 5.0

def normalize_level(level: Union[str, float, None]) -> str:
    """
    Normalize course level names to standardized categories
    
    Args:
        level: Course level string
        
    Returns:
        Normalized level string
    """
    if pd.isna(level):
        return 'All Levels'
    
    level = str(level).lower().strip()
    
    if any(word in level for word in ['beginner', 'pemula', 'basic', 'dasar', 'introduction']):
        return 'Beginner Level'
    elif any(word in level for word in ['intermediate', 'menengah', 'medium']):
        return 'Intermediate Level'
    elif any(word in level for word in ['advanced', 'lanjut', 'expert', 'mahir', 'professional']):
        return 'Advanced Level'
    elif any(word in level for word in ['all', 'semua', 'any']):
        return 'All Levels'
    else:
        return 'All Levels'

def categorize_duration(hours: Union[float, None]) -> str:
    """
    Categorize course duration into short, medium, or long
    
    Args:
        hours: Duration in hours
        
    Returns:
        Duration category as string
    """
    if pd.isna(hours):
        return 'medium'
    
    if hours < 5:
        return 'short'
    elif hours < 20:
        return 'medium'
    return 'long'

def get_language(text: str) -> str:
    """
    Simple heuristic to detect if text is more likely English or Indonesian
    
    Args:
        text: Input text
        
    Returns:
        Detected language code ('en' or 'id')
    """
    # Common Indonesian words that don't appear in English
    indo_words = ['dan', 'atau', 'dengan', 'untuk', 'dalam', 'yang', 'ini', 'itu', 'pada', 'tidak']
    
    text = text.lower()
    indo_count = sum(1 for word in indo_words if word in text.split())
    
    # If at least 2 Indonesian words are found, classify as Indonesian
    if indo_count >= 2:
        return 'id'
    return 'en'

def safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Safely divide two numbers, returning 0 if denominator is 0
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        
    Returns:
        Result of division or 0 if denominator is 0
    """
    return numerator / denominator if denominator != 0 else 0.0

def normalize_weights(weights: List[float]) -> List[float]:
    """
    Normalize a list of weights to sum to 1.0
    
    Args:
        weights: List of weight values
        
    Returns:
        Normalized weights that sum to 1.0
    """
    total = sum(weights)
    if total == 0:
        # If all weights are 0, return equal weights
        return [1.0/len(weights)] * len(weights)
    return [w/total for w in weights]

def cosine_similarity_safe(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors safely
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return safe_division(dot_product, norm_vec1 * norm_vec2) 