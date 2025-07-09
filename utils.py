import re
import numpy as np
import pandas as pd
from typing import Union, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(text: Union[str, float, None]) -> str:
    """ 
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

def convert_k_to_number(value: Union[str, float, None]) -> float:
    """
    Convert values like '1.5k', '39.3k', '244', '14k' to numeric values
    
    Args:
        value: String or numeric value, possibly with 'k' suffix
        
    Returns:
        Numeric value after conversion
    """
    if pd.isna(value):
        return 0
    
    # Convert to string and clean
    value_str = str(value).strip().lower()
    
    if value_str == '' or value_str == 'nan':
        return 0
    
    # Handle 'k' or 'K' suffix
    if value_str.endswith('k'):
        try:
            number = float(value_str[:-1])
            return int(number * 1000)
        except ValueError:
            return 0
    else:
        try:
            return int(float(value_str))
        except ValueError:
            return 0

def extract_duration_hours(duration_str: Union[str, float, None]) -> float:
    """ 
    Returns:
        Duration in hours as a float
    """
    if pd.isna(duration_str):
        return 5.0
    
    duration_str = str(duration_str).lower()
    
    # Look for X-Y Months pattern
    months_range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*months?', duration_str)
    if months_range_match:
        start_month = float(months_range_match.group(1))
        end_month = float(months_range_match.group(2))
        # Take average and convert to hours (assuming 160 hours per month)
        avg_months = (start_month + end_month) / 2
        return avg_months * 160
    
    # Look for hours pattern
    hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|jam)', duration_str)
    if hours_match:
        return float(hours_match.group(1))
    
    # Look for days pattern and convert to hours (assuming 8 hours per day)
    days_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:days?|hari)', duration_str)
    if days_match:
        return float(days_match.group(1)) * 8
    
    # Look for weeks pattern and convert to hours (assuming 40 hours per week)
    weeks_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:weeks?|minggu)', duration_str)
    if weeks_match:
        return float(weeks_match.group(1)) * 40
    
    # Look for single month pattern
    single_month_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:months?|bulan)', duration_str)
    if single_month_match:
        return float(single_month_match.group(1)) * 160  # assuming 160 hours per month
    
    # If it's just a number
    numbers = re.findall(r'\d+(?:\.\d+)?', duration_str)
    if numbers:
        return float(numbers[0])
    
    # Default value if no pattern is found
    return 5.0

def normalize_level(level: Union[str, float, None]) -> str:

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
    Returns:
        Duration category as string
    """
    if pd.isna(hours):
        return 'medium'
    
    if hours < 10:
        return 'short'
    elif 10 < hours and hours < 35:
        return 'medium'
    return 'long'


def safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Returns:
        Result of division or 0 if denominator is 0
    """
    return numerator / denominator if denominator != 0 else 0.0

def normalize_weights(weights: List[float]) -> List[float]:
    """
    Returns:
        Normalized weights that sum to 1.0
    """
    total = sum(weights)
    if total == 0:
        # If all weights are 0, return equal weights
        return [1.0/len(weights)] * len(weights)
    return [w/total for w in weights]

# def cosine_similarity_safe(vec1: np.ndarray, vec2: np.ndarray) -> float:
#     """

#     Returns:
#         Cosine similarity value between -1 and 1
#     """
#     if np.all(vec1 == 0) or np.all(vec2 == 0):
#         return 0.0
    
#     dot_product = np.dot(vec1, vec2)
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
    
#     return safe_division(dot_product, norm_vec1 * norm_vec2) 