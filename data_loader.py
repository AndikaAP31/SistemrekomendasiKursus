import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import os
from pathlib import Path

from utils import clean_text, extract_duration_hours, normalize_level, categorize_duration, convert_k_to_number

# Configure logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and preprocessing course data from different platforms
    """
    def __init__(self, verbose: bool = True):
     
        self.verbose = verbose
        self.datasets = {}
        self.combined_df = None
    
    def load_udemy_data(self, file_path: str) -> pd.DataFrame:
     
        try:
            df_udemy = pd.read_csv(file_path)
            if self.verbose:
                logger.info(f"Udemy dataset loaded: {len(df_udemy)} courses")
            
            # Create mapped DataFrame with consistent column names
            udemy_mapped = pd.DataFrame()
            
            # Map course title
            udemy_mapped['course_title'] = df_udemy.get('course_title', 'Unknown Course')
            
            # Map price - handle numeric conversion
            udemy_mapped['price'] = pd.to_numeric(df_udemy.get('price', 0), errors='coerce').fillna(0)
            
            # Map content duration
            udemy_mapped['content_duration'] = pd.to_numeric(df_udemy.get('content_duration', 5), errors='coerce').fillna(5)
            
            # Map level
            udemy_mapped['level'] = df_udemy.get('level', 'All Levels')
            
            # Map paid status
            udemy_mapped['is_paid'] = df_udemy.get('is_paid', True)
            
            # Map subject/category
            udemy_mapped['subject'] = df_udemy.get('subject', 'General')
            
            # Add platform identifier
            udemy_mapped['platform'] = 'Udemy'
            
            # Additional potentially useful columns if available
            if 'num_subscribers' in df_udemy.columns:
                udemy_mapped['num_subscribers'] = df_udemy['num_subscribers'].apply(convert_k_to_number)
            
            if 'num_reviews' in df_udemy.columns:
                udemy_mapped['num_reviews'] = df_udemy['num_reviews'].apply(convert_k_to_number)
            
            if 'num_lectures' in df_udemy.columns:
                udemy_mapped['num_lectures'] = df_udemy['num_lectures'].apply(convert_k_to_number)
            
            # Extract course URLs if available
            if 'url' in df_udemy.columns:
                udemy_mapped['course_url'] = df_udemy['url']
            
            # Store the dataset
            self.datasets['udemy'] = udemy_mapped
            return udemy_mapped
            
        except Exception as e:
            logger.error(f"Error loading Udemy data: {e}")
            return pd.DataFrame()
    
    def load_dicoding_data(self, file_path: str) -> pd.DataFrame:
   
        try:
            df_dicoding = pd.read_csv(file_path, sep='|')
            if self.verbose:
                logger.info(f"Dicoding dataset loaded: {len(df_dicoding)} courses")
            
            # Create mapped DataFrame with consistent column names
            dicoding_mapped = pd.DataFrame()
            
            # Define possible column mappings
            title_cols = ['course_name', 'title', 'course_title', 'name']
            duration_cols = ['duration', 'course_duration_hours', 'length', 'time']
            level_cols = ['level', 'difficulty_level', 'course_level']
            subject_cols = ['category', 'subject', 'topic', 'course_category']
            
            # Map course title
            for col in title_cols:
                if col in df_dicoding.columns:
                    dicoding_mapped['course_title'] = df_dicoding[col]
                    break
            else:
                dicoding_mapped['course_title'] = 'Unknown Course'
            
            # Map duration
            for col in duration_cols:
                if col in df_dicoding.columns:
                    dicoding_mapped['content_duration'] = df_dicoding[col].apply(extract_duration_hours)
                    break
            else:
                dicoding_mapped['content_duration'] = 5.0
            
            # Map level
            for col in level_cols:
                if col in df_dicoding.columns:
                    dicoding_mapped['level'] = df_dicoding[col]
                    break
            else:
                dicoding_mapped['level'] = 'All Levels'
            
            # Map subject/category
            for col in subject_cols:
                if col in df_dicoding.columns:
                    dicoding_mapped['subject'] = df_dicoding[col]
                    break
            else:
                dicoding_mapped['subject'] = 'Programming'
            
            # Set price (Dicoding courses are mostly free)
            dicoding_mapped['price'] = 0
            dicoding_mapped['is_paid'] = False
            
            # Add platform identifier
            dicoding_mapped['platform'] = 'Dicoding'
            
            # Extract additional metadata if available
            if 'course_link' in df_dicoding.columns:
                dicoding_mapped['course_url'] = df_dicoding['course_link']
            
            if 'rating' in df_dicoding.columns:
                dicoding_mapped['rating'] = pd.to_numeric(df_dicoding['rating'], errors='coerce')
            
            if 'num_enrolled_students' in df_dicoding.columns:
                dicoding_mapped['num_subscribers'] = df_dicoding['num_enrolled_students'].apply(convert_k_to_number)
            
            if 'num_modules' in df_dicoding.columns:
                dicoding_mapped['num_lectures'] = df_dicoding['num_modules'].apply(convert_k_to_number)
            
            # Store the dataset
            self.datasets['dicoding'] = dicoding_mapped
            return dicoding_mapped
            
        except Exception as e:
            logger.error(f"Error loading Dicoding data: {e}")
            return pd.DataFrame()
    
    def _calculate_coursera_price_by_level_duration(self, level, duration):
   
        # Normalize level for consistent mapping
        if pd.isna(level):
            level = "Intermediate"
        
        level = str(level).lower()
        
        # Ensure duration is a number
        if pd.isna(duration):
            duration = 10.0
        else:
            try:
                duration = float(duration)
            except (ValueError, TypeError):
                duration = 10.0
        
        # Categorize duration
        if duration < 160:
            duration_category = "Short"
        elif duration < 320:
            duration_category = "Medium" 
        else:
            duration_category = "Long"
        
        # Base price matrix based on level and duration
        price_matrix = {
            "beginner": {
                "Short": 30,
                "Medium": 40,
                "Long": 50
            },
            "intermediate": {
                "Short": 45,
                "Medium": 60,
                "Long": 75
            },
            "advanced": {
                "Short": 70,
                "Medium": 90,
                "Long": 110
            },
            "all levels": {
                "Short": 40,
                "Medium": 55,
                "Long": 70
            }
        }
        

        seed = hash(f"{level}_{duration_category}") % 10000
        rng = np.random.RandomState(seed)
        
        # Find the closest level match
        if "beginner" in level:
            level_key = "beginner"
        elif "intermediate" in level:
            level_key = "intermediate"
        elif "advanced" in level:
            level_key = "advanced"
        elif "all" in level:
            level_key = "all levels"
        else:
            level_key = "intermediate"  # Default to intermediate
        
        # Get base price from matrix
        base_price = price_matrix[level_key][duration_category]
        
        # Add small variation (-5 to +5)
        variation = rng.randint(-5, 6)
        final_price = base_price + variation
        
        # Make sure price is not negative
        return max(0, final_price)
    
   
    def load_coursera_data(self, file_path: str) -> pd.DataFrame:
   
        try:
            df_coursera = pd.read_csv(file_path)
            if self.verbose:
                logger.info(f"Coursera dataset loaded: {len(df_coursera)} courses")
            
            # Create mapped DataFrame with consistent column names
            coursera_mapped = pd.DataFrame()
            
            # Define possible column mappings
            title_cols = ['course_title', 'course_name', 'title', 'name']
            duration_cols = ['course_time', 'duration', 'course_duration', 'length', 'time', 'estimated_time']
            level_cols = ['course_difficulty', 'level', 'difficulty', 'course_level', 'skill_level']
            subject_cols = ['course_skills', 'category', 'subject', 'topic', 'course_category', 'skills']
            
            # Map course title
            for col in title_cols:
                if col in df_coursera.columns:
                    coursera_mapped['course_title'] = df_coursera[col]
                    break
            else:
                coursera_mapped['course_title'] = 'Unknown Course'
            
            # Map duration - extract hours from text descriptions
            for col in duration_cols:
                if col in df_coursera.columns:
                    coursera_mapped['content_duration'] = df_coursera[col].apply(extract_duration_hours)
                    break
            else:
                # Coursera courses are typically 4-6 weeks, ~3-5 hours per week
                coursera_mapped['content_duration'] = 20.0  
            
            # Map difficulty level
            for col in level_cols:
                if col in df_coursera.columns:
                    coursera_mapped['level'] = df_coursera[col]
                    break
            else:
                coursera_mapped['level'] = 'Intermediate'  
            
            # Map subject/category - handle list format in Coursera data
            for col in subject_cols:
                if col in df_coursera.columns:
                    if df_coursera[col].dtype == 'object':
                        # If it's a string that might be a list, extract first item
                        coursera_mapped['subject'] = df_coursera[col].apply(
                            lambda x: x.split(',')[0].strip("[]'\"") if isinstance(x, str) and ',' in x else x
                        )
                    else:
                        coursera_mapped['subject'] = df_coursera[col]
                    break
            else:
                coursera_mapped['subject'] = 'Computer Science'  
            
            # Menggunakan level dan durasi untuk menentukan harga
            coursera_mapped['price'] = coursera_mapped.apply(
                lambda row: self._calculate_coursera_price_by_level_duration(
                    row['level'], 
                    row['content_duration']
                ),
                axis=1
            )
                               
            coursera_mapped['is_paid'] = coursera_mapped['price'] > 0

            coursera_mapped['platform'] = 'Coursera'
            
            # Extract additional metadata if available
            if 'course_url' in df_coursera.columns:
                coursera_mapped['course_url'] = df_coursera['course_url']
            elif 'url' in df_coursera.columns:
                coursera_mapped['course_url'] = df_coursera['url']
            
            if 'course_rating' in df_coursera.columns:
                coursera_mapped['rating'] = pd.to_numeric(df_coursera['course_rating'], errors='coerce')
            elif 'rating' in df_coursera.columns:
                coursera_mapped['rating'] = pd.to_numeric(df_coursera['rating'], errors='coerce')
            
            if 'course_students_enrolled' in df_coursera.columns:
                # Menghapus koma dari nilai dan mengkonversi ke numerik
                coursera_mapped['num_subscribers'] = df_coursera['course_students_enrolled'].astype(str).str.replace(',', '').str.replace('"', '').astype(float).fillna(0).astype(int)
              
            if 'course_reviews_num' in df_coursera.columns:
                coursera_mapped['num_reviews'] = df_coursera['course_reviews_num'].apply(convert_k_to_number)
            
            if self.verbose:
                price_counts = coursera_mapped['price'].value_counts().head(10)
                logger.info(f"Final Coursera price distribution (top 10): {dict(zip(price_counts.index.round(0), price_counts.values))}")
            
            # Store the dataset
            self.datasets['coursera'] = coursera_mapped
            return coursera_mapped
            
        except Exception as e:
            logger.error(f"Error loading Coursera data: {e}")
            return pd.DataFrame()
    
    def combine_data(self) -> pd.DataFrame:
       
        if not self.datasets:
            logger.warning("No datasets have been loaded yet")
            return pd.DataFrame()
        
        self.combined_df = pd.concat(self.datasets.values(), ignore_index=True)
        
        if self.verbose:
            logger.info(f"Combined {len(self.combined_df)} courses from {len(self.datasets)} platforms")
            if self.combined_df is not None:
                platform_counts = self.combined_df['platform'].value_counts().to_dict()
                logger.info(f"Platform distribution: {platform_counts}")
        
        return self.combined_df
    
    def preprocess_combined_data(self) -> pd.DataFrame:
       
        if self.combined_df is None or self.combined_df.empty:
            logger.warning("No combined data available for preprocessing")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original
        df = self.combined_df.copy()
        
        # Clean and normalize text fields
        df['course_title'] = df['course_title'].apply(clean_text).fillna('unknown course')
        df['subject'] = df['subject'].apply(clean_text).fillna('general')
        
        # Normalize level categories
        df['level'] = df['level'].apply(normalize_level)
        
        # Ensure numeric fields are properly formatted
        df['content_duration'] = pd.to_numeric(df['content_duration'], errors='coerce').fillna(5.0)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
        
        # Pastikan harga selalu bulat
        df['price'] = df['price'].round(0).astype(int)
        
        df['is_paid'] = df['is_paid'].fillna(True)
        
        # Add duration category
        df['duration_category'] = df['content_duration'].apply(categorize_duration)
        
        # Ensure rating is in a consistent range (0-5)
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
            # Normalize ratings that might be on a 0-10 or 0-100 scale
            mask_0_10 = (df['rating'] > 5) & (df['rating'] <= 10)
            mask_0_100 = df['rating'] > 10
            df.loc[mask_0_10, 'rating'] = df.loc[mask_0_10, 'rating'] / 2
            df.loc[mask_0_100, 'rating'] = df.loc[mask_0_100, 'rating'] / 20
        
        # Create a combined text field for text-based recommendation
        df['combined_text'] = (
            df['course_title'] + ' ' +
            df['level'] + ' ' +
            df['subject'] + ' ' +
            df['duration_category'] + ' ' +
            df['platform']
        ).str.lower()
        
        # Update the combined dataset
        self.combined_df = df
        
        return df
    
    def load_and_preprocess_all(self, 
                               udemy_path: Optional[str] = None, 
                               dicoding_path: Optional[str] = None,
                               coursera_path: Optional[str] = None) -> pd.DataFrame:
        
        datasets_loaded = []
        
        if udemy_path and os.path.exists(udemy_path):
            self.load_udemy_data(udemy_path)
            datasets_loaded.append('Udemy')
        
        if dicoding_path and os.path.exists(dicoding_path):
            self.load_dicoding_data(dicoding_path)
            datasets_loaded.append('Dicoding')
        
        if coursera_path and os.path.exists(coursera_path):
            self.load_coursera_data(coursera_path)
            datasets_loaded.append('Coursera')
        
        if not datasets_loaded:
            logger.warning("No valid dataset paths provided")
            return pd.DataFrame()
        
        logger.info(f"Loaded datasets from: {', '.join(datasets_loaded)}")
        
        # Combine all loaded datasets
        self.combine_data()
        
        # Preprocess the combined dataset
        return self.preprocess_combined_data()
    
    def get_data_stats(self) -> Dict:
        if self.combined_df is None or self.combined_df.empty:
            logger.warning("No data available for statistics")
            return {}
        
        stats = {
            'total_courses': len(self.combined_df),
            'platform_distribution': self.combined_df['platform'].value_counts().to_dict(),
            'level_distribution': self.combined_df['level'].value_counts().to_dict(),
            'free_courses': int((self.combined_df['is_paid'] == False).sum()),
            'paid_courses': int((self.combined_df['is_paid'] == True).sum()),
            'avg_price': float(self.combined_df[self.combined_df['price'] > 0]['price'].mean()),
            'avg_duration': float(self.combined_df['content_duration'].mean()),
            'duration_distribution': self.combined_df['duration_category'].value_counts().to_dict(),
        }
           
        return stats 