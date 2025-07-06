import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
import os
from pathlib import Path

from utils import clean_text, extract_duration_hours, normalize_level, categorize_duration

# Configure logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and preprocessing course data from different platforms
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize DataLoader
        
        Args:
            verbose: Whether to print additional information
        """
        self.verbose = verbose
        self.datasets = {}
        self.combined_df = None
    
    def load_udemy_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess Udemy dataset
        
        Args:
            file_path: Path to Udemy CSV file
            
        Returns:
            Preprocessed Udemy DataFrame
        """
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
                udemy_mapped['num_subscribers'] = pd.to_numeric(df_udemy.get('num_subscribers', 0), errors='coerce').fillna(0)
            
            if 'num_reviews' in df_udemy.columns:
                udemy_mapped['num_reviews'] = pd.to_numeric(df_udemy.get('num_reviews', 0), errors='coerce').fillna(0)
            
            if 'num_lectures' in df_udemy.columns:
                udemy_mapped['num_lectures'] = pd.to_numeric(df_udemy.get('num_lectures', 0), errors='coerce').fillna(0)
            
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
        """
        Load and preprocess Dicoding dataset
        
        Args:
            file_path: Path to Dicoding CSV file
            
        Returns:
            Preprocessed Dicoding DataFrame
        """
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
                dicoding_mapped['num_subscribers'] = pd.to_numeric(df_dicoding['num_enrolled_students'], errors='coerce').fillna(0)
            
            if 'num_modules' in df_dicoding.columns:
                dicoding_mapped['num_lectures'] = pd.to_numeric(df_dicoding['num_modules'], errors='coerce').fillna(0)
            
            # Store the dataset
            self.datasets['dicoding'] = dicoding_mapped
            return dicoding_mapped
            
        except Exception as e:
            logger.error(f"Error loading Dicoding data: {e}")
            return pd.DataFrame()
    
    def _get_price_from_cert_type(self, cert_type):
        """
        Determine price based on certificate type
        
        Args:
            cert_type: Certificate type string
            
        Returns:
            Estimated price for the course
        """
        if pd.isna(cert_type):
            return 0.0
            
        cert_type = str(cert_type).lower()
        
        # Gunakan hash dari cert_type untuk seed agar konsisten tapi berbeda per kursus
        seed = hash(cert_type) % 10000
        rng = np.random.RandomState(seed)
        
        # Harga yang lebih realistis dan bervariasi berdasarkan riset pasar Coursera
        if 'free' in cert_type:
            return 0.0
        elif 'professional' in cert_type:
            return rng.choice([79.99, 89.99, 99.99])  # Variasi harga
        elif 'specialization' in cert_type:
            return rng.choice([59.99, 64.99, 69.99, 74.99])  # Variasi harga
        elif 'certificate' in cert_type:
            return rng.choice([39.99, 44.99, 49.99, 54.99])  # Variasi harga
        elif 'degree' in cert_type:
            return rng.choice([399.99, 449.99, 499.99])  # Variasi harga
        elif 'master' in cert_type:
            return rng.choice([499.99, 599.99, 699.99])  # Variasi harga
        else:
            # Cek kata kunci lain
            if 'guided project' in cert_type:
                return rng.choice([9.99, 14.99, 19.99])  # Variasi harga
            elif 'course' in cert_type:
                return rng.choice([29.99, 34.99, 39.99, 44.99, 49.99])  # Variasi harga lebih banyak
            else:
                # Tentukan harga base berdasarkan karakteristik cert_type
                # Lebih banyak kata = kursus lebih komprehensif = harga lebih tinggi
                base_price = 20 + len(cert_type) % 10 * 2  # 20-38 base price
                # Tambahkan variasi acak
                variation = rng.uniform(-5, 5)
                # Format ke format harga $X.99
                return round(base_price + variation - 0.01) + 0.99  # Variasi harga
    
    def load_coursera_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess Coursera dataset
        
        Args:
            file_path: Path to Coursera CSV file
            
        Returns:
            Preprocessed Coursera DataFrame
        """
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
            price_cols = ['price', 'cost', 'fee', 'course_price']
            cert_type_cols = ['course_certificate_type', 'certificate_type', 'cert_type']
            is_free_cols = ['is_free', 'free', 'is_paid']
            
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
                coursera_mapped['level'] = 'Intermediate'  # Most Coursera courses are intermediate
            
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
                coursera_mapped['subject'] = 'Computer Science'  # Most Coursera courses are in CS
            
            # Map certificate type
            cert_type = None
            for col in cert_type_cols:
                if col in df_coursera.columns:
                    cert_type = df_coursera[col]
                    if self.verbose:
                        logger.info(f"Using '{col}' for certificate type information")
                    break
            
            # Strategi multi-level untuk harga:
            logger.info("Starting Coursera pricing determination process")
            
            # 1. Periksa kolom harga eksplisit
            price_found = False
            for col in price_cols:
                if col in df_coursera.columns:
                    # Check if price column has actual data (not all NaN or zeros)
                    if not df_coursera[col].isnull().all() and (df_coursera[col] > 0).any():
                        # Clean price strings if needed (remove currency symbols, commas)
                        if df_coursera[col].dtype == 'object':
                            coursera_mapped['price'] = df_coursera[col].str.replace('$', '', regex=False) \
                                                                    .str.replace(',', '', regex=False) \
                                                                    .astype(float)
                        else:
                            coursera_mapped['price'] = pd.to_numeric(df_coursera[col], errors='coerce').fillna(0)
                        
                        # Set is_paid based on price
                        coursera_mapped['is_paid'] = coursera_mapped['price'] > 0
                        price_found = True
                        logger.info(f"Using explicit price column '{col}' from dataset")
                        break
            
            # 2. Jika tidak ada harga eksplisit, gunakan tipe sertifikat
            if not price_found and cert_type is not None:
                # Set price based on certificate type
                coursera_mapped['price'] = cert_type.apply(self._get_price_from_cert_type)
                coursera_mapped['is_paid'] = coursera_mapped['price'] > 0
                price_found = True
                logger.info(f"Using certificate type for pricing. Price range: {coursera_mapped['price'].min():.2f}-{coursera_mapped['price'].max():.2f}")
            
            # 3. Jika masih tidak ada, periksa indikator is_free
            if not price_found:
                for col in is_free_cols:
                    if col in df_coursera.columns:
                        # If is_free is True, price is 0, otherwise use a default
                        if col == 'is_paid':
                            # is_paid is inverted logic of is_free
                            coursera_mapped['price'] = np.where(df_coursera[col].astype(bool), 49.99, 0.0)
                        else:
                            # Normal is_free logic
                            coursera_mapped['price'] = np.where(df_coursera[col].astype(bool), 0.0, 49.99)
                        
                        coursera_mapped['is_paid'] = coursera_mapped['price'] > 0
                        price_found = True
                        logger.info(f"Using '{col}' indicator for pricing")
                        break
            
            # 4. Jika semua di atas gagal, tentukan harga berdasarkan karakteristik lainnya
            if not price_found:
                # Set realistic price based on course attributes
                # Harga default berbasis level kesulitan dan subject
                if 'level' in coursera_mapped.columns:
                    # Seed untuk konsistensi antar run
                    seed_value = hash(str(coursera_mapped['course_title'].iloc[0])) % 10000
                    np.random.seed(seed_value)
                    logger.info(f"Using level and subject-based pricing (seed: {seed_value})")
                    
                    # Beginner courses tend to be cheaper
                    level_price_map = {
                        'Beginner': [29.99, 34.99, 39.99],
                        'Intermediate': [39.99, 44.99, 49.99],
                        'Advanced': [49.99, 59.99, 69.99],
                        'Expert': [59.99, 69.99, 79.99],
                        'All Levels': [34.99, 44.99, 54.99]
                    }
                    
                    # Subject-based price variations
                    subject_premium = {
                        'data science': 10.0,
                        'machine learning': 15.0,
                        'artificial intelligence': 20.0,
                        'programming': 5.0,
                        'computer science': 5.0,
                        'business': 0.0,
                        'finance': 8.0,
                        'marketing': 0.0,
                        'design': 5.0
                    }
                    
                    def get_price_from_level_and_subject(row):
                        level = row.get('level')
                        subject = str(row.get('subject', '')).lower()
                        
                        # Default price range
                        price_range = [39.99, 49.99, 59.99]
                        
                        # Adjust by level
                        if not pd.isna(level):
                            for level_key, prices in level_price_map.items():
                                if level_key.lower() in str(level).lower():
                                    price_range = prices
                                    break
                        
                        # Get base price from range
                        base_price = np.random.choice(price_range)
                        
                        # Add premium for high-demand subjects
                        premium = 0.0
                        for key, value in subject_premium.items():
                            if key in subject:
                                premium = max(premium, value)  # Take highest premium if multiple matches
                        
                        # Random variation (±$5)
                        variation = (np.random.random() - 0.5) * 10
                        
                        # Calculate final price (ensure it's at least $19.99)
                        final_price = max(19.99, base_price + premium + variation)
                        
                        # Round to nearest $0.99 price point
                        return round(final_price - 0.01, 0) + 0.99
                    
                    # Apply pricing function row by row
                    coursera_mapped['price'] = coursera_mapped.apply(get_price_from_level_and_subject, axis=1)
                    
                    # Log price distribution
                    price_min = coursera_mapped['price'].min()
                    price_max = coursera_mapped['price'].max()
                    price_mean = coursera_mapped['price'].mean()
                    price_unique = coursera_mapped['price'].nunique()
                    logger.info(f"Generated varied prices: min=${price_min:.2f}, max=${price_max:.2f}, avg=${price_mean:.2f}, {price_unique} unique prices")
                    
                else:
                    # Default price if we can't determine anything else
                    # Create varied prices instead of fixed 49.99
                    np.random.seed(42)  # For reproducibility
                    coursera_mapped['price'] = np.random.choice(
                        [29.99, 39.99, 44.99, 49.99, 54.99, 59.99],
                        size=len(coursera_mapped)
                    )
                    logger.info("Using random price distribution from preset values")
                
                coursera_mapped['is_paid'] = True
            
            # Add platform identifier
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
                coursera_mapped['num_subscribers'] = pd.to_numeric(df_coursera['course_students_enrolled'], errors='coerce').fillna(0)
            elif 'students_enrolled' in df_coursera.columns:
                coursera_mapped['num_subscribers'] = pd.to_numeric(df_coursera['students_enrolled'], errors='coerce').fillna(0)
            elif 'num_enrolled' in df_coursera.columns:
                coursera_mapped['num_subscribers'] = pd.to_numeric(df_coursera['num_enrolled'], errors='coerce').fillna(0)
            
            # Verify final price distribution
            if self.verbose:
                price_counts = coursera_mapped['price'].value_counts().head(10)
                logger.info(f"Final Coursera price distribution (top 10): {dict(zip(price_counts.index.round(2), price_counts.values))}")
            
            # Store the dataset
            self.datasets['coursera'] = coursera_mapped
            return coursera_mapped
            
        except Exception as e:
            logger.error(f"Error loading Coursera data: {e}")
            return pd.DataFrame()
    
    def combine_data(self) -> pd.DataFrame:
        """
        Combine all loaded datasets into a single DataFrame
        
        Returns:
            Combined DataFrame with all courses
        """
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
        """
        Apply preprocessing to the combined dataset
        
        Returns:
            Preprocessed combined DataFrame
        """
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
        """
        Load all specified datasets and preprocess them
        
        Args:
            udemy_path: Path to Udemy CSV file
            dicoding_path: Path to Dicoding CSV file
            coursera_path: Path to Coursera CSV file
            
        Returns:
            Preprocessed combined DataFrame
        """
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
        """
        Get statistics about the loaded data
        
        Returns:
            Dictionary with dataset statistics
        """
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
        
        # Add subject distribution (top 10)
        subject_counts = self.combined_df['subject'].value_counts().head(10).to_dict()
        stats['subject_distribution'] = subject_counts
        
        return stats 