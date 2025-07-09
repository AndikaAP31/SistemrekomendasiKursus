import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from gensim.models import Word2Vec, FastText
import warnings
import time
import re

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')



class FeatureEngineering:
    
    def __init__(self, 
                verbose: bool = True, 
                random_state: int = 42,
                tfidf_params: Optional[Dict] = None,
                w2v_params: Optional[Dict] = None):
        
        self.verbose = verbose
        self.random_state = random_state
        
        # Initialize encoders for categorical features
        self.encoders = {
            'level': LabelEncoder(),
            'duration': LabelEncoder(),
            'paid': LabelEncoder(),
            'subject': LabelEncoder(),
            'platform': LabelEncoder()
        }
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Set default parameters for TF-IDF
        self.tfidf_params = {
            'max_features': None,  # No limit on features
            'ngram_range': (1, 2),
            'stop_words': None,  # We'll handle stop words in preprocessing
            'min_df': 1,  # Include all terms
            'max_df': 0.95,  # Remove only extremely common terms
            'strip_accents': 'unicode',
            'lowercase': True
        }
        
        # Override with user-provided parameters if any
        if tfidf_params:
            self.tfidf_params.update(tfidf_params)
        
        # Set default parameters for Word2Vec
        self.w2v_params = {
            'vector_size': 100,
            'window': 5,
            'min_count': 1,
            'workers': 4,
            'sg': 1  # Skip-gram model
        }
        
        # Override with user-provided parameters if any
        if w2v_params:
            self.w2v_params.update(w2v_params)
        
        # Initialize text processing models
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.word2vec_model = None
        self.fasttext_model = None
        self.count_vectorizer = None
        self.lsa_model = None
        
        # Lists to store feature names
        self.categorical_features = []
        self.numerical_features = []
        self.text_features = []
        self.derived_features = []
        
        # Load stop words
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except:
            import nltk
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
    
    def _log_time(self, start_time: float, operation: str):

        if self.verbose:
            elapsed = time.time() - start_time
            logger.info(f"{operation} completed in {elapsed:.2f} seconds")
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
      
        start_time = time.time()
        
        # Make a copy to avoid modifying the input
        df_encoded = df.copy()
        
        # Encode level
        if 'level' in df.columns:
            df_encoded['level_encoded'] = self.encoders['level'].fit_transform(df['level'])
            self.categorical_features.append('level_encoded')
        
        # Encode duration category
        if 'duration_category' in df.columns:
            df_encoded['duration_encoded'] = self.encoders['duration'].fit_transform(df['duration_category'])
            self.categorical_features.append('duration_encoded')
        
        # Encode paid status
        if 'is_paid' in df.columns:
            df_encoded['paid_encoded'] = self.encoders['paid'].fit_transform(df['is_paid'].astype(str))
            self.categorical_features.append('paid_encoded')
        
        # Encode subject (may have many unique values)
        if 'subject' in df.columns:
            df_encoded['subject_encoded'] = self.encoders['subject'].fit_transform(df['subject'])
            self.categorical_features.append('subject_encoded')
        
        # Encode platform
        if 'platform' in df.columns:
            df_encoded['platform_encoded'] = self.encoders['platform'].fit_transform(df['platform'])
            self.categorical_features.append('platform_encoded')
        
        self._log_time(start_time, "Categorical encoding")
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:

        start_time = time.time()
        
        # Make a copy to avoid modifying the input
        df_scaled = df.copy()
        
        # Identify numerical features
        numerical_cols = ['price', 'content_duration']
        if 'rating' in df.columns:
            numerical_cols.append('rating')
        if 'num_subscribers' in df.columns:
            numerical_cols.append('num_subscribers')
        if 'num_reviews' in df.columns:
            numerical_cols.append('num_reviews')
        if 'num_lectures' in df.columns:
            numerical_cols.append('num_lectures')
        
        # Filter to only include columns that are actually in the DataFrame
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        self.numerical_features = numerical_cols.copy()  # Store original columns
        
        if not numerical_cols:
            logger.warning("No numerical features found for scaling")
            return df_scaled
        
        # Check if method is valid
        if method not in self.scalers:
            logger.warning(f"Scaling method '{method}' not found, using 'standard' instead")
            method = 'standard'
        
        # Create a copy of the numerical features
        scaled_data = df_scaled[numerical_cols].copy()
        
        # Handle missing values before scaling
        scaled_data.fillna(0, inplace=True)
        
        # Scale the data
        scaler = self.scalers[method]
        scaled_values = scaler.fit_transform(scaled_data)
        
        # Get the actual shape of the scaled values
        n_samples, n_features = scaled_values.shape
        
        # Ensure that numerical_cols matches the number of columns in scaled_values
        if len(numerical_cols) != n_features:
            logger.warning(f"Mismatch between numerical_cols ({len(numerical_cols)}) and scaled_values columns ({n_features})")
            # Truncate numerical_cols if needed
            numerical_cols = numerical_cols[:n_features]
        
        # Now safely iterate over the columns
        for i, col in enumerate(numerical_cols):
            df_scaled[f"{col}_scaled"] = scaled_values[:, i]
            self.numerical_features.append(f"{col}_scaled")
        
        self._log_time(start_time, f"Numerical scaling ({method})")
        
        return df_scaled
    
    def create_tfidf_features(self, df: pd.DataFrame, text_col: str = 'combined_text') -> Tuple[pd.DataFrame, np.ndarray]:

        start_time = time.time()
        
        if text_col not in df.columns:
            logger.warning(f"Text column '{text_col}' not found in DataFrame")
            return df, None
        
        # Preprocess text data
        processed_texts = df[text_col].fillna('').apply(self.preprocess_text)
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)
        
        # Fit and transform the text data
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Add TF-IDF feature flag to DataFrame
        df_with_tfidf = df.copy()
        df_with_tfidf['has_tfidf'] = True
        
        self.text_features.append('tfidf')
        
        # Log feature information
        if self.verbose:
            logger.info(f"Created TF-IDF matrix with shape: {self.tfidf_matrix.shape}")
            logger.info(f"Number of features: {len(self.tfidf_vectorizer.get_feature_names_out())}")
            
            # Log vocabulary statistics
            vocab = self.tfidf_vectorizer.get_feature_names_out()
            logger.info(f"Sample vocabulary terms: {vocab[:10]}")
            
            # Log sparsity
            sparsity = 1.0 - (self.tfidf_matrix.nnz / float(self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))
            logger.info(f"TF-IDF matrix sparsity: {sparsity:.4f}")
        
        self._log_time(start_time, "TF-IDF feature creation")
        
        return df_with_tfidf, self.tfidf_matrix
    
    def train_word2vec(self, df: pd.DataFrame, text_col: str = 'combined_text') -> Tuple[pd.DataFrame, Word2Vec]:

        start_time = time.time()
        
        if text_col not in df.columns:
            logger.warning(f"Text column '{text_col}' not found in DataFrame")
            return df, None
        
        # Prepare sentences for Word2Vec training
        sentences = [text.split() for text in df[text_col].fillna('')]
        
        # Train Word2Vec model
        logger.info("Starting Word2Vec model training...")
        self.word2vec_model = Word2Vec(sentences, **self.w2v_params)
        
        # Add Word2Vec feature flag to DataFrame
        df_with_w2v = df.copy()
        df_with_w2v['has_w2v'] = True
        
        self.text_features.append('word2vec')
        
        # Log model information
        if self.verbose:
            vocab_size = len(self.word2vec_model.wv.index_to_key)
            vector_size = self.word2vec_model.wv.vector_size
            logger.info(f"Trained Word2Vec model with vocabulary size: {vocab_size}, vector size: {vector_size}")
        
        self._log_time(start_time, "Word2Vec model training")
        
        return df_with_w2v, self.word2vec_model
    
    def train_fasttext(self, df: pd.DataFrame, text_col: str = 'combined_text') -> Tuple[pd.DataFrame, FastText]:
       
        start_time = time.time()
        
        if text_col not in df.columns:
            logger.warning(f"Text column '{text_col}' not found in DataFrame")
            return df, None
        
        # Prepare sentences for FastText training
        sentences = [text.split() for text in df[text_col].fillna('')]
        
        # Modify Word2Vec params for FastText
        fasttext_params = self.w2v_params.copy()
        
        # Train FastText model using gensim's FastText implementation
        self.fasttext_model = FastText(sentences, **fasttext_params)
        
        # Add FastText feature flag to DataFrame
        df_with_fasttext = df.copy()
        df_with_fasttext['has_fasttext'] = True
        
        self.text_features.append('fasttext')
        
        # Log model information
        if self.verbose:
            vocab_size = len(self.fasttext_model.wv.index_to_key)
            vector_size = self.fasttext_model.wv.vector_size
            logger.info(f"Trained FastText model with vocabulary size: {vocab_size}, vector size: {vector_size}")
        
        self._log_time(start_time, "FastText model training")
        
        return df_with_fasttext, self.fasttext_model
    
    def create_lsa_features(self, df: pd.DataFrame, text_col: str = 'combined_text', n_components: int = 50) -> Tuple[pd.DataFrame, np.ndarray]:
        start_time = time.time()
        
        if text_col not in df.columns:
            logger.warning(f"Text column '{text_col}' not found in DataFrame")
            return df, None
        
        # Create Count Vectorizer
        self.count_vectorizer = CountVectorizer(stop_words='english', min_df=2)
        
        # Fit and transform the text data
        count_matrix = self.count_vectorizer.fit_transform(df[text_col].fillna(''))
        
        # Apply Truncated SVD (LSA)
        self.lsa_model = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        lsa_features = self.lsa_model.fit_transform(count_matrix)
        
        # Add LSA feature flag to DataFrame
        df_with_lsa = df.copy()
        df_with_lsa['has_lsa'] = True
        
        self.text_features.append('lsa')
        
        # Log feature information
        if self.verbose:
            logger.info(f"Created LSA features with shape: {lsa_features.shape}")
            explained_var = self.lsa_model.explained_variance_ratio_.sum()
            logger.info(f"Explained variance: {explained_var:.4f}")
        
        self._log_time(start_time, "LSA feature creation")
        
        return df_with_lsa, lsa_features
    
    def create_combined_features(self, df: pd.DataFrame) -> pd.DataFrame:

        start_time = time.time()
        
        df_combined = df.copy()
        
        # Create price-to-duration ratio (value for money)
        if 'price' in df.columns and 'content_duration' in df.columns:
            df_combined['price_per_hour'] = df['price'] / df['content_duration'].replace(0, 0.1)
            self.derived_features.append('price_per_hour')
        
        # Create popularity score combining subscribers and reviews if available
        if 'num_subscribers' in df.columns and 'num_reviews' in df.columns:
            # Normalize both metrics to 0-1 range
            max_subscribers = df['num_subscribers'].max() or 1
            max_reviews = df['num_reviews'].max() or 1
            
            normalized_subscribers = df['num_subscribers'] / max_subscribers
            normalized_reviews = df['num_reviews'] / max_reviews
            
            # Combine with more weight on reviews (which are harder to get)
            df_combined['popularity_score'] = (0.6 * normalized_subscribers + 0.4 * normalized_reviews)
            self.derived_features.append('popularity_score')
        
        # Create a complexity score based on level and duration
        if 'level_encoded' in df.columns and 'content_duration' in df.columns:
            # Normalize duration
            max_duration = df['content_duration'].max() or 1
            normalized_duration = df['content_duration'] / max_duration
            
            # Normalize level (assuming higher value means more advanced)
            max_level = df['level_encoded'].max() or 1
            normalized_level = df['level_encoded'] / max_level
            
            # Combine for complexity score
            df_combined['complexity_score'] = (0.7 * normalized_level + 0.3 * normalized_duration)
            self.derived_features.append('complexity_score')
        
        self._log_time(start_time, "Combined feature creation")
        
        return df_combined
    
    def create_text_vectors(self, text: str, model_type: str = 'tfidf') -> np.ndarray:

        # Preprocess text
        text = self.preprocess_text(text)
        logger.info(f"Preprocessed text for {model_type}: {text}")
        
        try:
            if model_type == 'tfidf':
                if self.tfidf_vectorizer is None:
                    logger.warning("TF-IDF vectorizer not initialized")
                    return np.zeros(100)  # Default size
                
                # Transform text using TF-IDF
                vector = self.tfidf_vectorizer.transform([text]).toarray()[0]
                logger.info(f"Created TF-IDF vector with shape: {vector.shape}")
                logger.info(f"Vector sum: {np.sum(vector)}")
                logger.info(f"Non-zero elements: {np.count_nonzero(vector)}")
                
                if np.all(vector == 0):
                    logger.warning("TF-IDF vector is all zeros")
                
                return vector
            
            elif model_type == 'word2vec':
                if self.word2vec_model is None:
                    logger.warning("Word2Vec model not initialized")
                    return np.zeros(100)  # Default size
                
                # Split text into words and get vectors
                words = text.split()
                vectors = []
                for word in words:
                    if word in self.word2vec_model.wv:
                        vectors.append(self.word2vec_model.wv[word])
                
                # Average word vectors
                if vectors:
                    vector = np.mean(vectors, axis=0)
                else:
                    logger.warning("No words found in Word2Vec vocabulary")
                    vector = np.zeros(self.word2vec_model.vector_size)
                
                logger.info(f"Created Word2Vec vector with shape: {vector.shape}")
                return vector
            
            elif model_type == 'fasttext':
                if self.fasttext_model is None:
                    logger.warning("FastText model not initialized")
                    return np.zeros(100)  # Default size
                
                # Split text into words and get vectors
                words = text.split()
                vectors = []
                for word in words:
                    if word in self.fasttext_model.wv:
                        vectors.append(self.fasttext_model.wv[word])
                
                # Average word vectors
                if vectors:
                    vector = np.mean(vectors, axis=0)
                else:
                    logger.warning("No words found in FastText vocabulary")
                    vector = np.zeros(self.fasttext_model.vector_size)
                
                logger.info(f"Created FastText vector with shape: {vector.shape}")
                return vector
            
            elif model_type == 'lsa':
                if self.count_vectorizer is None or self.lsa_model is None:
                    logger.warning("LSA model components not initialized")
                    return np.zeros(50)  # Default size
                
                # Transform text using Count Vectorizer
                    count_vector = self.count_vectorizer.transform([text])
                vector = self.lsa_model.transform(count_vector)[0]
                
                logger.info(f"Created LSA vector with shape: {vector.shape}")
                return vector
            
            else:
                logger.warning(f"Unsupported model type: {model_type}")
                return np.zeros(100)  # Default size
        
        except Exception as e:
            logger.error(f"Error creating {model_type} vector: {e}")
            return np.zeros(100)  # Default size
    
    def create_weighted_word_embedding(self, text: str) -> np.ndarray:
    
        if self.tfidf_vectorizer is None or self.word2vec_model is None:
            logger.warning("TF-IDF vectorizer or Word2Vec model not initialized")
            return np.array([])
        
        # Get TF-IDF weights for words
        tfidf_vector = self.create_text_vectors(text, 'tfidf')
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_dict = {feature_names[i]: tfidf_vector[i] for i in range(len(feature_names))}
        
        # Calculate weighted Word2Vec vector
        words = text.lower().split()
        weighted_vectors = []
        total_weight = 0
        
        for word in words:
            if word in self.word2vec_model.wv:
                # Use TF-IDF weight if available, otherwise use small default weight
                weight = tfidf_dict.get(word, 0.1)
                weighted_vectors.append(self.word2vec_model.wv[word] * weight)
                total_weight += weight
        
        if weighted_vectors and total_weight > 0:
            return np.sum(weighted_vectors, axis=0) / total_weight
        else:
            return np.zeros(self.word2vec_model.vector_size)
    
    def process_all_features(self, df: pd.DataFrame, text_col: str = 'combined_text') -> pd.DataFrame:
        
        try:
            # Encode categorical features
            df_processed = self.encode_categorical_features(df)
            
            try:
                # Scale numerical features
                df_processed = self.scale_numerical_features(df_processed)
            except IndexError as e:
                import logging
                logging.warning(f"Error scaling numerical features: {e}")
                # Continue without scaling if there's an issue
            
            # Create TF-IDF features
            df_processed, _ = self.create_tfidf_features(df_processed, text_col)
            
            # Train Word2Vec model
            df_processed, _ = self.train_word2vec(df_processed, text_col)
            
            # Train FastText model
            df_processed, _ = self.train_fasttext(df_processed, text_col)
            
            # Create LSA features
            df_processed, _ = self.create_lsa_features(df_processed, text_col)
            
            # Create combined features
            df_processed = self.create_combined_features(df_processed)
            
            return df_processed
        except Exception as e:
            import logging
            logging.error(f"Error processing features: {e}")
            # Return original DataFrame if processing fails
            return df
    
    def get_feature_info(self) -> Dict[str, List[str]]:
      
        return {
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'text_features': self.text_features,
            'derived_features': self.derived_features
        } 
    
    def preprocess_text(self, text: str) -> str:
 
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stop words
        if hasattr(self, 'stop_words'):
            words = text.split()
            words = [w for w in words if w not in self.stop_words]
            text = ' '.join(words)
            
        return text 