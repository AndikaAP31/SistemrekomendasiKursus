import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from pathlib import Path
import pickle
import scipy.sparse

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BaseRecommender:
    """Base class for all recommender models"""
    
    def __init__(self, name: str, verbose: bool = True):

        self.name = name
        self.verbose = verbose
        self.is_fitted = False
        self.data = None
        
    def _log_time(self, start_time: float, operation: str):
        if self.verbose:
            elapsed = time.time() - start_time
            logger.info(f"[{self.name}] {operation} completed in {elapsed:.2f} seconds")
    
    def fit(self, data: pd.DataFrame, **kwargs):

        raise NotImplementedError("Subclasses must implement this method")
    
    def recommend(self, query: Any, top_n: int = 5, **kwargs) -> pd.DataFrame:

        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, path: Union[str, Path]):
 
        if not self.is_fitted:
            logger.warning(f"[{self.name}] Model not fitted, nothing to save")
            return
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"[{self.name}] Model saved to {path}")
        except Exception as e:
            logger.error(f"[{self.name}] Error saving model: {e}")
    
    @classmethod
    def load(cls, path: Union[str, Path]):

        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None


class TFIDFRecommender(BaseRecommender):
    """Recommender based on TF-IDF vectors"""
    
    def __init__(self, verbose: bool = True):
        """Initialize TF-IDF recommender"""
        super().__init__(name="TF-IDF Recommender", verbose=verbose)
        self.tfidf_matrix = None
        self.feature_names = None
    
    def fit(self, data: pd.DataFrame, tfidf_matrix: np.ndarray, feature_names: List[str], **kwargs):
 
        start_time = time.time()
        
        self.data = data
        self.tfidf_matrix = tfidf_matrix
        self.feature_names = feature_names
        self.is_fitted = True
        
        self._log_time(start_time, "Model fitting")
        return self
    
    def recommend(self, query_vector: np.ndarray, top_n: int = 5, 
                 filter_dict: Optional[Dict] = None, **kwargs) -> pd.DataFrame:

        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return pd.DataFrame()
        
        start_time = time.time()
        
        # Check if query vector is valid
        if query_vector is None or query_vector.size == 0:
            logger.warning("Query vector is empty or None. Cannot generate recommendations.")
            return pd.DataFrame()
        
        # Ensure query vector has the same dimension as the TF-IDF matrix
        if query_vector.shape[0] != self.tfidf_matrix.shape[1]:
            logger.warning(f"Query vector dimension mismatch: got {query_vector.shape[0]}, expected {self.tfidf_matrix.shape[1]}")
            # Try to pad or truncate to make it compatible
            if query_vector.shape[0] < self.tfidf_matrix.shape[1]:
                # Pad with zeros
                padded_vector = np.zeros(self.tfidf_matrix.shape[1])
                padded_vector[:query_vector.shape[0]] = query_vector
                query_vector = padded_vector
            else:
                # Truncate
                query_vector = query_vector[:self.tfidf_matrix.shape[1]]
        
        # Apply pre-filtering if requested
        if filter_dict and isinstance(filter_dict, dict):
            filtered_indices = self._filter_data(filter_dict)
            filtered_matrix = self.tfidf_matrix[filtered_indices]
            data_subset = self.data.iloc[filtered_indices].copy()
        else:
            filtered_matrix = self.tfidf_matrix
            data_subset = self.data.copy()
        
        if len(data_subset) == 0:
            logger.warning("No courses match the filtering criteria")
            return pd.DataFrame()
        
        try:
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector.reshape(1, -1), filtered_matrix)[0]
            
            # Add similarity scores
            data_subset['similarity_score'] = similarities
            
            # Sort by similarity and return top N
            recommendations = data_subset.sort_values('similarity_score', ascending=False).head(top_n)
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            # Return some default recommendations if similarity calculation fails
            recommendations = data_subset.sample(min(top_n, len(data_subset))).copy()
            recommendations['similarity_score'] = 0.0
        
        self._log_time(start_time, "Recommendation generation")
        
        return recommendations
    
    def _filter_data(self, filter_dict: Dict) -> List[int]:

        df = self.data
        mask = pd.Series([True] * len(df), index=df.index)
        
        for key, value in filter_dict.items():
            if key in df.columns:
                if isinstance(value, list):
                    mask &= df[key].isin(value)
                else:
                    mask &= (df[key] == value)
            elif key == 'max_price' and 'price' in df.columns:
                mask &= (df['price'] <= value)
            elif key == 'min_rating' and 'rating' in df.columns:
                mask &= (df['rating'] >= value)
        
        return df[mask].index.tolist()



class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender that combines multiple feature matrices"""
    
    def __init__(self, verbose: bool = True):
        """Initialize content-based recommender"""
        super().__init__(name="Content-Based Recommender", verbose=verbose)
        self.feature_matrices = {}
        self.feature_weights = {}
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame, feature_matrices: Dict[str, np.ndarray], 
           feature_weights: Dict[str, float], feature_names: Optional[List[str]] = None, **kwargs):
        
        self.data = data
        self.feature_matrices = feature_matrices
        self.feature_weights = feature_weights
        self.feature_names = feature_names
        self.is_fitted = True
        
        if self.verbose:
            logger.info(f"ContentBasedRecommender fitted with {len(feature_matrices)} features")
            for feature, matrix in feature_matrices.items():
                logger.info(f"{feature} matrix shape: {matrix.shape}")

    def recommend(self, query_vectors: Dict[str, np.ndarray], top_n: int = 5, 
                 filter_dict: Optional[Dict] = None, **kwargs) -> pd.DataFrame:

        start_time = time.time()
        
        # Filter data if needed
        if filter_dict:
            valid_indices = self._filter_data(filter_dict)
            data_subset = self.data.iloc[valid_indices].copy()
            feature_matrices_subset = {
                feature: matrix[valid_indices] 
                for feature, matrix in self.feature_matrices.items()
            }
        else:
            data_subset = self.data.copy()
            feature_matrices_subset = self.feature_matrices
        
        # Initialize similarity arrays
        combined_similarity = np.zeros(len(data_subset))
        tfidf_contribution = np.zeros(len(data_subset))
        word2vec_contribution = np.zeros(len(data_subset))
        
        # Calculate similarity for each feature
        for feature_name, query_vector in query_vectors.items():
            try:
                if feature_name not in feature_matrices_subset:
                    logger.warning(f"Feature matrix for {feature_name} not found")
                    continue
                
                feature_matrix = feature_matrices_subset[feature_name]
                
                # Ensure query vector has correct shape
                if len(query_vector.shape) == 1:
                    query_vector = query_vector.reshape(1, -1)
                
                # Calculate cosine similarity using numpy operations
                # Convert sparse matrix to dense if needed
                if scipy.sparse.issparse(feature_matrix):
                    feature_matrix = feature_matrix.toarray()
                if scipy.sparse.issparse(query_vector):
                    query_vector = query_vector.toarray()
                
                # Normalize vectors
                query_norm = np.linalg.norm(query_vector)
                matrix_norm = np.linalg.norm(feature_matrix, axis=1)
                
                # Avoid division by zero
                query_norm = np.maximum(query_norm, 1e-10)
                matrix_norm = np.maximum(matrix_norm, 1e-10)
                
                # Calculate normalized dot product
                similarity = np.dot(feature_matrix, query_vector.T).ravel() / (matrix_norm * query_norm)
                
                # Store individual contributions
                if feature_name == 'tfidf':
                    tfidf_contribution = similarity
                elif feature_name == 'word2vec':
                    word2vec_contribution = similarity
                
                # Apply weight to similarity for combined score
                weight = self.feature_weights.get(feature_name, 1.0 / len(query_vectors))
                combined_similarity += similarity * weight
                
            except Exception as e:
                logger.error(f"Error calculating similarity for {feature_name}: {e}")
        
        # Get top N indices
        top_indices = np.argsort(combined_similarity)[::-1][:top_n]
        
        # Get recommendations
        recommendations = data_subset.iloc[top_indices].copy()
        
        # Add raw similarity scores (unweighted)
        recommendations['similarity_score'] = combined_similarity[top_indices]
        recommendations['tfidf_score'] = tfidf_contribution[top_indices]
        recommendations['word2vec_score'] = word2vec_contribution[top_indices]
        
        # Get weights
        tfidf_weight = self.feature_weights.get('tfidf', 0.3)
        word2vec_weight = self.feature_weights.get('word2vec', 0.7)
        
        # Calculate weighted scores
        weighted_tfidf = recommendations['tfidf_score'] * tfidf_weight
        weighted_word2vec = recommendations['word2vec_score'] * word2vec_weight
        
        # Calculate total weighted score
        total_weighted = weighted_tfidf + weighted_word2vec
        
        # Set similarity_score to be the weighted sum
        recommendations['similarity_score'] = total_weighted
        
        # Calculate contribution percentages based on weighted scores
        recommendations['tfidf_contribution'] = (weighted_tfidf / total_weighted * 100).fillna(0)
        recommendations['word2vec_contribution'] = (weighted_word2vec / total_weighted * 100).fillna(0)
        
        # Replace NaN with 0
        recommendations = recommendations.fillna(0)
        
        # Log weights and contributions for verification
        logger.info(f"Weights used - TF-IDF: {tfidf_weight}, Word2Vec: {word2vec_weight}")
        logger.info(f"Average contributions - TF-IDF: {recommendations['tfidf_contribution'].mean():.2f}%, Word2Vec: {recommendations['word2vec_contribution'].mean():.2f}%")
        
        self._log_time(start_time, "Recommendation generation")
        
        return recommendations
    
    def _filter_data(self, filter_dict: Dict) -> List[int]:
        df = self.data
        mask = pd.Series([True] * len(df), index=df.index)
        
        for key, value in filter_dict.items():
            if key in df.columns:
                if isinstance(value, list):
                    mask &= df[key].isin(value)
                else:
                    mask &= (df[key] == value)
            elif key == 'max_price' and 'price' in df.columns:
                mask &= (df['price'] <= value)
            elif key == 'min_rating' and 'rating' in df.columns:
                mask &= (df['rating'] >= value)
        
        return df[mask].index.tolist()
