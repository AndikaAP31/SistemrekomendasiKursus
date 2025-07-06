import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import joblib
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
import scipy.sparse

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BaseRecommender:
    """Base class for all recommender models"""
    
    def __init__(self, name: str, verbose: bool = True):
        """
        Initialize base recommender
        
        Args:
            name: Name of the recommender
            verbose: Whether to print additional information
        """
        self.name = name
        self.verbose = verbose
        self.is_fitted = False
        self.data = None
        
    def _log_time(self, start_time: float, operation: str):
        """
        Log time taken for an operation
        
        Args:
            start_time: Start time
            operation: Name of the operation
        """
        if self.verbose:
            elapsed = time.time() - start_time
            logger.info(f"[{self.name}] {operation} completed in {elapsed:.2f} seconds")
    
    def fit(self, data: pd.DataFrame, **kwargs):
        """
        Fit the recommender model
        
        Args:
            data: Input DataFrame
            **kwargs: Additional keyword arguments
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def recommend(self, query: Any, top_n: int = 5, **kwargs) -> pd.DataFrame:
        """
        Generate recommendations
        
        Args:
            query: Query input
            top_n: Number of recommendations to return
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommendations
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, path: Union[str, Path]):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
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
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
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
        """
        Fit the TF-IDF recommender
        
        Args:
            data: Input DataFrame
            tfidf_matrix: TF-IDF matrix
            feature_names: TF-IDF feature names
            **kwargs: Additional keyword arguments
        """
        start_time = time.time()
        
        self.data = data
        self.tfidf_matrix = tfidf_matrix
        self.feature_names = feature_names
        self.is_fitted = True
        
        self._log_time(start_time, "Model fitting")
        return self
    
    def recommend(self, query_vector: np.ndarray, top_n: int = 5, 
                 filter_dict: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        """
        Generate recommendations based on TF-IDF similarity
        
        Args:
            query_vector: Query vector in TF-IDF space
            top_n: Number of recommendations to return
            filter_dict: Dictionary of filters to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommendations
        """
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
        """
        Filter data based on criteria
        
        Args:
            filter_dict: Dictionary of filters to apply
            
        Returns:
            List of indices matching the filters
        """
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


class Word2VecRecommender(BaseRecommender):
    """Recommender based on Word2Vec embeddings"""
    
    def __init__(self, verbose: bool = True):
        """Initialize Word2Vec recommender"""
        super().__init__(name="Word2Vec Recommender", verbose=verbose)
        self.course_vectors = None
    
    def fit(self, data: pd.DataFrame, course_vectors: np.ndarray, **kwargs):
        """
        Fit the Word2Vec recommender
        
        Args:
            data: Input DataFrame
            course_vectors: Course vectors from Word2Vec
            **kwargs: Additional keyword arguments
        """
        start_time = time.time()
        
        self.data = data
        self.course_vectors = course_vectors
        self.is_fitted = True
        
        self._log_time(start_time, "Model fitting")
        return self
    
    def recommend(self, query_vector: np.ndarray, top_n: int = 5, 
                 filter_dict: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        """
        Recommend courses based on Word2Vec similarity
        
        Args:
            query_vector: Query vector
            top_n: Number of recommendations to return
            filter_dict: Dictionary of filters to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with recommendations
        """
        start_time = time.time()
        
        # Check if model is fitted
        if not self.is_fitted:
            logger.error("Model not fitted yet")
            return pd.DataFrame()
        
        # Apply filters if any
        data_subset = self.data.copy()
        if filter_dict:
            for key, value in filter_dict.items():
                if key in data_subset.columns:
                    if key == 'max_price':
                        data_subset = data_subset[data_subset['price'] <= value]
                    else:
                        data_subset = data_subset[data_subset[key] == value]
        
        # Get indices of filtered data
        filtered_indices = data_subset.index.tolist()
        
        # Check if we have any courses left after filtering
        if not filtered_indices:
            logger.warning("No courses left after filtering")
            return pd.DataFrame()
        
        # Get course vectors for filtered courses
        filtered_vectors = self.course_vectors[filtered_indices]
        
        # Check query vector
        if query_vector is None or query_vector.size == 0:
            logger.warning("Query vector is empty")
            return data_subset.head(top_n)
        
        # Normalize query vector for better similarity calculation
        query_vector_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # Calculate similarity
        try:
            similarities = cosine_similarity(query_vector_norm.reshape(1, -1), filtered_vectors)[0]
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return data_subset.head(top_n)
        
        # Get top N indices
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Get recommendations
        recommendations = data_subset.iloc[top_indices].copy()
        
        # Add similarity scores
        recommendations['similarity_score'] = similarities[top_indices]
        recommendations['w2v_score'] = similarities[top_indices]
        
        self._log_time(start_time, "Recommendation generation")
        
        return recommendations


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
        """
        Fit the recommender with feature matrices and weights
        
        Args:
            data: DataFrame with course data
            feature_matrices: Dictionary of feature matrices
            feature_weights: Dictionary of feature weights
            feature_names: Optional list of feature names
        """
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
        """
        Get recommendations based on query vectors
        
        Args:
            query_vectors: Dictionary of query vectors for each feature
            top_n: Number of recommendations to return
            filter_dict: Dictionary of filters to apply
            
        Returns:
            DataFrame with recommendations
        """
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
        """
        Filter data based on criteria
        
        Args:
            filter_dict: Dictionary of filters to apply
            
        Returns:
            List of indices matching the filters
        """
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


def evaluate_models(models, test_data, test_queries, actual_selections, metrics=['precision', 'recall', 'ndcg'], top_n=5):
    """
    Evaluate multiple recommendation models
        
        Args:
        models: Dictionary of recommendation models
        test_data: Test data
        test_queries: Test queries
        actual_selections: Actual selections for each query
        metrics: List of evaluation metrics
        top_n: Number of recommendations to consider
            
        Returns:
        Dictionary with evaluation results for each model
    """
    results = {}
    
    for model_name, model in models.items():
        if not model.is_fitted:
            logger.warning(f"Model {model_name} not fitted, skipping evaluation")
        #continue
                
        model_results = {}
        
        for metric in metrics:
            if metric == 'precision':
                precision_values = []
                for query, actual in zip(test_queries, actual_selections):
                    recommendations = model.recommend(query, top_n=top_n)
                    if recommendations.empty:
                        precision_values.append(0.0)
                        continue
                    
                    recommended_indices = recommendations.index.tolist()
                    hits = len(set(recommended_indices) & set(actual))
                    precision = hits / min(len(recommended_indices), top_n)
                    precision_values.append(precision)
                
                model_results['precision'] = np.mean(precision_values)
            
            elif metric == 'recall':
                recall_values = []
                for query, actual in zip(test_queries, actual_selections):
                    if not actual:
                        recall_values.append(0.0)
                        continue
                        
                    recommendations = model.recommend(query, top_n=top_n)
                    if recommendations.empty:
                        recall_values.append(0.0)
                        continue
                    
                    recommended_indices = recommendations.index.tolist()
                    hits = len(set(recommended_indices) & set(actual))
                    recall = hits / len(actual)
                    recall_values.append(recall)
                
                model_results['recall'] = np.mean(recall_values)
            
            elif metric == 'ndcg':
                ndcg_values = []
                for query, actual in zip(test_queries, actual_selections):
                    if not actual:
                        ndcg_values.append(0.0)
                        continue
                        
                    recommendations = model.recommend(query, top_n=top_n)
                    if recommendations.empty:
                        ndcg_values.append(0.0)
                        continue
                    
                    recommended_indices = recommendations.index.tolist()
                    
                    # Calculate DCG
                    dcg = 0.0
                    for i, idx in enumerate(recommended_indices):
                        if idx in actual:
                            # Use 1/log2(i+2) as the relevance weight (i+2 because i is 0-indexed)
                            dcg += 1.0 / np.log2(i + 2)
                    
                    # Calculate ideal DCG (IDCG)
                    idcg = 0.0
                    for i in range(min(len(actual), top_n)):
                        idcg += 1.0 / np.log2(i + 2)
                    
                    # Calculate NDCG
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    ndcg_values.append(ndcg)
                
                model_results['ndcg'] = np.mean(ndcg_values)
                
            elif metric == 'map':
                map_values = []
                for query, actual in zip(test_queries, actual_selections):
                    if not actual:
                        map_values.append(0.0)
                        continue
                        
                    recommendations = model.recommend(query, top_n=top_n)
                    if recommendations.empty:
                        map_values.append(0.0)
                        continue
                    
                    recommended_indices = recommendations.index.tolist()
                    
                    # Calculate average precision
                    ap = 0.0
                    hits = 0
                    
                    for i, idx in enumerate(recommended_indices):
                        if idx in actual:
                            hits += 1
                            ap += hits / (i + 1)
                    
                    if hits == 0:
                        map_values.append(0.0)
                    else:
                        map_values.append(ap / min(len(actual), top_n))
                
                model_results['map'] = np.mean(map_values)
        
        results[model_name] = model_results
        
        return results 

def plot_evaluation_results(results, metric='ndcg', figsize=(10, 6)):
    """
    Plot evaluation results
    
    Args:
        results: Dictionary with evaluation results
        metric: Metric to plot
        figsize: Figure size
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    models = list(results.keys())
    scores = [results[model].get(metric, 0.0) for model in models]
    
    sns.barplot(x=models, y=scores)
    plt.title(f'{metric.upper()} Comparison')
    plt.ylabel(f'{metric.upper()} Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{metric}_comparison.png')
    plt.close() 