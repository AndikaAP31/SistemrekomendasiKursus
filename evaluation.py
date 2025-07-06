import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from sklearn.model_selection import KFold, train_test_split
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from pathlib import Path
from models import BaseRecommender

# Configure logging
logger = logging.getLogger(__name__)

class RecommenderEvaluator:
    """Class for evaluating recommendation models"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize evaluator
        
        Args:
            verbose: Whether to print additional information
        """
        self.verbose = verbose
        self.results = {}
        self.metrics = []
        
    def precision_at_k(self, actual: List[int], predicted: List[int], k: int = 5) -> float:
        """
        Calculate precision at k
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of items to consider
            
        Returns:
            Precision at k
        """
        if len(predicted) == 0 or k <= 0:
            return 0.0
            
        k = min(k, len(predicted))
        predicted_k = predicted[:k]
        
        # Calculate number of relevant items in predictions
        num_relevant = len(set(actual) & set(predicted_k))
        
        return num_relevant / k
    
    def recall_at_k(self, actual: List[int], predicted: List[int], k: int = 5) -> float:
        """
        Calculate recall at k
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of items to consider
            
        Returns:
            Recall at k
        """
        if len(actual) == 0 or len(predicted) == 0 or k <= 0:
            return 0.0
            
        k = min(k, len(predicted))
        predicted_k = predicted[:k]
        
        # Calculate number of relevant items in predictions
        num_relevant = len(set(actual) & set(predicted_k))
        
        return num_relevant / len(actual)
    
    def f1_at_k(self, actual: List[int], predicted: List[int], k: int = 5) -> float:
        """
        Calculate F1 score at k
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of items to consider
            
        Returns:
            F1 score at k
        """
        precision = self.precision_at_k(actual, predicted, k)
        recall = self.recall_at_k(actual, predicted, k)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, actual: List[int], predicted: List[int], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of items to consider
            
        Returns:
            NDCG at k
        """
        if len(actual) == 0 or len(predicted) == 0 or k <= 0:
            return 0.0
            
        k = min(k, len(predicted))
        predicted_k = predicted[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(predicted_k):
            if item in actual:
                # Use binary relevance: 1 if in actual, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG
        idcg = 0.0
        for i in range(min(len(actual), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        # Normalize
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    def map_at_k(self, actual: List[int], predicted: List[int], k: int = 5) -> float:
        """
        Calculate Mean Average Precision at k
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            k: Number of items to consider
            
        Returns:
            MAP at k
        """
        if len(actual) == 0 or len(predicted) == 0 or k <= 0:
            return 0.0
            
        k = min(k, len(predicted))
        predicted_k = predicted[:k]
        
        # Calculate average precision
        ap = 0.0
        hits = 0
        
        for i, item in enumerate(predicted_k):
            if item in actual:
                hits += 1
                ap += hits / (i + 1)
        
        if hits == 0:
            return 0.0
            
        return ap / min(len(actual), k)
    
    def mrr(self, actual: List[int], predicted: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            actual: List of actual relevant items
            predicted: List of predicted items
            
        Returns:
            MRR value
        """
        if len(actual) == 0 or len(predicted) == 0:
            return 0.0
            
        # Find the rank of the first relevant item
        for i, item in enumerate(predicted):
            if item in actual:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def diversity(self, predictions: List[List[int]], feature_matrix: np.ndarray) -> float:
        """
        Calculate diversity of recommendations
        
        Args:
            predictions: List of lists of predicted items
            feature_matrix: Matrix of item features
            
        Returns:
            Diversity score
        """
        if len(predictions) == 0 or feature_matrix.shape[0] == 0:
            return 0.0
            
        # Calculate average pairwise distance between items in each recommendation list
        diversity_scores = []
        
        for pred_list in predictions:
            if len(pred_list) <= 1:
                continue
                
            # Get feature vectors for predicted items
            pred_features = feature_matrix[pred_list]
            
            # Calculate pairwise distances
            n = len(pred_list)
            total_distance = 0.0
            count = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    # Use Euclidean distance
                    dist = np.linalg.norm(pred_features[i] - pred_features[j])
                    total_distance += dist
                    count += 1
            
            if count > 0:
                avg_distance = total_distance / count
                diversity_scores.append(avg_distance)
        
        if len(diversity_scores) == 0:
            return 0.0
            
        return np.mean(diversity_scores)
    
    def coverage(self, predictions: List[List[int]], total_items: int) -> float:
        """
        Calculate catalog coverage
        
        Args:
            predictions: List of lists of predicted items
            total_items: Total number of items in the catalog
            
        Returns:
            Coverage percentage
        """
        if len(predictions) == 0 or total_items == 0:
            return 0.0
            
        # Get unique items recommended across all users
        unique_items = set()
        for pred_list in predictions:
            unique_items.update(pred_list)
        
        return len(unique_items) / total_items
    
    def evaluate_model(self, model: BaseRecommender, test_data: pd.DataFrame, 
                      query_fn: Callable, actual_col: str, k_values: List[int] = [5, 10, 20],
                      filter_dict: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a recommendation model
        
        Args:
            model: Recommender model to evaluate
            test_data: Test data
            query_fn: Function to generate query from test data
            actual_col: Column in test_data containing actual selections
            k_values: List of k values for evaluation
            filter_dict: Optional dictionary of filters
            
        Returns:
            Dictionary with evaluation results
        """
        if not model.is_fitted:
            logger.warning(f"Model {model.name} is not fitted")
            return {}
        
        start_time = time.time()
        results = defaultdict(dict)
        
        # Store metrics
        metrics = ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr']
        self.metrics = metrics
        
        # Initialize result storage
        for k in k_values:
            for metric in metrics:
                results[f"{metric}@{k}"] = []
        
        results['mrr'] = []
        
        # Collect all predictions for diversity and coverage
        all_predictions = []
        all_actual = []
        
        # Evaluate on each test instance
        for _, row in test_data.iterrows():
            # Get actual selections
            if isinstance(row[actual_col], list):
                actual = row[actual_col]
            elif isinstance(row[actual_col], str):
                try:
                    # Try to parse as JSON
                    actual = json.loads(row[actual_col])
                except:
                    # Fall back to comma-separated list
                    actual = [int(x.strip()) for x in row[actual_col].split(',')]
            else:
                actual = [row[actual_col]]
            
            # Generate query
            query = query_fn(row)
            
            # Get recommendations
            recommendations = model.recommend(query, top_n=max(k_values), filter_dict=filter_dict)
            if recommendations.empty:
                continue
            
            predicted = recommendations.index.tolist()
            
            # Store for diversity and coverage calculation
            all_predictions.append(predicted)
            all_actual.append(actual)
            
            # Calculate metrics for each k
            for k in k_values:
                results[f"precision@{k}"].append(self.precision_at_k(actual, predicted, k))
                results[f"recall@{k}"].append(self.recall_at_k(actual, predicted, k))
                results[f"f1@{k}"].append(self.f1_at_k(actual, predicted, k))
                results[f"ndcg@{k}"].append(self.ndcg_at_k(actual, predicted, k))
                results[f"map@{k}"].append(self.map_at_k(actual, predicted, k))
            
            # Calculate MRR
            results['mrr'].append(self.mrr(actual, predicted))
        
        # Calculate mean for each metric
        final_results = {}
        for metric, values in results.items():
            if values:
                final_results[metric] = np.mean(values)
            else:
                final_results[metric] = 0.0
        
        # Add diversity if we have feature matrix
        if hasattr(model, 'feature_matrices') and model.feature_matrices:
            # Use the first available feature matrix
            feature_matrix = next(iter(model.feature_matrices.values()))
            final_results['diversity'] = self.diversity(all_predictions, feature_matrix)
        
        # Add coverage
        if hasattr(model, 'data') and model.data is not None:
            final_results['coverage'] = self.coverage(all_predictions, len(model.data))
        
        # Store the results
        self.results[model.name] = final_results
        
        if self.verbose:
            elapsed = time.time() - start_time
            logger.info(f"Evaluation of {model.name} completed in {elapsed:.2f} seconds")
            logger.info(f"Results: {final_results}")
        
        return final_results
    
    def cross_validate(self, model_factory: Callable[[], BaseRecommender], data: pd.DataFrame, 
                      query_fn: Callable, actual_col: str, n_splits: int = 5, 
                      k_values: List[int] = [5, 10, 20], random_state: int = 42) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            model_factory: Function to create a new model instance
            data: Data to use for cross-validation
            query_fn: Function to generate query from test data
            actual_col: Column in data containing actual selections
            n_splits: Number of cross-validation folds
            k_values: List of k values for evaluation
            random_state: Random seed
            
        Returns:
            Dictionary with cross-validation results
        """
        start_time = time.time()
        
        # Initialize K-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Storage for results
        cv_results = defaultdict(list)
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            if self.verbose:
                logger.info(f"Starting fold {fold+1}/{n_splits}")
            
            # Split data
            train_data = data.iloc[train_idx].reset_index(drop=True)
            test_data = data.iloc[test_idx].reset_index(drop=True)
            
            # Create and fit model
            model = model_factory()
            model.fit(train_data)
            
            # Evaluate model
            fold_results = self.evaluate_model(
                model, test_data, query_fn, actual_col, k_values
            )
            
            # Store results
            for metric, value in fold_results.items():
                cv_results[metric].append(value)
            
            if self.verbose:
                logger.info(f"Fold {fold+1} results: {fold_results}")
        
        # Calculate mean and std for each metric
        final_results = {}
        for metric, values in cv_results.items():
            final_results[f"{metric}_mean"] = np.mean(values)
            final_results[f"{metric}_std"] = np.std(values)
        
        if self.verbose:
            elapsed = time.time() - start_time
            logger.info(f"Cross-validation completed in {elapsed:.2f} seconds")
            logger.info(f"Final results: {final_results}")
        
        return final_results
    
    def compare_models(self, models: Dict[str, Tuple[BaseRecommender, Callable]], 
                      test_data: pd.DataFrame, actual_col: str, 
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple recommendation models
        
        Args:
            models: Dictionary mapping model names to (model, query_fn) tuples
            test_data: Test data
            actual_col: Column in test_data containing actual selections
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary with comparison results
        """
        start_time = time.time()
        
        comparison_results = {}
        
        for name, (model, query_fn) in models.items():
            if self.verbose:
                logger.info(f"Evaluating model: {name}")
            
            # Evaluate model
            results = self.evaluate_model(
                model, test_data, query_fn, actual_col, k_values
            )
            
            comparison_results[name] = results
        
        if self.verbose:
            elapsed = time.time() - start_time
            logger.info(f"Model comparison completed in {elapsed:.2f} seconds")
        
        return comparison_results
    
    def plot_comparison(self, metric: str = 'ndcg@10', figsize: Tuple[int, int] = (10, 6)):
        """
        Plot comparison of models for a specific metric
        
        Args:
            metric: Metric to plot
            figsize: Figure size
        """
        if not self.results:
            logger.warning("No results to plot")
            return
        
        plt.figure(figsize=figsize)
        
        # Extract values for the metric
        model_names = list(self.results.keys())
        metric_values = [results.get(metric, 0) for results in self.results.values()]
        
        # Create bar plot
        sns.barplot(x=model_names, y=metric_values)
        plt.title(f"Comparison of Models: {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_metrics(self, model_name: str, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot all metrics for a specific model
        
        Args:
            model_name: Name of the model
            figsize: Figure size
        """
        if model_name not in self.results:
            logger.warning(f"No results for model: {model_name}")
            return
        
        results = self.results[model_name]
        
        plt.figure(figsize=figsize)
        
        # Group metrics by type
        metrics_by_k = defaultdict(dict)
        standalone_metrics = {}
        
        for metric, value in results.items():
            if '@' in metric:
                base_metric, k = metric.split('@')
                metrics_by_k[base_metric][int(k)] = value
            else:
                standalone_metrics[metric] = value
        
        # Plot metrics by k
        n_metrics = len(metrics_by_k)
        if n_metrics > 0:
            fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
            
            if n_metrics == 1:
                axes = [axes]
            
            for i, (base_metric, values_by_k) in enumerate(metrics_by_k.items()):
                k_values = sorted(values_by_k.keys())
                metric_values = [values_by_k[k] for k in k_values]
                
                axes[i].plot(k_values, metric_values, 'o-')
                axes[i].set_title(f"{base_metric.upper()} by k")
                axes[i].set_xlabel("k")
                axes[i].set_ylabel(base_metric)
                axes[i].grid(True)
            
            plt.tight_layout()
        
        # Plot standalone metrics
        if standalone_metrics:
            plt.figure(figsize=(8, 4))
            sns.barplot(x=list(standalone_metrics.keys()), y=list(standalone_metrics.values()))
            plt.title(f"Other Metrics for {model_name}")
            plt.ylabel("Value")
            plt.tight_layout()
        
        return plt.gcf()
    
    def save_results(self, output_dir: Union[str, Path], prefix: str = ''):
        """
        Save evaluation results to files
        
        Args:
            output_dir: Directory to save results
            prefix: Prefix for filenames
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        results_file = output_dir / f"{prefix}evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        if self.verbose:
            logger.info(f"Results saved to {results_file}")
        
        # Save comparison plots
        for metric in ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map@5']:
            if any(metric in results for results in self.results.values()):
                fig = self.plot_comparison(metric)
                if fig:
                    plot_file = output_dir / f"{prefix}comparison_{metric.replace('@', '_')}.png"
                    fig.savefig(plot_file)
                    plt.close(fig)
        
        # Save individual model plots
        for model_name in self.results:
            fig = self.plot_metrics(model_name)
            if fig:
                plot_file = output_dir / f"{prefix}metrics_{model_name.replace(' ', '_')}.png"
                fig.savefig(plot_file)
                plt.close(fig) 