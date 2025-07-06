import os
import sys
import logging
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from models import (
    TFIDFRecommender, Word2VecRecommender, ContentBasedRecommender
)
from utils import clean_text, normalize_weights

def load_or_train_models(force_retrain=False):
    """
    Load pre-trained models or train new ones
    
    Args:
        force_retrain: Whether to force retraining even if models exist
        
    Returns:
        Dictionary with trained models
    """
    model_path = Path("models/recommenders.pkl")
    
    if model_path.exists() and not force_retrain:
        logger.info("Loading pre-trained models...")
        try:
            with open(model_path, 'rb') as f:
                models = pickle.load(f)
            logger.info("Models loaded successfully")
            return models
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.info("Training new models...")
    else:
        logger.info("Training new models...")
    
    # Load data
    logger.info("Loading and combining data...")
    
    # Define paths to data files
    udemy_path = "udemy_courses.csv"
    dicoding_path = "dicoding_courses.csv"
    coursera_path = "coursera_courses.csv"
    
    # Load data
    data_loader = DataLoader(verbose=True)
    df = data_loader.load_and_preprocess_all(
        udemy_path=udemy_path,
        dicoding_path=dicoding_path,
        coursera_path=coursera_path
    )
    
    logger.info(f"Combined dataset: {len(df)} courses")
    
    # Feature engineering
    feature_eng = FeatureEngineering(verbose=True)
    df_processed = feature_eng.process_all_features(df)
    
    # Create TF-IDF recommender
    start_time = time.time()
    tfidf_recommender = TFIDFRecommender(verbose=True)
    tfidf_recommender.fit(
        df_processed, 
        feature_eng.tfidf_matrix, 
        feature_eng.tfidf_vectorizer.get_feature_names_out()
    )
    logger.info(f"TF-IDF recommender trained in {time.time() - start_time:.2f} seconds")
    
    # Create Word2Vec recommender
    start_time = time.time()
    word2vec_vectors = np.array([
        feature_eng.create_text_vectors(text, 'word2vec') 
        for text in df_processed['combined_text']
    ])
    word2vec_recommender = Word2VecRecommender(verbose=True)
    word2vec_recommender.fit(df_processed, word2vec_vectors)
    logger.info(f"Word2Vec recommender trained in {time.time() - start_time:.2f} seconds")
    
    # Create content-based recommender
    start_time = time.time()
    feature_matrices = {
        'tfidf': feature_eng.tfidf_matrix.toarray(),
        'word2vec': word2vec_vectors
    }
    
    feature_weights = {
        'tfidf': 0.4,
        'word2vec': 0.6
    }
    
    # Get feature names for TF-IDF
    feature_names = feature_eng.tfidf_vectorizer.get_feature_names_out()
    
    content_recommender = ContentBasedRecommender(verbose=True)
    content_recommender.fit(
        df_processed, 
        feature_matrices, 
        feature_weights,
        feature_names=feature_names
    )
    logger.info(f"Content-based recommender trained in {time.time() - start_time:.2f} seconds")
    
    # Save models
    models = {
        'tfidf_recommender': tfidf_recommender,
        'word2vec_recommender': word2vec_recommender,
        'content_recommender': content_recommender,
        'tfidf_vectorizer': feature_eng.tfidf_vectorizer,
        'word2vec_model': feature_eng.word2vec_model,
        'data': df_processed,
        'feature_eng': feature_eng
    }
    
    try:
        os.makedirs('models', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)
        logger.info(f"Models saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
    
    return models

def get_recommendations(query_text, models, top_n=5, level=None, platform=None, is_paid=None, max_price=None):
    """
    Get recommendations based on query text
    
    Args:
        query_text: Query text
        models: Dictionary with trained models
        top_n: Number of recommendations to return
        level: Filter by level
        platform: Filter by platform
        is_paid: Filter by paid status
        max_price: Filter by maximum price
        
    Returns:
        Dictionary with recommendations from different models
    """
    # Create filter dictionary
    filter_dict = {}
    if level:
        filter_dict['level'] = level
    if platform:
        filter_dict['platform'] = platform
    if is_paid is not None:
        filter_dict['is_paid'] = is_paid
    if max_price is not None:
        filter_dict['max_price'] = max_price
    
    # Get models
    tfidf_recommender = models['tfidf_recommender']
    word2vec_recommender = models['word2vec_recommender']
    content_recommender = models['content_recommender']
    tfidf_vectorizer = models['tfidf_vectorizer']
    feature_eng = models['feature_eng']
    
    # Clean and process query text
    query_text = clean_text(query_text)
    
    # Create vectors for the query
    tfidf_vector = tfidf_vectorizer.transform([query_text]).toarray()[0]
    word2vec_vector = feature_eng.create_text_vectors(query_text, 'word2vec')
    
    # Create content-based vectors
    content_vectors = {
        'tfidf': tfidf_vector,
        'word2vec': word2vec_vector
    }
    
    # Get recommendations from each model
    tfidf_recs = tfidf_recommender.recommend(tfidf_vector, top_n=top_n, filter_dict=filter_dict)
    word2vec_recs = word2vec_recommender.recommend(word2vec_vector, top_n=top_n, filter_dict=filter_dict)
    content_recs = content_recommender.recommend(content_vectors, top_n=top_n, filter_dict=filter_dict)
    
    return {
        'tfidf': tfidf_recs,
        'word2vec': word2vec_recs,
        'content': content_recs
    }

def print_recommendations(recommendations, model_name):
    """
    Print recommendations from a model
    
    Args:
        recommendations: DataFrame with recommendations
        model_name: Name of the model
    """
    print(f"\n{'-'*20} {model_name} Recommendations {'-'*20}")
    
    if recommendations.empty:
        print("No recommendations found.")
        return
    
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {row['course_title']} ({row['platform']}, {row['level']})")
        print(f"   Price: ${row['price']:.2f}, Duration: {row['content_duration']:.1f} hours")
        print(f"   Similarity Score: {row['similarity_score']:.4f}")
        
        # Print individual feature similarities if available
        for col in recommendations.columns:
            if col.endswith('_similarity') and col != 'similarity_score':
                print(f"   {col}: {row[col]:.4f}")
        print()

def evaluate_model_accuracy(models, test_queries, top_n=10):
    """
    Evaluate model accuracy on test queries
    
    Args:
        models: Dictionary with trained models
        test_queries: List of test queries
        top_n: Number of recommendations to consider
        
    Returns:
        Dictionary with evaluation results
    """
    # Extract recommender models
    recommenders = {
        'TF-IDF': models['tfidf_recommender'],
        'Word2Vec': models['word2vec_recommender'],
        'Content-Based': models['content_recommender']
    }
    
    # Create simulated ground truth by finding courses that match keywords in the queries
    data = models['data']
    actual_selections = []
    
    for query in test_queries:
        keywords = query.lower().split()
        # Find courses that contain at least 2 of the keywords in their title or description
        matches = []
        for idx, row in data.iterrows():
            title = row['course_title'].lower()
            desc = row['description'].lower() if 'description' in row else ''
            combined = title + " " + desc
            if sum(1 for keyword in keywords if keyword in combined) >= 2:
                matches.append(idx)
        # Limit to 10 matches per query
        actual_selections.append(matches[:10])
    
    # Prepare query vectors for evaluation
    tfidf_vectorizer = models['tfidf_vectorizer']
    feature_eng = models['feature_eng']
    
    tfidf_queries = [tfidf_vectorizer.transform([clean_text(q)]).toarray()[0] for q in test_queries]
    w2v_queries = [feature_eng.create_text_vectors(clean_text(q), 'word2vec') for q in test_queries]
    content_queries = [{'tfidf': tfidf_vectorizer.transform([clean_text(q)]).toarray()[0],
                       'word2vec': feature_eng.create_text_vectors(clean_text(q), 'word2vec')} 
                      for q in test_queries]
    
    # Create evaluation queries dictionary
    eval_queries = {
        'TF-IDF': tfidf_queries,
        'Word2Vec': w2v_queries,
        'Content-Based': content_queries
    }
    
    # Evaluate each model
    results = {}
    for model_name, model in recommenders.items():
        print(f"\nEvaluating {model_name} model...")
        model_results = {}
        
        # Get queries for this model
        queries = eval_queries[model_name]
        
        # Calculate precision
        precision_values = []
        for query, actual in zip(queries, actual_selections):
            recommendations = model.recommend(query, top_n=top_n)
            if recommendations.empty:
                precision_values.append(0.0)
                continue
            
            recommended_indices = recommendations.index.tolist()
            hits = len(set(recommended_indices) & set(actual))
            precision = hits / len(recommended_indices) if recommended_indices else 0
            precision_values.append(precision)
        
        model_results['precision'] = np.mean(precision_values)
        print(f"Precision@{top_n}: {model_results['precision']:.4f}")
        
        # Calculate recall
        recall_values = []
        for query, actual in zip(queries, actual_selections):
            if not actual:
                recall_values.append(0.0)
                continue
                
            recommendations = model.recommend(query, top_n=top_n)
            if recommendations.empty:
                recall_values.append(0.0)
                continue
            
            recommended_indices = recommendations.index.tolist()
            hits = len(set(recommended_indices) & set(actual))
            recall = hits / len(actual) if actual else 0
            recall_values.append(recall)
        
        model_results['recall'] = np.mean(recall_values)
        print(f"Recall@{top_n}: {model_results['recall']:.4f}")
        
        # Calculate NDCG
        ndcg_values = []
        for query, actual in zip(queries, actual_selections):
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
        print(f"NDCG@{top_n}: {model_results['ndcg']:.4f}")
        
        results[model_name] = model_results
    
    return results

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test recommendation models')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining of models')
    args = parser.parse_args()
    
    # Load or train models
    models_dict = load_or_train_models(force_retrain=args.force_retrain)
    
    # Extract recommender models
    recommenders = {
        'TF-IDF': models_dict['tfidf_recommender'],
        'Word2Vec': models_dict['word2vec_recommender'],
        'Content-Based': models_dict['content_recommender']
    }
    
    # Test queries and simulated ground truth for evaluation
    test_queries = [
        "Machine learning and artificial intelligence",
        "Web development with JavaScript and React",
        "Data science and Python programming",
        "Business management and leadership",
        "Mobile app development for beginners"
    ]
    
    # For demonstration purposes, we'll create some simulated ground truth data
    # In a real scenario, this would come from user interactions or expert annotations
    data = models_dict['data']
    
    # Create simulated ground truth by finding courses that match keywords in the queries
    actual_selections = []
    for query in test_queries:
        keywords = query.lower().split()
        # Find courses that contain at least 2 of the keywords in their title or description
        matches = []
        for idx, row in data.iterrows():
            title = row['course_title'].lower()
            desc = row['description'].lower() if 'description' in row else ''
            combined = title + " " + desc
            if sum(1 for keyword in keywords if keyword in combined) >= 2:
                matches.append(idx)
        # Limit to 10 matches per query
        actual_selections.append(matches[:10])
    
    # Evaluate models
    print("\nEvaluating recommendation models...")
    
    # Prepare query vectors for evaluation
    tfidf_vectorizer = models_dict['tfidf_vectorizer']
    feature_eng = models_dict['feature_eng']
    
    tfidf_queries = [tfidf_vectorizer.transform([clean_text(q)]).toarray()[0] for q in test_queries]
    w2v_queries = [feature_eng.create_text_vectors(clean_text(q), 'word2vec') for q in test_queries]
    content_queries = [{'tfidf': tfidf_vectorizer.transform([clean_text(q)]).toarray()[0],
                       'word2vec': feature_eng.create_text_vectors(clean_text(q), 'word2vec')} 
                      for q in test_queries]
    
    # Create evaluation queries dictionary
    eval_queries = {
        'TF-IDF': tfidf_queries,
        'Word2Vec': w2v_queries,
        'Content-Based': content_queries
    }
    
    # Evaluate each model
    results = {}
    for model_name, model in recommenders.items():
        print(f"\nEvaluating {model_name} model...")
        model_results = {}
        
        # Get queries for this model
        queries = eval_queries[model_name]
        
        # Calculate precision
        precision_values = []
        for query, actual in zip(queries, actual_selections):
            recommendations = model.recommend(query, top_n=10)
            if recommendations.empty:
                precision_values.append(0.0)
                continue
            
            recommended_indices = recommendations.index.tolist()
            hits = len(set(recommended_indices) & set(actual))
            precision = hits / len(recommended_indices) if recommended_indices else 0
            precision_values.append(precision)
        
        model_results['precision'] = np.mean(precision_values)
        print(f"Precision@10: {model_results['precision']:.4f}")
        
        # Calculate recall
        recall_values = []
        for query, actual in zip(queries, actual_selections):
            if not actual:
                recall_values.append(0.0)
                continue
                
            recommendations = model.recommend(query, top_n=10)
            if recommendations.empty:
                recall_values.append(0.0)
                continue
            
            recommended_indices = recommendations.index.tolist()
            hits = len(set(recommended_indices) & set(actual))
            recall = hits / len(actual) if actual else 0
            recall_values.append(recall)
        
        model_results['recall'] = np.mean(recall_values)
        print(f"Recall@10: {model_results['recall']:.4f}")
        
        # Calculate NDCG
        ndcg_values = []
        for query, actual in zip(queries, actual_selections):
            if not actual:
                ndcg_values.append(0.0)
                continue
                
            recommendations = model.recommend(query, top_n=10)
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
            for i in range(min(len(actual), 10)):
                idcg += 1.0 / np.log2(i + 2)
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_values.append(ndcg)
        
        model_results['ndcg'] = np.mean(ndcg_values)
        print(f"NDCG@10: {model_results['ndcg']:.4f}")
        
        results[model_name] = model_results
    
    # Get recommendations for a sample query
    sample_query = "Machine learning for beginners"
    print(f"\n\nGetting recommendations for query: '{sample_query}'")
    
    # Get recommendations
    recommendations = get_recommendations(sample_query, models_dict, top_n=5)
    
    # Display recommendations
    for model_name, recs in recommendations.items():
        print_recommendations(recs, model_name)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 