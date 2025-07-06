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
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
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

def load_combined_data():
    """Load and combine data from all sources"""
    logger.info("Loading and combining data...")
    
    # Define paths to data files
    udemy_path = "udemy_courses.csv"
    dicoding_path = "dicoding_courses.csv"
    coursera_path = "coursera_courses.csv"
    
    # Load data
    data_loader = DataLoader(verbose=True)
    combined_df = data_loader.load_and_preprocess_all(
        udemy_path=udemy_path,
        dicoding_path=dicoding_path,
        coursera_path=coursera_path
    )
    
    logger.info(f"Combined dataset: {len(combined_df)} courses")
    return combined_df

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
    df = load_combined_data()
    
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
        'tfidf': 0.3,  # Weights seperti pada combinestream.py
        'word2vec': 0.7
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
    w2v_recommender = models['word2vec_recommender']
    content_recommender = models['content_recommender']
    tfidf_vectorizer = models['tfidf_vectorizer']
    word2vec_model = models['word2vec_model']
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
    w2v_recs = w2v_recommender.recommend(word2vec_vector, top_n=top_n, filter_dict=filter_dict)
    content_recs = content_recommender.recommend(content_vectors, top_n=top_n, filter_dict=filter_dict)
    
    return {
        'tfidf': tfidf_recs,
        'word2vec': w2v_recs,
        'content': content_recs
    }

def compare_recommendations(recommendations):
    """
    Compare recommendations from different models
    
    Args:
        recommendations: Dictionary with recommendations from different models
        
    Returns:
        None
    """
    # Calculate overlap between models
    models = list(recommendations.keys())
    overlap = {}
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            if model1 != model2:
                recs1 = set(recommendations[model1].index)
                recs2 = set(recommendations[model2].index)
                intersection = recs1.intersection(recs2)
                union = recs1.union(recs2)
                jaccard = len(intersection) / len(union) if union else 0
                
                overlap[f"{model1}_vs_{model2}"] = {
                    'intersection': len(intersection),
                    'jaccard': jaccard
                }
    
    print("\nOverlap between models:")
    for key, value in overlap.items():
        print(f"{key}: {value['intersection']} items, Jaccard index: {value['jaccard']:.2f}")

def visualize_recommendations(recommendations):
    """
    Visualize recommendations from different models
    
    Args:
        recommendations: Dictionary with recommendations from different models
        
    Returns:
        None
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()
    
    for i, (model_name, recs) in enumerate(recommendations.items()):
        ax = axes[i]
        
        # Extract platforms and prices
        platforms = recs['platform'].value_counts()
        
        # Create bar chart
        platforms.plot(kind='bar', ax=ax, color=sns.color_palette("Set2"))
        ax.set_title(f"{model_name.capitalize()} Recommendations by Platform")
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Add price information as text
        avg_price = recs['price'].mean()
        ax.text(0.5, 0.9, f"Avg. Price: ${avg_price:.2f}", 
                transform=ax.transAxes, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/recommendation_comparison.png')
    plt.close()
    
    # Create price distribution plot
    plt.figure(figsize=(10, 6))
    for model_name, recs in recommendations.items():
        sns.kdeplot(recs['price'], label=model_name.capitalize())
    
    plt.title('Price Distribution of Recommendations')
    plt.xlabel('Price ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('plots/price_distribution.png')
    plt.close()
    
    # Create similarity score comparison
    plt.figure(figsize=(10, 6))
    
    for model_name, recs in recommendations.items():
        if 'similarity_score' in recs.columns:
            plt.scatter(range(len(recs)), recs['similarity_score'], 
                      label=f"{model_name.capitalize()} Similarity")
    
    plt.title('Similarity Scores Comparison')
    plt.xlabel('Recommendation Index')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.savefig('plots/similarity_comparison.png')
    plt.close()
    
    # Visualize TF-IDF dan Word2Vec contribution percentages untuk content-based model
    if 'content' in recommendations and 'tfidf_contribution_pct' in recommendations['content'].columns:
        content_recs = recommendations['content']
        
        # Buat bar chart untuk perbandingan kontribusi TF-IDF dan Word2Vec untuk setiap rekomendasi
        plt.figure(figsize=(12, 8))
        
        indices = range(len(content_recs))
        width = 0.35
        
        plt.bar([i - width/2 for i in indices], content_recs['tfidf_contribution_pct'], 
                width=width, label='TF-IDF Contribution (%)', color='#ff9999')
        plt.bar([i + width/2 for i in indices], content_recs['word2vec_contribution_pct'], 
                width=width, label='Word2Vec Contribution (%)', color='#66b3ff')
        
        plt.title('TF-IDF vs Word2Vec Contribution in Content-Based Recommendations')
        plt.xlabel('Course Index')
        plt.ylabel('Contribution (%)')
        plt.xticks(indices, [f"Course {i+1}" for i in indices])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('plots/feature_contribution.png')
        plt.close()
        
        # Buat pie chart untuk rata-rata kontribusi
        avg_tfidf = content_recs['tfidf_contribution_pct'].mean()
        avg_word2vec = content_recs['word2vec_contribution_pct'].mean()
        
        plt.figure(figsize=(8, 8))
        plt.pie([avg_tfidf, avg_word2vec], 
                labels=['TF-IDF', 'Word2Vec'],
                autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'],
                explode=(0.1, 0.1),
                shadow=True,
                startangle=90)
        plt.axis('equal')
        plt.title('Average Feature Contribution in Content-Based Recommendations')
        
        plt.tight_layout()
        plt.savefig('plots/average_contribution.png')
        plt.close()

def print_recommendations(recommendations, model_name):
    """
    Print recommendations in a readable format
    
    Args:
        recommendations: DataFrame with recommendations
        model_name: Name of the model
        
    Returns:
        None
    """
    print(f"\n{model_name} Recommendations:")
    print("=" * 80)
    
    if recommendations.empty:
        print("No recommendations found.")
        return
    
    for i, (idx, row) in enumerate(recommendations.iterrows()):
        print(f"{i+1}. {row['course_title']} ({row['platform']})")
        print(f"   Level: {row['level']}, Price: ${row['price']:.2f}, Duration: {row['duration_category']}")
        print(f"   Similarity Score: {row['similarity_score']:.4f}")
        
        # Tampilkan skor TF-IDF dan Word2Vec jika tersedia
        if 'tfidf_score' in row:
            print(f"   TF-IDF Score: {row['tfidf_score']:.4f}")
        if 'word2vec_score' in row:
            print(f"   Word2Vec Score: {row['word2vec_score']:.4f}")
        if 'weighted_w2v_score' in row:
            print(f"   Weighted Word2Vec Score: {row['weighted_w2v_score']:.4f}")
        
        # Tampilkan persentase kontribusi jika tersedia
        if 'tfidf_contribution_pct' in row and 'word2vec_contribution_pct' in row:
            print(f"   TF-IDF Contribution: {row['tfidf_contribution_pct']:.1f}%, Word2Vec Contribution: {row['word2vec_contribution_pct']:.1f}%")
        
        print("-" * 80)

def evaluate_models(models, test_queries, actual_selections=None, metrics=['precision', 'recall', 'ndcg'], top_n=5):
    """
    Evaluate multiple recommendation models
    
    Args:
        models: Dictionary of recommendation models
        test_queries: Test queries
        actual_selections: Actual selections for each query (if None, will generate synthetic ground truth)
        metrics: List of evaluation metrics
        top_n: Number of recommendations to consider
        
    Returns:
        Dictionary with evaluation results for each model
    """
    results = {}
    
    # If no actual selections provided, create synthetic ground truth
    if actual_selections is None:
        logger.info("Generating synthetic ground truth for evaluation")
        data = models['data']
        actual_selections = []
        
        tfidf_recommender = models['tfidf_recommender']
        content_recommender = models['content_recommender']
        word2vec_recommender = models['word2vec_recommender']
        tfidf_vectorizer = models['tfidf_vectorizer']
        feature_eng = models['feature_eng']
        
        # Create ground truth using same queries but with a simple hybrid approach
        for query_text in test_queries:
            # Vectorize query
            query_text = clean_text(query_text)
            tfidf_vector = tfidf_vectorizer.transform([query_text]).toarray()[0]
            word2vec_vector = feature_eng.create_text_vectors(query_text, 'word2vec')
            
            # Get recommendations from all models
            tfidf_recs = tfidf_recommender.recommend(tfidf_vector, top_n=top_n*2)
            w2v_recs = word2vec_recommender.recommend(word2vec_vector, top_n=top_n*2)
            
            content_vectors = {
                'tfidf': tfidf_vector,
                'word2vec': word2vec_vector
            }
            content_recs = content_recommender.recommend(content_vectors, top_n=top_n*2)
            
            # Combine recommendations using simple voting
            all_recs = set()
            all_recs.update(tfidf_recs.index[:top_n])
            all_recs.update(w2v_recs.index[:top_n])
            all_recs.update(content_recs.index[:top_n])
            
            # Select a reasonable number for ground truth
            ground_truth = list(all_recs)[:top_n]
            actual_selections.append(ground_truth)
            
            logger.info(f"Generated ground truth for query '{query_text}': {len(ground_truth)} items")
    
    # Prepare query vectors for each model
    tfidf_queries = []
    w2v_queries = []
    content_queries = []
    
    # Get vectorizers
    tfidf_vectorizer = models['tfidf_vectorizer']
    feature_eng = models['feature_eng']
    
    # Vectorize queries
    for query in test_queries:
        # Clean query
        query = clean_text(query)
        
        # TF-IDF vector
        tfidf_vector = tfidf_vectorizer.transform([query]).toarray()[0]
        tfidf_queries.append(tfidf_vector)
        
        # Word2Vec vector
        word2vec_vector = feature_eng.create_text_vectors(query, 'word2vec')
        w2v_queries.append(word2vec_vector)
        
        # Content-based vectors
        content_vectors = {
            'tfidf': tfidf_vector,
            'word2vec': word2vec_vector
        }
        content_queries.append(content_vectors)
    
    # Get recommenders
    recommenders = {
        'TF-IDF Recommender': models['tfidf_recommender'],
        'Word2Vec Recommender': models['word2vec_recommender'],
        'Content-Based Recommender': models['content_recommender']
    }
    
    # Create evaluation queries dictionary
    eval_queries = {
        'TF-IDF Recommender': tfidf_queries,
        'Word2Vec Recommender': w2v_queries,
        'Content-Based Recommender': content_queries
    }
    
    # Evaluate each model
    for model_name, model in recommenders.items():
        if not model.is_fitted:
            logger.warning(f"Model {model_name} not fitted, skipping evaluation")
            continue
            
        model_results = {}
        
        # Get queries for this model
        queries = eval_queries[model_name]
        
        for metric in metrics:
            if metric == 'precision':
                precision_values = []
                for i, (query, actual) in enumerate(zip(queries, actual_selections)):
                    if not actual:
                        logger.warning(f"No ground truth for query {i+1}: '{test_queries[i]}'")
                        precision_values.append(0.0)
                        continue
                        
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
                for i, (query, actual) in enumerate(zip(queries, actual_selections)):
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
            
            elif metric == 'ndcg':
                ndcg_values = []
                for i, (query, actual) in enumerate(zip(queries, actual_selections)):
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
            
            elif metric == 'coverage':
                # Calculate coverage as the percentage of all items that appear in recommendations
                all_recommended = set()
                total_items = len(models['data'])
                
                for query in queries:
                    recommendations = model.recommend(query, top_n=top_n)
                    if not recommendations.empty:
                        all_recommended.update(recommendations.index.tolist())
                
                coverage = len(all_recommended) / total_items if total_items > 0 else 0
                model_results['coverage'] = coverage
        
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

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test recommendation models')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining of models')
    args = parser.parse_args()
    
    # Load or train models
    models = load_or_train_models(force_retrain=args.force_retrain)
    
    # Test queries
    test_queries = [
        "Machine learning for beginners",
        "Advanced web development with JavaScript",
        "Data science with Python",
        "Business management and leadership",
        "Mobile app development tutorial",
        "Learn artificial intelligence",
        "Beginner's guide to programming",
        "Cloud computing and AWS",
        "Digital marketing strategies",
        "Cybersecurity fundamentals"
    ]
    
    # Evaluate models with more comprehensive metrics
    metrics = ['precision', 'recall', 'ndcg', 'coverage']
    results = evaluate_models(models, test_queries, metrics=metrics, top_n=5)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"{'Model':<30} {'Precision@5':<15} {'Recall@5':<15} {'NDCG@5':<15} {'Coverage':<15}")
    print("-" * 80)
    for model_name, metrics in results.items():
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        ndcg = metrics.get('ndcg', 0.0)
        coverage = metrics.get('coverage', 0.0)
        print(f"{model_name:<30} {precision:<15.3f} {recall:<15.3f} {ndcg:<15.3f} {coverage:<15.3f}")
    
    # Plot evaluation results
    plot_evaluation_results(results, metric='precision')
    plot_evaluation_results(results, metric='recall')
    plot_evaluation_results(results, metric='ndcg')
    
    # Get recommendations for a sample query
    sample_query = "Machine learning for beginners"
    print(f"\nGetting recommendations for query: '{sample_query}'")
    
    # Get recommendations
    recommendations = get_recommendations(sample_query, models, top_n=5)
    
    # Compare recommendations
    compare_recommendations(recommendations)
    
    # Print recommendations
    for model_name, recs in recommendations.items():
        print_recommendations(recs, model_name)
    
    # Visualize recommendations
    visualize_recommendations(recommendations)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 