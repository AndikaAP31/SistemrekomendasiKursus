import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import logging
import time
import pickle
import os
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_word2vec_model(df, text_column='combined_text', vector_size=100, window=5, min_count=1, workers=4):
  
    start_time = time.time()
    
    if text_column not in df.columns:
        logger.warning(f"Text column '{text_column}' not found in DataFrame")
        return None
    
    # Prepare sentences for Word2Vec training
    logger.info(f"Preparing sentences from {text_column} column")
    sentences = [text.split() for text in df[text_column].fillna('')]
    
    # Train Word2Vec model
    logger.info("Starting Word2Vec model training...")
    word2vec_model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    
    # Log model information
    vocab_size = len(word2vec_model.wv.index_to_key)
    logger.info(f"Trained Word2Vec model with vocabulary size: {vocab_size}, vector size: {vector_size}")
    logger.info(f"Word2Vec model training completed in {time.time() - start_time:.2f} seconds")
    
    return word2vec_model

def create_word2vec_vectors(df, word2vec_model, text_column='combined_text'):
    """
    Create Word2Vec vectors for each document in the DataFrame
    """
    start_time = time.time()
    
    if text_column not in df.columns:
        logger.warning(f"Text column '{text_column}' not found in DataFrame")
        return None
    
    # Function to convert text to Word2Vec vector
    def vectorize_text(text):
        words = text.lower().split()
        vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)
    
    # Create vectors for each document
    logger.info(f"Creating Word2Vec vectors for {len(df)} documents")
    vectors = np.array([vectorize_text(text) for text in df[text_column]])
    
    logger.info(f"Word2Vec vectors creation completed in {time.time() - start_time:.2f} seconds")
    
    return vectors

def fix_recommenders(model_path="models/recommenders.pkl"):
   
    # Load the existing recommenders
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                recommenders = pickle.load(f)
            logger.info("Loaded existing recommenders")
        except Exception as e:
            logger.error(f"Error loading recommenders: {e}")
            return None
    else:
        logger.error(f"Recommenders file not found at {model_path}")
        return None
    
    # Get the data
    df = recommenders['data']
    
    # Check if combined_text column exists
    if 'combined_text' not in df.columns:
        logger.info("Creating combined_text column")
        df['combined_text'] = (
            df['course_title'] + ' ' +
            df['level'] + ' ' +
            df['subject'] + ' ' +
            df['duration_category'] + ' ' +
            df['platform']
        ).str.lower()
    
    # Train Word2Vec model
    logger.info("Training Word2Vec model")
    word2vec_model = train_word2vec_model(df)
    
    # Create Word2Vec vectors
    logger.info("Creating Word2Vec vectors")
    word2vec_vectors = create_word2vec_vectors(df, word2vec_model)
    
    # Update recommenders
    recommenders['word2vec_model'] = word2vec_model
    recommenders['feature_eng'].word2vec_model = word2vec_model
    
    # Update Word2Vec recommender
    recommenders['word2vec_recommender'].course_vectors = word2vec_vectors
    
    # Update content-based recommender - dengan penanganan kesalahan yang lebih baik
    feature_matrices = {}
    
    # Tambahkan TF-IDF matrix dengan pengecekan
    try:
        if 'tfidf_recommender' in recommenders and hasattr(recommenders['tfidf_recommender'], 'tfidf_matrix'):
            tfidf_matrix = recommenders['tfidf_recommender'].tfidf_matrix
            if tfidf_matrix is not None:
                if hasattr(tfidf_matrix, 'toarray'):
                    feature_matrices['tfidf'] = tfidf_matrix.toarray()
                else:
                    feature_matrices['tfidf'] = tfidf_matrix  # Asumsikan sudah dalam bentuk array
            else:
                logger.warning("TF-IDF matrix is None, using zeros array instead")
                # Gunakan array kosong dengan dimensi yang sesuai
                feature_matrices['tfidf'] = np.zeros((len(df), len(recommenders['tfidf_vectorizer'].get_feature_names_out())))
        else:
            logger.warning("TF-IDF recommender not found or doesn't have tfidf_matrix attribute")
            # Gunakan array kosong dengan dimensi yang sesuai jika memungkinkan
            if 'tfidf_vectorizer' in recommenders:
                feature_matrices['tfidf'] = np.zeros((len(df), len(recommenders['tfidf_vectorizer'].get_feature_names_out())))
            else:
                # Jika tidak ada informasi dimensi, gunakan dimensi word2vec
                feature_matrices['tfidf'] = np.zeros((len(df), word2vec_vectors.shape[1]))
    except Exception as e:
        logger.error(f"Error processing TF-IDF matrix: {e}")
        # Fallback ke array kosong dengan dimensi word2vec
        feature_matrices['tfidf'] = np.zeros((len(df), word2vec_vectors.shape[1]))
    
    # Tambahkan Word2Vec vectors
    feature_matrices['word2vec'] = word2vec_vectors
    
    feature_weights = {
        'tfidf': 0.3,  # Same weights as in combinestream.py
        'word2vec': 0.7
    }
    
    recommenders['content_recommender'].feature_matrices = feature_matrices
    recommenders['content_recommender'].feature_weights = feature_weights
    
    # Save the fixed recommenders
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(recommenders, f)
        logger.info(f"Fixed recommenders saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving fixed recommenders: {e}")
    
    return recommenders

def create_query_vectors(query_text, recommenders):

    # Get models
    tfidf_vectorizer = recommenders['tfidf_vectorizer']
    word2vec_model = recommenders['word2vec_model']
    feature_eng = recommenders['feature_eng']
    
    # Clean and preprocess query text
    query_text = clean_text(query_text)
    logger.info(f"Cleaned query text: {query_text}")
    
    # Create TF-IDF vector
    try:
        # Try direct transformation first
        tfidf_vector = tfidf_vectorizer.transform([query_text]).toarray()[0]
        logger.info(f"TF-IDF vector created directly, shape: {tfidf_vector.shape}")
        logger.info(f"TF-IDF vector sum: {np.sum(tfidf_vector)}")
        
        # If all zeros, try feature engineering
        if np.all(tfidf_vector == 0):
            logger.info("TF-IDF vector is all zeros, trying feature engineering")
            tfidf_vector = feature_eng.create_text_vectors(query_text, 'tfidf')
            logger.info(f"TF-IDF vector created via feature engineering, shape: {tfidf_vector.shape}")
            logger.info(f"TF-IDF vector sum: {np.sum(tfidf_vector)}")
    except Exception as e:
        logger.warning(f"Error creating TF-IDF vector: {e}")
        # Use the correct dimension for zeros
        vocab_size = len(tfidf_vectorizer.get_feature_names_out())
        tfidf_vector = np.zeros(vocab_size)
        logger.warning(f"Using zero vector with shape: {tfidf_vector.shape}")
    
    # Create Word2Vec vector
    try:
        # Try feature engineering first
        word2vec_vector = feature_eng.create_text_vectors(query_text, 'word2vec')
        logger.info(f"Word2Vec vector created via feature engineering, shape: {word2vec_vector.shape}")
        
        # If all zeros, try direct word2vec
        if np.all(word2vec_vector == 0):
            logger.info("Word2Vec vector is all zeros, trying direct method")
            words = query_text.lower().split()
            vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
            word2vec_vector = np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)
            logger.info(f"Word2Vec vector created directly, shape: {word2vec_vector.shape}")
    except Exception as e:
        logger.warning(f"Error creating Word2Vec vector: {e}")
        word2vec_vector = np.zeros(word2vec_model.vector_size)
        logger.warning(f"Using zero vector with shape: {word2vec_vector.shape}")
    
    # Normalize vectors
    tfidf_norm = np.linalg.norm(tfidf_vector)
    word2vec_norm = np.linalg.norm(word2vec_vector)
    
    if tfidf_norm > 0:
        tfidf_vector = tfidf_vector / tfidf_norm
        logger.info("TF-IDF vector normalized")
    if word2vec_norm > 0:
        word2vec_vector = word2vec_vector / word2vec_norm
        logger.info("Word2Vec vector normalized")
    
    return {
        'tfidf': tfidf_vector,
        'word2vec': word2vec_vector
    }

def get_recommendations(query_text, recommenders, filter_dict=None, top_n=5):

    # Create query vectors
    query_vectors = create_query_vectors(query_text, recommenders)
    
    # Get recommendations
    content_recs = recommenders['content_recommender'].recommend(
        query_vectors,
        top_n=top_n,
        filter_dict=filter_dict
    )
    
    # Calculate contribution percentages
    if 'tfidf_score' in content_recs.columns and 'word2vec_score' in content_recs.columns:
        # Get scores
        tfidf_scores = content_recs['tfidf_score'].fillna(0)
        word2vec_scores = content_recs['word2vec_score'].fillna(0)
        
        # Calculate total scores
        total_scores = tfidf_scores + word2vec_scores
        
        # Calculate contributions (avoid division by zero)
        content_recs['tfidf_contribution'] = np.where(total_scores > 0, 
                                                     tfidf_scores / total_scores * 100, 0)
        content_recs['word2vec_contribution'] = np.where(total_scores > 0, 
                                                        word2vec_scores / total_scores * 100, 0)
    
    return content_recs

if __name__ == "__main__":
    # Fix recommenders
    recommenders = fix_recommenders()
    
    if recommenders is not None:
        print("Word2Vec model fixed successfully!")
        
        # Test with a query
        query = "belajar data"
        recs = get_recommendations(query, recommenders)
        
        print(f"\nRecommendations for '{query}':")
        print(recs[['course_title', 'platform', 'similarity_score', 'tfidf_score', 'word2vec_score', 
                   'tfidf_contribution', 'word2vec_contribution']])
    else:
        print("Failed to fix Word2Vec model.")
