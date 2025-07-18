import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import FeatureEngineering
from gensim.models import Word2Vec
from models import TFIDFRecommender, ContentBasedRecommender
from data_loader import DataLoader


from word2vec_fix import fix_recommenders, get_recommendations, create_query_vectors
from utils import clean_text, normalize_weights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "models"
DATA_DIR = "data"
PLOT_DIR = "plots"

# Initialize paths
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Create a function to get translation
def get_translation(text, src_lang, dest_lang):
    """Translate text between languages"""
    try:
        from googletrans import Translator
        translator = Translator()
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

# Function to load or train models with fixed Word2Vec
@st.cache_resource
def load_or_train_models_fixed(udemy_path, dicoding_path, coursera_path, force_retrain=False):
    """Load pretrained models or train new ones with fixed Word2Vec"""
    model_file = os.path.join(MODEL_DIR, "recommenders.pkl")
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if os.path.exists(model_file) and not force_retrain:
        try:
            recommenders = fix_recommenders(model_file)
            if recommenders is not None:
                st.success("Sistem siap Digunakan")
                return recommenders, recommenders['data']
        except Exception as e:
            st.info("📚Selamat Datang Di Sistem Rekomendasi Kursus Online📚")
    
    # Implementasi load_or_train_models langsung di sini
    recommenders, df = load_or_train_models(udemy_path, dicoding_path, coursera_path, force_retrain=False)
      
    return recommenders, df

# Function to load or train models
def load_or_train_models(udemy_path, dicoding_path, coursera_path, force_retrain=False):
    """Load pretrained models or train new ones"""
    model_file = os.path.join(MODEL_DIR, "recommenders.pkl")
    
    if os.path.exists(model_file) and not force_retrain:
        try:
            with open(model_file, 'rb') as f:
                recommenders = pickle.load(f)
            st.success("Sistem siap Digunakan")
            return recommenders, recommenders['data']
        except Exception as e:
            st.warning(f"Error loading models: {e}")
            st.info("Training new models...")
    
    # Load and preprocess data
    data_loader = DataLoader(verbose=True)
    df = data_loader.load_and_preprocess_all(udemy_path, dicoding_path, coursera_path)
    
    if df.empty:
        st.error("Failed to load course data!")
        return None, None
    
    feature_eng = FeatureEngineering(verbose=True)
    df_processed = feature_eng.process_all_features(df)
    
    
    if 'combined_text' not in df_processed.columns:
        df_processed['combined_text'] = (
            df_processed['course_title'] + ' ' +
            df_processed['level'] + ' ' +
            df_processed['subject'] + ' ' +
            df_processed['duration_category'] + ' ' +
            df_processed['platform']
        ).str.lower()
    
    # Train Word2Vec model explicitly
    sentences = [text.split() for text in df_processed['combined_text']]
    # from gensim.models import Word2Vec
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # Store the model in feature_eng
    feature_eng.word2vec_model = word2vec_model
    
    
    tfidf_recommender = TFIDFRecommender(verbose=True)
    tfidf_recommender.fit(
        df_processed, 
        feature_eng.tfidf_matrix, 
        feature_eng.tfidf_vectorizer.get_feature_names_out()
    )
    
    word2vec_vectors = np.array([
        feature_eng.create_text_vectors(text, 'word2vec') 
        for text in df_processed['combined_text']
    ])
    
    
    # Create content-based recommender
    feature_matrices = {
        'tfidf': feature_eng.tfidf_matrix.toarray(),
        'word2vec': word2vec_vectors
    }
    
    feature_weights = {
        'tfidf': 0.3,  # Same weights as in combinestream.py
        'word2vec': 0.7
    }
    
    content_recommender = ContentBasedRecommender(verbose=True)
    content_recommender.fit(
        df_processed, 
        feature_matrices, 
        feature_weights,
        feature_names=feature_eng.tfidf_vectorizer.get_feature_names_out()
    )
    
    # Create a dictionary to hold all recommenders
    recommenders = {
        'tfidf_recommender': tfidf_recommender,
        'content_recommender': content_recommender,
        'tfidf_vectorizer': feature_eng.tfidf_vectorizer,
        'word2vec_model': feature_eng.word2vec_model,
        'data': df_processed,
        'feature_eng': feature_eng
    }
    
    # Save the models
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(recommenders, f)
        st.success("Models trained and saved successfully!")
    except Exception as e:
        st.warning(f"Error saving models: {e}")
    
    return recommenders, df_processed

def show_recommendation_page_fixed():
    """Show recommendation page with fixed Word2Vec"""
    st.title("Sistem Rekomendasi Kursus Online")
    
    if 'recommenders' not in st.session_state or 'df' not in st.session_state:
        st.warning("Silakan tekan tombol mulai pada sidebar terlebih dahulu.")
        return
    
    # Inisialisasi state jika belum ada
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'has_searched' not in st.session_state:
        st.session_state.has_searched = False
    
    recommenders = st.session_state.recommenders
    df = st.session_state.df
    
    # Tampilkan tab untuk Pencarian dan Detail
    tab1, tab2 = st.tabs(["Pencarian Kursus", "Detail Kursus"])
    
    with tab1:
        st.subheader("Cari Kursus")
        
        # Get unique values for filters
        platforms = ["Semua"] + sorted(df['platform'].unique().tolist())
        levels = ["Semua"] + sorted(df['level'].unique().tolist())
        
        # Create duration descriptions
        duration_descriptions = {
            "short": "Short (< 10 jam)",
            "medium": "Medium (10 - 35 jam)",
            "long": "Long (> 35 jam)"
        }
        
        # Create duration options with descriptions
        durations = ["Semua"] + [
            duration_descriptions.get(dur, dur) 
            for dur in sorted(df['duration_category'].unique().tolist())
        ]
        
        # Create columns for filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_platform = st.selectbox("Platform", platforms)
        
        with col2:
            selected_level = st.selectbox("Tingkat Kesulitan", levels)
        
        with col3:
            selected_duration = st.selectbox("Durasi", durations)
            
        # Convert selected duration back to category for filtering
        if selected_duration != "Semua":
            selected_duration = next(
                (cat for cat, desc in duration_descriptions.items() 
                 if desc == selected_duration),
                selected_duration
            )
        
        # Additional filters
        col1, col2 = st.columns(2)
        
        with col1:
            price_range = st.slider(
                "Rentang Harga (USD)", 
                min_value=0, 
                max_value=int(df['price'].max()) + 1,
                value=(0, int(df['price'].max())),
                step=10
            )
        
        with col2:
            selected_language = st.selectbox(
                "Bahasa", 
                ["Semua", "Inggris", "Indonesia"]
            )
        
        # Learning goal input
        learning_goal = st.text_area(
            "Deskripsikan tujuan belajar Anda", 
            height=100,
            help="Masukkan apa yang ingin Anda pelajari, keterampilan yang ingin Anda peroleh, atau topik yang Anda tertarik."
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Jumlah Rekomendasi", 
            min_value=1, 
            max_value=20, 
            value=5
        )
        
        # Create filter dictionary
        filter_dict = {}
        
        if selected_platform != "Semua":
            filter_dict['platform'] = selected_platform
        
        if selected_level != "Semua":
            filter_dict['level'] = selected_level
        
        if selected_duration != "Semua":
            filter_dict['duration_category'] = selected_duration
        
        filter_dict['max_price'] = price_range[1]
        
        # Function to get recommendations with fixed Word2Vec
        def get_recommendations_fixed():
            if not learning_goal.strip():
                st.warning("Silakan deskripsikan tujuan belajar Anda untuk mendapatkan rekomendasi yang lebih baik.")
                return None
            
            with st.spinner("Mencari kursus terbaik untuk Anda..."):
                # Determine if translation is needed based on selected language
                translated_goal = learning_goal
                
                # Translate if language is English or All (Semua)
                if selected_language == "Inggris" or selected_language == "Semua":
                    translated_goal = get_translation(learning_goal, src_lang='id', dest_lang='en')
                    st.info(f"Tujuan belajar diterjemahkan ke Bahasa Inggris: {translated_goal}")
                elif selected_language == "Indonesia":
                    # Keep original text, no translation needed
                    translated_goal = learning_goal
                
                # Validasi: Jika input kosong, tampilkan peringatan
                if not translated_goal or translated_goal.strip() == "":
                    st.warning("Silakan masukkan tujuan belajar sebelum mendapatkan rekomendasi.")
                    return None
                
                # Clean and normalize the query text
                query_text = clean_text(translated_goal)
                
                # Use the fixed Word2Vec implementation to get recommendations
                return get_recommendations(query_text, recommenders, filter_dict, num_recommendations)
        
        # Show recommendations button
        if st.button("Dapatkan Rekomendasi"):
            recommendations = get_recommendations_fixed()
            if recommendations is not None:
                st.session_state.recommendations = recommendations
                st.session_state.has_searched = True
        
        # Tampilkan rekomendasi jika sudah ada
        if st.session_state.has_searched and st.session_state.recommendations is not None:
            recommendations = st.session_state.recommendations
            
            if recommendations.empty:
                st.warning("Tidak ada kursus yang sesuai dengan kriteria Anda. Coba lakukan penyesuaian pada filter.")
            else:
                st.success(f"Ditemukan {len(recommendations)} kursus untuk Anda!")
                
                # Create a subset of columns to display
                display_df = recommendations[['course_title', 'platform', 'level', 'duration_category', 'price', 'similarity_score']].copy()
                
                # Add score columns if they exist
                if 'tfidf_score' in recommendations.columns:
                    display_df['tfidf_score'] = recommendations['tfidf_score']
                if 'word2vec_score' in recommendations.columns:
                    display_df['word2vec_score'] = recommendations['word2vec_score']
                
                # Create column name mappings
                column_mappings = {
                    'course_title': 'Judul Kursus',
                    'platform': 'Platform',
                    'level': 'Level',
                    'duration_category': 'Durasi',
                    'price': 'Harga (USD)',
                    'similarity_score': 'Skor Kesamaan',
                    'tfidf_score': 'Skor TF-IDF',
                    'word2vec_score': 'Skor Word2Vec'
                }
                
                # Rename columns
                display_df = display_df.rename(columns=column_mappings)
                
                # Format numeric columns
                for col in display_df.columns:
                    if col in ['Skor Kesamaan', 'Skor TF-IDF', 'Skor Word2Vec']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                    elif col == 'Harga (USD)':
                        display_df[col] = display_df[col].apply(lambda x: f"${int(x)} (Rp {int(x)*16000:,})" if x > 0 else "Free")
                
                # Display recommendations
                st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        st.subheader("Detail Kursus")
        
        if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
            recommendations = st.session_state.recommendations
            
            if not recommendations.empty:
                # Create a dropdown to select a course
                course_titles = recommendations['course_title'].tolist()
                selected_course = st.selectbox("Pilih kursus untuk melihat detail", course_titles)
                
                # Get the selected course details
                selected_course_data = recommendations[recommendations['course_title'] == selected_course].iloc[0]
                
                # Get the full course data from the original dataset
                df = st.session_state.df
                full_course_data = df[df['course_title'] == selected_course].iloc[0] if len(df[df['course_title'] == selected_course]) > 0 else selected_course_data
                
                # Display course details in an attractive format
                st.markdown(f"## {selected_course}")
                
                # Platform badge with color
                platform_colors = {
                    "Udemy": "#EB524F",
                    "Coursera": "#0056D2",
                    "Dicoding": "#2D3E50"
                }
                platform = selected_course_data['platform']
                platform_color = platform_colors.get(platform, "#6c757d")
                st.markdown(f"<span style='background-color:{platform_color};color:white;padding:5px 10px;border-radius:4px;font-weight:bold;'>{platform}</span>", unsafe_allow_html=True)
                
                # Price display
                if selected_course_data['price'] == 0:
                    st.markdown("### 🎁 **GRATIS**")
                else:
                    usd_price = int(selected_course_data['price'])
                    idr_price = usd_price * 16000
                    st.markdown(f"### 💰 **${usd_price}** (Rp {idr_price:,})")
                
                st.markdown("---")
                
                # Basic information in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### ℹ️ Informasi Dasar")
                    
                    # Level
                    if 'level' in selected_course_data and not pd.isna(selected_course_data['level']) and str(selected_course_data['level']).strip() != '':
                        st.markdown(f"**Level:** {selected_course_data['level']}")
                    else:
                        st.markdown("**Level:** Tidak tersedia")
                    
                    # Durasi
                    if 'content_duration' in selected_course_data and not pd.isna(selected_course_data['content_duration']):
                        try:
                            st.markdown(f"**Durasi:** {float(selected_course_data['content_duration']):.1f} jam")
                        except (ValueError, TypeError):
                            st.markdown(f"**Durasi:** {selected_course_data['content_duration']}")
                    elif 'duration_category' in selected_course_data and not pd.isna(selected_course_data['duration_category']) and str(selected_course_data['duration_category']).strip() != '':
                        st.markdown(f"**Durasi:** {selected_course_data['duration_category']}")
                    else:
                        st.markdown("**Durasi:** Tidak tersedia")
                    
                    # Kategori Durasi
                    if 'duration_category' in selected_course_data and not pd.isna(selected_course_data['duration_category']) and str(selected_course_data['duration_category']).strip() != '':
                        st.markdown(f"**Kategori Durasi:** {selected_course_data['duration_category']}")
                    else:
                        st.markdown("**Kategori Durasi:** Tidak tersedia")
                    
                    # Subjek
                    if 'subject' in selected_course_data and not pd.isna(selected_course_data['subject']) and str(selected_course_data['subject']).strip() != '':
                        st.markdown(f"**Subjek:** {selected_course_data['subject']}")
                    else:
                        st.markdown("**Subjek:** Tidak tersedia")
                    
                    # Bahasa - set based on platform
                    platform = selected_course_data['platform']
                    if platform == "Dicoding":
                        st.markdown("**Bahasa:** Bahasa Indonesia")
                    else:  # Udemy or Coursera
                        st.markdown("**Bahasa:** Bahasa Inggris")
                
                with col2:
                    st.markdown("### 📊 Statistik")
                    
                    # Jumlah Peserta
                    if 'num_subscribers' in full_course_data and not pd.isna(full_course_data['num_subscribers']) and str(full_course_data['num_subscribers']).strip() != '':
                        try:
                            st.markdown(f"**Jumlah Peserta:** {int(full_course_data['num_subscribers']):,}")
                        except (ValueError, TypeError):
                            st.markdown(f"**Jumlah Peserta:** {full_course_data['num_subscribers']}")
                    else:
                        st.markdown("**Jumlah Peserta:** Tidak tersedia")
                    
                    # Jumlah Ulasan
                    if 'num_reviews' in full_course_data and not pd.isna(full_course_data['num_reviews']) and str(full_course_data['num_reviews']).strip() != '':
                        try:
                            st.markdown(f"**Jumlah Ulasan:** {int(full_course_data['num_reviews']):,}")
                        except (ValueError, TypeError):
                            st.markdown(f"**Jumlah Ulasan:** {full_course_data['num_reviews']}")
                    else:
                        st.markdown("**Jumlah Ulasan:** Tidak tersedia")
                    
                    # Jumlah Materi
                    if 'num_lectures' in full_course_data and not pd.isna(full_course_data['num_lectures']) and str(full_course_data['num_lectures']).strip() != '':
                        try:
                            st.markdown(f"**Jumlah Materi:** {int(full_course_data['num_lectures']):,}")
                        except (ValueError, TypeError):
                            st.markdown(f"**Jumlah Materi:** {full_course_data['num_lectures']}")
                    else:
                        st.markdown("**Jumlah Materi:** Tidak tersedia")
                    
                    # Rating
                    if 'rating' in full_course_data and not pd.isna(full_course_data['rating']) and str(full_course_data['rating']).strip() != '':
                        try:
                            rating_value = float(full_course_data['rating'])
                            stars = "⭐" * int(round(rating_value))
                            st.markdown(f"**Rating:** {stars} ({rating_value:.1f}/5.0)")
                        except (ValueError, TypeError):
                            st.markdown(f"**Rating:** {full_course_data['rating']}")
                    else:
                        st.markdown("**Rating:** Tidak tersedia")
                
                with col3:
                    st.markdown("### 🔍 Skor Relevansi")
                    
                    # Skor Kesamaan
                    if 'similarity_score' in selected_course_data and not pd.isna(selected_course_data['similarity_score']):
                        try:
                            st.markdown(f"**Skor Kesamaan:** {float(selected_course_data['similarity_score']):.4f}")
                        except (ValueError, TypeError):
                            st.markdown(f"**Skor Kesamaan:** {selected_course_data['similarity_score']}")
                    else:
                        st.markdown("**Skor Kesamaan:** Tidak tersedia")
                    
                    # Skor TF-IDF
                    if 'tfidf_score' in selected_course_data and not pd.isna(selected_course_data['tfidf_score']):
                        try:
                            tfidf_score = float(selected_course_data['tfidf_score'])
                            st.markdown(f"**Skor TF-IDF:** {tfidf_score:.4f}")
                        except (ValueError, TypeError):
                            st.markdown(f"**Skor TF-IDF:** {selected_course_data['tfidf_score']}")
                    else:
                        st.markdown("**Skor TF-IDF:** Tidak tersedia")
                    
                    # Skor Word2Vec
                    if 'word2vec_score' in selected_course_data and not pd.isna(selected_course_data['word2vec_score']):
                        try:
                            word2vec_score = float(selected_course_data['word2vec_score'])
                            st.markdown(f"**Skor Word2Vec:** {word2vec_score:.4f}")
                        except (ValueError, TypeError):
                            st.markdown(f"**Skor Word2Vec:** {selected_course_data['word2vec_score']}")
                    else:
                        st.markdown("**Skor Word2Vec:** Tidak tersedia")
                    
                    # Kontribusi TF-IDF
                    if 'tfidf_contribution' in selected_course_data and not pd.isna(selected_course_data['tfidf_contribution']):
                        try:
                            tfidf_contrib = float(selected_course_data['tfidf_contribution'])
                            st.markdown(f"**Kontribusi TF-IDF:** {tfidf_contrib:.2f}%")
                        except (ValueError, TypeError):
                            st.markdown(f"**Kontribusi TF-IDF:** {selected_course_data['tfidf_contribution']}")
                    else:
                        st.markdown("**Kontribusi TF-IDF:** Tidak tersedia")
                    
                    # Kontribusi Word2Vec
                    if 'word2vec_contribution' in selected_course_data and not pd.isna(selected_course_data['word2vec_contribution']):
                        try:
                            word2vec_contrib = float(selected_course_data['word2vec_contribution'])
                            st.markdown(f"**Kontribusi Word2Vec:** {word2vec_contrib:.2f}%")
                        except (ValueError, TypeError):
                            st.markdown(f"**Kontribusi Word2Vec:** {selected_course_data['word2vec_contribution']}")
                    else:
                        st.markdown("**Kontribusi Word2Vec:** Tidak tersedia")
                
                st.markdown("---")
                
                # Show link to course if available
                if 'course_url' in full_course_data and not pd.isna(full_course_data['course_url']):
                    st.markdown(f"<a href='{full_course_data['course_url']}' target='_blank' style='display:inline-block;background-color:#4CAF50;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;font-weight:bold;'>Lihat Kursus</a>", unsafe_allow_html=True)
            else:
                st.info("Silakan lakukan pencarian terlebih dahulu untuk melihat detail kursus.")
        else:
            st.info("Silakan lakukan pencarian terlebih dahulu untuk melihat detail kursus.")
    
def show_statistics_page():
    st.title("Statistik Dataset")
    
    if 'df' not in st.session_state:
        st.warning("Silakan tekan tombol mulai pada sidebar terlebih dahulu.")
        return
    
    df = st.session_state.df
    
    # Display overall statistics
    st.header("Statistik Umum")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Kursus", f"{len(df):,}")
    
    with col2:
        free_courses = (df['price'] == 0).sum()
        st.metric("Kursus Gratis", f"{free_courses:,}")
    
    with col3:
        paid_courses = (df['price'] > 0).sum()
        st.metric("Kursus Berbayar", f"{paid_courses:,}")
    
    with col4:
        avg_price = df[df['price'] > 0]['price'].mean()
        st.metric("Harga Rata-rata", f"${avg_price:.2f}")
    
    st.write(f"""
    Dataset terdiri dari total **{len(df):,}** kursus, dengan **{free_courses:,}** kursus gratis dan **{paid_courses:,}** kursus berbayar.
    Rata-rata harga untuk kursus berbayar adalah **${avg_price:.2f}**.
    """)
    
    # Course distribution by platform
    st.subheader("Distribusi Kursus Berdasarkan Platform")
    platform_counts = df['platform'].value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Jumlah']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(platform_counts.set_index('Platform'))
        
        # Add descriptive text for platform distribution
        platform_text = []
        for _, row in platform_counts.iterrows():
            percentage = (row['Jumlah'] / len(df)) * 100
            platform_text.append(f"**{row['Platform']}**: {row['Jumlah']:,} kursus ({percentage:.1f}%)")
        
        st.write("Distribusi kursus per platform:")
        st.write("\n".join(platform_text))
    
    with col2:
        # Create a pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            platform_counts['Jumlah'], 
            labels=platform_counts['Platform'],
            autopct='%1.1f%%',
            startangle=90,
            shadow=True
        )
        ax.axis('equal')
        ax.set_title('Distribusi Platform')
        st.pyplot(fig)
    
    # Level distribution
    st.subheader("Distribusi Kursus Berdasarkan Tingkat Kesulitan")
    level_counts = df['level'].value_counts().reset_index()
    level_counts.columns = ['Tingkat', 'Jumlah']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(level_counts.set_index('Tingkat'))
        
        # Add descriptive text for level distribution
        level_text = []
        for _, row in level_counts.iterrows():
            percentage = (row['Jumlah'] / len(df)) * 100
            level_text.append(f"**{row['Tingkat']}**: {row['Jumlah']:,} kursus ({percentage:.1f}%)")
        
        st.write("Distribusi kursus berdasarkan tingkat kesulitan:")
        st.write("\n".join(level_text))
    
    with col2:
        # Create a pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            level_counts['Jumlah'], 
            labels=level_counts['Tingkat'],
            autopct='%1.1f%%',
            startangle=90,
            shadow=True
        )
        ax.axis('equal')
        ax.set_title('Distribusi Tingkat Kesulitan')
        st.pyplot(fig)
    
    # Duration distribution
    st.subheader("Distribusi Kursus Berdasarkan Durasi")
    duration_counts = df['duration_category'].value_counts().reset_index()
    duration_counts.columns = ['Durasi', 'Jumlah']
    
    st.bar_chart(duration_counts.set_index('Durasi'))
    
    # Add descriptive text for duration distribution
    duration_text = []
    for _, row in duration_counts.iterrows():
        percentage = (row['Jumlah'] / len(df)) * 100
        duration_text.append(f"**{row['Durasi']}**: {row['Jumlah']:,} kursus ({percentage:.1f}%)")
    
    st.write("Distribusi kursus berdasarkan kategori durasi:")
    st.write("\n".join(duration_text))
    
    # Price distribution
    st.subheader("Distribusi Harga")
    
    # Filter for better visualization
    price_filter = st.slider(
        "Harga Maksimum untuk Visualisasi", 
        min_value=0, 
        max_value=int(df['price'].max()),
        value=min(200, int(df['price'].max())),
        step=10
    )
    
    filtered_prices = df[df['price'] <= price_filter]['price']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_prices, bins=30, kde=True, ax=ax)
    ax.set_xlabel('Harga (USD)')
    ax.set_ylabel('Jumlah Kursus')
    ax.set_title('Distribusi Harga Kursus')
    st.pyplot(fig)
    
    # Add descriptive text for price distribution
    price_stats = {
        'free': (df['price'] == 0).sum(),
        'paid': (df['price'] > 0).sum(),
        'min_price': df[df['price'] > 0]['price'].min(),
        'max_price': df['price'].max(),
        'mean_price': df[df['price'] > 0]['price'].mean(),
        'median_price': df[df['price'] > 0]['price'].median()
    }
    
    st.write(f"""
    Statistik harga kursus:
    - Kursus gratis: **{price_stats['free']:,}** kursus
    - Kursus berbayar: **{price_stats['paid']:,}** kursus
    - Harga terendah: **${price_stats['min_price']:.2f}**
    - Harga tertinggi: **${price_stats['max_price']:.2f}**
    - Harga rata-rata: **${price_stats['mean_price']:.2f}**
    - Harga median: **${price_stats['median_price']:.2f}**
    """)
    
    
    # Platform comparison
    st.subheader("Perbandingan Antar Platform")
    
    # Average price by platform
    avg_price_by_platform = df.groupby('platform')['price'].mean().reset_index()
    avg_price_by_platform.columns = ['Platform', 'Harga Rata-rata']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Harga Rata-rata per Platform")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Platform', y='Harga Rata-rata', data=avg_price_by_platform, ax=ax)
        ax.set_title('Harga Rata-rata per Platform')
        ax.set_ylabel('Harga (USD)')
        st.pyplot(fig)
        
        # Add descriptive text for average price
        st.write("Harga rata-rata kursus per platform:")
        for _, row in avg_price_by_platform.iterrows():
            st.write(f"**{row['Platform']}**: ${row['Harga Rata-rata']:.2f}")
    
    # Average duration by platform
    avg_duration_by_platform = df.groupby('platform')['content_duration'].mean().reset_index()
    avg_duration_by_platform.columns = ['Platform', 'Durasi Rata-rata']
    
    with col2:
        st.subheader("Durasi Rata-rata per Platform")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Platform', y='Durasi Rata-rata', data=avg_duration_by_platform, ax=ax)
        ax.set_title('Durasi Rata-rata per Platform')
        ax.set_ylabel('Durasi (jam)')
        st.pyplot(fig)
        
        # Add descriptive text for average duration
        st.write("Durasi rata-rata kursus per platform:")
        for _, row in avg_duration_by_platform.iterrows():
            st.write(f"**{row['Platform']}**: {row['Durasi Rata-rata']:.1f} jam")


def show_dashboard_page():
    """Show dashboard page with course platform information"""
    st.title("Dashboard Kursus Online")
    
    if 'df' not in st.session_state:
        st.warning("Silakan tekan tombol mulai pada sidebar terlebih dahulu.")
        return

    # Initialize session state for pagination
    if 'page_dicoding' not in st.session_state:
        st.session_state.page_dicoding = 0
    if 'page_udemy' not in st.session_state:
        st.session_state.page_udemy = 0
    if 'page_coursera' not in st.session_state:
        st.session_state.page_coursera = 0
    
    ITEMS_PER_PAGE = 10
        
    def display_course_list(courses, platform_name, page_key):
        total_pages = len(courses) // ITEMS_PER_PAGE + (1 if len(courses) % ITEMS_PER_PAGE > 0 else 0)
        
        # Display platform header
        st.header(platform_name)
        
        # Display platform description and logo in a more compact layout
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.image(f"Foto/{platform_name.lower()}.{'jpeg' if platform_name == 'Dicoding' else 'png' if platform_name == 'Udemy' else 'svg'}", width=150)
        
        with col2:
            descriptions = {
                'Dicoding': "Dicoding adalah platform belajar digital asal Indonesia yang fokus menyediakan pelatihan di bidang teknologi, khususnya pemrograman, pengembangan aplikasi, kecerdasan buatan, dan cloud computing. Platform ini didesain untuk mendukung talenta digital Indonesia agar siap menghadapi kebutuhan industri. Dicoding bekerja sama dengan perusahaan global seperti Google, AWS, dan IBM untuk menyusun kelas-kelas yang relevan dan berkualitas. Materi disampaikan dalam bahasa Indonesia, sehingga mudah diakses oleh pemula maupun profesional. Selain itu, Dicoding juga menyediakan sistem sertifikasi yang diakui industri dan telah mendukung banyak program pemerintah seperti Kampus Merdeka dan Digital Talent Scholarship.",
                'Udemy': "Udemy adalah marketplace kursus online global yang memungkinkan siapa saja, baik individu maupun lembaga untuk membuat dan menjual kursus mereka secara bebas. Platform ini menawarkan ribuan kelas dalam berbagai topik, seperti pengembangan web, desain grafis, bisnis, fotografi, hingga keterampilan pengembangan diri. Keunggulan Udemy terletak pada fleksibilitas harga dan akses seumur hidup terhadap materi kursus yang telah dibeli. Banyak kursus di Udemy dijual dengan harga sangat terjangkau karena sering ada diskon besar. Walaupun kualitas materi bisa bervariasi tergantung pada instruktur, banyak kursus populer di Udemy telah terbukti efektif dan mendapat rating tinggi dari jutaan pengguna di seluruh dunia.",
                'Coursera': "Coursera adalah platform kursus online berskala internasional yang menyediakan akses pembelajaran dari universitas dan institusi ternama dunia, seperti Stanford University, University of London, Google, IBM, dan Meta. Kursus di Coursera mencakup berbagai bidang, mulai dari ilmu komputer, bisnis, kesehatan, hingga seni dan ilmu sosial. Platform ini menyediakan pilihan pembelajaran gratis dan berbayar, termasuk program sertifikat profesional hingga gelar sarjana atau magister secara online. Pembelajaran dilakukan dalam bahasa Inggris, namun banyak kursus menyediakan subtitle dalam berbagai bahasa, termasuk bahasa Indonesia. Coursera sangat cocok bagi pelajar dan profesional yang ingin memperdalam keahlian dengan kurikulum akademik dari institusi bereputasi tinggi.."
            }
            st.write(descriptions[platform_name])
        
        with col3:
            st.link_button(f"Kunjungi {platform_name}", f"https://www.{platform_name.lower()}.com/", use_container_width=True)
        
        # st.markdown("---")
        
        # Add table header
        st.subheader("Daftar Kursus yang Tersedia:")
        cols = st.columns([3, 1, 1, 1, 1, 1])
        cols[0].write("**Nama Kursus**")
        cols[1].write("**Level**")
        cols[2].write("**Durasi**")
        cols[3].write("**Kategori**")
        cols[4].write("**Harga**")
        # cols[5].write("**Aksi**")
        st.markdown("---")
        
        # Create DataFrame for display
        start_idx = st.session_state[page_key] * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        page_courses = courses.iloc[start_idx:end_idx]
        
        # Display each course in a container
        for _, course in page_courses.iterrows():
            with st.container():
                cols = st.columns([3, 1, 1, 1, 1, 1])
                with cols[0]:
                    st.write(course['course_title'])
                with cols[1]:
                    st.write(course['level'])
                with cols[2]:
                    duration = f"{course['content_duration']:.1f}h" if pd.notna(course['content_duration']) and course['content_duration'] > 0 else "N/A"
                    st.write(duration)
                with cols[3]:
                    st.write(course['subject'])
                with cols[4]:
                    if pd.notna(course['price']) and course['price'] > 0:
                        usd_price = int(course['price'])
                        idr_price = usd_price * 16000
                        st.write(f"${usd_price}\n(Rp {idr_price:,})")
                    else:
                        st.write("Free")
                with cols[5]:
                    if pd.notna(course['course_url']):
                        st.link_button("Daftar", course['course_url'], type="primary", use_container_width=True)
                    else:
                        st.button("Tidak Tersedia", disabled=True, use_container_width=True)
                st.markdown("---")
        
        # Pagination controls in a single line
        cols = st.columns([2, 1, 2])
        with cols[0]:
            if st.session_state[page_key] > 0:
                if st.button("← Previous", key=f"prev_{platform_name}", use_container_width=True):
                    st.session_state[page_key] -= 1
                    st.rerun()
        with cols[1]:
            st.write(f"Halaman {st.session_state[page_key] + 1} dari {total_pages}")
        with cols[2]:
            if st.session_state[page_key] < total_pages - 1:
                if st.button("Next →", key=f"next_{platform_name}", use_container_width=True):
                    st.session_state[page_key] += 1
                    st.rerun()
    
    # Display courses for each platform
    platforms = {
        'Dicoding': st.session_state.df[st.session_state.df['platform'] == 'Dicoding'],
        'Udemy': st.session_state.df[st.session_state.df['platform'] == 'Udemy'].sample(n=min(200, len(st.session_state.df[st.session_state.df['platform'] == 'Udemy']))),
        'Coursera': st.session_state.df[st.session_state.df['platform'] == 'Coursera'].sample(n=min(200, len(st.session_state.df[st.session_state.df['platform'] == 'Coursera'])))
    }
    
    for platform_name, courses in platforms.items():
        display_course_list(courses, platform_name, f'page_{platform_name.lower()}')
        st.markdown("---")

def show_welcome_message():
    """Show welcome message when system is not yet started"""
    st.title("📚 Selamat Datang di Sistem Rekomendasi Kursus Online")
    
    # Create three columns for platform logos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("Foto/udemy.png", caption="Udemy", use_container_width=True)
    with col2:
        st.image("Foto/coursera.svg", caption="Coursera", use_container_width=True)
    with col3:
        st.image("Foto/dicoding.jpeg", caption="Dicoding", use_container_width=True)
    
    st.markdown("""
    ### 🎯 Tentang Sistem
    Sistem ini dirancang untuk membantu Anda menemukan kursus online yang paling sesuai dengan kebutuhan belajar Anda. 
    Dengan menggunakan teknologi AI dan machine learning, sistem akan menganalisis preferensi Anda dan memberikan rekomendasi 
    kursus yang paling relevan dari berbagai platform seperti:
    
    - 🎓 **Coursera** - Kursus dari universitas dan institusi ternama dunia
    - 💻 **Dicoding** - Platform pembelajaran coding terkemuka di Indonesia
    - 📱 **Udemy** - Marketplace kursus online global dengan berbagai topik
    
    ### 🚀 Cara Memulai
    1. Klik tombol **"Mulai"** di sidebar untuk memuat sistem
    2. Pilih halaman yang ingin Anda jelajahi:
       - 📊 **Dashboard** - Lihat daftar lengkap kursus dari setiap platform
       - 🎯 **Rekomendasi** - Dapatkan rekomendasi kursus sesuai kebutuhan Anda
       - 📈 **Statistik** - Eksplorasi statistik dan insight dari dataset
    
    ### ⚡ Fitur Utama
    - 🔍 Pencarian kursus berdasarkan tujuan pembelajaran
    - 💡 Rekomendasi personal menggunakan AI
    - 📊 Analisis mendalam tentang kursus
    - 💰 Filter berdasarkan harga, level, durasi, dan platform
    - 🌐 Dukungan multi-bahasa (Indonesia & Inggris)
    """)

def main():
    """Main function for the Streamlit app"""
    # Page configuration
    st.set_page_config(
        page_title="Sistem Rekomendasi Kursus",
        page_icon="📚",
        layout="wide"
    )

    # Sidebar
    st.sidebar.title("Sistem Rekomendasi Kursus Online")
    
    # Add load model button to sidebar
    if st.sidebar.button("Mulai"):
        with st.spinner("Loading"):
            recommenders, df = load_or_train_models_fixed(
                "udemy_courses.csv",
                "dicoding_courses.csv",
                "coursera_courses.csv"
            )
            st.session_state.recommenders = recommenders
            st.session_state.df = df
    
    # Navigation
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["Dashboard", "Rekomendasi", "Statistik"]
    )
    
    # Check if system is started
    if 'recommenders' not in st.session_state or 'df' not in st.session_state:
        show_welcome_message()
        return
    
    # Display selected page
    if page == "Dashboard":
        show_dashboard_page()
    elif page == "Rekomendasi":
        show_recommendation_page_fixed()
    elif page == "Statistik":
        show_statistics_page()

if __name__ == "__main__":
    main() 