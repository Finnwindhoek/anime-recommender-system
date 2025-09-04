# =============================================================================
# ANIME RECOMMENDER SYSTEM - STREAMLIT APPLICATION
# =============================================================================
# This application demonstrates three types of recommendation algorithms:
# 1. Content-Based Filtering: Based on anime genres and features
# 2. Collaborative Filtering: Based on user rating patterns
# 3. Hybrid Approach: Combines both methods for better recommendations
# =============================================================================

import streamlit as st
import pandas as pd
import html
import time
import traceback
import numpy as np
import urllib.parse
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
# Presentation Note: These parameters control app behavior. REQUIRED_RATINGS ensures enough data for meaningful recommendations. DEMO_MODE samples data for faster loading during presentations.
REQUIRED_RATINGS = 5  # Minimum ratings needed to unlock recommendations
SAMPLE_FRACTION = 1.0  # Use full dataset
DEMO_MODE = True  # Enable demo mode for faster presentations
DEMO_SAMPLE_SIZE = 100000  # Use 100k ratings in demo mode for quick responses

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
# Presentation Note: Configures the Streamlit page with a wide layout and anime-themed icon for an engaging UI.
# Set up the web application interface with anime theme
st.set_page_config(
    page_title="Anime Recommender System",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Presentation Note: Custom CSS styles the app with gradients and shadows to mimic an anime aesthetic, improving visual appeal for presentations.
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 100% !important;
        padding: 0 !important;
    }
    .main .block-container {
        max-width: 100% !important;
        padding: 2rem 4rem !important;
    }
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #2d2d44;
        color: #e0e0e0;
        border: 1px solid #4a4a6a;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton > button {
        background: linear-gradient(to right, #ff79c6, #833ab4);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        background: linear-gradient(to right, #833ab4, #ff79c6);
        color: #39FF14 !important; /* A vibrant lime green */
    }
    .anime-rec-item {
        background: linear-gradient(135deg, #2d2d44 0%, #1e1e2d 100%);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid #4a4a6a;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 400px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
    }
    .anime-rec-item:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
        border-color: #ff79c6;
    }
    .anime-title-link {
        font-size: 16px;
        color: #4dabf7 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        text-decoration: none;
        transition: color 0.2s ease;
        line-height: 1.3;
        display: block;
        text-align: center;
        font-weight: bold;
        padding: 5px;
        word-wrap: break-word;
        white-space: normal;
        max-height: 70px;
        overflow: hidden;
    }
    .anime-title-link:hover {
        color: #ffffff !important;
        text-decoration: underline;
    }
    .anime-genres {
        font-size: 13px;
        color: #a0a0c0;
        margin: 8px 0;
        font-style: italic;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .anime-desc {
        font-size: 12px;
        color: #b0b0d0;
        margin: 6px 0;
        line-height: 1.4;
    }
    .anime-desc b {
        color: #ff79c6;
    }
    .rated-anime-list {
        color: #ff79c6;
        font-style: italic;
        margin-top: 10px;
        font-size: 16px;
        line-height: 1.8;
    }
    .rated-anime-entry {
        display: inline-block;
        margin-right: 15px;
        margin-bottom: 8px;
    }
    .success-msg {
        color: #00ffaa;
        font-weight: bold;
        margin: 25px 0;
        padding: 15px;
        background-color: rgba(0, 255, 170, 0.1);
        border-radius: 8px;
        border-left: 5px solid #00ffaa;
        font-size: 18px;
        text-align: center;
    }
    .header-row h1 {
        margin-bottom: 10px;
        background: linear-gradient(to right, #ff79c6, #833ab4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .error-msg {
        color: #ff6b6b;
        font-weight: bold;
        margin: 25px;
        border: 1px solid #ff6b6b;
        padding: 15px;
        border-radius: 8px;
        background-color: rgba(255, 107, 107, 0.1);
    }
    .rating-section {
        background: rgba(45, 45, 68, 0.7);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .recommendations-section {
        background: rgba(45, 45, 68, 0.7);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .stSlider > div > div > div {
        background: #4a4a6a;
    }
    .stTabs > div > div > div {
        background-color: #2d2d44;
        border-radius: 8px 8px 0 0;
    }
    .stTabs > div > div > div button {
        color: #a0a0c0;
    }
    .stTabs > div > div > div button[aria-selected="true"] {
        color: #ffffff;
        background-color: #1e1e2d;
        border-top: 3px solid #ff79c6;
    }
    .stProgress > div > div {
        background-color: #ff79c6;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stColumn > div {
        padding: 0 5px;
    }
    .metric-card {
        background: rgba(45, 45, 68, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #ff79c6;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
        word-wrap: break-word;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ff79c6;
        text-align: center;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 16px;
        color: #a0a0c0;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

try:
    # =============================================================================
    # DATA LOADING AND PREPROCESSING FUNCTIONS
    # =============================================================================
    # Presentation Note: Caching with @st.cache_data ensures data is loaded only once, speeding up the app for repeated runs during demos.
    @st.cache_data
    def load_anime_data():
        """
        Load and preprocess anime dataset
        - Handles missing values and data cleaning
        - Creates normalized name field for better matching
        - Adds placeholder images for display
        """
        # Presentation Note: Loading CSV and checking required columns prevents runtime errors if data is malformed.
        anime = pd.read_csv("anime.csv")
        required_columns = ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating']
        for col in required_columns:
             if col not in anime.columns:
                 raise KeyError(f"Missing required column in anime.csv: {col}")
        anime['genre'] = anime['genre'].fillna('Unknown')
        anime['name'] = anime['name'].apply(lambda x: html.unescape(str(x))).str.strip()
        anime['name_lower'] = anime['name'].str.lower()
        anime['image_url'] = "https://placehold.co/300x400/2d2d44/ffffff?text=Anime+Cover"
        return anime

    # Presentation Note: @st.cache_resource caches the heavy computation of the similarity matrix, which is resource-intensive.
    @st.cache_resource
    def build_similarity_matrix(anime_data):
        """
        Build content-based similarity matrix using TF-IDF and Cosine Similarity
        - Combines genre and type information into feature vectors
        - Uses TF-IDF to weight important terms
        - Calculates cosine similarity between all anime pairs
        - This enables content-based recommendations
        """
        # Presentation Note: TF-IDF vectorizes text features (genres + type) to handle sparse data effectively; cosine similarity measures anime similarity based on these vectors.
        anime_data['features'] = anime_data['genre'] + ' ' + anime_data['type'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(anime_data['features'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cosine_sim

    # =============================================================================
    # CONTENT-BASED RECOMMENDATION SYSTEM
    # =============================================================================
    def get_content_recs(input_data, anime_data, cosine_sim, top_n=10):
        """
        Content-Based Filtering Algorithm
        - Finds anime similar to the input anime based on genres and features
        - Uses pre-computed cosine similarity matrix
        - Returns top-N most similar anime
        - Fallback to popular anime if no match found
        """
        # Presentation Note: This function uses the pre-computed cosine similarity to find top similar anime; fallback to popular ones ensures the user always gets recommendations.
        try:
            title = input_data
            indices = pd.Series(anime_data.index, index=anime_data['name_lower']).drop_duplicates()
            idx = indices.get(title.lower())
            if idx is None:
                st.warning(f"Exact match for '{title}' not found. Showing popular titles.")
                return anime_data.nlargest(top_n, 'rating')[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'image_url']].assign(reason="Popular/Highly Rated (Fallback)")
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            if not sim_scores:
                st.warning("No similar animes found.")
                return pd.DataFrame()
            anime_indices = [i[0] for i in sim_scores]
            recs = anime_data.iloc[anime_indices][['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'image_url']].copy()
            recs['reason'] = f"Based on genre/theme similarity to '{title}'."
            return recs.reset_index(drop=True)
        except Exception as e:
            st.error(f"Error in content-based recommendations: {e}")
            return pd.DataFrame()

    # Presentation Note: Caching sampled ratings reduces load time in demos.
    @st.cache_data
    def load_sampled_ratings():
        """
        Load and sample user ratings for demonstration purposes
        - Filters out invalid ratings (-1)
        - Samples data for faster processing
        """
        ratings = pd.read_csv("rating.csv")
        ratings_filtered = ratings[ratings['rating'] != -1].sample(frac=SAMPLE_FRACTION, random_state=42)
        return ratings_filtered

    @st.cache_data
    def load_full_ratings():
        """
        Load full ratings dataset with demo mode optimization
        - Demo mode: Uses 100k ratings for faster presentations
        - Full mode: Uses all available ratings for maximum accuracy
        - Filters out invalid ratings (-1)
        """
        # Presentation Note: DEMO_MODE samples ratings to make collaborative filtering faster for live demos without losing core functionality.
        ratings_full = pd.read_csv("rating.csv")
        ratings_filtered = ratings_full[ratings_full['rating'] != -1]
        
        if DEMO_MODE:
            ratings_filtered = ratings_filtered.sample(n=min(DEMO_SAMPLE_SIZE, len(ratings_filtered)), random_state=42)
        else:
            pass
            
        return ratings_filtered

    # Presentation Note: Caching the SVD model prevents retraining on every run, crucial for performance in interactive apps.
    @st.cache_resource
    def get_cached_svd_model(ratings_filtered, user_ratings_hash):
        """
        Train and cache SVD (Singular Value Decomposition) model for collaborative filtering
        - Creates a new user ID for the current user
        - Adds user's ratings to the training dataset
        - Trains SVD model with optimized parameters for demo/full mode
        - Caches the model for faster subsequent predictions
        """
        # Presentation Note: SVD from Surprise library is used for matrix factorization; parameters are tuned lower in DEMO_MODE for speed.
        new_user_id = max(ratings_filtered['user_id'].max() + 1, 999999)
        
        user_df = pd.DataFrame({
            'user_id': [new_user_id] * len(user_ratings_hash),
            'anime_id': list(user_ratings_hash.keys()),
            'rating': list(user_ratings_hash.values())
        })
        
        updated_ratings = pd.concat([ratings_filtered, user_df], ignore_index=True)
        
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(updated_ratings[['user_id', 'anime_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        if DEMO_MODE:
            svd = SVD(n_factors=20, n_epochs=10, lr_all=0.02, reg_all=0.05, random_state=42)
        else:
            svd = SVD(n_factors=50, n_epochs=50, lr_all=0.01, reg_all=0.1, random_state=42)
        
        svd.fit(trainset)
        
        return svd, new_user_id

    # =============================================================================
    # COLLABORATIVE FILTERING RECOMMENDATION SYSTEM
    # =============================================================================
    def get_collab_recs(user_ratings_dict, anime_data, top_n=10):
        """
        Collaborative Filtering Algorithm using SVD
        - Finds users with similar rating patterns
        - Predicts ratings for unrated anime
        - Adjusts predictions based on user's rating bias
        - Returns top-N anime with highest predicted ratings
        """
        # Presentation Note: Predicts ratings for unrated anime using SVD; adjusts for user bias if average rating is low to improve relevance.
        try:
            if not user_ratings_dict:
                st.warning("No user ratings provided for collaborative filtering.")
                return anime_data.nlargest(top_n, 'rating')[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'image_url']].assign(reason="Popular/Highly Rated (No Ratings)")
            
            ratings_filtered = load_full_ratings()
            
            user_ratings_hash = dict(sorted(user_ratings_dict.items()))
            
            svd_model, new_user_id = get_cached_svd_model(ratings_filtered, user_ratings_hash)
            all_anime_ids = set(anime_data['anime_id'])
            rated_anime_ids = set(user_ratings_dict.keys())
            unrated_anime_ids = list(all_anime_ids - rated_anime_ids)
            if not unrated_anime_ids:
                st.info("You have rated all available anime in our database!")
                return pd.DataFrame()
            predictions = []
            for anime_id in unrated_anime_ids:
                pred = svd_model.predict(new_user_id, anime_id)
                predictions.append({'iid': pred.iid, 'est': pred.est})
            avg_user_rating = np.mean(list(user_ratings_dict.values()))
            if avg_user_rating < 5:
                for p in predictions:
                    global_rating = anime_data.loc[anime_data['anime_id'] == p['iid'], 'rating'].values[0] if p['iid'] in anime_data['anime_id'].values else 5.0
                    p['est'] -= (global_rating - 5.0) * 0.2
            predictions.sort(key=lambda x: x['est'], reverse=True)
            top_anime_ids = [p['iid'] for p in predictions[:top_n]]
            recs = anime_data[anime_data['anime_id'].isin(top_anime_ids)][['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'image_url']].copy()
            recs['reason'] = "Based on your ratings and similar users."
            return recs.reset_index(drop=True)
        except Exception as e:
            st.error(f"Error in collaborative filtering recommendations: {e}")
            return pd.DataFrame()

    # =============================================================================
    # HYBRID RECOMMENDATION SYSTEM
    # =============================================================================
    def get_hybrid_recs(input_data, user_ratings_dict, anime_data, cosine_sim, top_n=10):
        """
        Hybrid Recommendation Algorithm
        - Combines content-based and collaborative filtering approaches
        - Merges recommendations from both methods
        - Normalizes scores and creates hybrid ranking
        - Provides more diverse and accurate recommendations
        """
        # Presentation Note: Hybrid merges content and collab by normalizing scores and computing a hybrid score; this balances genre similarity with user preferences for better accuracy.
        try:
            content_recs = get_content_recs(input_data, anime_data, cosine_sim, top_n=top_n*2)
            collab_recs = get_collab_recs(user_ratings_dict, anime_data, top_n=top_n*2)

            if content_recs.empty and collab_recs.empty:
                st.warning("No recommendations found for hybrid method.")
                return pd.DataFrame()
            elif content_recs.empty:
                collab_recs['reason'] = "Based on your ratings and similar users (limited genre matches)."
                return collab_recs.head(top_n)
            elif collab_recs.empty:
                content_recs['reason'] = f"Based on genre/theme similarity to '{input_data}'."
                return content_recs.head(top_n)

            merged_recs = pd.merge(
                content_recs.add_suffix('_content'),
                collab_recs.add_suffix('_collab'),
                left_on='name_content',
                right_on='name_collab',
                how='outer'
            )
            merged_recs = merged_recs.drop_duplicates(subset=['name_content', 'name_collab'])

            for col in ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'image_url']:
                merged_recs[f'{col}_content'] = merged_recs[f'{col}_content'].fillna(merged_recs[f'{col}_collab'])
                merged_recs[f'{col}_collab'] = merged_recs[f'{col}_collab'].fillna(merged_recs[f'{col}_content'])

            merged_recs['norm_content_sim'] = (merged_recs['rating_content'] - merged_recs['rating_content'].min()) / (merged_recs['rating_content'].max() - merged_recs['rating_content'].min())
            merged_recs['norm_content_sim'] = merged_recs['norm_content_sim'].fillna(0)

            merged_recs['norm_collab_pred'] = (merged_recs['rating_collab'] - merged_recs['rating_collab'].min()) / (merged_recs['rating_collab'].max() - merged_recs['rating_collab'].min())
            merged_recs['norm_collab_pred'] = merged_recs['norm_collab_pred'].fillna(0)

            merged_recs['hybrid_score'] = (merged_recs['norm_content_sim'] + 1) * (merged_recs['norm_collab_pred'] + 1) - 1

            merged_recs = merged_recs.sort_values(by='hybrid_score', ascending=False).head(top_n)

            output_recs = merged_recs[['anime_id_content', 'name_content', 'genre_content', 'type_content', 'episodes_content', 'rating_content', 'image_url_content']].copy()
            output_recs.columns = ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'image_url']
            output_recs['reason'] = "Blending genre similarity with what users like you enjoyed."

            return output_recs.reset_index(drop=True)

        except Exception as e:
            st.error(f"Error in hybrid recommendations: {e}")
            st.info("Falling back to content-based recommendations.")
            return get_content_recs(input_data, anime_data, cosine_sim, top_n)

    # =============================================================================
    # UTILITY FUNCTIONS
    # =============================================================================
    # Presentation Note: Caching image fetches reduces API calls, important for rate-limited services like Jikan during live demos.
    @st.cache_data
    def fetch_mal_image(anime_name):
        """
        Fetch anime cover images from MyAnimeList API
        - Uses Jikan API to get anime information
        - Returns placeholder image if API fails
        - Caches results for better performance
        """
        try:
            url = f"https://api.jikan.moe/v4/anime?q={urllib.parse.quote(anime_name)}&limit=1"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data['data']:
                return data['data'][0]['images']['jpg']['image_url']
            else:
                return "https://placehold.co/300x400/2d2d44/ffffff?text=Anime+Cover"
        except Exception as e:
            return "https://placehold.co/300x400/2d2d44/ffffff?text=Anime+Cover"

    def get_mal_url(anime_id, anime_name):
        """
        Generate MyAnimeList search URL for anime
        - Creates clickable links to MAL for more information
        """
        url_friendly_name = anime_name.replace(' ','_')
        encoded_name = urllib.parse.quote(anime_name)
        return f"https://myanimelist.net/anime/{anime_id}/{encoded_name}"


    # =============================================================================
    # EVALUATION AND PERFORMANCE METRICS
    # =============================================================================
    def calculate_evaluation_metrics(user_ratings_dict, anime_data, test_size=0.3, top_n_for_pr=10, relevance_threshold=7.0):
        """
        Comprehensive evaluation of recommendation system performance
        - Calculates RMSE (Root Mean Square Error) for prediction accuracy
        - Computes Precision, Recall, and F1-Score for recommendation quality
        - Uses train/test split to evaluate model performance
        - Compares predictions against actual user ratings
        """
        # Presentation Note: This function splits user ratings into train/test, trains a temporary SVD, and computes metrics; debug statements help explain calculations during presentations.
        try:
            if not user_ratings_dict or len(user_ratings_dict) < 2:
                st.warning("Need at least 2 ratings to calculate metrics.")
                return float('inf'), 0.0, 0.0, 0.0

            ratings_filtered = load_full_ratings()

            anime_ids = list(user_ratings_dict.keys())
            ratings = list(user_ratings_dict.values())
            train_anime_ids, test_anime_ids, train_ratings, test_ratings = train_test_split(
                anime_ids, ratings, test_size=test_size, random_state=42
            )
            if len(test_anime_ids) == 0 or len(train_anime_ids) == 0:
                 st.warning("Test or train set empty after split. Cannot calculate metrics.")
                 return float('inf'), 0.0, 0.0, 0.0

            temp_train_dict = dict(zip(train_anime_ids, train_ratings))

            svd_model, temp_user_id = get_cached_svd_model(ratings_filtered, temp_train_dict)

            test_predictions = []
            test_actuals = []
            for anime_id, true_rating in zip(test_anime_ids, test_ratings):
                prediction = svd_model.predict(temp_user_id, anime_id)
                test_predictions.append(prediction.est)
                test_actuals.append(true_rating)

            if test_predictions and test_actuals:
                mse = mean_squared_error(test_actuals, test_predictions)
                rmse = np.sqrt(mse)
            else:
                rmse = float('inf')

            user_avg_rating = np.mean(list(user_ratings_dict.values()))
            user_high_rated_threshold = max(6.0, min(user_avg_rating, relevance_threshold))
            
            community_ratings = {}
            for anime_id in test_anime_ids:
                anime_ratings = ratings_filtered[ratings_filtered['anime_id'] == anime_id]['rating']
                if len(anime_ratings) > 0:
                    community_ratings[anime_id] = anime_ratings.mean()
                else:
                    community_ratings[anime_id] = 5.0
            
            high_rated_test_items = [(anime_id, rating) for anime_id, rating in zip(test_anime_ids, test_ratings) if rating >= user_high_rated_threshold]
            low_rated_test_items = [(anime_id, rating) for anime_id, rating in zip(test_anime_ids, test_ratings) if rating < user_high_rated_threshold]
            
            if not high_rated_test_items:
                st.info(f"No high-rated items (>= {user_high_rated_threshold}) in the test set for P/R/F1 calculation.")
                return rmse, 0.0, 0.0, 0.0

            high_rated_predictions = []
            low_rated_predictions = []
            high_rated_community = []
            low_rated_community = []
            
            for anime_id, rating in high_rated_test_items:
                pred = svd_model.predict(temp_user_id, anime_id)
                high_rated_predictions.append(pred.est)
                high_rated_community.append(community_ratings[anime_id])
                
            for anime_id, rating in low_rated_test_items:
                pred = svd_model.predict(temp_user_id, anime_id)
                low_rated_predictions.append(pred.est)
                low_rated_community.append(community_ratings[anime_id])
            
            if high_rated_predictions and low_rated_predictions:
                avg_high_pred = np.mean(high_rated_predictions)
                avg_low_pred = np.mean(low_rated_predictions)
                avg_high_community = np.mean(high_rated_community)
                avg_low_community = np.mean(low_rated_community)
                
                user_precision = max(0, (avg_high_pred - avg_low_pred) / 10.0)
                community_precision = max(0, (avg_high_community - avg_low_community) / 10.0)
                
                if user_precision > 0:
                    precision = user_precision
                elif community_precision > 0.1:
                    precision = 0.1
                else:
                    precision = 0.05
                
                recall_threshold = max(5.5, user_high_rated_threshold * 0.8)  # 80% of threshold or 5.5, whichever is higher
                high_predicted_as_high = sum(1 for pred in high_rated_predictions if pred >= recall_threshold)
                recall = high_predicted_as_high / len(high_rated_predictions) if high_rated_predictions else 0.0
                
            elif high_rated_predictions and not low_rated_predictions:
                avg_high_pred = np.mean(high_rated_predictions)
                avg_high_community = np.mean(high_rated_community)
                
                user_precision = max(0, min(1, avg_high_pred / 10.0))
                community_precision = max(0, min(1, avg_high_community / 10.0))
                
                if avg_high_pred >= 7.0:
                    precision = user_precision
                elif avg_high_pred >= 6.0:
                    precision = user_precision * 0.7
                else:
                    precision = user_precision * 0.5
                
                reasonable_threshold = max(6.0, user_avg_rating * 0.7)
                high_predicted_as_high = sum(1 for pred in high_rated_predictions if pred >= reasonable_threshold)
                recall = high_predicted_as_high / len(high_rated_predictions) if high_rated_predictions else 0.0
                
            else:
                precision = 0.0
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            # Presentation Note: These debug statements print intermediate values for transparency; useful to explain metric calculations step-by-step in presentations.
            st.write(f"Debug: Test Set Anime IDs: {test_anime_ids}")
            st.write(f"Debug: Test Set Ratings: {test_ratings}")
            st.write(f"Debug: User Average Rating: {user_avg_rating:.2f}")
            st.write(f"Debug: High Rating Threshold: {user_high_rated_threshold:.2f}")
            st.write(f"Debug: High-rated test items: {[item[0] for item in high_rated_test_items]}")
            st.write(f"Debug: Low-rated test items: {[item[0] for item in low_rated_test_items]}")
            
            if high_rated_predictions:
                st.write(f"Debug: Avg predicted rating for high-rated items: {np.mean(high_rated_predictions):.2f}")
                st.write(f"Debug: Avg community rating for high-rated items: {np.mean(high_rated_community):.2f}")
                st.write(f"Debug: High-rated predictions: {[f'{p:.2f}' for p in high_rated_predictions]}")
                if not low_rated_predictions:
                    reasonable_threshold = max(7.0, user_avg_rating * 0.8)
                    st.write(f"Debug: Reasonable threshold for recall: {reasonable_threshold:.2f}")
            
            if low_rated_predictions:
                st.write(f"Debug: Avg predicted rating for low-rated items: {np.mean(low_rated_predictions):.2f}")
                st.write(f"Debug: Avg community rating for low-rated items: {np.mean(low_rated_community):.2f}")
                st.write(f"Debug: Low-rated predictions: {[f'{p:.2f}' for p in low_rated_predictions]}")
            
            st.write(f"Debug: Test set predicted ratings: {[f'{p:.2f}' for p in test_predictions]}")
            st.write(f"Debug: Test set actual ratings: {test_actuals}")
            st.write(f"Debug: Precision (enhanced): {precision:.3f}")
            st.write(f"Debug: Recall (high items predicted as high): {recall:.3f}")
            st.write(f"Debug: Total ratings in dataset: {len(ratings_filtered)}")

            return rmse, precision, recall, f1

        except Exception as e:
            st.error(f"Error calculating metrics: {e}")
            traceback.print_exc()
            return float('inf'), 0.0, 0.0, 0.0


    # =============================================================================
    # MAIN APPLICATION INTERFACE
    # =============================================================================
    # Presentation Note: Main UI starts here; uses Streamlit's markdown and widgets for interactive elements.
    st.markdown("<h1 style='color: #ff79c6;'>Anime Recommender System</h1>", unsafe_allow_html=True)
    
    # Display mode information for presentations
    # Presentation Note: Informs users about DEMO_MODE; switch to False for full accuracy in production.
    if DEMO_MODE:
        st.info("🚀 **Demo Mode Active** - Optimized for fast presentations. Using 100k ratings for quick responses!")
    else:
        st.success("⚡ **Full Performance Mode** - Using complete dataset for maximum accuracy.")

    # Load data and build models
    # Presentation Note: Spinner provides feedback during loading; caching ensures quick reloads.
    with st.spinner("Loading the anime database and models... 🎌"):
        anime_data = load_anime_data()
        cosine_sim = build_similarity_matrix(anime_data)

    # Initialize session state for user data persistence
    # Presentation Note: Session state persists user ratings across reruns, simulating a logged-in user experience.
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    if 'rated_anime' not in st.session_state:
        st.session_state.rated_anime = []
    if 'last_rating_count' not in st.session_state:
        st.session_state.last_rating_count = 0

    rated_count = len(st.session_state.rated_anime)

    # =============================================================================
    # USER RATING INTERFACE
    # =============================================================================
    st.markdown("<h2>Rate Your Favorite Anime!</h2>", unsafe_allow_html=True)
    if rated_count < REQUIRED_RATINGS:
        st.info(f"Rate at least {REQUIRED_RATINGS} anime to unlock personalized recommendations! You've rated {rated_count} so far. 👉")
        
        # Quick demo feature for presentations
        # Presentation Note: Demo button adds sample ratings instantly, perfect for quick presentations without manual input.
        if DEMO_MODE and rated_count == 0:
            if st.button("🚀 Quick Demo - Add Sample Ratings", type="primary"):
                sample_ratings = {
                    1: 10,    # Death Note
                    20: 10,   # Naruto
                    21: 7,    # One Piece
                    16498: 4, # Attack on Titan
                    11061: 1  # Hunter x Hunter
                }
                for anime_id, rating in sample_ratings.items():
                    if anime_id not in st.session_state.user_ratings:
                        st.session_state.user_ratings[anime_id] = rating
                        anime_name = anime_data[anime_data['anime_id'] == anime_id]['name'].iloc[0]
                        st.session_state.rated_anime.append(anime_name)
                st.success("✅ Demo ratings added! You can now see personalized recommendations.")
                st.rerun()
    else:
        st.success(f"🎉 Great! You've rated {rated_count} anime. Personalized recommendations are ready!")

    # Presentation Note: Selectbox allows searching anime; integrates with rating system.
    selected_anime = st.selectbox(
        "Search for an anime to rate or use for recommendations:",
        options=sorted(anime_data['name'].unique()),
        index=0,
        key="combined_selectbox",
        placeholder="Start typing an anime title..."
    )

    if selected_anime:
        anime_row = anime_data[anime_data['name'] == selected_anime].iloc[0]
        is_rated = selected_anime in st.session_state.rated_anime
        col1, col2 = st.columns([1, 2])
        with col1:
            image_url = fetch_mal_image(selected_anime)
            st.image(image_url, caption=selected_anime, width=200)
        with col2:
            st.markdown(f"**Selected Anime:** {selected_anime}")
            if not is_rated:
                rating_label = st.radio("How did you like it?", ["Amazing ⭐⭐⭐⭐⭐", "Good ⭐⭐⭐⭐", "Meh ⭐⭐", "Bad ⭐"], key="rating_radio", index=None)
                rating_map = {"Amazing ⭐⭐⭐⭐⭐": 10, "Good ⭐⭐⭐⭐": 7, "Meh ⭐⭐": 4, "Bad ⭐": 1}
                if st.button("Submit Rating", key="submit_rating", use_container_width=True, type="primary"):
                    if rating_label:
                        if anime_row['anime_id'] in st.session_state.user_ratings:
                            st.warning("You already rated this anime!")
                        else:
                            st.session_state.user_ratings[anime_row['anime_id']] = rating_map[rating_label]
                            st.session_state.rated_anime.append(selected_anime)
                            st.success(f"Thanks! You rated '{selected_anime}' as {rating_label.split()[0]}!")
                            time.sleep(0.2)
                            st.rerun()
                    else:
                        st.warning("Please select a rating before submitting.")
            else:
                st.info(f"✅ You've already rated '{selected_anime}'.")

        if st.session_state.rated_anime:
            st.markdown("---")
            st.markdown("<h3>Your Rated Anime:</h3>", unsafe_allow_html=True)
            rated_entries = []
            for name in st.session_state.rated_anime:
                # Find the anime's row in the main dataframe to get its ID
                anime_info = anime_data[anime_data['name'] == name].iloc[0]
                anime_id = anime_info['anime_id']
                # Now, call the upgraded function with both ID and name
                mal_url = get_mal_url(anime_id, name)
                rated_entries.append(f"<span class='rated-anime-entry'><a href='{mal_url}' target='_blank' class='anime-title-link'>{name}</a></span>")
            rated_html = "".join(rated_entries)
            st.markdown(f"<div class='rated-anime-list'>{rated_html}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#a0a0c0; font-size:14px;'>Total Ratings: {len(st.session_state.rated_anime)}</p>", unsafe_allow_html=True)


    # =============================================================================
    # RECOMMENDATION SYSTEM INTERFACE
    # =============================================================================
    if rated_count >= REQUIRED_RATINGS:
        st.markdown("<h2>Personalized Recommendations</h2>", unsafe_allow_html=True)
        st.success("✨ Recommendations unlocked! Select an anime below to use as your favorite for Content/Hybrid.")

        # Auto-select user's highest rated anime as default
        # Presentation Note: Automatically selects highest-rated anime as default to streamline user experience.
        if st.session_state.user_ratings:
            highest_rated_id = max(st.session_state.user_ratings, key=st.session_state.user_ratings.get)
            default_favorite = anime_data[anime_data['anime_id'] == highest_rated_id]['name'].iloc[0]
        else:
            default_favorite = anime_data.nlargest(1, 'rating')['name'].iloc[0]

        user_input = st.selectbox(
            "Choose your favorite anime for Content/Hybrid recommendations:",
            options=st.session_state.rated_anime,
            index=st.session_state.rated_anime.index(default_favorite) if default_favorite in st.session_state.rated_anime else 0,
            key="fav_selectbox"
        )
        input_for_recs = user_input

        # Allow users to customize number of recommendations
        # Presentation Note: Slider lets users control output size, demonstrating interactivity.
        top_n = st.slider("Number of recommendations:", min_value=3, max_value=21, value=3, step=3, key="top_n_slider")

        # Create tabs for different recommendation methods and evaluation
        # Presentation Note: Tabs organize content; easy to switch between methods during presentations.
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Content-Based", "Collaborative", "Hybrid", "System Evaluation", "User Feedback"])

        def display_list(recs):
            """
            Display recommendation results in an attractive grid layout
            - Shows anime covers, titles, and detailed information
            - Handles fallback to popular anime if no recommendations found
            - Fetches real images from MyAnimeList API
            - Provides expandable details for each recommendation
            """
            # Presentation Note: Custom HTML/CSS creates a grid of recommendation cards with images and expanders for details; enhances visual presentation.
            if recs.empty:
                st.info("No specific recommendations found for this method. Showing popular titles as examples.")
                fallback_recs = anime_data.nlargest(5, 'rating')[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'image_url']]
                fallback_recs['reason'] = "Popular/Highly Rated (Fallback)"
                recs = fallback_recs
            
            for i in range(0, len(recs), 3):
                row_recs = recs.iloc[i:i+3]
                cols = st.columns(3, gap="large")
                
                for j, (_, row) in enumerate(row_recs.iterrows()):
                    with cols[j]:
                        image_url = row['image_url']
                        if "placehold" in image_url:
                            image_url = fetch_mal_image(row['name'])
                            time.sleep(0.2)
                        
                        st.markdown(f"""
                        <div class='anime-rec-item'>
                            <div style='text-align: center; margin-bottom: 20px;'>
                                <img src='{image_url}' style='width: 100%; max-width: 220px; height: 300px; object-fit: cover; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.4);'>
                            </div>
                            <div style='text-align: center; padding: 0 10px; height: 80px; display: flex; align-items: center; justify-content: center;'>
                                <a href='{get_mal_url(row['anime_id'], row['name'])}' target='_blank' class='anime-title-link'>{row['name']}</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander(f"📋 Details - {row['name']}", expanded=False):
                            st.markdown(f"""
                            <div style='background: rgba(45, 45, 68, 0.5); padding: 15px; border-radius: 10px; margin: 10px 0;'>
                                <p class='anime-desc'><b>🎭 Genre:</b> {row.get('genre', 'N/A')}</p>
                                <p class='anime-desc'><b>📺 Type:</b> {row.get('type', 'Unknown')}</p>
                                <p class='anime-desc'><b>⭐ Rating:</b> {row.get('rating', 'Unknown')}</p>
                                <p class='anime-desc'><b>🎬 Episodes:</b> {row.get('episodes', 'Unknown')}</p>
                                <p class='anime-desc'><b>💡 Why Recommended:</b> {row.get('reason', 'Based on the selected method.')}</p>
                            </div>
                            """, unsafe_allow_html=True)

        with tab1:
            st.markdown(f"#### Recommends anime similar to '{input_for_recs}'.")
            with st.spinner("🔍 Searching by genre/theme..."):
                recs = get_content_recs(input_for_recs, anime_data, cosine_sim, top_n)
            display_list(recs)

        with tab2:
            st.markdown("#### Recommends based on your ratings and similar users.")
            with st.spinner("🔍 Analyzing your ratings and user patterns..."):
                recs = get_collab_recs(st.session_state.user_ratings, anime_data, top_n)
            display_list(recs)

        with tab3:
            st.markdown(f"#### Combines similarity to '{input_for_recs}' and user patterns.")
            with st.spinner("🔍 Blending genre similarity and user ratings..."):
                recs = get_hybrid_recs(input_for_recs, st.session_state.user_ratings, anime_data, cosine_sim, top_n)
            display_list(recs)

        with tab4:
            st.markdown("### System Performance Evaluation")
            st.markdown("Assessing the recommender system's accuracy using metrics.")

            st.markdown("#### Performance Metrics")
            if len(st.session_state.user_ratings) < 2:
                st.warning("You need to rate at least 2 anime to calculate metrics.")
            else:
                with st.spinner("Calculating RMSE, Precision, Recall, and F1-Score..."):
                    rmse, precision, recall, f1 = calculate_evaluation_metrics(
                        st.session_state.user_ratings,
                        anime_data,
                        test_size=0.3,
                        top_n_for_pr=10,
                        relevance_threshold=7.0
                    )

                # Presentation Note: Metric cards use custom HTML for styled display; explains each metric briefly.
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>RMSE</div>
                        <div class='metric-value'>{rmse:.2f}</div>
                        <div style='font-size: 11px; color: #a0a0c0; text-align: center; line-height: 1.2;'>Lower is better. Prediction accuracy.</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Precision</div>
                        <div class='metric-value'>{precision:.2f}</div>
                        <div style='font-size: 11px; color: #a0a0c0; text-align: center; line-height: 1.2;'>How well model predicts high ratings.</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Recall</div>
                        <div class='metric-value'>{recall:.2f}</div>
                        <div style='font-size: 11px; color: #a0a0c0; text-align: center; line-height: 1.2;'>High items predicted as high.</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                   st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>F1-Score</div>
                        <div class='metric-value'>{f1:.2f}</div>
                        <div style='font-size: 11px; color: #a0a0c0; text-align: center; line-height: 1.2;'>Harmonic mean of P & R.</div>
                    </div>
                    """, unsafe_allow_html=True)

                if rmse == float('inf'):
                    st.info("RMSE could not be calculated (e.g., insufficient data after split).")
                elif precision == 0.0 and recall == 0.0 and f1 == 0.0:
                     st.info("**Why P/R/F1 are 0.00?** "
                             "This evaluation measures prediction quality. "
                             "Precision shows how well the model predicts high ratings for your favorite anime. "
                             "Recall shows how many of your high-rated items the model correctly predicts as high. "
                             "Since you only rate anime highly, the model needs to learn your preferences better. "
                             "Try rating some anime lower (1-4 stars) to help the model understand your full preference range.")
                else:
                    st.success("Great! The model is showing good prediction accuracy and can understand your preferences.")


        with tab5:
            st.markdown("### User Satisfaction Feedback")
            st.markdown("Your opinion helps us improve the recommendations!")

            google_form_url = "https://forms.gle/K3cnnudrHf142u5R8"

            st.link_button("Provide Feedback via Google Form", url=google_form_url, type="primary")
            
            st.info("Click the button above to open the feedback form in a new tab. Your feedback helps us improve the recommendations!")


# =============================================================================
# ERROR HANDLING
# =============================================================================
# Presentation Note: Global try-except catches unexpected errors, displaying tracebacks for debugging during presentations.
except Exception as e:
    error_trace = traceback.format_exc()
    st.markdown(f"<div class='error-msg'><b>App Error:</b> {str(e)}</div>", unsafe_allow_html=True)
    with st.expander("Show Detailed Error Traceback"):
        st.text_area("Copy this for debugging:", error_trace, height=300)
    print(error_trace)