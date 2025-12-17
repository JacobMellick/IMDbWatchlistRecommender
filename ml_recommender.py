import pandas as pd
import requests
import json
import os
import numpy as np
import time
from dotenv import load_dotenv

# ML Imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error

# -- CONFIGURATION --
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
WATCHLIST_PATH = "watchlist.csv"
RATINGS_PATH = "ratings.csv"
CACHE_FILE = "omdb_cache.json"


class MLMovieRecommender:
    def __init__(self):
        self.watchlist = pd.DataFrame()
        self.ratings = pd.DataFrame()
        self.omdb_cache = self.load_cache()
        self.model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)

        # NLP Vectorizer for Plot
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=300)
        self.mlb_genre = MultiLabelBinarizer()

    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with open(CACHE_FILE, 'w') as f:
            json.dump(self.omdb_cache, f)

    def fetch_omdb_data(self, imdb_id):
        if imdb_id in self.omdb_cache:
            return self.omdb_cache[imdb_id]

        # Rate limiting sleep
        time.sleep(0.1)

        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'True':
                    self.omdb_cache[imdb_id] = data
                    return data
        except Exception:
            pass
        return {}

    def clean_money(self, value):
        """Converts '$32,181,230' to integer 32181230"""
        if not isinstance(value, str) or value == 'N/A':
            return 0
        return int(value.replace('$', '').replace(',', ''))

    def parse_metascore(self, value):
        if not isinstance(value, str) or value == 'N/A':
            return 0
        return int(value)

    def prepare_features(self, df, is_training=False):
        """
        Transforms raw DataFrame into a Matrix of features for the ML model.
        This is where we extract OMDb info.
        """
        print(
            f"Enriching and processing features for {len(df)} items... (This may take time)")

        # Lists to hold extracted data
        plots = []
        metascores = []
        box_offices = []
        directors_avg_scores = []  # Feature Engineering: Director Track Record

        # We need a dictionary of Director -> Avg Rating from the TRAINING set
        if is_training:
            # Create a lookup for director averages
            self.director_stats = df.groupby(
                'Directors')['Your Rating'].mean().to_dict()

        for idx, row in df.iterrows():
            imdb_id = row['Const']
            omdb = self.fetch_omdb_data(imdb_id)

            # 1. Text Data
            plots.append(omdb.get('Plot', ''))

            # 2. Numerical Data from OMDb
            metascores.append(self.parse_metascore(
                omdb.get('Metascore', 'N/A')))
            box_offices.append(self.clean_money(omdb.get('BoxOffice', 'N/A')))

        # Save cache after loop
        self.save_cache()

        # --- FEATURE CONSTRUCTION ---

        # 1. Base Numerical Features from CSV
        features = df[[
            'Year', 'Runtime (mins)', 'IMDb Rating', 'Num Votes']].copy()
        features['Metascore'] = metascores
        # Log transform BoxOffice to reduce skew
        features['BoxOffice'] = np.log1p(box_offices)
        features = features.fillna(0)

        # 2. Genre (One Hot Encoding)
        # Parse "Action, Drama" -> ["Action", "Drama"]
        genres_list = df['Genres'].astype(str).str.split(', ')
        if is_training:
            genres_encoded = pd.DataFrame(self.mlb_genre.fit_transform(genres_list),
                                          columns=self.mlb_genre.classes_, index=df.index)
        else:
            # Handle unknown genres in watchlist gracefully
            genres_encoded = pd.DataFrame(self.mlb_genre.transform(genres_list),
                                          columns=self.mlb_genre.classes_, index=df.index)

        # 3. Plot (TF-IDF)
        if is_training:
            plot_vectors = self.tfidf.fit_transform(plots)
        else:
            plot_vectors = self.tfidf.transform(plots)

        plot_df = pd.DataFrame(plot_vectors.toarray(), index=df.index)
        # Rename columns to avoid collisions
        plot_df.columns = [f"word_{i}" for i in range(plot_df.shape[1])]

        # 4. Director "Target Encoding"
        # Map the director to the average rating
        global_avg = 7.0  # Fallback
        if hasattr(self, 'director_stats'):
            features['Director_Score'] = df['Directors'].map(
                self.director_stats).fillna(global_avg)
        else:
            features['Director_Score'] = global_avg

        # Combine all features
        X = pd.concat([features, genres_encoded, plot_df], axis=1)

        # Clean column names for ML model
        X.columns = X.columns.astype(str)

        return X

    def run(self):
        # 1. LOAD DATA
        print("Loading CSVs...")
        ratings = pd.read_csv(RATINGS_PATH)
        ratings = ratings[ratings['Title Type'] == 'Movie'].copy()

        # 2. PREPARE TRAINING DATA
        # We need to drop movies with no rating (just in case)
        ratings = ratings.dropna(subset=['Your Rating'])
        y = ratings['Your Rating']

        print(
            f"Training on {len(ratings)} rated movies. (Fetching OMDb data if not cached...)")
        X_train = self.prepare_features(ratings, is_training=True)

        # 3. TRAIN MODEL
        print("Training Gradient Boosting Model...")
        X_t, X_v, y_t, y_v = train_test_split(
            X_train, y, test_size=0.2, random_state=42)
        self.model.fit(X_t, y_t)

        # Validate
        preds = self.model.predict(X_v)
        mae = mean_absolute_error(y_v, preds)
        print(f"Model Trained. Average Prediction Error: +/- {mae:.2f} stars")

        # Retrain on full data for best results
        self.model.fit(X_train, y)

        # 4. PREPARE WATCHLIST
        watchlist = pd.read_csv(WATCHLIST_PATH)
        watchlist = watchlist[watchlist['Title Type'] == 'Movie'].copy()

        print(f"Predicting ratings for {len(watchlist)} watchlist items...")
        X_watchlist = self.prepare_features(watchlist, is_training=False)

        # 5. PREDICT
        watchlist['Predicted_Rating'] = self.model.predict(X_watchlist)

        # 6. RESULTS
        results = watchlist.sort_values(
            by='Predicted_Rating', ascending=False).head(15)

        print("\n==========================================")
        print("       AI RECOMMENDED MOVIES       ")
        print("==========================================\n")

        for i, row in results.iterrows():
            print(f"Predicted Rating: {row['Predicted_Rating']:.2f}/10")
            print(f"{row['Title']} ({row['Year']})")
            print(f"Genre: {row['Genres']}")
            omdb_data = self.omdb_cache.get(row['Const'], {})
            print(f"Plot: {omdb_data.get('Plot', 'N/A')}")
            print("-" * 40)


if __name__ == "__main__":
    app = MLMovieRecommender()
    app.run()
