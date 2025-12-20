import pandas as pd
import requests
import json
import os
import numpy as np
import time
from dotenv import load_dotenv
from tqdm import tqdm
import sys

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


class OMDbLimitReached(Exception):
    """Custom exception for when API limit is hit"""
    pass


class MLMovieRecommender:
    def __init__(self):
        self.watchlist = pd.DataFrame()
        self.ratings = pd.DataFrame()
        self.omdb_cache = self.load_cache()
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42
        )

        # NLP Vectorizer for Plot
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            max_df=0.5,
            min_df=2
        )
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
        cached_data = self.omdb_cache.get(imdb_id)
        if cached_data:
            plot = cached_data.get('Plot', 'N/A')
            if plot and plot != "N/A":
                return cached_data

        # Rate limiting sleep
        time.sleep(0.2)

        url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&i={imdb_id}"
        try:
            response = requests.get(url)
            if response.status_code == 401:
                raise OMDbLimitReached(
                    "OMDb API limit reached or invalid API key.")
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'True':
                    self.omdb_cache[imdb_id] = data
                    return data
                else:
                    print(f"OMDb API error for {imdb_id}: {data.get('Error')}")
            else:
                print(f"OMDb API error for {imdb_id}: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Network error for {imdb_id}: {e}")
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

        # Director stats for target encoding
        if is_training:
            # Create a lookup for director averages
            self.director_stats = df.groupby(
                'Directors')['Your Rating'].mean().to_dict()

        try:
            loop_desc = "Training Data" if is_training else "Watchlist Data"
            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=loop_desc, unit="mov"):
                imdb_id = row['Const']
                omdb = self.fetch_omdb_data(imdb_id)

                # 1. Text Data
                plots.append(omdb.get('Plot', ''))

                # 2. Numerical Data from OMDb
                metascores.append(self.parse_metascore(
                    omdb.get('Metascore', 'N/A')))
                box_offices.append(self.clean_money(
                    omdb.get('BoxOffice', 'N/A')))
        except OMDbLimitReached as e:
            print("\n" + "!" * 50)
            print(f"CRITICAL: {e}")
            print("Saving current progress to cache...")
            self.save_cache()
            print("Cache saved. Exiting program. Please try again later/tomorrow.")
            print("!" * 50)
            sys.exit(1)

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

    def train(self):
        """Loads ratings, prepares features, and trains the model."""
        print("Loading and Training Model...")

        # 1. Load Data
        self.ratings = pd.read_csv(RATINGS_PATH)
        self.ratings = self.ratings[self.ratings['Title Type'] == 'Movie'].copy(
        )
        self.ratings = self.ratings.dropna(subset=['Your Rating'])

        y = self.ratings['Your Rating']

        # 2. Prepare Features (is_training=True)
        # Note: This uses the existing prepare_features method
        X_train = self.prepare_features(self.ratings, is_training=True)

        # 3. Fit Model
        # We fit on the whole dataset for the final application
        self.model.fit(X_train, y)
        print("Model trained successfully.")

    def predict(self):
        """Loads watchlist, generates features, and returns predictions."""
        if not hasattr(self.model, "estimators_"):
            print("Error: Model not trained yet. Call train() first.")
            return pd.DataFrame()

        # 1. Load Watchlist
        self.watchlist = pd.read_csv(WATCHLIST_PATH)
        self.watchlist = self.watchlist[self.watchlist['Title Type'] == 'Movie'].copy(
        )

        # 2. Prepare Features (is_training=False)
        X_watchlist = self.prepare_features(self.watchlist, is_training=False)

        # 3. Predict
        self.watchlist['Predicted_Rating'] = self.model.predict(X_watchlist)

        # 4. Attach Plot/Actors for display purposes
        # We do a quick lookup in the cache we just built
        plots = []
        for idx, row in self.watchlist.iterrows():
            data = self.omdb_cache.get(row['Const'], {})
            plots.append(data.get('Plot', 'N/A'))
        self.watchlist['Plot'] = plots

        # Return sorted DataFrame
        return self.watchlist.sort_values(by='Predicted_Rating', ascending=False)


if __name__ == "__main__":
    app = MLMovieRecommender()
    app.run()
