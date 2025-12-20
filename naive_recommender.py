import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv
import time

# -- CONFIGURATION --
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
WATCHLIST_PATH = "watchlist.csv"
RATINGS_PATH = "ratings.csv"
CACHE_FILE = "omdb_cache.json"

# Weights for scoring algorithm
WEIGHT_IMDB = 0.5
WEIGHT_GENRE = 0.3
BONUS_DIRECTOR = 1.5
BONUS_ACTOR = 1.0


class MovieRecommender:
    def __init__(self):
        self.watchlist = pd.DataFrame()
        self.ratings = pd.DataFrame()
        self.genre_preferences = {}
        self.favorite_directors = set()
        self.favorite_actors = set()
        self.omdb_cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with open(CACHE_FILE, 'w') as f:
            json.dump(self.omdb_cache, f)

    def load_data(self):
        """Loads and cleans CSV Data"""
        try:
            self.watchlist = pd.read_csv(WATCHLIST_PATH)
            self.ratings = pd.read_csv(RATINGS_PATH)

            # filter for movies only
            self.watchlist = self.watchlist[self.watchlist['Title Type'] == 'Movie'].copy(
            )
            self.ratings = self.ratings[self.ratings['Title Type'] == 'Movie'].copy(
            )

            print(
                f"Loaded {len(self.watchlist)} movies from watchlist and {len(self.ratings)} ratings.")
        except Exception as e:
            print(f"Error loading data: {e}")
            exit(1)

    def analyze_user_preferences(self):
        """Builds a user profile based on the user's past ratings"""
        print("Building your taste profile...")

        # 1. Genre Preferences
        temp_ratings = self.ratings.copy()
        temp_ratings['Genres'] = temp_ratings['Genres'].str.split(', ')
        exploded = temp_ratings.explode('Genres')

        # Calculate average rating per genre
        genre_stats = exploded.groupby(
            'Genres')['Your Rating'].agg(['mean', 'count'])
        # Minimum 3 ratings
        genre_stats = genre_stats[genre_stats['count'] >= 5]
        self.genre_preferences = genre_stats['mean'].to_dict()

        director_stats = self.ratings.groupby(
            'Directors')['Your Rating'].agg(['mean', 'count'])
        director_stats = director_stats[director_stats['count'] >= 3]
        self.favorite_directors = set(
            director_stats[director_stats['mean'] > 7.5].index)

    def fetch_omdb_data(self, imdb_id):
        """Fetches data from OMDb or loads from local cache."""
        if imdb_id in self.omdb_cache:
            return self.omdb_cache[imdb_id]

        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'True':
                    self.omdb_cache[imdb_id] = data
                    return data
        except Exception as e:
            print(f"Error fetching {imdb_id}: {e}")
        return None

    def calculate_score(self, row):
        """Calculates a recommendation score for a single movie row."""
        score = 0

        # 1. Base Score: IMDb Rating (Normalized to match user rating scale roughly)
        # Using the CSV IMDb rating
        imdb_rating = row.get('IMDb Rating', 0)
        score += imdb_rating * WEIGHT_IMDB

        # 2. Genre Match Score
        genres = str(row.get('Genres', '')).split(', ')
        # Default to 6.0 if unknown genre
        genre_scores = [self.genre_preferences.get(g, 6.0) for g in genres]
        if genre_scores:
            avg_genre_score = sum(genre_scores) / len(genre_scores)
            score += avg_genre_score * WEIGHT_GENRE

        # 3. Director Bonus
        directors = str(row.get('Directors', '')).split(', ')
        for d in directors:
            if d in self.favorite_directors:
                score += BONUS_DIRECTOR
                break  # Apply bonus once

        return score

    def enrich_and_recommend(self, top_n=50):
        """
        1. Scores all watchlist items based on CSV data.
        2. Takes the top N items.
        3. Fetches OMDb data for those N items to find Actors/RottenTomatoes.
        4. Re-scores and prints final recommendation.
        """

        # Initial Scoring based on CSV data only
        self.watchlist['Calc_Score'] = self.watchlist.apply(
            self.calculate_score, axis=1)

        # Sort by initial score to get candidates
        candidates = self.watchlist.sort_values(
            by='Calc_Score', ascending=False).head(top_n).copy()

        print(
            f"\nFetching OMDb data for top {top_n} candidates to refine scores...")

        final_scores = []

        for index, row in candidates.iterrows():
            imdb_id = row['Const']
            omdb_data = self.fetch_omdb_data(imdb_id)

            # Start with existing CSV score
            final_score = row['Calc_Score']
            actors_str = "N/A"
            plot_str = "No plot available"

            if omdb_data:
                # OMDb Enrichment Logic

                # 1. Metascore / Rotten Tomatoes boost
                ratings_sources = {r['Source']: r['Value']
                                   for r in omdb_data.get('Ratings', [])}
                if 'Rotten Tomatoes' in ratings_sources:
                    rt_score = int(
                        ratings_sources['Rotten Tomatoes'].replace('%', ''))
                    if rt_score > 80:
                        final_score += 1.0  # Boost for critical acclaim

                # 2. Actor Logic (Simple keyword check, can be expanded)
                actors_str = omdb_data.get('Actors', 'N/A')
                plot_str = omdb_data.get('Plot', 'N/A')

            final_scores.append({
                'Title': row['Title'],
                'Year': row['Year'],
                'IMDb': row['IMDb Rating'],
                'Genres': row['Genres'],
                'Final_Score': round(final_score, 2),
                'Actors': actors_str,
                'Plot': plot_str
            })

            # Be nice to the API
            if imdb_id not in self.omdb_cache:
                time.sleep(0.1)

        # Save cache for next time
        self.save_cache()

        # Create Final DataFrame
        recommendations = pd.DataFrame(final_scores)
        recommendations = recommendations.sort_values(
            by='Final_Score', ascending=False)

        return recommendations.head(5)


def main():
    rec = MovieRecommender()
    rec.load_data()
    rec.analyze_user_preferences()

    # Show user their genre profile
    print("\n--- Your Top Genres ---")
    sorted_genres = sorted(rec.genre_preferences.items(),
                           key=lambda x: x[1], reverse=True)[:5]
    for g, s in sorted_genres:
        print(f"{g}: {s:.2f} avg rating")

    top_movies = rec.enrich_and_recommend(top_n=20)

    print("\n==========================================")
    print("       TOP RECOMMENDATIONS FOR YOU       ")
    print("==========================================\n")

    for i, row in top_movies.iterrows():
        print(f"{i+1}. {row['Title']} ({row['Year']})")
        print(f"   Score: {row['Final_Score']} | IMDb: {row['IMDb']}")
        print(f"   Genres: {row['Genres']}")
        print(f"   Starring: {row['Actors']}")
        print(f"   Plot: {row['Plot']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
