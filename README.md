# IMDbWatchlistRecommender

A simple movie recommender for people who just can't choose.

## Overview

You can run either:

- Naive scoring recommender: [`MovieRecommender`](naive_recommender.py)
- ML-based recommender: [`MLMovieRecommender`](ml_recommender.py)

## Requirements

- Python 3.9+
- Dependencies: see [requirements.txt](requirements.txt)
- Files required at runtime (not committed): `watchlist.csv`, `ratings.csv`, `omdb_cache.json`
- OMDb API key required in a `.env` file: `OMDB_API_KEY=your_key`
- Get a free OMDb API key at https://www.omdbapi.com/ (free tier allows up to 1,000 requests/day)

## Data Export (IMDb)

1. Export Watchlist: IMDb → Your Lists → Export → save as `watchlist.csv`.
2. Export Ratings: IMDb → Your Ratings → Export → save as `ratings.csv`.

## Setup

```sh
python -m venv .venv
. .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

Create `.env` with your OMDb key:

```sh
OMDB_API_KEY=your_key
```

## Option A: Naive Recommender

Runs a weighted heuristic based on IMDb ratings, genres, and director bonuses.

- Weights in [naive_recommender.py](naive_recommender.py):
  - [`WEIGHT_IMDB`](naive_recommender.py), [`WEIGHT_GENRE`](naive_recommender.py), [`BONUS_DIRECTOR`](naive_recommender.py)
- Core flow: [`MovieRecommender.load_data`](naive_recommender.py) → [`MovieRecommender.analyze_user_preferences`](naive_recommender.py) → [`MovieRecommender.enrich_and_recommend`](naive_recommender.py)

Run:

```sh
python naive_recommender.py
```

Adjust candidate pool via `top_n` in [`MovieRecommender.enrich_and_recommend`](naive_recommender.py).

## Option B: ML Recommender

Trains a Gradient Boosting model on your ratings and predicts your watchlist scores.

- Pipeline: [`MLMovieRecommender.run`](ml_recommender.py)
- Feature building: [`MLMovieRecommender.prepare_features`](ml_recommender.py)
- OMDb fetch/caching: [`MLMovieRecommender.fetch_omdb_data`](ml_recommender.py)

Run:

```sh
python ml_recommender.py
```

## Output

- Naive: Top 5 with score, IMDb, genres, actors, plot.
- ML: Top 15 with predicted rating, genres, plot.

## Caching & API

- OMDb responses cached in `omdb_cache.json`.
- If the cache exists, it’s reused automatically.
- Free OMDb keys allow up to 1,000 requests per day; both recommenders minimize calls and cache responses to stay within limits, but large watchlists/ratings may require multiple runs over several days.

## Troubleshooting

- Missing OMDb key: set `OMDB_API_KEY` in `.env`.
- CSV columns differ: ensure IMDb exports match expected headers.
- API throttling: scripts already sleep between requests.

## Project Structure

- [naive_recommender.py](naive_recommender.py): heuristic scorer via [`MovieRecommender`](naive_recommender.py)
- [ml_recommender.py](ml_recommender.py): ML pipeline via [`MLMovieRecommender`](ml_recommender.py)
- watchlist.csv / ratings.csv: IMDb exports (user-provided; not committed)
- [requirements.txt](requirements.txt)
