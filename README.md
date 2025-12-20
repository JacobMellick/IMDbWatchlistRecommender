# IMDb Watchlist Recommender

A smart movie recommender system for people who have too many movies in their watchlist and can't decide what to watch.

## Overview

This application analyzes your IMDb **Ratings** and **Watchlist** to provide personalized recommendations. It features a unified CLI with two distinct recommendation engines:

1.  **Weighted Heuristic Engine:** Best for finding movies that match your specific taste profile (favorite directors, preferred genres, critical acclaim).
2.  **Machine Learning Engine (Gradient Boosting):** Best for discovering "hidden gems" by learning your implicit rating patterns based on plot keywords, metadata, and voting history.

**Key Features:**

- Interactive CLI with rich visuals (tables, progress bars).
- Automatic caching of OMDb API data to minimize requests.
- Graceful handling of API rate limits (saves progress automatically).
- Post-recommendation filtering (by Runtime, Genre, etc.).

## Requirements

- Python 3.9+
- **Data Files (User Provided):** `watchlist.csv`, `ratings.csv`
- **API Key:** OMDb API key (Free tier allowed).

## Setup

### 1. Environment

```sh
# Create virtual environment
python -m venv .venv

# Activate (Windows)
. .venv/Scripts/activate
# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. OMDb API Key

Get a free key at [omdbapi.com](https://www.omdbapi.com/) (1,000 requests/day).
Create a `.env` file in the root directory:

```env
OMDB_API_KEY=your_key_here
```

### 3. Data Export (IMDb)

You must export your data from IMDb and place the CSV files in the project root.

1.  **Watchlist:** IMDb → Your Lists → Export → save as `watchlist.csv`.
2.  **Ratings:** IMDb → Your Ratings → Export → save as `ratings.csv`.

> **Note:** Ensure headers match standard IMDb exports (e.g., 'Const', 'Your Rating', 'Title Type', 'Directors', etc.).

## Usage

Run the unified controller:

```sh
python main.py
```

Follow the on-screen prompts:

1.  **Select Engine:** Choose between _Weighted Heuristic_ or _Gradient Boosting AI_.
2.  **Processing:**
    - The AI engine will train on your ratings and predict watchlist scores.
    - The Heuristic engine will analyze your preferences and score candidates.
    - _Note: First run will be slower as it fetches metadata from OMDb._
3.  **Filtering:** Optionally filter results by max runtime (e.g., "< 120 mins") or specific moods.
4.  **View Details:** Select specific movies from the result table to see plot summaries, actors, and more.

## The Engines

### Naive Recommender (Heuristic)

- **Logic:** Calculates a score based on weighted factors:
  - IMDb Rating (50%)
  - Genre match against your history (30%)
  - Bonus points for your favorite Directors and Actors.
  - Bonus points for high Rotten Tomatoes scores (via OMDb).
- **Best for:** "I want to watch something safe that aligns with what I usually like."

### ML Recommender (Gradient Boosting)

- **Logic:** Uses `sklearn.ensemble.GradientBoostingRegressor`.
  - **Features:** Year, Runtime, Box Office, Metascore, One-Hot Encoded Genres, Target Encoded Directors, and **TF-IDF Vectorized Plot Summaries**.
  - It learns _why_ you rate movies high or low based on the content of the plot and metadata.
- **Best for:** "Surprise me with something I might like, even if it's a genre I rarely watch."

## Caching & API Limits

- **Cache:** All OMDb API responses are saved to `omdb_cache.json`. This file is loaded automatically on every run.
- **Rate Limits:** The free OMDb key has a 1,000 call/day limit.
  - The scripts include sleep timers to be polite to the API.
  - If you hit the daily limit (HTTP 401), the application will **automatically save your progress** and exit safely. You can simply run it again tomorrow to pick up where you left off.

## Project Structure

- `main.py`: The CLI controller and UI logic.
- `ml_recommender.py`: Machine Learning pipeline class.
- `naive_recommender.py`: Heuristic rule-based class.
- `watchlist.csv` / `ratings.csv`: Your data (git-ignored).
- `omdb_cache.json`: Local data cache (git-ignored).
