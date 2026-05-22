from __future__ import annotations

from collections import defaultdict
import gzip
import math
from pathlib import Path
import pickle
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MOVIES_FILE = DATA_DIR / "Netflix_Dataset_Movie.csv"
RATINGS_FILE = DATA_DIR / "Netflix_Dataset_Rating.csv"
OUTPUT_FILE = ARTIFACTS_DIR / "recommender_slim.pkl.gz"
TOP_MOVIES = 1350
MIN_USER_ACTIVITY = 4
MIN_RATING = 4
MIN_OVERLAP = 8
MAX_NEIGHBORS = 120


def normalize_text(text: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text.casefold())
    return " ".join(normalized.split())


def read_dataset_csv(path: Path) -> pd.DataFrame:
    with path.open("rb") as file_obj:
        signature = file_obj.read(2)
    if signature == b"PK":
        return pd.read_csv(path, compression="zip")
    return pd.read_csv(path)


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    movies = read_dataset_csv(MOVIES_FILE)
    ratings = read_dataset_csv(RATINGS_FILE)

    movies["Movie_ID"] = movies["Movie_ID"].astype(int)
    movies["Year"] = pd.to_numeric(movies["Year"], errors="coerce").fillna(0).astype(int)
    movies["Name"] = movies["Name"].fillna("").astype(str).str.strip()
    movies = movies[movies["Name"] != ""]

    ratings["Movie_ID"] = ratings["Movie_ID"].astype(int)
    ratings["User_ID"] = ratings["User_ID"].astype(int)
    ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce").fillna(0).astype(int)

    positive_ratings = ratings[ratings["Rating"] >= 3].copy()
    movie_stats = (
        positive_ratings.groupby("Movie_ID")
        .agg(rating_count=("Rating", "size"), avg_rating=("Rating", "mean"))
        .reset_index()
    )
    top_movie_ids = set(
        movie_stats.sort_values(["rating_count", "avg_rating"], ascending=[False, False])
        .head(TOP_MOVIES)["Movie_ID"]
        .astype(int)
        .tolist()
    )

    positive_ratings = positive_ratings[positive_ratings["Movie_ID"].isin(top_movie_ids)]
    user_activity = positive_ratings.groupby("User_ID").size()
    active_users = set(user_activity[user_activity >= MIN_USER_ACTIVITY].index.astype(int).tolist())
    positive_ratings = positive_ratings[positive_ratings["User_ID"].isin(active_users)]

    movie_stats = (
        positive_ratings.groupby("Movie_ID")
        .agg(rating_count=("Rating", "size"), avg_rating=("Rating", "mean"))
        .reset_index()
    )
    movies = movies.merge(movie_stats, on="Movie_ID", how="inner")
    movies = movies.sort_values(["rating_count", "avg_rating", "Name"], ascending=[False, False, True]).reset_index(drop=True)

    strong_ratings = positive_ratings[positive_ratings["Rating"] >= MIN_RATING][["User_ID", "Movie_ID"]].copy()
    movie_to_users = {
        int(movie_id): set(group["User_ID"].astype(int).tolist())
        for movie_id, group in strong_ratings.groupby("Movie_ID")
    }

    movies_by_id = {}
    movie_lookup = {}
    public_movies = []
    for row in movies.to_dict("records"):
        item = {
            "movie_id": int(row["Movie_ID"]),
            "title": row["Name"],
            "year": int(row["Year"]),
            "rating_count": int(row["rating_count"]),
            "avg_rating": round(float(row["avg_rating"]), 2),
            "normalized_title": normalize_text(row["Name"]),
        }
        movies_by_id[item["movie_id"]] = item
        movie_lookup[item["title"].casefold()] = item
        public_movies.append(
            {
                "movie_id": item["movie_id"],
                "title": item["title"],
                "year": item["year"],
                "avg_rating": item["avg_rating"],
                "rating_count": item["rating_count"],
            }
        )

    movie_ids = [int(movie_id) for movie_id in movies["Movie_ID"].tolist() if int(movie_id) in movie_to_users]
    neighbors: dict[int, list[dict[str, float | int]]] = defaultdict(list)

    for index, left_id in enumerate(movie_ids):
        left_users = movie_to_users[left_id]
        if not left_users:
            continue

        for right_id in movie_ids[index + 1:]:
            right_users = movie_to_users[right_id]
            overlap = len(left_users & right_users)
            if overlap < MIN_OVERLAP:
                continue

            similarity = overlap / math.sqrt(len(left_users) * len(right_users))
            left_score = similarity * 100 + min(overlap, 500) / 10
            right_score = left_score

            neighbors[left_id].append(
                {"movie_id": right_id, "score": round(left_score, 4), "support": overlap}
            )
            neighbors[right_id].append(
                {"movie_id": left_id, "score": round(right_score, 4), "support": overlap}
            )

    similar_movies = {}
    for movie_id, entries in neighbors.items():
        entries.sort(key=lambda entry: (entry["score"], entry["support"]), reverse=True)
        similar_movies[movie_id] = entries[:MAX_NEIGHBORS]

    payload = {
        "movies_df": movies,
        "movie_lookup": movie_lookup,
        "movies_by_id": movies_by_id,
        "similar_movies": similar_movies,
        "popular_movies": public_movies[:200],
    }

    with gzip.open(OUTPUT_FILE, "wb") as file_obj:
        pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

    print(OUTPUT_FILE)
    print(OUTPUT_FILE.stat().st_size)


if __name__ == "__main__":
    main()
