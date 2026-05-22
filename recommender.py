from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
from pathlib import Path
import pickle
import re
from typing import Any

import pandas as pd

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:  # pragma: no cover
    kagglehub = None
    KaggleDatasetAdapter = None


MOVIES_FILE = "Netflix_Dataset_Movie.csv"
RATINGS_FILE = "Netflix_Dataset_Rating.csv"
KAGGLE_DATASET = "rishitjavia/netflix-movie-rating-dataset"


@dataclass
class DatasetState:
    ready: bool
    message: str


class MovieRecommender:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.cache_path = base_dir / "artifacts" / "recommender_cache.pkl.gz"
        self.data_dir.mkdir(exist_ok=True)

        self.ready = False
        self.state = DatasetState(False, "Dataset not loaded yet.")

        self.movies_df = pd.DataFrame()
        self.movie_lookup: dict[str, dict[str, Any]] = {}
        self.movies_by_id: dict[int, dict[str, Any]] = {}
        self.movie_to_users: dict[int, set[int]] = {}
        self.user_to_movies: dict[int, list[tuple[int, int]]] = {}
        self.movie_popularity: dict[int, int] = {}
        self.movie_avg_rating: dict[int, float] = {}

        self._load()

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text.casefold())
        return " ".join(normalized.split())

    @staticmethod
    def _public_movie(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "movie_id": item["movie_id"],
            "title": item["title"],
            "year": item["year"],
            "avg_rating": item["avg_rating"],
            "rating_count": item["rating_count"],
        }

    def dataset_status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "message": self.state.message,
            "data_dir": str(self.data_dir),
            "movie_count": len(self.movies_by_id),
            "indexed_movie_count": len(self.movie_to_users),
            "user_count": len(self.user_to_movies),
        }

    def _load(self) -> None:
        try:
            if self.cache_path.exists():
                self._load_cache()
            else:
                movies_path, ratings_path = self._resolve_dataset_paths()
                movies = self._read_dataset_csv(movies_path)
                ratings = self._read_dataset_csv(ratings_path)
                self._build_indexes(movies, ratings)
            self.ready = True
            self.state = DatasetState(True, "Dataset loaded successfully.")
        except Exception as exc:  # pragma: no cover
            self.ready = False
            self.state = DatasetState(
                False,
                f"Dataset could not be loaded automatically. Add CSVs to {self.data_dir} or install kagglehub. Error: {exc}",
            )

    def _resolve_dataset_paths(self) -> tuple[Path, Path]:
        local_movies = self.data_dir / MOVIES_FILE
        local_ratings = self.data_dir / RATINGS_FILE
        if local_movies.exists() and local_ratings.exists():
            return local_movies, local_ratings

        if kagglehub is None or KaggleDatasetAdapter is None:
            raise FileNotFoundError("kagglehub is not installed and local dataset files were not found.")

        movies_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            KAGGLE_DATASET,
            MOVIES_FILE,
        )
        ratings_path = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / RATINGS_FILE
        movies_df.to_csv(local_movies, index=False)
        local_ratings.write_bytes(ratings_path.read_bytes())
        return local_movies, local_ratings

    def _read_dataset_csv(self, path: Path) -> pd.DataFrame:
        with path.open("rb") as file_obj:
            signature = file_obj.read(2)

        if signature == b"PK":
            return pd.read_csv(path, compression="zip")
        return pd.read_csv(path)

    def _load_cache(self) -> None:
        with gzip.open(self.cache_path, "rb") as file_obj:
            payload = pickle.load(file_obj)

        self.movies_df = payload["movies_df"]
        self.movie_lookup = payload["movie_lookup"]
        self.movies_by_id = payload["movies_by_id"]
        self.movie_to_users = defaultdict(set, payload["movie_to_users"])
        self.user_to_movies = defaultdict(list, payload["user_to_movies"])
        self.movie_popularity = payload["movie_popularity"]
        self.movie_avg_rating = payload["movie_avg_rating"]

    def _build_indexes(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
        movies = movies.copy()
        ratings = ratings.copy()

        movies["Movie_ID"] = movies["Movie_ID"].astype(int)
        movies["Year"] = pd.to_numeric(movies["Year"], errors="coerce").fillna(0).astype(int)
        movies["Name"] = movies["Name"].fillna("").astype(str).str.strip()
        movies = movies[movies["Name"] != ""]

        ratings["Movie_ID"] = ratings["Movie_ID"].astype(int)
        ratings["User_ID"] = ratings["User_ID"].astype(int)
        ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce").fillna(0).astype(int)
        ratings = ratings[ratings["Rating"] >= 3]

        rating_counts = ratings.groupby("Movie_ID").size().rename("rating_count")
        avg_ratings = ratings.groupby("Movie_ID")["Rating"].mean().rename("avg_rating")
        movie_stats = pd.concat([rating_counts, avg_ratings], axis=1).reset_index()
        top_movie_ids = set(
            movie_stats.sort_values(["rating_count", "avg_rating"], ascending=[False, False])
            .head(1500)["Movie_ID"]
            .astype(int)
            .tolist()
        )

        ratings = ratings[ratings["Movie_ID"].isin(top_movie_ids)]
        user_activity = ratings.groupby("User_ID").size()
        active_users = set(user_activity[user_activity >= 4].index.astype(int).tolist())
        ratings = ratings[ratings["User_ID"].isin(active_users)]

        movie_stats = (
            ratings.groupby("Movie_ID")
            .agg(rating_count=("Rating", "size"), avg_rating=("Rating", "mean"))
            .reset_index()
        )
        movies = movies.merge(movie_stats, on="Movie_ID", how="inner")
        movies = movies.sort_values(["rating_count", "avg_rating", "Name"], ascending=[False, False, True])
        self.movies_df = movies.reset_index(drop=True)

        self.movie_lookup = {}
        self.movies_by_id = {}
        for row in self.movies_df.to_dict("records"):
            movie_id = int(row["Movie_ID"])
            normalized_title = self._normalize_text(row["Name"])
            item = {
                "movie_id": movie_id,
                "title": row["Name"],
                "year": int(row["Year"]),
                "rating_count": int(row["rating_count"]),
                "avg_rating": round(float(row["avg_rating"]), 2),
                "normalized_title": normalized_title,
            }
            self.movies_by_id[movie_id] = item
            self.movie_lookup[row["Name"].casefold()] = item

        self.movie_to_users = defaultdict(set)
        self.user_to_movies = defaultdict(list)
        self.movie_popularity = {}
        self.movie_avg_rating = {}

        for row in ratings.itertuples(index=False):
            movie_id = int(row.Movie_ID)
            user_id = int(row.User_ID)
            rating = int(row.Rating)
            self.movie_to_users[movie_id].add(user_id)
            self.user_to_movies[user_id].append((movie_id, rating))

        for row in self.movies_df.itertuples(index=False):
            self.movie_popularity[int(row.Movie_ID)] = int(row.rating_count)
            self.movie_avg_rating[int(row.Movie_ID)] = float(row.avg_rating)

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not self.ready:
            return []

        query_folded = query.casefold().strip()
        normalized_query = self._normalize_text(query)
        query_tokens = normalized_query.split()
        results = []

        for item in self.movies_by_id.values():
            title_folded = item["title"].casefold()
            normalized_title = item["normalized_title"]
            title_words = normalized_title.split()
            score = 0.0

            if query_folded and query_folded in title_folded:
                score += 120.0
            if normalized_query and normalized_query in normalized_title:
                score += 100.0
            if normalized_query and normalized_title.startswith(normalized_query):
                score += 20.0

            for token in query_tokens:
                if token in normalized_title:
                    score += 30.0
                if any(word.startswith(token) for word in title_words):
                    score += 12.0

            if score <= 0:
                continue

            results.append(
                (
                    score,
                    item["rating_count"],
                    item["avg_rating"],
                    self._public_movie(item),
                )
            )

        results.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
        return [entry[3] for entry in results[:limit]]

    def recommend_from_titles(self, titles: list[str], limit: int = 12) -> dict[str, Any]:
        clean_titles = [title.strip() for title in titles if isinstance(title, str) and title.strip()]
        if not self.ready:
            return {
                "status": "error",
                "message": self.state.message,
                "selected": [],
                "recommendations": [],
            }
        if not clean_titles:
            return {
                "status": "error",
                "message": "좋아하는 영화를 최소 1편 이상 선택하세요.",
                "selected": [],
                "recommendations": [],
            }

        selected_movies = []
        selected_ids = set()
        missing = []
        for title in clean_titles:
            item = self.movie_lookup.get(title.casefold())
            if item is None:
                missing.append(title)
                continue
            selected_movies.append(item)
            selected_ids.add(item["movie_id"])

        if not selected_ids:
            return {
                "status": "error",
                "message": "선택한 영화가 추천 인덱스에 없습니다. 다른 영화를 선택해 주세요.",
                "selected": [],
                "recommendations": [],
            }

        candidate_scores: Counter[int] = Counter()
        candidate_support: Counter[int] = Counter()

        for movie in selected_movies:
            base_movie_id = movie["movie_id"]
            users = self.movie_to_users.get(base_movie_id, set())
            if not users:
                continue

            overlap_weight = max(len(users), 1) ** 0.5
            for user_id in users:
                for candidate_id, rating in self.user_to_movies.get(user_id, []):
                    if candidate_id in selected_ids or rating < 4:
                        continue
                    candidate_scores[candidate_id] += rating / overlap_weight
                    candidate_support[candidate_id] += 1

        recommendations = []
        for movie_id, score in candidate_scores.most_common(limit * 5):
            movie = self.movies_by_id.get(movie_id)
            if movie is None:
                continue
            popularity = self.movie_popularity.get(movie_id, 1)
            avg_rating = self.movie_avg_rating.get(movie_id, 0.0)
            blended_score = round(score + avg_rating * 2 + min(popularity, 5000) / 1000, 3)
            recommendations.append(
                {
                    **self._public_movie(movie),
                    "support": int(candidate_support[movie_id]),
                    "score": blended_score,
                }
            )

        recommendations = sorted(
            recommendations,
            key=lambda item: (item["score"], item["support"], item["avg_rating"], item["rating_count"]),
            reverse=True,
        )[:limit]

        message = "추천 결과를 만들었습니다."
        if missing:
            message += f" 일부 제목은 찾지 못했습니다: {', '.join(missing[:3])}"

        return {
            "status": "ok",
            "message": message,
            "selected": selected_movies,
            "recommendations": recommendations,
        }
