from __future__ import annotations

from collections import Counter
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
        self.slim_cache_path = base_dir / "artifacts" / "recommender_slim.pkl.gz"

        self.ready = False
        self.state = DatasetState(False, "Dataset not loaded yet.")

        self.movies_df = pd.DataFrame()
        self.movie_lookup: dict[str, dict[str, Any]] = {}
        self.movies_by_id: dict[int, dict[str, Any]] = {}
        self.similar_movies: dict[int, list[dict[str, Any]]] = {}
        self.popular_movies: list[dict[str, Any]] = []

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
            "indexed_movie_count": len(self.similar_movies),
        }

    def _load(self) -> None:
        try:
            if self.slim_cache_path.exists():
                self._load_slim_cache()
            else:
                self._load_minimal_dataset()
            self.ready = True
            self.state = DatasetState(True, "Dataset loaded successfully.")
        except Exception as exc:  # pragma: no cover
            self.ready = False
            self.state = DatasetState(
                False,
                f"Dataset could not be loaded automatically. Error: {exc}",
            )

    def _load_slim_cache(self) -> None:
        with gzip.open(self.slim_cache_path, "rb") as file_obj:
            payload = pickle.load(file_obj)

        self.movies_df = payload["movies_df"]
        self.movie_lookup = payload["movie_lookup"]
        self.movies_by_id = payload["movies_by_id"]
        self.similar_movies = payload["similar_movies"]
        self.popular_movies = payload["popular_movies"]

    def _load_minimal_dataset(self) -> None:
        local_movies, _ = self._resolve_dataset_paths()
        movies = pd.read_csv(local_movies)
        movies["Movie_ID"] = movies["Movie_ID"].astype(int)
        movies["Year"] = pd.to_numeric(movies["Year"], errors="coerce").fillna(0).astype(int)
        movies["Name"] = movies["Name"].fillna("").astype(str).str.strip()
        movies = movies[movies["Name"] != ""].copy()
        movies["rating_count"] = 0
        movies["avg_rating"] = 0.0
        self.movies_df = movies.head(1000).reset_index(drop=True)
        self.movie_lookup = {}
        self.movies_by_id = {}
        for row in self.movies_df.to_dict("records"):
            item = {
                "movie_id": int(row["Movie_ID"]),
                "title": row["Name"],
                "year": int(row["Year"]),
                "rating_count": int(row["rating_count"]),
                "avg_rating": round(float(row["avg_rating"]), 2),
                "normalized_title": self._normalize_text(row["Name"]),
            }
            self.movie_lookup[row["Name"].casefold()] = item
            self.movies_by_id[item["movie_id"]] = item
        self.similar_movies = {}
        self.popular_movies = [self._public_movie(item) for item in self.movies_by_id.values()]

    def _resolve_dataset_paths(self) -> tuple[Path, Path]:
        self.data_dir.mkdir(exist_ok=True)
        local_movies = self.data_dir / MOVIES_FILE
        local_ratings = self.data_dir / RATINGS_FILE
        if local_movies.exists() and local_ratings.exists():
            return local_movies, local_ratings

        if kagglehub is None or KaggleDatasetAdapter is None:
            raise FileNotFoundError("Dataset files were not found.")

        movies_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            KAGGLE_DATASET,
            MOVIES_FILE,
        )
        ratings_path = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / RATINGS_FILE
        movies_df.to_csv(local_movies, index=False)
        local_ratings.write_bytes(ratings_path.read_bytes())
        return local_movies, local_ratings

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

            results.append((score, item["rating_count"], item["avg_rating"], self._public_movie(item)))

        results.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
        return [entry[3] for entry in results[:limit]]

    def recommend_from_titles(self, titles: list[str], limit: int = 12) -> dict[str, Any]:
        clean_titles = [title.strip() for title in titles if isinstance(title, str) and title.strip()]
        if not self.ready:
            return {"status": "error", "message": self.state.message, "selected": [], "recommendations": []}
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
            selected_movies.append(self._public_movie(item))
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

        for movie_id in selected_ids:
            for neighbor in self.similar_movies.get(movie_id, []):
                candidate_id = neighbor["movie_id"]
                if candidate_id in selected_ids:
                    continue
                candidate_scores[candidate_id] += float(neighbor["score"])
                candidate_support[candidate_id] += int(neighbor["support"])

        recommendations = []
        if candidate_scores:
            for candidate_id, score in candidate_scores.most_common(limit * 8):
                movie = self.movies_by_id.get(candidate_id)
                if movie is None:
                    continue
                recommendations.append(
                    {
                        **self._public_movie(movie),
                        "support": int(candidate_support[candidate_id]),
                        "score": round(float(score), 3),
                    }
                )
            recommendations.sort(
                key=lambda item: (item["score"], item["support"], item["avg_rating"], item["rating_count"]),
                reverse=True,
            )
        else:
            for movie in self.popular_movies:
                if movie["movie_id"] not in selected_ids:
                    recommendations.append({**movie, "support": 0, "score": 0.0})

        message = "추천 결과를 만들었습니다."
        if missing:
            message += f" 일부 제목은 찾지 못했습니다: {', '.join(missing[:3])}"

        return {
            "status": "ok",
            "message": message,
            "selected": selected_movies,
            "recommendations": recommendations[:limit],
        }
