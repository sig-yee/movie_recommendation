from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from recommender import MovieRecommender


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
recommender = MovieRecommender(BASE_DIR)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "ready": recommender.ready,
            "dataset": recommender.dataset_status(),
        }
    )


@app.get("/api/search")
def search_movies():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    return jsonify(recommender.search(query))


@app.post("/api/recommend")
def recommend_movies():
    payload = request.get_json(silent=True) or {}
    titles = payload.get("titles", [])
    limit = int(payload.get("limit", 12))
    return jsonify(recommender.recommend_from_titles(titles=titles, limit=limit))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
