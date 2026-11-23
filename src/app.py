import os
from pathlib import Path

from flask import Flask, jsonify, request

from itemknn import ItemKNNRecommender

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/home/ilya/Workspace/ML_Project/model/itemknn_v1"))
MAX_N = 50
MAX_HISTORY = 500

rec = ItemKNNRecommender(MODEL_DIR)
app = Flask(__name__)
app.json.ensure_ascii = False
app.json.sort_keys = False


class ApiError(Exception):
    def __init__(self, message, status=400):
        self.message, self.status = message, status


@app.errorhandler(ApiError)
def handle_api_error(e):
    return jsonify(error=e.message), e.status


@app.errorhandler(404)
def handle_404(e):
    return jsonify(error="not found"), 404


@app.errorhandler(405)
def handle_405(e):
    return jsonify(error="method not allowed"), 405


def get_n():
    try:
        n = int(request.args.get("n", 10))
    except ValueError:
        raise ApiError("n must be an integer")
    if not 1 <= n <= MAX_N:
        raise ApiError(f"n must be between 1 and {MAX_N}")
    return n


def respond(df, **extra):
    items = [
        {"movie_id": int(r.movieId), "title": r.title, "genres": r.genres, "score": round(float(r.score), 6)}
        for r in df.itertuples()
    ]
    return jsonify(model_version=rec.meta.get("version", "unknown"), **extra, count=len(items), items=items)


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=rec.meta.get("version", "unknown"),
                   n_users=rec.hist.shape[0], n_items=rec.sim.shape[0])


@app.get("/v1/recommend/user/<int:user_id>")
def recommend_user(user_id):
    n = get_n()
    if user_id not in rec.uid_of:
        raise ApiError(f"unknown user_id {user_id}, use /v1/recommend/history instead", 404)
    return respond(rec.recommend_user(user_id, n), user_id=user_id, n=n)


@app.get("/v1/recommend/history")
def recommend_history():
    n = get_n()
    try:
        ids = [int(x) for part in request.args.getlist("movie_ids") for x in part.split(",") if x.strip()]
    except ValueError:
        raise ApiError("movie_ids must be integers, e.g. movie_ids=1210,2018")
    if not ids:
        raise ApiError("movie_ids is required, e.g. movie_ids=1210,2018")
    if len(ids) > MAX_HISTORY:
        raise ApiError(f"at most {MAX_HISTORY} movie_ids")

    known = [m for m in ids if m in rec.iid_of]
    if not known:
        raise ApiError("none of movie_ids are known to the model", 422)
    ignored = [m for m in ids if m not in rec.iid_of]
    return respond(rec.recommend_history(known, n), n=n, used_movie_ids=known, ignored_movie_ids=ignored)


@app.get("/v1/similar/<int:movie_id>")
def similar(movie_id):
    n = get_n()
    if movie_id not in rec.iid_of:
        raise ApiError(f"unknown movie_id {movie_id}", 404)
    return respond(rec.similar(movie_id, n), movie_id=movie_id, n=n)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
