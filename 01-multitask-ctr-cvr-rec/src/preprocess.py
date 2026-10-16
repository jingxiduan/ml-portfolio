"""
MovieLens-1M -> multi-task CTR/CVR training sample construction script

Transforms an implicit-feedback dataset into an "impression-click-conversion" multi-task
scenario (ESMM-style task dependency):
  - CTR task (clicked/interacted or not): positive samples = movies the user actually rated;
    negative samples = negative sampling over movies the user never interacted with for each
    positive sample (simulating impressions without a click).
  - CVR task (converted/satisfied or not): defined only on CTR=1 (clicked) samples, with
    rating>=4 treated as "conversion" (corresponding to the branch-routing + sample-mask task
    dependency between retention/acquisition shared samples).

This is the common approach for reproducing a "multi-task tower modeling" scenario on a public
dataset: MovieLens itself has no impression/click/conversion three-stage log, so proxy labels
("rated = clicked, high rating = converted") plus negative sampling to simulate un-clicked
impressions are used instead. This is explicitly stated in the README to avoid any confusion.
"""
import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ml-1m")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
NEG_PER_POS = 2          # number of negative samples per positive sample
CONVERT_RATING_THRESH = 4
RANDOM_SEED = 42


def load_raw():
    users = pd.read_csv(
        os.path.join(RAW_DIR, "users.dat"), sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"], encoding="latin-1",
    )
    movies = pd.read_csv(
        os.path.join(RAW_DIR, "movies.dat"), sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    ratings = pd.read_csv(
        os.path.join(RAW_DIR, "ratings.dat"), sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1",
    )
    return users, movies, ratings


def build_features(users, movies, ratings):
    rng = np.random.default_rng(RANDOM_SEED)

    # User behavior features: historical average rating, historical interaction count
    # (simulating real-time user-side features)
    user_stats = ratings.groupby("user_id")["rating"].agg(
        user_avg_rating="mean", user_rating_cnt="count"
    ).reset_index()

    # Movie popularity features (simulating media/campaign-side features)
    movie_stats = ratings.groupby("movie_id")["rating"].agg(
        movie_avg_rating="mean", movie_rating_cnt="count"
    ).reset_index()

    top_genres = ["Action", "Comedy", "Drama", "Thriller", "Romance",
                  "Sci-Fi", "Horror", "Animation", "Adventure", "Children's"]
    for g in top_genres:
        movies[f"genre_{g}"] = movies["genres"].str.contains(g).astype(int)

    all_movie_ids = movies["movie_id"].unique()
    user_pos_movies = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

    pos_rows = ratings.copy()
    pos_rows["label_click"] = 1
    pos_rows["label_convert"] = (pos_rows["rating"] >= CONVERT_RATING_THRESH).astype(int)

    # Negative sampling: simulates "impressed but not clicked" samples; label_convert is set
    # to -1 (masked out during training, corresponding to the sample mask)
    neg_user_list, neg_movie_list = [], []
    for uid, pos_set in user_pos_movies.items():
        n_neg = min(len(pos_set) * NEG_PER_POS, len(all_movie_ids) - len(pos_set))
        if n_neg <= 0:
            continue
        candidates = rng.choice(all_movie_ids, size=n_neg * 2, replace=True)
        chosen = []
        for m in candidates:
            if m not in pos_set:
                chosen.append(m)
            if len(chosen) >= n_neg:
                break
        neg_user_list.extend([uid] * len(chosen))
        neg_movie_list.extend(chosen)

    neg_rows = pd.DataFrame({"user_id": neg_user_list, "movie_id": neg_movie_list})
    neg_rows["rating"] = 0
    neg_rows["timestamp"] = 0
    neg_rows["label_click"] = 0
    neg_rows["label_convert"] = -1  # mask

    full = pd.concat([
        pos_rows[["user_id", "movie_id", "label_click", "label_convert", "timestamp"]],
        neg_rows[["user_id", "movie_id", "label_click", "label_convert", "timestamp"]],
    ], ignore_index=True)

    full = full.merge(users, on="user_id", how="left")
    full = full.merge(movies, on="movie_id", how="left")
    full = full.merge(user_stats, on="user_id", how="left")
    full = full.merge(movie_stats, on="movie_id", how="left")

    # Negative samples "leaking" into aggregate stats that include themselves has negligible
    # impact overall; fill missing values simply
    full["movie_avg_rating"] = full["movie_avg_rating"].fillna(full["movie_avg_rating"].mean())
    full["movie_rating_cnt"] = full["movie_rating_cnt"].fillna(0)
    full["user_avg_rating"] = full["user_avg_rating"].fillna(full["user_avg_rating"].mean())
    full["user_rating_cnt"] = full["user_rating_cnt"].fillna(0)

    # Bucket age/occupation as categorical features, encode gender
    full["age"] = full["age"].astype("category").cat.codes
    full["occupation"] = full["occupation"].astype("category").cat.codes
    full["gender"] = (full["gender"] == "M").astype(int)

    genre_cols = [f"genre_{g}" for g in top_genres]
    sparse_cols = ["user_id", "movie_id", "gender", "age", "occupation"]
    dense_cols = ["user_avg_rating", "user_rating_cnt", "movie_avg_rating", "movie_rating_cnt"] + genre_cols

    full = full.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    n = len(full)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = full.iloc[:n_train]
    val = full.iloc[n_train:n_train + n_val]
    test = full.iloc[n_train + n_val:]

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {"sparse_cols": sparse_cols, "dense_cols": dense_cols}
    import json
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    keep_cols = sparse_cols + dense_cols + ["label_click", "label_convert"]
    train[keep_cols].to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val[keep_cols].to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test[keep_cols].to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print(f"total={n} train={len(train)} val={len(val)} test={len(test)}")
    print("click positive ratio:", full["label_click"].mean())
    clicked = full[full["label_click"] == 1]
    print("convert positive ratio (within clicked):", clicked["label_convert"].mean())


if __name__ == "__main__":
    users, movies, ratings = load_raw()
    build_features(users, movies, ratings)
