"""
Offline Evaluation, Required Test Cases, and Visualization Generator
for the Hotel and Homestay Review-Aware Recommender (CS467 - MP-A13).

Run:
    python src/evaluate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for reliable PNG saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow importing sibling modules regardless of working directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_pipeline import ASPECT_KEYWORDS, StayNLPPipeline, detect_query_aspects
from recommender import (
    ContentBasedRecommender,
    HybridReviewAwareRecommender,
    PopularityRecommender,
    UserBasedCFRecommender,
)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load items.csv and ratings.csv from the data/ directory."""
    items_df = pd.read_csv(DATA_DIR / "items.csv")
    ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")
    return items_df, ratings_df


def train_test_split_by_user(
    ratings_df: pd.DataFrame, test_ratio: float = 0.25, min_test_per_user: int = 3, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out a fixed proportion of ratings per user for realistic offline evaluation.
    Guarantees every user retains ratings in both train and test splits.
    """
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for _, group in ratings_df.groupby("user_id"):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_test = max(min_test_per_user, int(round(len(indices) * test_ratio)))
        n_test = min(n_test, max(1, len(indices) - 2))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        test_parts.append(ratings_df.loc[test_idx])
        train_parts.append(ratings_df.loc[train_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def build_models(
    items_df: pd.DataFrame, ratings_df: pd.DataFrame
) -> Tuple[
    StayNLPPipeline,
    PopularityRecommender,
    ContentBasedRecommender,
    UserBasedCFRecommender,
    HybridReviewAwareRecommender,
]:
    """Fit the NLP pipeline and all recommendation models strictly on the provided ratings_df."""
    nlp = StayNLPPipeline().fit(items_df, ratings_df)
    pop = PopularityRecommender(min_ratings=3).fit(items_df, ratings_df)
    cb = ContentBasedRecommender(nlp).fit(ratings_df)
    cf = UserBasedCFRecommender(k_neighbors=4, min_similarity=0.05, shrinkage=3.0).fit(items_df, ratings_df)
    hybrid = HybridReviewAwareRecommender(
        nlp_pipeline=nlp,
        popularity_model=pop,
        content_model=cb,
        cf_model=cf,
        w_cf=0.40,
        w_cb=0.45,
        w_aspect=0.15,
    )
    return nlp, pop, cb, cf, hybrid


def evaluate_models(
    items_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 5,
    relevance_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Evaluate Popularity Baseline, NLP Content-Based, User-Based CF, and Review-Aware Hybrid
    on Precision@5, Recall@5, RMSE, and MAE using the held-out test split (no data leakage).
    """
    _, pop, cb, cf, hybrid = build_models(items_df, train_df)

    models = [
        ("Popularity Baseline", pop, lambda u: pop.recommend(user_id=u, top_k=k)),
        ("NLP Content-Based", cb, lambda u: cb.recommend_for_user(user_id=u, top_k=k)),
        ("User-Based CF", cf, lambda u: cf.recommend(user_id=u, top_k=k)),
        ("Review-Aware Hybrid", hybrid, lambda u: hybrid.recommend_for_user(user_id=u, top_k=k)),
    ]

    rows = []
    test_users = sorted(test_df["user_id"].unique())

    for name, model, rec_fn in models:
        precisions: List[float] = []
        recalls: List[float] = []

        for uid in test_users:
            user_test = test_df[test_df["user_id"] == uid]
            relevant_items = set(
                user_test.loc[user_test["rating"] >= relevance_threshold, "item_id"].astype(str)
            )
            if not relevant_items:
                continue

            recs = rec_fn(uid)
            recommended_items = recs["item_id"].astype(str).tolist()[:k]
            hits = sum(1 for iid in recommended_items if iid in relevant_items)
            precisions.append(hits / float(k))
            recalls.append(hits / float(len(relevant_items)))

        # Numeric rating prediction accuracy (RMSE & MAE) over all held-out ratings
        sq_errors = []
        abs_errors = []
        for _, row in test_df.iterrows():
            actual = float(row["rating"])
            pred = model.predict_rating(str(row["user_id"]), str(row["item_id"]))
            err = pred - actual
            sq_errors.append(err ** 2)
            abs_errors.append(abs(err))

        rows.append({
            "Method": name,
            f"Precision@{k}": round(float(np.mean(precisions)), 4),
            f"Recall@{k}": round(float(np.mean(recalls)), 4),
            "RMSE": round(float(np.sqrt(np.mean(sq_errors))), 4),
            "MAE": round(float(np.mean(abs_errors)), 4),
        })

    return pd.DataFrame(rows)


def generate_test_case_report(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    """Run all 3 required test cases on the full dataset and return formatted report text."""
    nlp, pop, cb, cf, hybrid = build_models(items_df, ratings_df)

    lines: List[str] = []
    sep = "=" * 88
    sub = "-" * 88

    n_users = ratings_df["user_id"].nunique()
    n_items = items_df["item_id"].nunique()
    n_ratings = len(ratings_df)
    sparsity = (1.0 - n_ratings / (n_users * n_items)) * 100.0

    lines.append(sep)
    lines.append("CS467 ADVANCED AI (MP-A13) - HOTEL & HOMESTAY REVIEW-AWARE RECOMMENDER")
    lines.append("Student: Nguyen Ngoc Hoang Thai | ID: 29211151440 | Section: CS467A (No. 13)")
    lines.append(sep)
    lines.append(
        f"Dataset Summary: {n_items} stays | {n_users} guests | {n_ratings} ratings | "
        f"Matrix Sparsity: {sparsity:.2f}% missing entries"
    )
    lines.append("")

    # Offline Evaluation Summary
    lines.append(sub)
    lines.append("OFFLINE EVALUATION RESULTS (Per-User Holdout Test Split, Relevant >= 4 Stars)")
    lines.append(sub)
    lines.append(eval_df.to_string(index=False))
    lines.append("")

    # ------------------------------------------------------------------
    # TEST CASE 1: Required Free-Text Stay Query
    # ------------------------------------------------------------------
    tc1_query = "quiet homestay near beach with strong wifi and friendly host"
    tc1_weights = detect_query_aspects(tc1_query)
    tc1_cb = cb.recommend_by_query(tc1_query, top_k=5)
    tc1_hybrid = hybrid.recommend_by_query(tc1_query, top_k=5)

    lines.append(sep)
    lines.append("TEST CASE 1: STAY QUERY RECOMMENDATION")
    lines.append(f'Query: "{tc1_query}"')
    lines.append(
        "Detected Query Aspect Weights: "
        + ", ".join(f"{k}={v:.2f}" for k, v in tc1_weights.items())
    )
    lines.append(sub)
    lines.append("[1A] NLP Content-Based (TF-IDF Cosine Similarity) - Top 5:")
    for rank, r in tc1_cb.iterrows():
        lines.append(
            f"  #{rank+1} [{r['item_id']}] {r['stay_name']} ({r['area']}, {r['price_tier']}) "
            f"| TF-IDF Score: {r['score']:.4f}"
        )
        lines.append(f"      Keywords: {r['matching_keywords']}")

    lines.append("")
    lines.append("[1B] Review-Aware Hybrid (Query TF-IDF + Aspect Sentiment + Popularity) - Top 5:")
    for rank, r in tc1_hybrid.iterrows():
        lines.append(
            f"  #{rank+1} [{r['item_id']}] {r['stay_name']} ({r['area']}, {r['price_tier']}) "
            f"| Hybrid Score: {r['score']:.4f} | TF-IDF: {r['tfidf_sim']:.4f} | Aspect Sent: {r['aspect_sentiment']:+.3f}"
        )
        lines.append(f"      Details: {r['explanation']}")
    lines.append("")

    # ------------------------------------------------------------------
    # TEST CASE 2: Review-Aware Comparison vs. Plain Star Average
    # ------------------------------------------------------------------
    plain_top = hybrid.compare_plain_vs_review_aware(top_k=8, sort_by="plain")
    review_top = hybrid.compare_plain_vs_review_aware(top_k=8, sort_by="review_aware")

    lines.append(sep)
    lines.append("TEST CASE 2: REVIEW-AWARE RANKING VS. PLAIN STAR AVERAGE")
    lines.append(
        "Demonstrates how NLP aspect sentiment penalizes star-inflated stays with negative "
        "cleanliness/quietness/service reviews and promotes genuinely high-quality stays."
    )
    lines.append(sub)
    lines.append("[2A] Top 8 Ranked by PLAIN STAR AVERAGE (Ignoring Review Text):")
    cols_view = [
        "plain_rank",
        "review_aware_rank",
        "rank_change",
        "item_id",
        "stay_name",
        "avg_rating",
        "overall_aspect_sentiment",
        "review_aware_score",
    ]
    lines.append(plain_top[cols_view].to_string(index=False))

    lines.append("")
    lines.append("[2B] Top 8 Ranked by REVIEW-AWARE SCORE (55% Star Avg + 45% Aspect Sentiment):")
    lines.append(review_top[cols_view].to_string(index=False))

    # Highlight specific star-inflated stays that dropped sharply
    full_cmp = hybrid.compare_plain_vs_review_aware(top_k=len(items_df), sort_by="plain")
    dropped = full_cmp[full_cmp["rank_change"] <= -5].sort_values("rank_change")
    if not dropped.empty:
        lines.append("")
        lines.append("Key Insight - Star-Inflated Stays Demoted by Aspect Sentiment:")
        for _, r in dropped.head(4).iterrows():
            lines.append(
                f"  - [{r['item_id']}] {r['stay_name']}: Plain Avg {r['avg_rating']:.2f}★ (Rank #{int(r['plain_rank'])}) "
                f"-> Review-Aware {r['review_aware_score']:.2f} (Rank #{int(r['review_aware_rank'])}, "
                f"change {int(r['rank_change'])}) | "
                f"clean={r['cleanliness_score']:+.2f}, loc={r['location_score']:+.2f}, "
                f"quiet={r['quietness_score']:+.2f}, serv={r['service_score']:+.2f}"
            )
    lines.append("")

    # ------------------------------------------------------------------
    # TEST CASE 3: Cold-Start Test (Brand-New Guest, No Rating History)
    # ------------------------------------------------------------------
    cold_guest_id = "NEW_GUEST_01"
    cold_query = (
        "peaceful heritage homestay near ancient town with spotless room, "
        "kind local host, and reliable wifi"
    )
    cold_weights = detect_query_aspects(cold_query)
    cold_recs = hybrid.recommend_by_query(cold_query, top_k=5, user_id=cold_guest_id)
    cold_cb = cb.recommend_by_query(cold_query, top_k=5)

    lines.append(sep)
    lines.append("TEST CASE 3: COLD-START RECOMMENDATION FOR A BRAND-NEW GUEST")
    lines.append(f"Guest ID: {cold_guest_id} (0 prior ratings in ratings.csv)")
    lines.append(f'Natural-Language Request: "{cold_query}"')
    lines.append(
        "Detected Query Aspect Weights: "
        + ", ".join(f"{k}={v:.2f}" for k, v in cold_weights.items())
    )
    lines.append(sub)
    lines.append(
        "Note: Pure User-Based CF has no rating history for NEW_GUEST_01. "
        "Our Review-Aware Cold-Start pipeline falls back seamlessly to Query TF-IDF + "
        "Query-Weighted Aspect Sentiment + Popularity Prior:"
    )
    lines.append("")
    lines.append("[3A] Review-Aware Cold-Start Hybrid - Top 5:")
    for rank, r in cold_recs.iterrows():
        lines.append(
            f"  #{rank+1} [{r['item_id']}] {r['stay_name']} ({r['area']}, {r['price_tier']}) "
            f"| Hybrid Score: {r['score']:.4f} | TF-IDF: {r['tfidf_sim']:.4f} | Aspect Sent: {r['aspect_sentiment']:+.3f}"
        )
        lines.append(f"      Details: {r['explanation']}")

    lines.append("")
    lines.append("[3B] Pure NLP Content-Based Cold-Start - Top 5:")
    for rank, r in cold_cb.iterrows():
        lines.append(
            f"  #{rank+1} [{r['item_id']}] {r['stay_name']} ({r['area']}, {r['price_tier']}) "
            f"| TF-IDF Score: {r['score']:.4f} | Keywords: {r['matching_keywords']}"
        )

    lines.append("")
    lines.append(sep)
    return "\n".join(lines), full_cmp


def generate_charts(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    full_cmp_df: pd.DataFrame,
) -> List[Path]:
    """Create and save 3 publication-quality evaluation and analysis charts in results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    saved_paths: List[Path] = []

    # ------------------------------------------------------------------
    # Chart 1: Dataset Overview (Rating Distribution & Stays by Area/Tier)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    rating_counts = ratings_df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    bar_colors = ["#D9534F", "#F0AD4E", "#F7DC6F", "#5BC0DE", "#2E86AB"]
    bars = ax1.bar(rating_counts.index.astype(str), rating_counts.values, color=bar_colors, edgecolor="#333333", linewidth=0.6)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{int(h)}", ha="center", va="bottom", fontweight="bold")
    n_users = ratings_df["user_id"].nunique()
    n_items = items_df["item_id"].nunique()
    sparsity = (1.0 - len(ratings_df) / (n_users * n_items)) * 100
    ax1.set_title(f"Guest Rating Distribution (n={len(ratings_df)}, Sparsity={sparsity:.1f}%)")
    ax1.set_xlabel("Star Rating (1–5)")
    ax1.set_ylabel("Number of Guest Reviews")
    ax1.set_ylim(0, rating_counts.max() * 1.18)

    area_tier = pd.crosstab(items_df["area"], items_df["price_tier"]).reindex(
        columns=["budget", "mid-range", "luxury"], fill_value=0
    )
    area_tier["total"] = area_tier.sum(axis=1)
    area_tier = area_tier.sort_values("total", ascending=True).drop(columns=["total"])

    area_tier.plot(
        kind="barh",
        stacked=True,
        ax=ax2,
        color=["#66C2A5", "#4C72B0", "#DD8452"],
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax2.set_title(f"Stay Inventory by Area & Price Tier (n={n_items} Stays)")
    ax2.set_xlabel("Number of Stays")
    ax2.set_ylabel("")
    ax2.legend(title="Price Tier", loc="lower right")

    plt.tight_layout()
    chart1 = RESULTS_DIR / "data_distribution.png"
    fig.savefig(chart1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(chart1)

    # ------------------------------------------------------------------
    # Chart 2: Offline Evaluation Model Comparison (Precision@5 & RMSE/MAE)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    methods = eval_df["Method"].tolist()
    short_labels = [m.replace(" ", "\n") for m in methods]
    x = np.arange(len(methods))

    p5_vals = eval_df["Precision@5"].values
    colors_m = ["#95A5A6", "#3498DB", "#9B59B6", "#27AE60"]
    bars1 = ax1.bar(x, p5_vals, width=0.55, color=colors_m, edgecolor="#333333", linewidth=0.7)
    for bar, val in zip(bars1, p5_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels)
    ax1.set_ylabel("Precision@5 (Higher is Better)")
    ax1.set_title("Top-5 Recommendation Accuracy (Precision@5)")
    ax1.set_ylim(0, max(p5_vals) * 1.25 + 0.05)

    width = 0.35
    rmse_vals = eval_df["RMSE"].values
    mae_vals = eval_df["MAE"].values
    b_rmse = ax2.bar(x - width / 2, rmse_vals, width=width, label="RMSE", color="#E74C3C", edgecolor="#333333", linewidth=0.6)
    b_mae = ax2.bar(x + width / 2, mae_vals, width=width, label="MAE", color="#F39C12", edgecolor="#333333", linewidth=0.6)
    for bar in b_rmse:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in b_mae:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels)
    ax2.set_ylabel("Prediction Error in Stars (Lower is Better)")
    ax2.set_title("Rating Prediction Error Comparison (RMSE & MAE)")
    ax2.set_ylim(0, max(rmse_vals) * 1.25)
    ax2.legend()

    plt.tight_layout()
    chart2 = RESULTS_DIR / "model_comparison.png"
    fig.savefig(chart2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(chart2)

    # ------------------------------------------------------------------
    # Chart 3: Test Case #2 - Plain Star Average vs. Review-Aware Score
    # ------------------------------------------------------------------
    top_plain = full_cmp_df.sort_values("avg_rating", ascending=False).head(4)
    top_review = full_cmp_df.sort_values("review_aware_score", ascending=False).head(5)
    spotlight = pd.concat([top_review, top_plain]).drop_duplicates(subset=["item_id"]).head(8)
    spotlight = spotlight.sort_values("review_aware_score", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1.0]})
    y = np.arange(len(spotlight))
    h_bar = 0.36

    ax1.barh(y - h_bar / 2, spotlight["avg_rating"], height=h_bar, label="Plain Star Average", color="#B0BEC5", edgecolor="#333333", linewidth=0.5)
    ax1.barh(y + h_bar / 2, spotlight["review_aware_score"], height=h_bar, label="Review-Aware Score", color="#1E88E5", edgecolor="#333333", linewidth=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r['stay_name']} ({r['item_id']})" for _, r in spotlight.iterrows()], fontsize=9)
    ax1.invert_yaxis()  # Match top-to-bottom order of the heatmap on the right
    ax1.set_xlabel("Score (1–5 Scale)")
    ax1.set_xlim(2.0, 5.25)
    ax1.set_title("Plain Star Average vs. Review-Aware Score")
    ax1.legend(loc="lower right")

    # Right subplot: Heatmap-style aspect sentiment breakdown (aligned row-for-row)
    aspect_cols = ["cleanliness_score", "location_score", "quietness_score", "service_score"]
    aspect_matrix = spotlight[aspect_cols].values
    im = ax2.imshow(aspect_matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["Cleanliness", "Location", "Quietness", "Service"], fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(spotlight["item_id"].tolist())
    ax2.set_title("NLP Aspect Sentiment Breakdown (-1 to +1)")
    ax2.grid(False)

    for i in range(aspect_matrix.shape[0]):
        for j in range(aspect_matrix.shape[1]):
            val = aspect_matrix[i, j]
            txt_color = "black" if -0.55 < val < 0.65 else "white"
            ax2.text(j, i, f"{val:+.2f}", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Aspect Sentiment")

    plt.tight_layout()
    chart3 = RESULTS_DIR / "aspect_sentiment_comparison.png"
    fig.savefig(chart3, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(chart3)

    return saved_paths


def main() -> None:
    items_df, ratings_df = load_data()
    train_df, test_df = train_test_split_by_user(ratings_df, test_ratio=0.25, min_test_per_user=3, seed=42)

    print(f"Loaded {len(items_df)} stays and {len(ratings_df)} ratings.")
    print(f"Offline Split: {len(train_df)} train ratings | {len(test_df)} test ratings\n")

    eval_df = evaluate_models(items_df, train_df, test_df, k=5, relevance_threshold=4.0)
    report_text, full_cmp_df = generate_test_case_report(items_df, ratings_df, eval_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "sample_outputs.txt"
    out_file.write_text(report_text, encoding="utf-8")

    chart_paths = generate_charts(items_df, ratings_df, eval_df, full_cmp_df)

    print(report_text)
    print(f"\nSaved sample outputs to: {out_file}")
    for path in chart_paths:
        print(f"Saved chart to: {path}")


if __name__ == "__main__":
    main()
