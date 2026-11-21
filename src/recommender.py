"""
Recommendation Engine for Hotel and Homestay Review-Aware Recommender (CS467 - MP-A13).

Implements and compares three core approaches:
1. Popularity Baseline (Bayesian weighted rating with minimum rating count filter)
2. NLP Content-Based Recommender (TF-IDF cosine similarity for user profiles,
   item-to-item similarity, and free-text natural-language queries)
3. Project-Specific Extension:
   - User-Based Collaborative Filtering (mean-centered cosine similarity)
   - Review-Aware Hybrid Recommender combining CF, NLP Content-Based, and
     NLP Aspect-Sentiment scores (plus Cold-Start query support and
     Plain Star Average vs. Review-Aware ranking comparison).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import ASPECT_KEYWORDS, StayNLPPipeline, detect_query_aspects


class PopularityRecommender:
    """
    Baseline recommender ranking stays by Bayesian weighted rating (and tracking plain
    star average) while filtering out stays with fewer than `min_ratings` reviews
    and items already rated by the target user.
    """

    def __init__(self, min_ratings: int = 3):
        self.min_ratings = min_ratings
        self.items_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.stats_df: Optional[pd.DataFrame] = None
        self.global_mean: float = 3.5

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "PopularityRecommender":
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy()
        self.global_mean = float(ratings_df["rating"].mean()) if len(ratings_df) > 0 else 3.5

        agg = (
            ratings_df.groupby("item_id")["rating"]
            .agg(rating_count="count", avg_rating="mean")
            .reset_index()
        )
        stats = items_df[["item_id", "stay_name", "area", "price_tier"]].merge(
            agg, on="item_id", how="left"
        )
        stats["rating_count"] = stats["rating_count"].fillna(0).astype(int)
        stats["avg_rating"] = stats["avg_rating"].fillna(self.global_mean)

        # Bayesian weighted popularity score: (v / (v + m)) * R + (m / (v + m)) * C
        v = stats["rating_count"].astype(float)
        m = float(self.min_ratings)
        stats["weighted_score"] = (v / (v + m)) * stats["avg_rating"] + (m / (v + m)) * self.global_mean
        self.stats_df = stats.sort_values(
            ["weighted_score", "rating_count"], ascending=False
        ).reset_index(drop=True)
        return self

    def _user_rated_items(self, user_id: Optional[str]) -> Set[str]:
        if user_id is None or self.ratings_df is None:
            return set()
        return set(self.ratings_df.loc[self.ratings_df["user_id"] == user_id, "item_id"].astype(str))

    def predict_rating(self, user_id: str, item_id: str) -> float:
        if self.stats_df is None:
            return self.global_mean
        row = self.stats_df[self.stats_df["item_id"] == item_id]
        if row.empty:
            return self.global_mean
        return float(row.iloc[0]["weighted_score"])

    def recommend(
        self,
        user_id: Optional[str] = None,
        top_k: int = 5,
        exclude_items: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        if self.stats_df is None:
            raise RuntimeError("PopularityRecommender must be fitted first.")

        excluded = self._user_rated_items(user_id)
        if exclude_items is not None:
            excluded |= set(exclude_items)

        candidates = self.stats_df[
            (self.stats_df["rating_count"] >= self.min_ratings)
            & (~self.stats_df["item_id"].isin(excluded))
        ].copy()

        # Fallback if strict threshold leaves fewer than top_k items
        if len(candidates) < top_k:
            candidates = self.stats_df[~self.stats_df["item_id"].isin(excluded)].copy()

        top = candidates.head(top_k).copy()
        top["score"] = top["weighted_score"].round(4)
        top["avg_rating"] = top["avg_rating"].round(3)
        top["explanation"] = top.apply(
            lambda r: f"Popular stay ({r['avg_rating']:.2f}★ across {int(r['rating_count'])} reviews)",
            axis=1,
        )
        return top[
            ["item_id", "stay_name", "area", "price_tier", "score", "avg_rating", "rating_count", "explanation"]
        ].reset_index(drop=True)


class ContentBasedRecommender:
    """
    NLP Content-Based Recommender powered by TF-IDF cosine similarity.
    Supports:
      (a) Free-text natural-language stay queries
      (b) User-profile recommendations from positively rated past stays
      (c) Item-to-item similar stay recommendations
    """

    def __init__(self, nlp_pipeline: StayNLPPipeline):
        self.nlp = nlp_pipeline
        self.ratings_df: Optional[pd.DataFrame] = None
        self.item_sim_matrix: Optional[np.ndarray] = None
        self.global_mean: float = 3.5

    def fit(self, ratings_df: pd.DataFrame) -> "ContentBasedRecommender":
        if not self.nlp.is_fitted:
            raise RuntimeError("StayNLPPipeline must be fitted before ContentBasedRecommender.fit().")
        self.ratings_df = ratings_df.copy()
        self.global_mean = float(ratings_df["rating"].mean()) if len(ratings_df) > 0 else 3.5
        self.item_sim_matrix = self.nlp.compute_item_similarity_matrix()
        return self

    def _build_user_profile_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Build a weighted TF-IDF user profile emphasizing stays rated >= 4."""
        if self.ratings_df is None:
            return None
        user_rows = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_rows.empty:
            return None

        liked = user_rows[user_rows["rating"] >= 4]
        source_rows = liked if not liked.empty else user_rows

        n_features = self.nlp.tfidf_matrix.shape[1]
        profile = np.zeros(n_features, dtype=float)
        total_weight = 0.0

        for _, row in source_rows.iterrows():
            iid = str(row["item_id"])
            if iid not in self.nlp.item_id_to_idx:
                continue
            idx = self.nlp.item_id_to_idx[iid]
            weight = max(0.5, float(row["rating"]) - 2.5)
            profile += weight * self.nlp.tfidf_matrix[idx].toarray().ravel()
            total_weight += weight

        if total_weight == 0.0 or np.linalg.norm(profile) == 0.0:
            return None
        return (profile / np.linalg.norm(profile)).reshape(1, -1)

    def recommend_by_query(
        self,
        query_text: str,
        top_k: int = 5,
        exclude_items: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Recommend Top-K stays matching a free-text query via TF-IDF cosine similarity."""
        query_vec = self.nlp.vectorize_query(query_text)
        sims = cosine_similarity(query_vec, self.nlp.tfidf_matrix).ravel()
        excluded = set(exclude_items) if exclude_items else set()

        rows = []
        for idx, sim in enumerate(sims):
            item_row = self.nlp.items_df.iloc[idx]
            iid = str(item_row["item_id"])
            if iid in excluded:
                continue
            keywords = self.nlp.get_matching_keywords(query_vec, idx, top_k=5)
            rows.append({
                "item_id": iid,
                "stay_name": item_row["stay_name"],
                "area": item_row["area"],
                "price_tier": item_row["price_tier"],
                "score": round(float(sim), 4),
                "matching_keywords": ", ".join(keywords) if keywords else "general semantic match",
                "explanation": (
                    f"Matched keywords: {', '.join(keywords)}"
                    if keywords
                    else "Low direct keyword overlap"
                ),
            })

        res = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)
        return res.reset_index(drop=True)

    def recommend_similar_items(self, item_id: str, top_k: int = 5) -> pd.DataFrame:
        """Item-to-item content-based recommendations."""
        if item_id not in self.nlp.item_id_to_idx:
            raise ValueError(f"Unknown item_id: {item_id}")
        target_idx = self.nlp.item_id_to_idx[item_id]
        target_vec = self.nlp.tfidf_matrix[target_idx]
        sims = self.item_sim_matrix[target_idx]

        rows = []
        for idx, sim in enumerate(sims):
            if idx == target_idx:
                continue
            item_row = self.nlp.items_df.iloc[idx]
            keywords = self.nlp.get_matching_keywords(target_vec, idx, top_k=5)
            rows.append({
                "item_id": str(item_row["item_id"]),
                "stay_name": item_row["stay_name"],
                "area": item_row["area"],
                "price_tier": item_row["price_tier"],
                "score": round(float(sim), 4),
                "matching_keywords": ", ".join(keywords),
                "explanation": f"Shared features: {', '.join(keywords)}",
            })

        return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_items: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Recommend Top-K unrated stays matching the user's TF-IDF taste profile."""
        profile_vec = self._build_user_profile_vector(user_id)
        if profile_vec is None:
            return pd.DataFrame(
                columns=["item_id", "stay_name", "area", "price_tier", "score", "matching_keywords", "explanation"]
            )

        sims = cosine_similarity(profile_vec, self.nlp.tfidf_matrix).ravel()
        rated_items = set(
            self.ratings_df.loc[self.ratings_df["user_id"] == user_id, "item_id"].astype(str)
        )
        excluded = rated_items | (set(exclude_items) if exclude_items else set())

        rows = []
        for idx, sim in enumerate(sims):
            item_row = self.nlp.items_df.iloc[idx]
            iid = str(item_row["item_id"])
            if iid in excluded:
                continue
            keywords = self.nlp.get_matching_keywords(profile_vec, idx, top_k=5)
            rows.append({
                "item_id": iid,
                "stay_name": item_row["stay_name"],
                "area": item_row["area"],
                "price_tier": item_row["price_tier"],
                "score": round(float(sim), 4),
                "matching_keywords": ", ".join(keywords),
                "explanation": f"Matches user taste keywords: {', '.join(keywords)}",
            })

        return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    def get_user_content_scores(self, user_id: str) -> Dict[str, float]:
        """Return raw TF-IDF cosine similarity for every stay relative to a user's profile."""
        profile_vec = self._build_user_profile_vector(user_id)
        if profile_vec is None:
            return {iid: 0.0 for iid in self.nlp.item_ids}
        sims = cosine_similarity(profile_vec, self.nlp.tfidf_matrix).ravel()
        return {iid: float(sims[idx]) for idx, iid in enumerate(self.nlp.item_ids)}

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating via TF-IDF item-to-item weighted average of the user's rated stays."""
        if self.ratings_df is None or item_id not in self.nlp.item_id_to_idx:
            return self.global_mean
        user_rows = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_rows.empty:
            return self.global_mean

        target_idx = self.nlp.item_id_to_idx[item_id]
        num, den = 0.0, 0.0
        for _, row in user_rows.iterrows():
            other_id = str(row["item_id"])
            if other_id == item_id or other_id not in self.nlp.item_id_to_idx:
                continue
            sim = float(self.item_sim_matrix[target_idx, self.nlp.item_id_to_idx[other_id]])
            if sim > 0:
                num += sim * float(row["rating"])
                den += sim

        if den == 0.0:
            return float(user_rows["rating"].mean())
        return float(np.clip(num / den, 1.0, 5.0))


class UserBasedCFRecommender:
    """
    User-Based Collaborative Filtering (UBCF) using co-rated Pearson correlation
    with significance weighting (shrinkage) over the sparse user-item rating matrix.
    """

    def __init__(self, k_neighbors: int = 4, min_similarity: float = 0.05, shrinkage: float = 3.0):
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.shrinkage = shrinkage
        self.items_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_means: Optional[pd.Series] = None
        self.item_means: Optional[pd.Series] = None
        self.user_sim_df: Optional[pd.DataFrame] = None
        self.global_mean: float = 3.5

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "UserBasedCFRecommender":
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy()
        self.global_mean = float(ratings_df["rating"].mean()) if len(ratings_df) > 0 else 3.5

        pivot = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating")
        self.user_item_matrix = pivot
        self.user_means = pivot.mean(axis=1)
        self.item_means = pivot.mean(axis=0)

        users = pivot.index.tolist()
        n_users = len(users)
        sim_mat = np.zeros((n_users, n_users), dtype=float)
        mat = pivot.values
        means = self.user_means.values

        for i in range(n_users):
            for j in range(i + 1, n_users):
                common = ~np.isnan(mat[i]) & ~np.isnan(mat[j])
                n_common = int(common.sum())
                if n_common < 2:
                    continue
                di = mat[i, common] - means[i]
                dj = mat[j, common] - means[j]
                denom = float(np.linalg.norm(di) * np.linalg.norm(dj))
                if denom == 0.0:
                    continue
                pearson = float(np.dot(di, dj) / denom)
                # Significance weighting down-weights noisy similarities from very few co-rated items
                weighted_sim = (n_common / (n_common + self.shrinkage)) * pearson
                sim_mat[i, j] = weighted_sim
                sim_mat[j, i] = weighted_sim

        self.user_sim_df = pd.DataFrame(sim_mat, index=users, columns=users)
        return self

    def get_similar_users(self, user_id: str, top_k: int = 3) -> List[tuple]:
        """Return top-k most similar neighbors with positive similarity."""
        if self.user_sim_df is None or user_id not in self.user_sim_df.index:
            return []
        sims = self.user_sim_df.loc[user_id]
        pos = sims[sims > 0].sort_values(ascending=False).head(top_k)
        return [(uid, round(float(val), 4)) for uid, val in pos.items()]

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating r_hat(u, i) from top-k positively correlated neighbors."""
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            if self.item_means is not None and item_id in self.item_means.index:
                return float(self.item_means[item_id])
            return self.global_mean

        u_mean = float(self.user_means[user_id])
        if item_id not in self.user_item_matrix.columns:
            return u_mean

        item_col = self.user_item_matrix[item_id]
        raters = item_col.dropna().index
        sims = self.user_sim_df.loc[user_id, raters]
        pos_neighbors = sims[sims >= self.min_similarity].sort_values(ascending=False).head(self.k_neighbors)
        if pos_neighbors.empty:
            pos_neighbors = sims[sims > 0].sort_values(ascending=False).head(self.k_neighbors)

        if pos_neighbors.empty:
            i_mean = float(self.item_means[item_id]) if item_id in self.item_means.index else self.global_mean
            return float(np.clip(0.5 * u_mean + 0.5 * i_mean, 1.0, 5.0))

        num = 0.0
        den = 0.0
        for neighbor_id, sim in pos_neighbors.items():
            deviation = float(self.user_item_matrix.loc[neighbor_id, item_id]) - float(self.user_means[neighbor_id])
            num += float(sim) * deviation
            den += abs(float(sim))

        pred = u_mean + (num / den) if den > 0 else u_mean
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_items: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Recommend Top-K unrated stays for an existing user via UBCF."""
        if self.items_df is None or self.ratings_df is None:
            raise RuntimeError("UserBasedCFRecommender must be fitted first.")

        rated_items = set(
            self.ratings_df.loc[self.ratings_df["user_id"] == user_id, "item_id"].astype(str)
        )
        excluded = rated_items | (set(exclude_items) if exclude_items else set())
        neighbors = self.get_similar_users(user_id, top_k=3)
        neighbor_str = ", ".join(f"{uid} ({s:.2f})" for uid, s in neighbors) or "fallback mean"

        rows = []
        for _, item_row in self.items_df.iterrows():
            iid = str(item_row["item_id"])
            if iid in excluded:
                continue
            pred = self.predict_rating(user_id, iid)
            rows.append({
                "item_id": iid,
                "stay_name": item_row["stay_name"],
                "area": item_row["area"],
                "price_tier": item_row["price_tier"],
                "score": round(pred, 4),
                "explanation": f"UBCF predicted {pred:.2f}★ from similar guests: {neighbor_str}",
            })

        return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


class HybridReviewAwareRecommender:
    """
    Review-Aware Hybrid Recommender combining:
      1. User-Based Collaborative Filtering (guest rating patterns)
      2. NLP Content-Based TF-IDF similarity (stay amenities, descriptions, queries)
      3. NLP Aspect-Sentiment scores (cleanliness, location, quietness, service)

    Gracefully handles:
      - Existing users (CF + Content + Aspect Sentiment)
      - Cold-start users entering only a natural-language stay request (Query TF-IDF +
        Query-Weighted Aspect Sentiment + Popularity prior)
      - Review-aware vs. plain star average ranking comparison (Test Case #2)
    """

    def __init__(
        self,
        nlp_pipeline: StayNLPPipeline,
        popularity_model: PopularityRecommender,
        content_model: ContentBasedRecommender,
        cf_model: UserBasedCFRecommender,
        w_cf: float = 0.40,
        w_cb: float = 0.45,
        w_aspect: float = 0.15,
    ):
        self.nlp = nlp_pipeline
        self.popularity = popularity_model
        self.content = content_model
        self.cf = cf_model
        self.w_cf = w_cf
        self.w_cb = w_cb
        self.w_aspect = w_aspect

    def _aspect_score_01(self, item_id: str, weights: Optional[Dict[str, float]] = None) -> float:
        """Map stay aspect sentiment from [-1, +1] to [0, 1], optionally weighted by aspect priorities."""
        scores = self.nlp.get_aspect_scores_for_item(item_id)
        if weights is None:
            raw = scores["overall"]
        else:
            raw = sum(weights[a] * scores[a] for a in ASPECT_KEYWORDS)
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    def _format_aspect_brief(self, item_id: str) -> str:
        scores = self.nlp.get_aspect_scores_for_item(item_id)
        return (
            f"clean={scores['cleanliness']:+.2f}, "
            f"loc={scores['location']:+.2f}, "
            f"quiet={scores['quietness']:+.2f}, "
            f"serv={scores['service']:+.2f}"
        )

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Review-aware hybrid numeric rating prediction combining UBCF, Content-Based
        item similarity, and a calibrated review aspect-sentiment adjustment.
        """
        cf_pred = self.cf.predict_rating(user_id, item_id)
        cb_pred = self.content.predict_rating(user_id, item_id)
        aspect_overall = self.nlp.get_aspect_scores_for_item(item_id)["overall"]
        # Blend UBCF (55%) + Content-Based (45%) with subtle aspect-sentiment calibration
        pred = 0.55 * cf_pred + 0.45 * cb_pred + 0.08 * aspect_overall
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_items: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Personalized Top-K recommendations for an existing user."""
        if self.cf.user_item_matrix is None or user_id not in self.cf.user_item_matrix.index:
            # Unknown user without query -> fall back to review-aware popularity
            return self.compare_plain_vs_review_aware(top_k=top_k, sort_by="review_aware")

        rated_items = set(
            self.cf.ratings_df.loc[self.cf.ratings_df["user_id"] == user_id, "item_id"].astype(str)
        )
        excluded = rated_items | (set(exclude_items) if exclude_items else set())

        cb_scores = self.content.get_user_content_scores(user_id)
        max_cb = max(cb_scores.values()) if cb_scores else 1.0
        max_cb = max(max_cb, 1e-6)
        profile_vec = self.content._build_user_profile_vector(user_id)

        rows = []
        for idx, item_row in self.nlp.items_df.iterrows():
            iid = str(item_row["item_id"])
            if iid in excluded:
                continue

            cf_pred = self.cf.predict_rating(user_id, iid)
            cf_norm = (cf_pred - 1.0) / 4.0
            cb_norm = cb_scores.get(iid, 0.0) / max_cb
            aspect_norm = self._aspect_score_01(iid)

            hybrid_score = (
                self.w_cf * cf_norm + self.w_cb * cb_norm + self.w_aspect * aspect_norm
            )
            keywords = (
                self.nlp.get_matching_keywords(profile_vec, idx, top_k=4)
                if profile_vec is not None
                else []
            )
            aspect_str = self._format_aspect_brief(iid)
            rows.append({
                "item_id": iid,
                "stay_name": item_row["stay_name"],
                "area": item_row["area"],
                "price_tier": item_row["price_tier"],
                "score": round(float(hybrid_score), 4),
                "cf_pred": round(cf_pred, 2),
                "cb_sim": round(cb_scores.get(iid, 0.0), 3),
                "aspect_sentiment": round(self.nlp.get_aspect_scores_for_item(iid)["overall"], 3),
                "matching_keywords": ", ".join(keywords),
                "explanation": f"CF={cf_pred:.2f}★ | Aspects({aspect_str}) | Keywords: {', '.join(keywords)}",
            })

        return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    def recommend_by_query(
        self,
        query_text: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        exclude_items: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Review-aware recommendations for a natural-language stay request.
        Works seamlessly for brand-new cold-start guests (`user_id=None`) as well as
        existing guests adding an explicit trip query.
        """
        query_vec = self.nlp.vectorize_query(query_text)
        query_sims = cosine_similarity(query_vec, self.nlp.tfidf_matrix).ravel()
        max_sim = max(float(query_sims.max()), 1e-6)
        query_aspect_weights = detect_query_aspects(query_text)

        excluded = set(exclude_items) if exclude_items else set()
        has_history = (
            user_id is not None
            and self.cf.user_item_matrix is not None
            and user_id in self.cf.user_item_matrix.index
        )
        if has_history:
            excluded |= set(
                self.cf.ratings_df.loc[self.cf.ratings_df["user_id"] == user_id, "item_id"].astype(str)
            )

        rows = []
        for idx, item_row in self.nlp.items_df.iterrows():
            iid = str(item_row["item_id"])
            if iid in excluded:
                continue

            raw_sim = float(query_sims[idx])
            sim_norm = raw_sim / max_sim
            aspect_norm = self._aspect_score_01(iid, weights=query_aspect_weights)
            pop_norm = (self.popularity.predict_rating(user_id or "", iid) - 1.0) / 4.0

            if has_history:
                cf_norm = (self.cf.predict_rating(user_id, iid) - 1.0) / 4.0
                hybrid_score = 0.45 * sim_norm + 0.25 * aspect_norm + 0.20 * cf_norm + 0.10 * pop_norm
            else:
                # Pure Cold-Start Review-Aware Query Ranking
                hybrid_score = 0.55 * sim_norm + 0.30 * aspect_norm + 0.15 * pop_norm

            keywords = self.nlp.get_matching_keywords(query_vec, idx, top_k=5)
            aspect_scores = self.nlp.get_aspect_scores_for_item(iid)
            aspect_str = self._format_aspect_brief(iid)
            rows.append({
                "item_id": iid,
                "stay_name": item_row["stay_name"],
                "area": item_row["area"],
                "price_tier": item_row["price_tier"],
                "score": round(float(hybrid_score), 4),
                "tfidf_sim": round(raw_sim, 4),
                "aspect_sentiment": round(aspect_scores["overall"], 3),
                "matching_keywords": ", ".join(keywords) if keywords else "general match",
                "explanation": (
                    f"TF-IDF={raw_sim:.3f} | Aspects({aspect_str}) | "
                    f"Keywords: {', '.join(keywords) if keywords else 'n/a'}"
                ),
            })

        return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    def compare_plain_vs_review_aware(
        self, top_k: int = 8, sort_by: str = "review_aware"
    ) -> pd.DataFrame:
        """
        Compare stays ranked by Plain Star Average vs. Review-Aware Score
        (combining star average with NLP aspect-sentiment converted to the 1-5 scale).
        Directly demonstrates Test Case #2.
        """
        stats = self.popularity.stats_df.copy()
        stats = stats[stats["rating_count"] >= self.popularity.min_ratings].copy()

        aspect_df = self.nlp.aspect_summary_df[
            [
                "item_id",
                "cleanliness_score",
                "location_score",
                "quietness_score",
                "service_score",
                "overall_aspect_sentiment",
            ]
        ]
        merged = stats.merge(aspect_df, on="item_id", how="left")

        # Map overall aspect sentiment [-1, +1] -> [1.0, 5.0] review-quality equivalent
        merged["aspect_star_equiv"] = 3.0 + 2.0 * merged["overall_aspect_sentiment"]
        # Review-Aware Score: 55% plain star average + 45% NLP aspect-sentiment star equivalent
        merged["review_aware_score"] = (
            0.55 * merged["avg_rating"] + 0.45 * merged["aspect_star_equiv"]
        )

        merged["plain_rank"] = (
            merged["avg_rating"].rank(ascending=False, method="min").astype(int)
        )
        merged["review_aware_rank"] = (
            merged["review_aware_score"].rank(ascending=False, method="min").astype(int)
        )
        merged["rank_change"] = merged["plain_rank"] - merged["review_aware_rank"]

        sort_col = "review_aware_score" if sort_by == "review_aware" else "avg_rating"
        merged = merged.sort_values([sort_col, "rating_count"], ascending=False).reset_index(drop=True)

        merged["avg_rating"] = merged["avg_rating"].round(3)
        merged["review_aware_score"] = merged["review_aware_score"].round(3)
        merged["overall_aspect_sentiment"] = merged["overall_aspect_sentiment"].round(3)
        merged["score"] = merged["review_aware_score"]
        return merged.head(top_k)
