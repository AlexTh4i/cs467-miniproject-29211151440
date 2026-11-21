"""
NLP Preprocessing, TF-IDF Vectorization, and Aspect-Sentiment Pipeline
for the Hotel and Homestay Review-Aware Recommender (CS467 - MP-A13).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Curated English stop-words for TF-IDF and keyword extraction.
# Negation words ("not", "no", "never", "without") are intentionally excluded
# from negation-aware sentiment stripping and handled separately.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after", "is",
    "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "this", "that", "these", "those", "it", "its",
    "they", "them", "their", "we", "us", "our", "you", "your", "i", "me", "my",
    "he", "him", "his", "she", "her", "as", "so", "than", "too", "very", "just",
    "also", "all", "any", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "then", "there", "here", "when", "where", "why",
    "how", "both", "during", "while", "before", "between", "through", "out",
    "off", "down", "above", "below", "again", "further", "once", "who", "which",
    "what", "whose", "whom", "get", "got", "made", "make", "really", "felt",
    "every", "even", "much", "well", "stay", "stayed", "place", "places", "room",
    "rooms", "one", "two", "minutes", "short",
}

NEGATION_WORDS = {"not", "no", "never", "hardly", "barely", "without", "cant", "cannot", "dont", "didnt", "wasnt", "werent"}

# Project-specific 4 hospitality aspects and their indicator keywords/phrases
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "cleanliness": [
        "clean", "cleanliness", "spotless", "immaculate", "pristine", "tidy",
        "sparkling", "hygiene", "fresh", "housekeeping", "bathroom", "linens",
        "sheets", "towels", "shower", "floor", "dirty", "dusty", "stained",
        "musty", "hair", "sanitary",
    ],
    "location": [
        "location", "beach", "beachfront", "oceanfront", "sea", "sand", "coast",
        "central", "downtown", "walkable", "walk", "steps", "close", "near",
        "convenient", "riverside", "river", "old town", "ancient town", "area",
        "neighborhood", "proximity", "access", "remote", "far", "isolated",
        "inconvenient", "strip", "alley", "lane", "bridge", "market",
    ],
    "quietness": [
        "quiet", "quietness", "peaceful", "calm", "serene", "silent", "silence",
        "tranquil", "soundproof", "sleep", "restful", "relaxing", "noisy",
        "noise", "loud", "bass", "traffic", "party", "construction", "chatter",
        "awake", "thin walls", "paper thin", "crowds", "music", "hum",
    ],
    "service": [
        "service", "host", "hosts", "staff", "reception", "front desk",
        "concierge", "butler", "owner", "hospitality", "check in", "checkout",
        "friendly", "welcoming", "helpful", "attentive", "kind", "warm",
        "polite", "courteous", "responsive", "caring", "gracious", "hospitable",
        "rude", "unhelpful", "unfriendly", "dismissive", "indifferent",
        "grumpy", "unresponsive",
    ],
}

# Positive and negative sentiment lexicons tailored to hospitality reviews
POSITIVE_WORDS = {
    "spotless", "immaculate", "pristine", "clean", "tidy", "sparkling", "fresh",
    "quiet", "peaceful", "calm", "serene", "silent", "tranquil", "soundproof",
    "restful", "relaxing", "friendly", "welcoming", "helpful", "attentive",
    "kind", "warm", "polite", "courteous", "responsive", "caring", "gracious",
    "hospitable", "strong", "fast", "reliable", "stable", "blazing", "excellent",
    "great", "amazing", "wonderful", "awesome", "super", "superb", "unbeatable",
    "ideal", "perfect", "prime", "convenient", "walkable", "effortless",
    "comfortable", "cozy", "charming", "lovely", "gorgeous", "stunning",
    "outstanding", "efficient", "smooth", "thoughtful", "generous", "polished",
    "loved", "love", "cherish", "rave", "praise", "good", "solid", "decent",
    "dependable", "steady", "quick", "prompt", "bright", "airy", "spacious",
}

NEGATIVE_WORDS = {
    "dirty", "dusty", "stained", "musty", "rushed", "noisy", "noise", "loud",
    "bass", "awake", "booming", "shaking", "rude", "unhelpful", "unfriendly",
    "dismissive", "indifferent", "grumpy", "unresponsive", "slow", "weak",
    "flaky", "dropping", "dropped", "broken", "terrible", "terribly", "poor",
    "bad", "disappointing", "letdowns", "letdown", "overhyped", "frustrating",
    "cramped", "remote", "isolated", "inconvenient", "far", "dated", "outdated",
    "aging", "basic", "steep", "pricey", "ruined", "disturbed", "impossible",
    "barely", "problem", "issues", "complaint", "thin", "construction",
    "traffic", "crowds", "highway", "hair",
}

# Direct inherent polarity for strongly opinionated aspect words
INHERENT_ASPECT_POLARITY: Dict[str, Dict[str, float]] = {
    "cleanliness": {
        "spotless": 1.0, "immaculate": 1.0, "pristine": 1.0, "sparkling": 1.0,
        "clean": 0.9, "tidy": 0.8, "fresh": 0.7,
        "dirty": -1.0, "dusty": -0.9, "stained": -1.0, "musty": -0.9, "hair": -0.8,
    },
    "location": {
        "walkable": 0.8, "convenient": 0.8, "prime": 0.9, "unbeatable": 1.0,
        "ideal": 0.9, "central": 0.6, "close": 0.6, "steps": 0.7,
        "remote": -0.7, "isolated": -0.9, "inconvenient": -1.0, "far": -0.6,
    },
    "quietness": {
        "quiet": 0.9, "peaceful": 1.0, "calm": 0.9, "serene": 1.0,
        "silent": 1.0, "tranquil": 1.0, "soundproof": 1.0, "restful": 0.9,
        "noisy": -1.0, "noise": -0.8, "loud": -1.0, "bass": -0.9,
        "awake": -0.9, "construction": -0.9, "thin walls": -0.9, "paper thin": -0.9,
    },
    "service": {
        "friendly": 1.0, "welcoming": 1.0, "helpful": 0.9, "attentive": 1.0,
        "kind": 0.9, "warm": 0.9, "polite": 0.8, "courteous": 0.9,
        "responsive": 0.9, "caring": 1.0, "gracious": 1.0, "hospitable": 1.0,
        "rude": -1.0, "unhelpful": -1.0, "unfriendly": -1.0, "dismissive": -1.0,
        "indifferent": -0.9, "grumpy": -1.0, "unresponsive": -1.0,
    },
}


def clean_text(text: str) -> str:
    """Lowercase text, normalize Wi-Fi variants, remove punctuation, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    cleaned = text.lower()
    cleaned = re.sub(r"\bwi[\s\-]?fi\b", "wifi", cleaned)
    cleaned = re.sub(r"[-/]", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize(text: str) -> List[str]:
    """Clean and split text into lowercase alphanumeric tokens."""
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def remove_stopwords(tokens: List[str], keep_negations: bool = False) -> List[str]:
    """Remove standard stop-words while optionally preserving negation words."""
    if keep_negations:
        return [tok for tok in tokens if tok not in STOP_WORDS or tok in NEGATION_WORDS]
    return [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 1]


def preprocess_text(text: str) -> str:
    """Full NLP preprocessing pipeline returning space-separated cleaned tokens."""
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens, keep_negations=False)
    return " ".join(filtered)


def _split_into_clauses(text: str) -> List[str]:
    """Split a review into clauses along punctuation and contrast conjunctions."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Split on sentence punctuation or contrast words so positive and negative halves stay separated
    pattern = r"[.!?;,]+|\b(?:but|however|yet|though|although|whereas)\b"
    raw_clauses = re.split(pattern, text.lower())
    return [c.strip() for c in raw_clauses if c and c.strip()]


def _has_negation_before(tokens: List[str], idx: int, window: int = 3) -> bool:
    """Check whether a negation word appears within `window` tokens before `idx`."""
    start = max(0, idx - window)
    return any(tok in NEGATION_WORDS for tok in tokens[start:idx])


def analyze_review_aspects(review_text: str) -> Dict[str, Optional[float]]:
    """
    Detect hospitality aspects (cleanliness, location, quietness, service) in a guest review
    and compute clause-level, negation-aware sentiment scores in [-1.0, +1.0].

    Returns a dict mapping each aspect name to its sentiment score (or None if not mentioned).
    """
    clauses = _split_into_clauses(review_text)
    aspect_scores: Dict[str, List[float]] = {aspect: [] for aspect in ASPECT_KEYWORDS}

    for clause in clauses:
        cleaned_clause = clean_text(clause)
        if not cleaned_clause:
            continue
        tokens = cleaned_clause.split()

        # General clause sentiment (with negation flipping)
        clause_sent_values: List[float] = []
        for idx, tok in enumerate(tokens):
            negated = _has_negation_before(tokens, idx, window=3)
            if tok in POSITIVE_WORDS:
                clause_sent_values.append(-1.0 if negated else 1.0)
            elif tok in NEGATIVE_WORDS:
                clause_sent_values.append(0.5 if negated else -1.0)

        clause_general_sentiment = (
            float(np.mean(clause_sent_values)) if clause_sent_values else 0.0
        )

        # Match each aspect in this clause
        for aspect, keywords in ASPECT_KEYWORDS.items():
            matched_keywords = []
            for kw in keywords:
                if " " in kw:
                    if kw in cleaned_clause:
                        matched_keywords.append(kw)
                elif kw in tokens:
                    matched_keywords.append(kw)

            if not matched_keywords:
                continue

            # Check inherent aspect-word polarity first (with negation check)
            inherent_vals: List[float] = []
            inherent_map = INHERENT_ASPECT_POLARITY.get(aspect, {})
            for kw in matched_keywords:
                if kw in inherent_map:
                    pol = inherent_map[kw]
                    if " " not in kw:
                        kw_idx = tokens.index(kw)
                        if _has_negation_before(tokens, kw_idx, window=3):
                            pol = -pol if pol > 0 else 0.5
                    inherent_vals.append(pol)

            if inherent_vals:
                score = float(np.mean(inherent_vals))
                # Blend lightly with clause context if clause has additional opinion cues
                if clause_sent_values:
                    score = 0.7 * score + 0.3 * clause_general_sentiment
            else:
                score = clause_general_sentiment

            aspect_scores[aspect].append(float(np.clip(score, -1.0, 1.0)))

    return {
        aspect: (round(float(np.mean(vals)), 4) if vals else None)
        for aspect, vals in aspect_scores.items()
    }


def detect_query_aspects(query_text: str) -> Dict[str, float]:
    """
    Identify which aspects a user query emphasizes and return normalized aspect weights.
    If no specific aspect keywords are found, return uniform weights across all 4 aspects.
    """
    cleaned = clean_text(query_text)
    tokens = set(cleaned.split())
    counts: Dict[str, int] = {}

    for aspect, keywords in ASPECT_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            if " " in kw and kw in cleaned:
                hits += 1
            elif kw in tokens:
                hits += 1
        counts[aspect] = hits

    total_hits = sum(counts.values())
    if total_hits == 0:
        uniform = 1.0 / len(ASPECT_KEYWORDS)
        return {aspect: uniform for aspect in ASPECT_KEYWORDS}

    # Base floor ensures all 4 aspects still contribute slightly while query aspects dominate
    weights = {aspect: 0.15 + 0.85 * (counts[aspect] / total_hits) for aspect in ASPECT_KEYWORDS}
    norm = sum(weights.values())
    return {aspect: round(w / norm, 4) for aspect, w in weights.items()}


def compute_item_aspect_summary(
    items_df: pd.DataFrame, ratings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate aspect-level sentiment scores per stay across all provided guest reviews.
    Only uses reviews present in `ratings_df` (pass train_ratings during offline evaluation
    to avoid data leakage).
    """
    per_item_aspects: Dict[str, Dict[str, List[float]]] = {
        item_id: {a: [] for a in ASPECT_KEYWORDS} for item_id in items_df["item_id"]
    }
    review_counts: Dict[str, int] = {item_id: 0 for item_id in items_df["item_id"]}

    for _, row in ratings_df.iterrows():
        item_id = row["item_id"]
        if item_id not in per_item_aspects:
            continue
        review_counts[item_id] += 1
        scores = analyze_review_aspects(str(row.get("guest_review", "")))
        for aspect, val in scores.items():
            if val is not None:
                per_item_aspects[item_id][aspect].append(val)

    records = []
    for item_id in items_df["item_id"]:
        rec: Dict[str, object] = {
            "item_id": item_id,
            "review_count": review_counts[item_id],
        }
        valid_aspect_means = []
        for aspect in ASPECT_KEYWORDS:
            vals = per_item_aspects[item_id][aspect]
            mean_val = float(np.mean(vals)) if vals else 0.0
            rec[f"{aspect}_score"] = round(mean_val, 4)
            rec[f"{aspect}_mentions"] = len(vals)
            if vals:
                valid_aspect_means.append(mean_val)

        rec["overall_aspect_sentiment"] = (
            round(float(np.mean(valid_aspect_means)), 4) if valid_aspect_means else 0.0
        )
        records.append(rec)

    return pd.DataFrame(records)


class StayNLPPipeline:
    """
    End-to-end NLP feature extraction pipeline for hospitality stays:
    - Combines and preprocesses stay metadata, amenities, descriptions, and guest reviews.
    - Fits a single shared TF-IDF vectorizer (supporting both item corpus and new queries).
    - Computes stay-level aspect-sentiment profiles.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), max_features: int = 1500):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=1,
        )
        self.items_df: Optional[pd.DataFrame] = None
        self.item_ids: List[str] = []
        self.item_id_to_idx: Dict[str, int] = {}
        self.feature_names: np.ndarray = np.array([])
        self.tfidf_matrix = None
        self.aspect_summary_df: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def _build_stay_documents(
        self, items_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """Construct rich preprocessed text documents combining stay metadata and guest reviews."""
        review_map: Dict[str, List[str]] = {}
        if ratings_df is not None and len(ratings_df) > 0:
            for item_id, group in ratings_df.groupby("item_id"):
                review_map[str(item_id)] = group["guest_review"].fillna("").astype(str).tolist()

        docs: List[str] = []
        for _, row in items_df.iterrows():
            item_id = str(row["item_id"])
            # Emphasize core stay identity, area, amenities, and description, plus guest reviews
            metadata_text = " ".join([
                str(row["stay_name"]),
                str(row["area"]),
                str(row["price_tier"]),
                str(row["amenities_text"]),
                str(row["amenities_text"]),  # slight boost to structured amenities
                str(row["description"]),
            ])
            reviews_text = " ".join(review_map.get(item_id, []))
            combined = f"{metadata_text} {reviews_text}"
            docs.append(preprocess_text(combined))
        return docs

    def fit(
        self, items_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None
    ) -> "StayNLPPipeline":
        """Fit TF-IDF vectorizer on stay documents and compute aspect-sentiment scores."""
        self.items_df = items_df.reset_index(drop=True).copy()
        self.item_ids = self.items_df["item_id"].astype(str).tolist()
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        documents = self._build_stay_documents(self.items_df, ratings_df)
        self.items_df["processed_text"] = documents

        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

        if ratings_df is not None and len(ratings_df) > 0:
            self.aspect_summary_df = compute_item_aspect_summary(self.items_df, ratings_df)
        else:
            empty_ratings = pd.DataFrame(columns=["user_id", "item_id", "rating", "guest_review"])
            self.aspect_summary_df = compute_item_aspect_summary(self.items_df, empty_ratings)

        self.is_fitted = True
        return self

    def vectorize_query(self, query_text: str):
        """
        Transform any new free-text query or stay description using the exact same
        TF-IDF vocabulary and IDF weights fitted on the dataset.
        """
        if not self.is_fitted:
            raise RuntimeError("StayNLPPipeline must be fitted before calling vectorize_query().")
        processed = preprocess_text(query_text)
        return self.vectorizer.transform([processed])

    def compute_query_similarities(self, query_text: str) -> np.ndarray:
        """Compute cosine similarity between a free-text query and every stay."""
        query_vec = self.vectorize_query(query_text)
        return cosine_similarity(query_vec, self.tfidf_matrix).ravel()

    def compute_item_similarity_matrix(self) -> np.ndarray:
        """Return the pairwise item-to-item cosine similarity matrix."""
        if not self.is_fitted:
            raise RuntimeError("StayNLPPipeline must be fitted first.")
        return cosine_similarity(self.tfidf_matrix)

    def get_matching_keywords(
        self, query_vec, item_idx: int, top_k: int = 5
    ) -> List[str]:
        """
        Extract the top overlapping TF-IDF terms contributing most to the cosine similarity
        between a query/user vector and an item vector.
        """
        if not self.is_fitted:
            raise RuntimeError("StayNLPPipeline must be fitted first.")

        q_arr = query_vec.toarray().ravel() if hasattr(query_vec, "toarray") else np.asarray(query_vec).ravel()
        item_arr = self.tfidf_matrix[item_idx].toarray().ravel()
        contributions = q_arr * item_arr

        nonzero_indices = np.where(contributions > 0)[0]
        if len(nonzero_indices) == 0:
            return []

        ranked = nonzero_indices[np.argsort(contributions[nonzero_indices])[::-1]]
        keywords: List[str] = []
        for idx in ranked:
            term = str(self.feature_names[idx])
            if term not in keywords:
                keywords.append(term)
            if len(keywords) >= top_k:
                break
        return keywords

    def get_aspect_scores_for_item(self, item_id: str) -> Dict[str, float]:
        """Retrieve aspect sentiment dictionary for a specific stay."""
        if self.aspect_summary_df is None:
            return {a: 0.0 for a in ASPECT_KEYWORDS}
        row = self.aspect_summary_df[self.aspect_summary_df["item_id"] == item_id]
        if row.empty:
            return {a: 0.0 for a in ASPECT_KEYWORDS}
        r = row.iloc[0]
        return {
            "cleanliness": float(r["cleanliness_score"]),
            "location": float(r["location_score"]),
            "quietness": float(r["quietness_score"]),
            "service": float(r["service_score"]),
            "overall": float(r["overall_aspect_sentiment"]),
        }
