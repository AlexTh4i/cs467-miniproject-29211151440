"""
Runnable CLI Demo for the Hotel and Homestay Review-Aware Recommender (CS467 - MP-A13).

Examples:
    python src/main.py
    python src/main.py --user U001
    python src/main.py --query "quiet homestay near beach with strong wifi and friendly host"
    python src/main.py --user U001 --query "peaceful oceanfront homestay with fast wifi"
    python src/main.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import build_models, load_data
from nlp_pipeline import detect_query_aspects


def print_header(title: str) -> None:
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


def print_top_recs(df, score_label: str = "Score") -> None:
    if df.empty:
        print("  (No recommendations available)")
        return
    for rank, r in df.iterrows():
        print(
            f"  #{rank + 1} [{r['item_id']}] {r['stay_name']} "
            f"({r['area']} | {r['price_tier']}) — {score_label}: {r['score']:.4f}"
        )
        if "explanation" in r and r["explanation"]:
            print(f"      -> {r['explanation']}")


def run_user_demo(user_id: str, pop, cb, cf, hybrid, ratings_df, top_k: int = 5) -> None:
    user_history = ratings_df[ratings_df["user_id"] == user_id]
    print_header(f"PERSONALIZED RECOMMENDATIONS FOR USER: {user_id}")
    if user_history.empty:
        print(f"  User '{user_id}' has no rating history (Cold-Start guest).")
        print("  Tip: Provide a natural-language --query for personalized cold-start results!")
        print("\n[Fallback] Top Stays by Review-Aware Quality:")
        fallback = hybrid.compare_plain_vs_review_aware(top_k=top_k, sort_by="review_aware")
        for rank, r in fallback.iterrows():
            print(
                f"  #{rank + 1} [{r['item_id']}] {r['stay_name']} ({r['area']} | {r['price_tier']}) "
                f"— Review-Aware: {r['review_aware_score']:.3f} (Plain: {r['avg_rating']:.2f}★, "
                f"Aspect: {r['overall_aspect_sentiment']:+.3f})"
            )
        return

    print(
        f"  Past history: {len(user_history)} stays rated | "
        f"Mean rating: {user_history['rating'].mean():.2f}★"
    )
    print("\n[1] Popularity Baseline (Top 5):")
    print_top_recs(pop.recommend(user_id=user_id, top_k=top_k), score_label="Weighted Rating")

    print("\n[2] NLP Content-Based User Profile (Top 5):")
    print_top_recs(cb.recommend_for_user(user_id=user_id, top_k=top_k), score_label="TF-IDF Cosine")

    print("\n[3] User-Based Collaborative Filtering (Top 5):")
    print_top_recs(cf.recommend(user_id=user_id, top_k=top_k), score_label="Pred Rating")

    print("\n[4] Review-Aware Hybrid Recommender (Top 5):")
    print_top_recs(hybrid.recommend_for_user(user_id=user_id, top_k=top_k), score_label="Hybrid Score")


def run_query_demo(query_text: str, cb, hybrid, user_id: str | None = None, top_k: int = 5) -> None:
    mode = f"User {user_id} + Query" if user_id else "Cold-Start / Free-Text Query"
    print_header(f"NATURAL-LANGUAGE STAY QUERY DEMO ({mode})")
    print(f'  Query: "{query_text}"')
    weights = detect_query_aspects(query_text)
    print("  Detected Aspect Weights: " + ", ".join(f"{k}={v:.2f}" for k, v in weights.items()))

    print("\n[A] NLP Content-Based (TF-IDF Cosine Similarity) - Top 5:")
    print_top_recs(cb.recommend_by_query(query_text, top_k=top_k), score_label="TF-IDF Cosine")

    print("\n[B] Review-Aware Hybrid (Query TF-IDF + Aspect Sentiment) - Top 5:")
    print_top_recs(
        hybrid.recommend_by_query(query_text, top_k=top_k, user_id=user_id),
        score_label="Hybrid Score",
    )


def run_required_test_cases(pop, cb, cf, hybrid, ratings_df) -> None:
    """Execute and display all 3 assignment-required test cases cleanly."""
    print_header("MP-A13 DEMO: HOTEL & HOMESTAY REVIEW-AWARE RECOMMENDER")
    print("Student: Nguyen Ngoc Hoang Thai (29211151440) | CS467A (No. 13)")

    # Test Case 1: Required Stay Query
    run_query_demo(
        "quiet homestay near beach with strong wifi and friendly host",
        cb=cb,
        hybrid=hybrid,
        user_id=None,
        top_k=5,
    )

    # Test Case 2: Review-Aware Comparison vs. Plain Star Average
    print_header("TEST CASE 2: REVIEW-AWARE RANKING VS. PLAIN STAR AVERAGE")
    plain_top = hybrid.compare_plain_vs_review_aware(top_k=6, sort_by="plain")
    review_top = hybrid.compare_plain_vs_review_aware(top_k=6, sort_by="review_aware")
    cols = [
        "plain_rank",
        "review_aware_rank",
        "rank_change",
        "item_id",
        "stay_name",
        "avg_rating",
        "overall_aspect_sentiment",
        "review_aware_score",
    ]
    print("\nTop 6 by PLAIN STAR AVERAGE:")
    print(plain_top[cols].to_string(index=False))
    print("\nTop 6 by REVIEW-AWARE SCORE (55% Star Avg + 45% Aspect Sentiment):")
    print(review_top[cols].to_string(index=False))

    # Test Case 3: Cold-Start Brand-New Guest
    run_query_demo(
        "peaceful heritage homestay near ancient town with spotless room, kind local host, and reliable wifi",
        cb=cb,
        hybrid=hybrid,
        user_id="NEW_GUEST_01",
        top_k=5,
    )

    # Bonus: Existing User Personalized Comparison
    run_user_demo("U001", pop=pop, cb=cb, cf=cf, hybrid=hybrid, ratings_df=ratings_df, top_k=5)


def run_interactive_mode(pop, cb, cf, hybrid, ratings_df) -> None:
    """Interactive terminal menu for live grading Q&A."""
    valid_users = sorted(ratings_df["user_id"].unique().tolist())
    print_header("INTERACTIVE MODE - HOTEL & HOMESTAY REVIEW-AWARE RECOMMENDER")
    print(f"Available existing users: {', '.join(valid_users)}")

    while True:
        print("\nChoose an option:")
        print("  1. Recommend by existing User ID (compare all 4 methods)")
        print("  2. Recommend by free-text stay query (Cold-Start / NLP + Aspect Sentiment)")
        print("  3. Compare Plain Star Average vs. Review-Aware Ranking")
        print("  4. Run all 3 required test cases")
        print("  q. Quit")

        choice = input("\nEnter choice [1-4/q]: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            print("Goodbye!")
            break
        elif choice == "1":
            uid = input(f"Enter User ID (e.g., {valid_users[0]}): ").strip().upper()
            run_user_demo(uid, pop, cb, cf, hybrid, ratings_df, top_k=5)
        elif choice == "2":
            q = input("Describe your ideal stay: ").strip()
            if q:
                uid = input("Optional User ID (press Enter for Cold-Start guest): ").strip().upper() or None
                run_query_demo(q, cb, hybrid, user_id=uid, top_k=5)
        elif choice == "3":
            cmp_df = hybrid.compare_plain_vs_review_aware(top_k=10, sort_by="review_aware")
            cols = [
                "plain_rank", "review_aware_rank", "rank_change", "item_id",
                "stay_name", "avg_rating", "overall_aspect_sentiment", "review_aware_score",
            ]
            print(cmp_df[cols].to_string(index=False))
        elif choice == "4":
            run_required_test_cases(pop, cb, cf, hybrid, ratings_df)
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or q.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hotel and Homestay Review-Aware Recommender CLI Demo (CS467 - MP-A13)"
    )
    parser.add_argument("--user", type=str, default=None, help="Existing User ID (e.g., U001)")
    parser.add_argument("--query", type=str, default=None, help="Free-text natural-language stay request")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations to return (default: 5)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal menu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items_df, ratings_df = load_data()
    _, pop, cb, cf, hybrid = build_models(items_df, ratings_df)

    if args.interactive:
        run_interactive_mode(pop, cb, cf, hybrid, ratings_df)
    elif args.query:
        run_query_demo(args.query, cb=cb, hybrid=hybrid, user_id=args.user, top_k=args.top_k)
    elif args.user:
        run_user_demo(args.user, pop=pop, cb=cb, cf=cf, hybrid=hybrid, ratings_df=ratings_df, top_k=args.top_k)
    else:
        run_required_test_cases(pop, cb, cf, hybrid, ratings_df)


if __name__ == "__main__":
    main()
