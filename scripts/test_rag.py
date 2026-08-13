from app.rag.retriever import LocalRetriever


def main() -> None:

    retriever = LocalRetriever()

    print("Knowledge Base")
    print("==============")

    info = retriever.info()

    print(
        f"Documents: {info['documents']}"
    )

    print(
        f"Chunks:    {info['chunks']}"
    )

    queries = [
        "What should happen when a high risk customer has multiple support complaints?",
        "Can an AI agent execute arbitrary SQL?",
        "What metrics are used to evaluate the churn model?",
        "What is the policy for secrets and API keys?",
    ]

    for query in queries:

        print("\n" + "=" * 70)

        print(
            f"QUERY: {query}"
        )

        print("=" * 70)

        results = retriever.search(
            query,
            top_k=3,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{rank}] "
                f"{result['title']} "
                f"(score={result['score']})"
            )

            print(
                result["text"][:600]
            )


if __name__ == "__main__":
    main()