from app.agent.tools import execute_tool
from app.db.database import SessionLocal


def main() -> None:

    db = SessionLocal()

    try:
        result = execute_tool(
            db=db,
            tool_name="search_enterprise_knowledge",
            arguments={
                "query": (
                    "What should happen when a high-risk "
                    "customer has multiple support complaints?"
                ),
                "top_k": 3,
            },
        )

        print("RAG TOOL RESULT")
        print("================")

        print(
            f"Query: {result['query']}"
        )

        results = result.get(
            "results",
            [],
        )

        if not results:
            print(
                "No relevant knowledge was retrieved."
            )
            return

        for rank, item in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{rank}] "
                f"{item['title']} "
                f"(score={item['score']})"
            )

            print(
                f"Document: {item['document_id']}"
            )

            print(
                f"Chunk: {item['chunk_id']}"
            )

            print(
                f"Source: {item['source']}"
            )

            print(
                "\n"
                + item["text"]
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()