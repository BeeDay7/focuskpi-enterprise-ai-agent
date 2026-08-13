from dataclasses import asdict
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.documents.loader import load_documents
from app.rag.chunker import Chunk, chunk_documents


class LocalRetriever:
    """
    Lightweight local semantic-style retriever using TF-IDF and cosine
    similarity.

    This interface is intentionally provider-neutral so it can later be
    replaced with an embedding + vector database implementation.
    """

    def __init__(
        self,
        knowledge_base_path: str = "data/knowledge_base",
    ) -> None:

        documents = load_documents(
            knowledge_base_path
        )

        self.chunks: list[Chunk] = chunk_documents(
    	    documents,
            chunk_size=120,
            overlap=25,
         )

        if not self.chunks:
            raise RuntimeError(
                "No knowledge-base documents were loaded."
            )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.matrix = self.vectorizer.fit_transform(
            [chunk.text for chunk in self.chunks]
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            return []

        query_vector = self.vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        )[0]

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            score = float(
                scores[index]
            )

            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "score": round(
                        score,
                        4,
                    ),
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "text": chunk.text,
                }
            )

        return results

    def info(self) -> dict[str, int]:
        return {
            "documents": len(
                set(
                    chunk.document_id
                    for chunk in self.chunks
                )
            ),
            "chunks": len(
                self.chunks
            ),
        }