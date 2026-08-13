from dataclasses import dataclass

from app.documents.loader import Document


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    source: str
    text: str


def chunk_document(
    document: Document,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Split a document into overlapping word-based chunks.
    """

    words = document.text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks: list[Chunk] = []

    start = 0
    chunk_number = 1

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        text = " ".join(
            words[start:end]
        ).strip()

        chunks.append(
            Chunk(
                chunk_id=(
                    f"{document.document_id}-"
                    f"{chunk_number:04d}"
                ),
                document_id=document.document_id,
                title=document.title,
                source=document.source,
                text=text,
            )
        )

        if end >= len(words):
            break

        start = end - overlap
        chunk_number += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[Chunk]:

    chunks: list[Chunk] = []

    for document in documents:

        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks