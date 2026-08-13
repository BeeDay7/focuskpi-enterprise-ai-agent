from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    document_id: str
    title: str
    source: str
    text: str


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
}


def load_documents(
    directory: str | Path,
) -> list[Document]:
    """
    Load supported text documents from a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory does not exist: {directory}"
        )

    documents: list[Document] = []

    for path in sorted(directory.iterdir()):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        document_id = path.stem.upper()

        documents.append(
            Document(
                document_id=document_id,
                title=path.stem.replace("_", " ").title(),
                source=str(path),
                text=text,
            )
        )

    return documents