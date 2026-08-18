import os
import pathlib
import re

from dotenv import load_dotenv
from src.core.db import upsert_document, store_chunks
from src.ingestion.docling_parser import parse_document

load_dotenv()

_TEXT_CHUNK_SIZE = 512
_TEXT_CHUNK_OVERLAP = 100


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Smart text splitting respecting sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    # Split by double newlines or periods
    sentences = re.split(r'(\n\n|\. )', text)
    chunks: list[str] = []
    current_chunk = ""

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        sep = sentences[i+1] if i+1 < len(sentences) else ""
        piece = sentence + sep

        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Overlap context setup
            overlap_str = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_str + piece

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def run_ingestion(file_path: str) -> dict:
    resolved = pathlib.Path(file_path).resolve()

    doc_id = upsert_document(resolved.name, str(resolved))
    print(f"[ingestion] doc_id={doc_id} file={file_path}")

    print(f"[ingestion] Parsing: {file_path}")
    parsed_elements = parse_document(file_path)
    print(f"[ingestion] Docling produced {len(parsed_elements)} elements")

    chunks: list[dict] = []
    for elem in parsed_elements:
    # Keep tables and images atomic — only split long text chunks
        if elem["content_type"] == "text" and len(elem["content"]) > _TEXT_CHUNK_SIZE:
            section_prefix = f"Section Context: {elem['metadata'].get('section', '')}\n" if elem['metadata'].get('section') else ""

            sub_splits = _split_text(elem["content"], _TEXT_CHUNK_SIZE, _TEXT_CHUNK_OVERLAP)
            for sub in sub_splits:
                final_sub_content = sub if sub.startswith("Section Context:") or sub.startswith("[") else f"{section_prefix}{sub}"
            chunks.append(
            {
            "content": final_sub_content,
            "content_type": elem["content_type"],
            "metadata": elem["metadata"],
            }
            )
        else:
            chunks.append(elem)

    print(f"[ingestion] {len(chunks)} enriched chunks ready for embedding")

    count = store_chunks(chunks, doc_id)
    print(f"[ingestion] Stored {count} chunks → multimodal_chunks")

    return {"status": "success", "doc_id": doc_id, "chunks_ingested": count}


if __name__ == "__main__":
    pdf_path = pathlib.Path("data/KB_Smart_Banking.pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path.resolve()}")

    result = run_ingestion(str(pdf_path))
    print(f"\nIngestion complete: {result}")