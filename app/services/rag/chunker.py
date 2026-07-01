"""
Chunking strategies:
  1. fixed_size     — simple token-aware sliding window (baseline)
  2. parent_child   — small child chunks for retrieval, large parent for context
"""

from dataclasses import dataclass, field
from uuid import uuid4

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_TOKENIZER.encode(text))


@dataclass
class Chunk:
    id: str
    document_id: str
    filename: str
    page: int | None
    text: str
    token_count: int
    chunk_index: int
    parent_id: str | None = None       # set for child chunks
    metadata: dict = field(default_factory=dict)


def fixed_size_chunks(
    text: str,
    document_id: str,
    filename: str,
    page: int | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Standard overlapping fixed-size chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    return [
        Chunk(
            id=str(uuid4()),
            document_id=document_id,
            filename=filename,
            page=page,
            text=chunk,
            token_count=_token_len(chunk),
            chunk_index=i,
        )
        for i, chunk in enumerate(raw_chunks)
    ]


def parent_child_chunks(
    text: str,
    document_id: str,
    filename: str,
    page: int | None = None,
    parent_size: int = 512,
    child_size: int = 128,
    overlap: int = 20,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Returns (parent_chunks, child_chunks).
    - Store BOTH in vector DB
    - Index child chunks for retrieval (higher precision)
    - When a child chunk matches, return its parent chunk as context (more content)
    """
    parents = fixed_size_chunks(
        text, document_id, filename, page,
        chunk_size=parent_size, chunk_overlap=overlap
    )

    all_children: list[Chunk] = []
    for parent in parents:
        children = fixed_size_chunks(
            parent.text, document_id, filename, page,
            chunk_size=child_size, chunk_overlap=overlap
        )
        for child in children:
            child.parent_id = parent.id
        all_children.extend(children)

    return parents, all_children
