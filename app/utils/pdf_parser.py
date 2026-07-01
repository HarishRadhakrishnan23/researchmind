"""
PDF text extraction using PyMuPDF (fitz).
Returns a list of page dicts: {page_num, text, char_count}
"""

from dataclasses import dataclass
import fitz  # PyMuPDF


@dataclass
class PageContent:
    page_num: int   # 1-indexed
    text: str
    char_count: int


def extract_pages(file_bytes: bytes, filename: str = "") -> list[PageContent]:
    """
    Extract text from a PDF's bytes.
    Filters out pages with fewer than 50 characters (likely images/covers).
    """
    pages: list[PageContent] = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if len(text) < 50:
                continue
            pages.append(PageContent(
                page_num=i,
                text=text,
                char_count=len(text),
            ))

    return pages


def extract_text_flat(file_bytes: bytes) -> str:
    """Single string of all page text joined by newlines. Useful for quick summaries."""
    pages = extract_pages(file_bytes)
    return "\n\n".join(p.text for p in pages)
