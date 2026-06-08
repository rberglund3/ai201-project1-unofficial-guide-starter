"""
Milestone 3 — Ingestion and Chunking
====================================

RAG pipeline stage for "The Unofficial Guide" (off-campus housing near Georgia Tech).

Pipeline implemented here:

    [ Document Ingestion ] -> (Local Files in data/raw/)
            |
            v
    [ Cleaning & Regex ] -> (BeautifulSoup + regex strip boilerplate)
            |
            v
    [ Chunking ] -> (LangChain RecursiveCharacterTextSplitter, token-based)
            |
            v
    [ Output JSON ] -> (data/processed/chunks.json)

Chunking strategy (from planning.md):
    - chunk_size  = 800 tokens
    - chunk_overlap = 150 tokens
    An 800-token chunk is large enough to hold a full Reddit rant or a complete
    pricing table without cutting it off mid-thought; the 150-token overlap keeps
    a complex's name (usually stated up top) attached to the complaints below it.

Run:
    python ingest_and_chunk.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/chunks.json")

# File types we know how to load from disk.
SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm"}

# Chunking parameters — see module docstring / planning.md for the reasoning.
CHUNK_SIZE = 250       # tokens
CHUNK_OVERLAP = 40    # tokens
ENCODING_NAME = "cl100k_base"  # tiktoken tokenizer used by modern embedding models

# Sanity bounds for the corpus — flag if we fall outside a "healthy" range.
MIN_HEALTHY_CHUNKS = 50
MAX_HEALTHY_CHUNKS = 2_000

# Boilerplate phrases that commonly survive HTML stripping. Matched
# case-insensitively against whole lines so we don't nuke legitimate prose.
BOILERPLATE_LINE_PATTERNS = [
    r"read more",
    r"show more",
    r"continue reading",
    r"share\b.*",
    r"\d+\s+comments?",
    r"\d+\s+(?:upvotes?|likes?|points?)",
    r"reply\b",
    r"sign (?:in|up)",
    r"log ?in",
    r"subscribe",
    r"cookie",
    r"we use cookies",
    r"accept (?:all )?cookies",
    r"privacy policy",
    r"terms (?:of service|& conditions)",
    r"all rights reserved",
    r"©.*",
    r"©\s*\d{4}.*",
    r"copyright.*",
    r"follow us",
    r"back to top",
    r"skip to (?:main )?content",
    r"menu\b",
    r"navigation",
    r"advertisement",
    r"sponsored",
    r"powered by.*",
]

# HTML tags that never contain substantive content — drop them wholesale
# (along with their children) before extracting text.
JUNK_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "button", "svg", "iframe",
]


# --------------------------------------------------------------------------- #
# Stage 1 — Load files from disk
# --------------------------------------------------------------------------- #
def load_raw_documents(raw_dir: Path) -> list[dict]:
    """Load every supported file in `raw_dir` into a uniform internal format.

    Returns a list of dicts: {"source": <filename>, "suffix": <ext>, "raw": <text>}.
    Keeping the raw text and the file suffix lets the cleaning stage decide
    whether to run the HTML parser.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory '{raw_dir}' not found. "
            f"Create it and drop your .txt/.md/.html files inside."
        )

    documents: list[dict] = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue  # silently skip unsupported files (e.g. .DS_Store, .pdf)
        # errors="ignore" guards against stray bad bytes in scraped pages.
        text = path.read_text(encoding="utf-8", errors="ignore")
        documents.append(
            {"source": path.name, "suffix": path.suffix.lower(), "raw": text}
        )

    if not documents:
        raise FileNotFoundError(
            f"No supported files ({sorted(SUPPORTED_SUFFIXES)}) found in '{raw_dir}'."
        )

    print(f"[LOAD] Loaded {len(documents)} document(s) from '{raw_dir}':")
    for doc in documents:
        print(f"       - {doc['source']} ({len(doc['raw']):,} chars)")
    return documents


# --------------------------------------------------------------------------- #
# Stage 2 — Rigorous cleaning
# --------------------------------------------------------------------------- #
def _strip_html(raw: str) -> str:
    """Remove junk tags and extract visible text from an HTML document."""
    soup = BeautifulSoup(raw, "html.parser")

    # Drop tags that structurally cannot hold content we care about.
    for tag in soup(JUNK_TAGS):
        tag.decompose()

    # Drop obvious cookie/ad/nav containers identified by class or id.
    junk_keywords = ("cookie", "banner", "ad-", "advert", "nav", "menu",
                     "footer", "header", "share", "social", "popup", "modal")
    for element in soup.find_all(attrs={"class": True}):
        # Make sure the element is a valid HTML tag and has attributes
        if hasattr(element, "attrs") and element.attrs is not None:
            classes = " ".join(element.get("class", []) or []).lower()
        else:
            classes = ""
        if any(kw in classes for kw in junk_keywords):
            element.decompose()
    for element in soup.find_all(attrs={"id": True}):
        if any(kw in element.get("id", "").lower() for kw in junk_keywords):
            element.decompose()

    # newline separator keeps paragraph/line boundaries for later line filtering.
    return soup.get_text(separator="\n")


def _looks_like_boilerplate(line: str) -> bool:
    """True if an entire line matches a known boilerplate pattern."""
    stripped = line.strip().lower()
    if not stripped:
        return False
    for pattern in BOILERPLATE_LINE_PATTERNS:
        # Anchor to the full line so "I want to share my experience" survives
        # but a standalone "Share" button label is removed.
        if re.fullmatch(pattern, stripped):
            return True
    return False


def clean_document(doc: dict) -> dict:
    """Aggressively clean one document's raw text.

    Removes: HTML tags/attributes, nav menus, cookie banners, ads, footers,
    repeated headers, "Read more"/share/comment-count lines, and HTML entities.
    Keeps: review text, opinions, ratings, descriptions, floor plans, pricing.
    """
    raw = doc["raw"]

    # 1. If it's HTML, parse it; otherwise keep the plain text as-is.
    if doc["suffix"] in {".html", ".htm"} or re.search(r"<\s*\w+[^>]*>", raw):
        text = _strip_html(raw)
    else:
        text = raw

    # 2. Decode leftover HTML entities (&amp;, &nbsp;, &#39;, ...) twice to
    #    catch double-encoded entities, then normalize non-breaking spaces.
    text = html.unescape(html.unescape(text))
    text = text.replace("\xa0", " ")

    # 3. Strip any HTML tags that slipped through (e.g. tags inside plain text).
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Collapse runs of spaces/tabs and trim trailing spaces on each line.
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Drop boilerplate lines and dedupe consecutive repeated lines
    #    (repeated site headers tend to appear as identical adjacent lines).
    cleaned_lines: list[str] = []
    previous_line: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if _looks_like_boilerplate(line):
            continue
        if line and line == previous_line:
            continue  # collapse immediate duplicate (repeated header/footer)
        cleaned_lines.append(line)
        previous_line = line

    # 6. Collapse 3+ blank lines down to a single blank line for readability.
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return {"source": doc["source"], "text": text}


# --------------------------------------------------------------------------- #
# Stage 3 — Cleaning verification (DEBUG PRINT #1)
# --------------------------------------------------------------------------- #
def debug_print_one_cleaned_doc(cleaned_docs: list[dict]) -> None:
    """Print exactly ONE fully cleaned document for visual inspection."""
    sample = cleaned_docs[0]
    print("\n" + "=" * 72)
    print("DEBUG #1 — ONE FULLY CLEANED DOCUMENT")
    print(f"source: {sample['source']}  ({len(sample['text']):,} chars)")
    print("=" * 72)
    print(sample["text"])
    print("=" * 72)
    print("^ Inspect above for leftover nav text, ads, or HTML artifacts.\n")


# --------------------------------------------------------------------------- #
# Stage 4 — Chunking
# --------------------------------------------------------------------------- #
def build_splitter() -> RecursiveCharacterTextSplitter:
    """Token-based recursive splitter configured per the chunking strategy."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def chunk_documents(
    cleaned_docs: list[dict], splitter: RecursiveCharacterTextSplitter
) -> list[dict]:
    """Split every cleaned document into overlapping token-based chunks.

    Each chunk carries metadata tracking its source file and position so the
    retrieval stage can attribute answers correctly (key for entity confusion).
    """
    chunks: list[dict] = []
    for doc in cleaned_docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "text": piece,
                    "metadata": {
                        "source": doc["source"],
                        "chunk_index": i,
                        "total_chunks_in_doc": len(pieces),
                    },
                }
            )
    return chunks


# --------------------------------------------------------------------------- #
# Stage 5 — Chunk verification (DEBUG PRINT #2 & #3)
# --------------------------------------------------------------------------- #
def debug_print_sample_chunks(chunks: list[dict], n: int = 5) -> None:
    """Print up to `n` representative chunks for standalone-readability checks."""
    print("\n" + "=" * 72)
    print(f"DEBUG #2 — {min(n, len(chunks))} REPRESENTATIVE CHUNKS")
    print("=" * 72)

    # Spread the samples across the corpus instead of taking the first 5,
    # so we inspect different sources rather than one document's opening.
    if len(chunks) <= n:
        sample_indices = list(range(len(chunks)))
    else:
        step = len(chunks) // n
        sample_indices = [i * step for i in range(n)]

    for rank, idx in enumerate(sample_indices, start=1):
        chunk = chunks[idx]
        meta = chunk["metadata"]
        print(f"\n--- Sample {rank} (corpus #{idx}) "
              f"| source={meta['source']} "
              f"| chunk {meta['chunk_index'] + 1}/{meta['total_chunks_in_doc']} ---")
        print(chunk["text"])
    print("\n" + "=" * 72)
    print("^ Each chunk should read as a complete, retrievable standalone thought.\n")


def debug_print_chunk_count(chunks: list[dict]) -> None:
    """Print the total chunk count and verify it is within a healthy range."""
    total = len(chunks)
    print("=" * 72)
    print("DEBUG #3 — TOTAL CHUNK COUNT")
    print("=" * 72)
    print(f"Total chunks generated across corpus: {total}")
    if MIN_HEALTHY_CHUNKS <= total <= MAX_HEALTHY_CHUNKS:
        print(f"[OK] Within healthy range "
              f"({MIN_HEALTHY_CHUNKS}–{MAX_HEALTHY_CHUNKS} chunks).")
    else:
        print(f"[WARN] Outside healthy range "
              f"({MIN_HEALTHY_CHUNKS}–{MAX_HEALTHY_CHUNKS}). "
              f"Add more source documents or revisit chunk_size if this is "
              f"unexpected.")
    print("=" * 72 + "\n")


# --------------------------------------------------------------------------- #
# Stage 6 — Export
# --------------------------------------------------------------------------- #
def export_chunks(chunks: list[dict], output_path: Path) -> None:
    """Write chunks to JSON as an array of {text, metadata} objects."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[EXPORT] Wrote {len(chunks)} chunks to '{output_path}'.")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    # Stage 1: load
    raw_docs = load_raw_documents(RAW_DIR)

    # Stage 2: clean
    cleaned_docs = [clean_document(doc) for doc in raw_docs]

    # Stage 3: cleaning verification (DEBUG #1)
    debug_print_one_cleaned_doc(cleaned_docs)

    # Stage 4: chunk
    splitter = build_splitter()
    chunks = chunk_documents(cleaned_docs, splitter)

    # Stage 5: chunk verification (DEBUG #2 & #3)
    debug_print_sample_chunks(chunks, n=5)
    debug_print_chunk_count(chunks)

    # Stage 6: export
    export_chunks(chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()
