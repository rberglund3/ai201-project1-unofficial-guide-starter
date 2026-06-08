"""
Milestone 4 — Embedding and Retrieval
======================================

RAG pipeline stage for "The Unofficial Guide" (off-campus housing near Georgia Tech).

Pipeline implemented here:

    [ JSON chunks ] -> [ Embedding (all-MiniLM-L6-v2) ]
            |
            v
    [ ChromaDB Vector Store (persistent) ]
            |
            v
    [ retrieve(query, k) -> top-k chunks + distances ]

Design notes:
    - Embeddings are produced by the *native* sentence-transformers library
      (no LangChain wrappers), and inserted into ChromaDB ourselves so we control
      the IDs, metadata, and distance metric.
    - The collection is configured to use COSINE distance (see create logic).
      Cosine distance lives in the [0, 2] range where 0 == identical; for normalized
      MiniLM embeddings, "good" hits are typically < ~0.6 and anything drifting
      toward 0.6-0.7+ is the failure threshold the rubric asks us to watch for.

Run:
    python embed_and_retrieve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_PATH = "data/chroma_db"          # on-disk location for the vector store
COLLECTION_NAME = "housing_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"         # 384-dim sentence embedding model
DEFAULT_TOP_K = 5

# MMR (Maximal Marginal Relevance) defaults. We over-fetch FETCH_K candidates,
# then greedily pick TOP_K that balance relevance to the query against diversity,
# so we don't return 5 near-duplicate sentences of the same complaint.
DEFAULT_FETCH_K = 20    # how many candidates to pull before MMR re-ranking
DEFAULT_LAMBDA = 0.5    # 1.0 = pure relevance, 0.0 = pure diversity


# --------------------------------------------------------------------------- #
# Stage 1 — Setup: load model and chunks
# --------------------------------------------------------------------------- #
# Load the embedding model once at import time so retrieve() can reuse it.
# Loading it here (not inside retrieve) avoids re-reading ~90MB from disk on
# every single query.
print(f"[SETUP] Loading embedding model '{MODEL_NAME}'...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_chunks(path: Path) -> list[dict]:
    """Load chunks.json and ensure each chunk's metadata records its position.

    The grading rubric wants accurate source attribution later, so we guarantee
    every chunk carries:
        - metadata["source"]      : originating file name (already set in M3)
        - metadata["chunk_index"] : 0-based position within its source document

    M3 already writes chunk_index, but we recompute a robust position here so the
    script is correct even if it's ever fed a barebones [{text, metadata}] file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Run ingest_and_chunk.py (Milestone 3) first."
        )

    with path.open(encoding="utf-8") as f:
        chunks = json.load(f)

    # Track how many chunks we've seen per source so we can number them in order.
    position_by_source: dict[str, int] = {}
    for chunk in chunks:
        meta = chunk.setdefault("metadata", {})
        source = meta.get("source", "unknown")
        # Assign this chunk the next position for its source document.
        position = position_by_source.get(source, 0)
        meta["source"] = source
        meta["chunk_index"] = meta.get("chunk_index", position)
        position_by_source[source] = position + 1

    print(f"[SETUP] Loaded {len(chunks)} chunks from '{path}'.")
    return chunks


# --------------------------------------------------------------------------- #
# Stage 2 — Vector database: build (or reuse) the Chroma collection
# --------------------------------------------------------------------------- #
def get_collection():
    """Create or open the persistent Chroma collection, populating it if empty.

    Returns a ready-to-query collection.
    """
    # PersistentClient writes the DB to disk under CHROMA_PATH, so embeddings
    # survive between runs and we don't re-embed every time.
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create_collection() returns the existing collection if it's already
    # on disk, otherwise creates a new one. We pin the distance metric to cosine
    # via the "hnsw:space" metadata key (default is L2/"squared euclidean", which
    # would put distances on a different scale than the 0.6-0.7 threshold we
    # want to reason about).
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # collection.count() returns how many documents are already stored. If the
    # collection already has data, we skip the (slow) embedding + insert step so
    # re-running the script is cheap and idempotent.
    existing = collection.count()
    if existing > 0:
        print(f"[VECTORDB] Collection '{COLLECTION_NAME}' already has "
              f"{existing} documents — skipping insertion.")
        return collection

    # --- Collection is empty: embed and insert. -------------------------------
    print(f"[VECTORDB] Collection '{COLLECTION_NAME}' is empty — embedding and "
          f"inserting chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    # Unique, stable IDs. Chroma requires a unique string ID per document; we
    # build one from the source file + position so it's both unique and readable.
    ids = [
        f"{c['metadata']['source']}::chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]

    # Encode all chunk texts in one batched call. normalize_embeddings=True gives
    # unit-length vectors, which is what cosine distance expects.
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()  # Chroma wants plain Python lists, not numpy arrays.

    # collection.add() persists everything in one call: the raw text (documents),
    # the precomputed vectors (embeddings), the per-chunk metadata, and the IDs.
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"[VECTORDB] Inserted {collection.count()} documents into "
          f"'{COLLECTION_NAME}'.")
    return collection


# Build/open the collection once at import time so retrieve() can use it directly.
collection = get_collection()


# --------------------------------------------------------------------------- #
# Stage 3 — Retrieval (metadata filtering + MMR re-ranking)
# --------------------------------------------------------------------------- #
def _distinct_sources() -> list[str]:
    """Return the distinct source filenames currently stored in the collection."""
    # collection.get() with no ids returns every document; we only need the
    # metadata to learn which source files exist.
    stored = collection.get(include=["metadatas"])
    return sorted({m["source"] for m in stored["metadatas"]})


def _build_where(source_filter: str | None) -> dict | None:
    """Translate a friendly substring into a Chroma metadata `where` filter.

    NOTE ON THE API: Chroma's metadata `where` filter does NOT support a
    `$contains` operator — that operator only exists for `where_document`, which
    matches the chunk TEXT, not the source file. To "isolate the search to a
    specific source file" we instead resolve the substring against the real
    filenames here (client-side) and emit a `$in` filter on the `source` field,
    which Chroma fully supports and which gives us the same "contains" behavior.
    """
    if not source_filter:
        return None  # None == search the whole collection

    matched = [
        s for s in _distinct_sources()
        if source_filter.lower() in s.lower()
    ]
    if not matched:
        raise ValueError(
            f"source_filter='{source_filter}' matched no source files. "
            f"Available sources: {_distinct_sources()}"
        )

    # {"source": {"$in": [...]}} keeps the search to just the matched files.
    return {"source": {"$in": matched}}


def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    k: int,
    lambda_mult: float = DEFAULT_LAMBDA,
) -> list[int]:
    """Maximal Marginal Relevance selection over candidate embeddings.

    Returns the indices (into `candidate_vecs`) of the k chunks that best trade
    off relevance to the query against novelty vs. already-selected chunks.
    All vectors are unit-normalized, so a dot product == cosine similarity.

        score(c) = lambda * sim(c, query) - (1 - lambda) * max sim(c, selected)
    """
    # Similarity of every candidate to the query (higher = more relevant).
    sim_to_query = candidate_vecs @ query_vec
    # Pairwise similarity between candidates (used for the diversity penalty).
    sim_between = candidate_vecs @ candidate_vecs.T

    selected: list[int] = []
    remaining = list(range(len(candidate_vecs)))

    while remaining and len(selected) < k:
        if not selected:
            # First pick: simply the most relevant candidate.
            best = max(remaining, key=lambda i: sim_to_query[i])
        else:
            # Penalize candidates similar to anything we've already chosen.
            def mmr_score(i: int) -> float:
                redundancy = max(sim_between[i][j] for j in selected)
                return lambda_mult * sim_to_query[i] - (1 - lambda_mult) * redundancy

            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return selected


def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    source_filter: str | None = None,
    use_mmr: bool = True,
    fetch_k: int = DEFAULT_FETCH_K,
    lambda_mult: float = DEFAULT_LAMBDA,
) -> dict:
    """Embed `query`, optionally restrict to a source file, and return top-k chunks.

    Args:
        query:         natural-language question.
        k:             number of chunks to return.
        source_filter: substring of a source filename (e.g. "rambler"). When set,
                       the search is isolated to matching file(s) — our fix for the
                       entity-confusion risk (don't attribute a Standard review to
                       the Rambler, etc.).
        use_mmr:       if True, over-fetch `fetch_k` candidates then MMR re-rank to
                       k diverse results; if False, return Chroma's raw top-k.
        fetch_k:       candidate pool size before MMR.
        lambda_mult:   MMR relevance/diversity trade-off (1.0 relevance, 0.0 diversity).

    Returns Chroma's native query result dict (lists keyed by 0 for our 1 query).
    """
    # Embed the query with the SAME model (and same normalization) used for the
    # documents — mixing models or normalization would make distances meaningless.
    query_embedding = model.encode([query], normalize_embeddings=True)

    where = _build_where(source_filter)

    # When MMR is on we pull a larger candidate pool to give it room to diversify;
    # otherwise we ask Chroma for exactly k. We also request "embeddings" so MMR
    # can compute candidate-to-candidate similarity locally.
    n_results = max(k, fetch_k) if use_mmr else k
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        where=where,  # None searches everything; otherwise restricts by source
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    if not use_mmr:
        return results

    candidate_vecs = np.asarray(results["embeddings"][0])
    if len(candidate_vecs) == 0:
        return results  # filter matched nothing retrievable; nothing to re-rank

    # Re-rank the candidate pool and keep only the MMR-selected order, trimmed to k.
    order = _mmr(query_embedding[0], candidate_vecs, k=k, lambda_mult=lambda_mult)
    return {
        "ids": [[results["ids"][0][i] for i in order]],
        "documents": [[results["documents"][0][i] for i in order]],
        "metadatas": [[results["metadatas"][0][i] for i in order]],
        "distances": [[results["distances"][0][i] for i in order]],
    }


# --------------------------------------------------------------------------- #
# Stage 4 — Debug / test execution
# --------------------------------------------------------------------------- #
def _print_results(query: str, results: dict, source_filter: str | None = None) -> None:
    """Pretty-print one query's results: text, source, position, and distance."""
    print("\n" + "=" * 78)
    print(f"QUERY: {query}")
    scope = f"source_filter='{source_filter}'" if source_filter else "ALL sources"
    print(f"SCOPE: {scope}  |  re-ranking: MMR (lambda={DEFAULT_LAMBDA})")
    print("=" * 78)

    # Chroma returns each field as a list-of-lists (one inner list per query).
    # We sent a single query, so index [0] gives that query's results.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        # Flag hits at/above the rubric's 0.6-0.7 cosine-distance failure band.
        flag = "  <-- WEAK MATCH (>= 0.6)" if dist >= 0.6 else ""
        snippet = doc.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        print(f"\n  [{rank}] distance={dist:.4f}{flag}")
        print(f"      source        : {meta.get('source')}")
        print(f"      chunk_index   : {meta.get('chunk_index')}")
        print(f"      text          : {snippet}")
    print()


if __name__ == "__main__":
    # Three evaluation queries from planning.md, each paired with the source_filter
    # that isolates it to the right document(s). Watch the distance scores: if the
    # top hits are >= ~0.6, retrieval is weak and the corpus likely lacks the answer
    # (or chunking split it badly). The filter + MMR combo is our entity-confusion fix.
    #
    # Q1: pin to the Rambler floor-plan page so pricing can't leak in from other
    #     luxury complexes (Standard/SQ5 tables look near-identical to the embedder).
    # Q2: no filter — it's an explicit cross-complex comparison, so we WANT breadth.
    # Q3: "based on Reddit" -> pin to reddit_the_standard.txt only. Filtering on
    #     "standard" alone would wrongly include the_standard.html, the Google
    #     reviews, AND standard_lease_boilerplate.txt.
    # test_cases = [
    #     ("What's the cheapest 4-bedroom at The Rambler right now?", "rambler"),
    #     ("Which apartment complexes are actually less than a 5-minute walk to campus?", None),
    #     ("What is the worst thing about living at The Standard based on Reddit?", "reddit_the_standard"),
    # ]

    # for query, source_filter in test_cases:
    #     results = retrieve(query, k=DEFAULT_TOP_K, source_filter=source_filter)
    #     _print_results(query, results, source_filter=source_filter)
        print("\n" + "="*50)
        print("🚀 MILESTONE 4: RETRIEVAL DIAGNOSTICS")
        print("="*50)

        def display_results(results):
        # Verify it is the expected ChromaDB dictionary format
            if isinstance(results, dict) and 'documents' in results:
                # We only passed one query, so we access the first list [0] inside each key
                documents = results.get('documents', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]

                if not documents:
                    print("  -> No results returned.")
                    return

                # Loop through the length of the documents list
                for i in range(len(documents)):
                    text = documents[i]
                    meta = metadatas[i] if metadatas else {}
                    dist = distances[i] if distances else 'N/A'

                    print(f"\n  [ Rank {i+1} | Distance: {dist} ]")
                    print(f"  📁 Source: {meta.get('source', 'Unknown')} (Chunk {meta.get('chunk_index', 'N/A')})")
                    print(f"  📝 Text: {text[:250]}...\n")
            else:
                print("  -> Unexpected format returned by retrieve function:")
                print(results)

        # Query 1: Filtered, pure similarity, k=2
        print("\n[ QUERY 1: The Rambler 4-Bedroom Pricing ]")
        q1 = "What's the cheapest 4-bedroom at The Rambler right now?"
        results1 = retrieve(q1, k=2, source_filter="rambler", use_mmr=False)
        display_results(results1)

        # Query 2: Unfiltered general search across the whole DB, k=3
        print("\n[ QUERY 2: 5-minute walk to campus ]")
        q2 = "Which apartment complexes are actually less than a 5-minute walk to campus?"
        results2 = retrieve(q2, k=3, source_filter=None, use_mmr=False)
        display_results(results2)

        # Query 3: Filtered, pure similarity, k=2
        print("\n[ QUERY 3: The Standard Complaints ]")
        q3 = "What is the worst thing about living at The Standard based on Reddit?"
        results3 = retrieve(q3, k=2, source_filter="reddit_the_standard", use_mmr=False)
        display_results(results3)
