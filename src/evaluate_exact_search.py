"""
V1 - Exact Vector Search Evaluation

Big Data Vector Search
======================

Evaluates exact semantic vector search using:

    sentence-transformers/all-MiniLM-L6-v2
    normalized 384-dimensional embeddings
    FAISS IndexFlatIP

The experiment measures:

    document embedding time
    document embedding throughput
    index construction time
    serialized index size
    query embedding time
    exact FAISS search latency
    relevance Recall@K
    relevance Hit Rate@K
    Python process RSS

The exact Top-K neighbors are saved for future ANN evaluation.

This version also records SHA-256 fingerprints of the benchmark
artifacts so stale exact-neighbor results can be detected later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BENCHMARK_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

DEFAULT_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EXPECTED_DIMENSION = 384
DEFAULT_BATCH_SIZE = 64

DEFAULT_RECALL_K = [
    1,
    5,
    10,
]


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact semantic vector search "
            "using FAISS IndexFlatIP."
        )
    )

    parser.add_argument(
        "--subset",
        required=True,
        help="Example: msmarco_1k",
    )

    parser.add_argument(
        "--benchmark-dir",
        default=str(DEFAULT_BENCHMARK_DIR),
        help="Default: data/processed",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def validate_args(args):

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.top_k < 1:
        raise ValueError(
            "--top-k must be >= 1"
        )

    if args.warmup_queries < 0:
        raise ValueError(
            "--warmup-queries must be >= 0"
        )


# ============================================================
# Readers
# ============================================================

def read_json(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def read_jsonl(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}."
                )

            rows.append(row)

    return rows


# ============================================================
# SHA-256 provenance
# ============================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file:\n{path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# Output helpers
# ============================================================

def ensure_writable(
    path: Path,
    overwrite: bool,
):

    if path.exists() and not overwrite:
        raise FileExistsError(
            "Output already exists:\n"
            f"{path}\n"
            "Use --overwrite to replace it."
        )


def write_json(
    path: Path,
    value,
    overwrite: bool,
):

    ensure_writable(
        path,
        overwrite,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            value,
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_jsonl(
    path: Path,
    rows,
    overwrite: bool,
):

    ensure_writable(
        path,
        overwrite,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            f.write("\n")


def append_csv(
    path: Path,
    row: dict,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    # Protect cumulative CSV schema.
    if exists:

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.reader(f)

            try:
                existing_header = next(reader)

            except StopIteration:
                existing_header = []

        expected_header = list(
            row.keys()
        )

        if (
            existing_header
            and existing_header
            != expected_header
        ):

            raise RuntimeError(
                "Existing cumulative CSV uses a different "
                "schema. Rename/remove the old file before "
                "running this updated experiment.\n"
                f"File: {path}"
            )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            ),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Utilities
# ============================================================

def current_rss_mb():

    process = psutil.Process(
        os.getpid()
    )

    return (
        process.memory_info().rss
        / (1024 * 1024)
    )


def percentile(
    values,
    percent,
):

    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            percent,
        )
    )


def subset_label(
    subset_name: str,
):

    prefix = "msmarco_"

    if subset_name.startswith(prefix):
        return subset_name[len(prefix):]

    return subset_name


def document_text(row):

    title = str(
        row.get(
            "title",
            "",
        )
        or ""
    ).strip()

    text = str(
        row.get(
            "text",
            "",
        )
        or ""
    ).strip()

    if title:
        return f"{title} {text}".strip()

    return text


# ============================================================
# Benchmark loading
# ============================================================

def load_benchmark(
    benchmark_dir: Path,
    subset_name: str,
):

    manifest_path = (
        benchmark_dir
        / "benchmark_manifest.json"
    )

    queries_path = (
        benchmark_dir
        / "queries.jsonl"
    )

    qrels_path = (
        benchmark_dir
        / "qrels.json"
    )

    corpus_path = (
        benchmark_dir
        / subset_name
        / "corpus.jsonl"
    )

    manifest = read_json(
        manifest_path
    )

    queries = read_jsonl(
        queries_path
    )

    qrels = read_json(
        qrels_path
    )

    corpus = read_jsonl(
        corpus_path
    )

    paths = {
        "manifest": manifest_path,
        "queries": queries_path,
        "qrels": qrels_path,
        "corpus": corpus_path,
    }

    return (
        manifest,
        queries,
        qrels,
        corpus,
        paths,
    )


# ============================================================
# Benchmark validation
# ============================================================

def validate_benchmark(
    manifest,
    queries,
    qrels,
    corpus,
    subset_name,
):

    if not queries:
        raise RuntimeError(
            "Query set is empty."
        )

    if not corpus:
        raise RuntimeError(
            "Corpus is empty."
        )

    if not isinstance(qrels, dict):
        raise RuntimeError(
            "qrels.json must contain a JSON object."
        )

    query_ids = [
        str(row["_id"])
        for row in queries
    ]

    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError(
            "Duplicate query IDs detected."
        )

    document_ids = [
        str(row["_id"])
        for row in corpus
    ]

    if (
        len(document_ids)
        != len(set(document_ids))
    ):
        raise RuntimeError(
            "Duplicate document IDs detected."
        )

    qrel_query_ids = {
        str(query_id)
        for query_id
        in qrels.keys()
    }

    if set(query_ids) != qrel_query_ids:
        raise RuntimeError(
            "Query/qrels ID mismatch."
        )

    required_document_ids = set()

    for documents in qrels.values():

        for document_id, score in (
            documents.items()
        ):

            if float(score) > 0:
                required_document_ids.add(
                    str(document_id)
                )

    corpus_ids = set(document_ids)

    missing = (
        required_document_ids
        - corpus_ids
    )

    if missing:
        raise RuntimeError(
            f"Corpus is missing "
            f"{len(missing):,} required "
            "relevant documents."
        )

    manifest_subsets = {
        item.get("name"): item
        for item
        in manifest.get(
            "subsets",
            [],
        )
    }

    if subset_name not in manifest_subsets:
        raise RuntimeError(
            f"Subset {subset_name!r} "
            "is not listed in the benchmark manifest."
        )

    expected_documents = (
        manifest_subsets[
            subset_name
        ].get(
            "documents"
        )
    )

    if expected_documents != len(corpus):
        raise RuntimeError(
            "Corpus size does not match benchmark "
            "manifest. "
            f"Expected={expected_documents:,}, "
            f"actual={len(corpus):,}."
        )


# ============================================================
# Retrieval quality
# ============================================================

def relevance_recall_at_k(
    query_ids,
    neighbor_doc_ids,
    qrels,
    k,
):

    per_query = []
    hits = 0

    for query_index, query_id in enumerate(
        query_ids
    ):

        relevant = {
            str(document_id)
            for document_id, score
            in qrels[
                query_id
            ].items()
            if float(score) > 0
        }

        retrieved = set(
            neighbor_doc_ids[
                query_index
            ][:k]
        )

        relevant_retrieved = (
            relevant
            & retrieved
        )

        if relevant_retrieved:
            hits += 1

        recall = (
            len(relevant_retrieved)
            / len(relevant)
        )

        per_query.append(recall)

    macro_recall = float(
        np.mean(per_query)
    )

    hit_rate = (
        hits
        / len(query_ids)
    )

    return (
        macro_recall,
        hit_rate,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    validate_args(args)

    benchmark_dir = Path(
        args.benchmark_dir
    )

    subset = args.subset

    label = subset_label(
        subset
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_json_path = (
        RESULTS_DIR
        / f"v1_exact_{label}.json"
    )

    neighbors_path = (
        RESULTS_DIR
        / (
            f"v1_exact_neighbors_"
            f"{label}.jsonl"
        )
    )

    cumulative_csv_path = (
        RESULTS_DIR
        / "v1_exact_search_experiments.csv"
    )

    # Fail before expensive computation.
    ensure_writable(
        result_json_path,
        args.overwrite,
    )

    ensure_writable(
        neighbors_path,
        args.overwrite,
    )

    print()
    print(
        "V1 - Exact Vector Search Evaluation"
    )

    print(
        "==================================="
    )

    print()

    print(
        f"Subset:             {subset}"
    )

    print(
        f"Benchmark:          "
        f"{benchmark_dir.resolve()}"
    )

    print(
        f"Model:              {args.model}"
    )

    print(
        f"Batch size:         {args.batch_size}"
    )

    print(
        f"Saved Top-K:        {args.top_k}"
    )

    total_start = time.perf_counter()

    rss_samples = [
        current_rss_mb()
    ]

    # ========================================================
    # Load benchmark
    # ========================================================

    print()
    print("Loading benchmark...")

    load_start = time.perf_counter()

    (
        manifest,
        queries,
        qrels,
        corpus,
        paths,
    ) = load_benchmark(
        benchmark_dir,
        subset,
    )

    validate_benchmark(
        manifest,
        queries,
        qrels,
        corpus,
        subset,
    )

    load_time = (
        time.perf_counter()
        - load_start
    )

    rss_samples.append(
        current_rss_mb()
    )

    document_ids = [
        str(row["_id"])
        for row in corpus
    ]

    document_texts = [
        document_text(row)
        for row in corpus
    ]

    query_ids = [
        str(row["_id"])
        for row in queries
    ]

    query_texts = [
        str(
            row.get(
                "text",
                "",
            )
            or ""
        )
        for row in queries
    ]

    documents = len(document_ids)
    query_count = len(query_ids)

    print(
        f"Documents:          {documents:,}"
    )

    print(
        f"Queries:            {query_count:,}"
    )

    print(
        f"Load time:          {load_time:.4f}s"
    )

    effective_top_k = min(
        args.top_k,
        documents,
    )

    # ========================================================
    # SHA-256 provenance
    # ========================================================

    print()
    print(
        "Calculating benchmark fingerprints..."
    )

    provenance_start = (
        time.perf_counter()
    )

    corpus_sha256 = sha256_file(
        paths["corpus"]
    )

    queries_sha256 = sha256_file(
        paths["queries"]
    )

    qrels_sha256 = sha256_file(
        paths["qrels"]
    )

    manifest_sha256 = sha256_file(
        paths["manifest"]
    )

    provenance_time = (
        time.perf_counter()
        - provenance_start
    )

    print(
        "Corpus SHA-256:     "
        f"{corpus_sha256[:16]}..."
    )

    print(
        "Queries SHA-256:    "
        f"{queries_sha256[:16]}..."
    )

    print(
        "Qrels SHA-256:      "
        f"{qrels_sha256[:16]}..."
    )

    print(
        "Manifest SHA-256:   "
        f"{manifest_sha256[:16]}..."
    )

    print(
        f"Hashing time:       "
        f"{provenance_time:.4f}s"
    )

    # ========================================================
    # Load model
    # ========================================================

    print()
    print(
        "Loading Sentence Transformer..."
    )

    model_load_start = (
        time.perf_counter()
    )

    model = SentenceTransformer(
        args.model
    )

    model_load_time = (
        time.perf_counter()
        - model_load_start
    )

    rss_samples.append(
        current_rss_mb()
    )

    print(
        f"Model load time:    "
        f"{model_load_time:.2f}s"
    )

    # ========================================================
    # Document embeddings
    # ========================================================

    print()
    print(
        "Embedding documents..."
    )

    embedding_start = (
        time.perf_counter()
    )

    document_embeddings = model.encode(
        document_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    document_embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    document_embeddings = np.asarray(
        document_embeddings,
        dtype=np.float32,
    )

    rss_samples.append(
        current_rss_mb()
    )

    if document_embeddings.ndim != 2:
        raise RuntimeError(
            "Document embeddings are not 2-D."
        )

    if (
        document_embeddings.shape[0]
        != documents
    ):
        raise RuntimeError(
            "Document embedding count mismatch."
        )

    embedding_dimension = int(
        document_embeddings.shape[1]
    )

    if (
        embedding_dimension
        != EXPECTED_DIMENSION
    ):
        raise RuntimeError(
            "Unexpected embedding dimension. "
            f"Expected={EXPECTED_DIMENSION}, "
            f"actual={embedding_dimension}."
        )

    if not np.isfinite(
        document_embeddings
    ).all():
        raise RuntimeError(
            "Non-finite document embeddings detected."
        )

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1,
    )

    if not np.allclose(
        document_norms,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Document embeddings are not "
            "properly normalized."
        )

    document_throughput = (
        documents
        / document_embedding_time
    )

    print(
        f"Embedding shape:    "
        f"{document_embeddings.shape}"
    )

    print(
        f"Embedding time:     "
        f"{document_embedding_time:.2f}s"
    )

    print(
        f"Throughput:         "
        f"{document_throughput:.2f} docs/s"
    )

    # ========================================================
    # FAISS exact index
    # ========================================================

    print()
    print(
        "Building FAISS IndexFlatIP..."
    )

    index_start = (
        time.perf_counter()
    )

    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    index.add(
        document_embeddings
    )

    index_build_time = (
        time.perf_counter()
        - index_start
    )

    rss_samples.append(
        current_rss_mb()
    )

    if index.ntotal != documents:
        raise RuntimeError(
            "FAISS index cardinality mismatch."
        )

    serialized_index = (
        faiss.serialize_index(index)
    )

    index_size_bytes = len(
        serialized_index
    )

    index_size_mb = (
        index_size_bytes
        / (1024 * 1024)
    )

    raw_vector_bytes = (
        document_embeddings.nbytes
    )

    raw_vector_mb = (
        raw_vector_bytes
        / (1024 * 1024)
    )

    print(
        f"Indexed vectors:    "
        f"{index.ntotal:,}"
    )

    print(
        f"Index build time:   "
        f"{index_build_time:.6f}s"
    )

    # ========================================================
    # Query embeddings
    # ========================================================

    print()
    print(
        "Embedding queries..."
    )

    query_embedding_start = (
        time.perf_counter()
    )

    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding_time = (
        time.perf_counter()
        - query_embedding_start
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    rss_samples.append(
        current_rss_mb()
    )

    expected_query_shape = (
        query_count,
        embedding_dimension,
    )

    if (
        query_embeddings.shape
        != expected_query_shape
    ):
        raise RuntimeError(
            "Query embedding shape mismatch. "
            f"Expected={expected_query_shape}, "
            f"actual={query_embeddings.shape}."
        )

    if not np.isfinite(
        query_embeddings
    ).all():
        raise RuntimeError(
            "Non-finite query embeddings detected."
        )

    query_embedding_per_query_ms = (
        query_embedding_time
        / query_count
        * 1000.0
    )

    print(
        f"Query embedding:    "
        f"{query_embedding_time:.2f}s"
    )

    print(
        f"Per query:          "
        f"{query_embedding_per_query_ms:.3f} ms"
    )

    # ========================================================
    # Warmup
    # ========================================================

    warmup_count = min(
        args.warmup_queries,
        query_count,
    )

    if warmup_count:

        index.search(
            query_embeddings[:warmup_count],
            effective_top_k,
        )

    # ========================================================
    # Exact search
    # ========================================================

    print()
    print(
        "Running exact Top-K search..."
    )

    latencies_ms = []
    all_scores = []
    all_indices = []

    search_wall_start = (
        time.perf_counter()
    )

    for query_vector in query_embeddings:

        query_matrix = (
            query_vector.reshape(
                1,
                -1,
            )
        )

        start_ns = (
            time.perf_counter_ns()
        )

        scores, indices = index.search(
            query_matrix,
            effective_top_k,
        )

        end_ns = (
            time.perf_counter_ns()
        )

        latency_ms = (
            end_ns
            - start_ns
        ) / 1_000_000.0

        latencies_ms.append(
            latency_ms
        )

        all_scores.append(
            scores[0].copy()
        )

        all_indices.append(
            indices[0].copy()
        )

    search_wall_time = (
        time.perf_counter()
        - search_wall_start
    )

    rss_samples.append(
        current_rss_mb()
    )

    mean_latency_ms = float(
        statistics.mean(
            latencies_ms
        )
    )

    median_latency_ms = float(
        statistics.median(
            latencies_ms
        )
    )

    p95_latency_ms = percentile(
        latencies_ms,
        95,
    )

    p99_latency_ms = percentile(
        latencies_ms,
        99,
    )

    min_latency_ms = float(
        min(latencies_ms)
    )

    max_latency_ms = float(
        max(latencies_ms)
    )

    search_queries_per_sec = (
        query_count
        / search_wall_time
    )

    print(
        f"Mean latency:       "
        f"{mean_latency_ms:.4f} ms"
    )

    print(
        f"Median latency:     "
        f"{median_latency_ms:.4f} ms"
    )

    print(
        f"P95 latency:        "
        f"{p95_latency_ms:.4f} ms"
    )

    print(
        f"P99 latency:        "
        f"{p99_latency_ms:.4f} ms"
    )

    # ========================================================
    # Convert FAISS positions to document IDs
    # ========================================================

    neighbor_doc_ids = []

    for indices in all_indices:

        ids = []

        for index_position in indices:

            index_position = int(
                index_position
            )

            if index_position < 0:
                continue

            ids.append(
                document_ids[
                    index_position
                ]
            )

        neighbor_doc_ids.append(ids)

    # ========================================================
    # Relevance metrics
    # ========================================================

    recall_metrics = {}

    for k in DEFAULT_RECALL_K:

        if k > effective_top_k:
            continue

        (
            macro_recall,
            hit_rate,
        ) = relevance_recall_at_k(
            query_ids,
            neighbor_doc_ids,
            qrels,
            k,
        )

        recall_metrics[
            f"recall_at_{k}"
        ] = macro_recall

        recall_metrics[
            f"hit_rate_at_{k}"
        ] = hit_rate

    # ========================================================
    # Exact-neighbor ground truth
    # ========================================================

    neighbor_rows = []

    for query_index, query_id in enumerate(
        query_ids
    ):

        neighbors = []

        for rank, (
            faiss_index,
            score,
        ) in enumerate(
            zip(
                all_indices[
                    query_index
                ],
                all_scores[
                    query_index
                ],
            ),
            start=1,
        ):

            faiss_index = int(
                faiss_index
            )

            if faiss_index < 0:
                continue

            neighbors.append({
                "rank": rank,
                "doc_id": document_ids[
                    faiss_index
                ],
                "score": float(score),
            })

        relevant_ids = {
            str(document_id)
            for document_id, score
            in qrels[
                query_id
            ].items()
            if float(score) > 0
        }

        neighbor_rows.append({
            "query_id": query_id,
            "neighbors": neighbors,
            "relevant_document_ids":
                sorted(relevant_ids),
        })

    # ========================================================
    # Result
    # ========================================================

    max_observed_rss_mb = max(
        rss_samples
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    result = {

        "experiment":
            "v1_exact_search",

        "timestamp":
            timestamp,

        "status":
            "SUCCESS",

        "subset":
            subset,

        "documents":
            documents,

        "queries":
            query_count,

        "embedding_model":
            args.model,

        "embedding_dimension":
            embedding_dimension,

        "normalized_embeddings":
            True,

        "index_type":
            "IndexFlatIP",

        "distance_metric":
            (
                "inner_product_on_"
                "l2_normalized_vectors"
            ),

        "batch_size":
            args.batch_size,

        "saved_top_k":
            effective_top_k,

        "benchmark_seed":
            manifest.get("seed"),

        "provenance": {

            "hash_algorithm":
                "sha256",

            "corpus_file":
                str(
                    paths[
                        "corpus"
                    ].resolve()
                ),

            "corpus_sha256":
                corpus_sha256,

            "queries_file":
                str(
                    paths[
                        "queries"
                    ].resolve()
                ),

            "queries_sha256":
                queries_sha256,

            "qrels_file":
                str(
                    paths[
                        "qrels"
                    ].resolve()
                ),

            "qrels_sha256":
                qrels_sha256,

            "manifest_file":
                str(
                    paths[
                        "manifest"
                    ].resolve()
                ),

            "manifest_sha256":
                manifest_sha256,

            "fingerprint_time_sec":
                round(
                    provenance_time,
                    6,
                ),
        },

        "timing": {

            "benchmark_load_time_sec":
                round(
                    load_time,
                    6,
                ),

            "model_load_time_sec":
                round(
                    model_load_time,
                    6,
                ),

            "document_embedding_time_sec":
                round(
                    document_embedding_time,
                    6,
                ),

            "document_embedding_docs_per_sec":
                round(
                    document_throughput,
                    6,
                ),

            "index_build_time_sec":
                round(
                    index_build_time,
                    6,
                ),

            "query_embedding_time_sec":
                round(
                    query_embedding_time,
                    6,
                ),

            "query_embedding_per_query_ms":
                round(
                    query_embedding_per_query_ms,
                    6,
                ),

            "search_wall_time_sec":
                round(
                    search_wall_time,
                    6,
                ),

            "search_queries_per_sec":
                round(
                    search_queries_per_sec,
                    6,
                ),

            "total_experiment_time_sec":
                round(
                    total_time,
                    6,
                ),
        },

        "search_latency_ms": {

            "mean":
                round(
                    mean_latency_ms,
                    6,
                ),

            "median":
                round(
                    median_latency_ms,
                    6,
                ),

            "p95":
                round(
                    p95_latency_ms,
                    6,
                ),

            "p99":
                round(
                    p99_latency_ms,
                    6,
                ),

            "min":
                round(
                    min_latency_ms,
                    6,
                ),

            "max":
                round(
                    max_latency_ms,
                    6,
                ),
        },

        "retrieval_quality": {
            key: round(
                value,
                6,
            )
            for key, value
            in recall_metrics.items()
        },

        "storage": {

            "raw_embedding_bytes":
                raw_vector_bytes,

            "raw_embedding_mb":
                round(
                    raw_vector_mb,
                    6,
                ),

            "serialized_index_bytes":
                index_size_bytes,

            "serialized_index_mb":
                round(
                    index_size_mb,
                    6,
                ),
        },

        "memory": {

            "max_observed_python_rss_mb":
                round(
                    max_observed_rss_mb,
                    6,
                ),
        },

        "environment": {

            "python_version":
                platform.python_version(),

            "python_executable":
                sys.executable,

            "platform":
                platform.platform(),

            "cpu_count":
                os.cpu_count(),

            "numpy_version":
                np.__version__,

            "faiss_version":
                getattr(
                    faiss,
                    "__version__",
                    "unknown",
                ),
        },
    }

    # ========================================================
    # Save detailed outputs first
    # ========================================================

    write_jsonl(
        neighbors_path,
        neighbor_rows,
        args.overwrite,
    )

    write_json(
        result_json_path,
        result,
        args.overwrite,
    )

    # ========================================================
    # Cumulative CSV
    # ========================================================

    csv_row = {

        "timestamp":
            timestamp,

        "subset":
            subset,

        "documents":
            documents,

        "queries":
            query_count,

        "embedding_model":
            args.model,

        "embedding_dimension":
            embedding_dimension,

        "index_type":
            "IndexFlatIP",

        "batch_size":
            args.batch_size,

        "document_embedding_time_sec":
            round(
                document_embedding_time,
                6,
            ),

        "document_embedding_docs_per_sec":
            round(
                document_throughput,
                6,
            ),

        "index_build_time_sec":
            round(
                index_build_time,
                6,
            ),

        "index_size_mb":
            round(
                index_size_mb,
                6,
            ),

        "query_embedding_time_sec":
            round(
                query_embedding_time,
                6,
            ),

        "mean_search_latency_ms":
            round(
                mean_latency_ms,
                6,
            ),

        "median_search_latency_ms":
            round(
                median_latency_ms,
                6,
            ),

        "p95_search_latency_ms":
            round(
                p95_latency_ms,
                6,
            ),

        "p99_search_latency_ms":
            round(
                p99_latency_ms,
                6,
            ),

        "recall_at_1":
            round(
                recall_metrics.get(
                    "recall_at_1",
                    0.0,
                ),
                6,
            ),

        "recall_at_5":
            round(
                recall_metrics.get(
                    "recall_at_5",
                    0.0,
                ),
                6,
            ),

        "recall_at_10":
            round(
                recall_metrics.get(
                    "recall_at_10",
                    0.0,
                ),
                6,
            ),

        "hit_rate_at_1":
            round(
                recall_metrics.get(
                    "hit_rate_at_1",
                    0.0,
                ),
                6,
            ),

        "hit_rate_at_5":
            round(
                recall_metrics.get(
                    "hit_rate_at_5",
                    0.0,
                ),
                6,
            ),

        "hit_rate_at_10":
            round(
                recall_metrics.get(
                    "hit_rate_at_10",
                    0.0,
                ),
                6,
            ),

        "max_observed_python_rss_mb":
            round(
                max_observed_rss_mb,
                6,
            ),

        "total_experiment_time_sec":
            round(
                total_time,
                6,
            ),

        "corpus_sha256":
            corpus_sha256,

        "queries_sha256":
            queries_sha256,

        "qrels_sha256":
            qrels_sha256,

        "manifest_sha256":
            manifest_sha256,

        "status":
            "SUCCESS",
    }

    append_csv(
        cumulative_csv_path,
        csv_row,
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print("=" * 72)

    print(
        "V1 Exact Vector Search Result"
    )

    print("=" * 72)

    print(
        f"Subset:                    {subset}"
    )

    print(
        f"Documents:                 {documents:,}"
    )

    print(
        f"Queries:                   {query_count:,}"
    )

    print(
        f"Embedding dimension:       "
        f"{embedding_dimension}"
    )

    print(
        "Index:                     "
        "FAISS IndexFlatIP"
    )

    print()

    print(
        f"Document embedding time:   "
        f"{document_embedding_time:.2f}s"
    )

    print(
        f"Embedding throughput:      "
        f"{document_throughput:.2f} docs/s"
    )

    print(
        f"Index build time:          "
        f"{index_build_time:.6f}s"
    )

    print(
        f"Serialized index size:     "
        f"{index_size_mb:.3f} MB"
    )

    print()

    print(
        f"Query embedding time:      "
        f"{query_embedding_time:.2f}s"
    )

    print(
        f"Query embedding/query:     "
        f"{query_embedding_per_query_ms:.3f} ms"
    )

    print()

    print(
        f"Mean search latency:       "
        f"{mean_latency_ms:.4f} ms"
    )

    print(
        f"Median search latency:     "
        f"{median_latency_ms:.4f} ms"
    )

    print(
        f"P95 search latency:        "
        f"{p95_latency_ms:.4f} ms"
    )

    print(
        f"P99 search latency:        "
        f"{p99_latency_ms:.4f} ms"
    )

    print()

    for k in DEFAULT_RECALL_K:

        recall_key = (
            f"recall_at_{k}"
        )

        hit_key = (
            f"hit_rate_at_{k}"
        )

        if recall_key in recall_metrics:

            print(
                f"Relevance Recall@{k:<2}:       "
                f"{recall_metrics[recall_key]:.4f}"
            )

            print(
                f"Relevant Hit Rate@{k:<2}:      "
                f"{recall_metrics[hit_key]:.4f}"
            )

    print()

    print(
        f"Max observed Python RSS:   "
        f"{max_observed_rss_mb:.2f} MB"
    )

    print(
        f"Total experiment time:     "
        f"{total_time:.2f}s"
    )

    print()

    print(
        "Benchmark fingerprints:"
    )

    print(
        f"  Corpus:   {corpus_sha256}"
    )

    print(
        f"  Queries:  {queries_sha256}"
    )

    print(
        f"  Qrels:    {qrels_sha256}"
    )

    print(
        f"  Manifest: {manifest_sha256}"
    )

    print()

    print(
        "Experiment result:"
    )

    print(
        f"  {result_json_path}"
    )

    print()

    print(
        "Exact-neighbor ground truth:"
    )

    print(
        f"  {neighbors_path}"
    )

    print()

    print(
        "Cumulative results:"
    )

    print(
        f"  {cumulative_csv_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()