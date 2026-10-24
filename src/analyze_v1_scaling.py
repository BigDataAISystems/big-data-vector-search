"""
V1 - Exact Vector Search Scaling Analysis

Big Data Vector Search
======================

Purpose
-------
Analyze V1 exact-search experiments produced by:

    src/evaluate_exact_search.py

The analyzer:

    - reads results/v1_exact_search_experiments.csv
    - fingerprints the current benchmark manifest
    - excludes experiments from obsolete benchmark manifests
    - keeps successful experiments only
    - selects the latest valid run for each corpus size
    - verifies common query/qrels provenance
    - produces a scaling summary
    - calculates 1K -> largest-corpus scaling factors
    - generates publication-friendly plots

Expected V1 scale points:

    1K
    10K
    25K
    50K
    100K

Outputs
-------
results/

    v1_scaling_summary.csv

    v1_embedding_time.png
    v1_embedding_throughput.png
    v1_index_size.png
    v1_search_latency.png
    v1_memory.png
    v1_recall.png

Important
---------
The cumulative experiment CSV may contain historical runs.

This analyzer deliberately does NOT average experiments across different
benchmark manifests.

Only rows whose manifest SHA-256 matches the current
data/processed/benchmark_manifest.json are considered current.

If multiple valid runs exist for the same subset under the current
benchmark provenance, the latest timestamp is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = (
    ROOT
    / "results"
)

DEFAULT_BENCHMARK_DIR = (
    ROOT
    / "data"
    / "processed"
)

DEFAULT_EXPERIMENT_CSV = (
    DEFAULT_RESULTS_DIR
    / "v1_exact_search_experiments.csv"
)

DEFAULT_MANIFEST = (
    DEFAULT_BENCHMARK_DIR
    / "benchmark_manifest.json"
)

EXPECTED_DOCUMENT_COUNTS = [
    1_000,
    10_000,
    25_000,
    50_000,
    100_000,
]


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze V1 exact vector-search "
            "scaling experiments."
        )
    )

    parser.add_argument(
        "--experiments",
        default=str(
            DEFAULT_EXPERIMENT_CSV
        ),
        help=(
            "Cumulative exact-search experiment CSV. "
            "Default: results/"
            "v1_exact_search_experiments.csv"
        ),
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
        help=(
            "Current benchmark manifest. "
            "Default: data/processed/"
            "benchmark_manifest.json"
        ),
    )

    parser.add_argument(
        "--results-dir",
        default=str(
            DEFAULT_RESULTS_DIR
        ),
        help=(
            "Analysis output directory. "
            "Default: results"
        ),
    )

    return parser.parse_args()


# ============================================================
# SHA-256
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

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# Input validation
# ============================================================

def validate_columns(
    df: pd.DataFrame,
):

    required = {

        "timestamp",
        "subset",
        "documents",
        "queries",

        "embedding_model",
        "embedding_dimension",
        "index_type",
        "batch_size",

        "document_embedding_time_sec",
        "document_embedding_docs_per_sec",

        "index_build_time_sec",
        "index_size_mb",

        "query_embedding_time_sec",

        "mean_search_latency_ms",
        "median_search_latency_ms",
        "p95_search_latency_ms",
        "p99_search_latency_ms",

        "recall_at_1",
        "recall_at_5",
        "recall_at_10",

        "hit_rate_at_1",
        "hit_rate_at_5",
        "hit_rate_at_10",

        "max_observed_python_rss_mb",
        "total_experiment_time_sec",

        "corpus_sha256",
        "queries_sha256",
        "qrels_sha256",
        "manifest_sha256",

        "status",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Experiment CSV is missing required "
            "columns:\n"
            + "\n".join(
                f"  {column}"
                for column
                in sorted(
                    missing
                )
            )
        )


# ============================================================
# Current-run selection
# ============================================================

def select_current_runs(
    df: pd.DataFrame,
    current_manifest_sha256: str,
):
    """
    Select only experiments belonging to the current benchmark.

    Rules:

        1. status == SUCCESS
        2. manifest_sha256 == current manifest SHA-256
        3. latest timestamp per subset
    """

    total_rows = len(
        df
    )

    successful = df[
        df[
            "status"
        ].astype(
            str
        ).str.upper()
        == "SUCCESS"
    ].copy()

    success_rows = len(
        successful
    )

    current = successful[
        successful[
            "manifest_sha256"
        ].astype(
            str
        )
        == current_manifest_sha256
    ].copy()

    current_rows = len(
        current
    )

    obsolete_rows = (
        success_rows
        - current_rows
    )

    if current.empty:

        raise RuntimeError(
            "No successful experiment rows match "
            "the current benchmark manifest SHA-256."
        )

    current[
        "_timestamp_parsed"
    ] = pd.to_datetime(
        current[
            "timestamp"
        ],
        errors="coerce",
    )

    if (
        current[
            "_timestamp_parsed"
        ].isna().any()
    ):

        raise RuntimeError(
            "One or more current experiment timestamps "
            "could not be parsed."
        )

    current = current.sort_values(
        [
            "documents",
            "_timestamp_parsed",
        ]
    )

    # Keep the latest valid run for each subset.
    selected = (
        current
        .groupby(
            "subset",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    selected = selected.sort_values(
        "documents"
    ).reset_index(
        drop=True
    )

    selected = selected.drop(
        columns=[
            "_timestamp_parsed"
        ]
    )

    diagnostics = {

        "total_rows":
            total_rows,

        "successful_rows":
            success_rows,

        "current_provenance_rows":
            current_rows,

        "obsolete_success_rows":
            obsolete_rows,

        "selected_rows":
            len(
                selected
            ),
    }

    return (
        selected,
        diagnostics,
    )


# ============================================================
# Provenance consistency
# ============================================================

def validate_selected_provenance(
    df: pd.DataFrame,
):

    # Query and qrels artifacts must be identical across
    # all scale points.

    query_hashes = set(
        df[
            "queries_sha256"
        ].astype(
            str
        )
    )

    qrels_hashes = set(
        df[
            "qrels_sha256"
        ].astype(
            str
        )
    )

    manifest_hashes = set(
        df[
            "manifest_sha256"
        ].astype(
            str
        )
    )

    if len(
        query_hashes
    ) != 1:

        raise RuntimeError(
            "Selected experiments do not share "
            "one query-set fingerprint."
        )

    if len(
        qrels_hashes
    ) != 1:

        raise RuntimeError(
            "Selected experiments do not share "
            "one qrels fingerprint."
        )

    if len(
        manifest_hashes
    ) != 1:

        raise RuntimeError(
            "Selected experiments do not share "
            "one manifest fingerprint."
        )

    # Each corpus size should have its own corpus fingerprint.
    corpus_pairs = (
        df[
            [
                "documents",
                "corpus_sha256",
            ]
        ]
        .drop_duplicates()
    )

    duplicate_size_hashes = (
        corpus_pairs
        .groupby(
            "documents"
        )[
            "corpus_sha256"
        ]
        .nunique()
    )

    if (
        duplicate_size_hashes
        > 1
    ).any():

        raise RuntimeError(
            "At least one document count has multiple "
            "corpus fingerprints."
        )


# ============================================================
# Scale-point validation
# ============================================================

def validate_scale_points(
    df: pd.DataFrame,
):

    actual = sorted(
        int(value)
        for value
        in df[
            "documents"
        ].tolist()
    )

    missing = [
        size
        for size
        in EXPECTED_DOCUMENT_COUNTS
        if size not in actual
    ]

    extra = [
        size
        for size
        in actual
        if size
        not in EXPECTED_DOCUMENT_COUNTS
    ]

    return (
        actual,
        missing,
        extra,
    )


# ============================================================
# Summary
# ============================================================

def build_summary(
    df: pd.DataFrame,
):

    summary = pd.DataFrame({

        "subset":
            df[
                "subset"
            ],

        "documents":
            df[
                "documents"
            ].astype(
                int
            ),

        "embedding_time_sec":
            df[
                "document_embedding_time_sec"
            ].astype(
                float
            ),

        "embedding_docs_per_sec":
            df[
                "document_embedding_docs_per_sec"
            ].astype(
                float
            ),

        "index_build_time_sec":
            df[
                "index_build_time_sec"
            ].astype(
                float
            ),

        "index_size_mb":
            df[
                "index_size_mb"
            ].astype(
                float
            ),

        "query_embedding_time_sec":
            df[
                "query_embedding_time_sec"
            ].astype(
                float
            ),

        "mean_search_latency_ms":
            df[
                "mean_search_latency_ms"
            ].astype(
                float
            ),

        "median_search_latency_ms":
            df[
                "median_search_latency_ms"
            ].astype(
                float
            ),

        "p95_search_latency_ms":
            df[
                "p95_search_latency_ms"
            ].astype(
                float
            ),

        "p99_search_latency_ms":
            df[
                "p99_search_latency_ms"
            ].astype(
                float
            ),

        "recall_at_1":
            df[
                "recall_at_1"
            ].astype(
                float
            ),

        "recall_at_5":
            df[
                "recall_at_5"
            ].astype(
                float
            ),

        "recall_at_10":
            df[
                "recall_at_10"
            ].astype(
                float
            ),

        "hit_rate_at_1":
            df[
                "hit_rate_at_1"
            ].astype(
                float
            ),

        "hit_rate_at_5":
            df[
                "hit_rate_at_5"
            ].astype(
                float
            ),

        "hit_rate_at_10":
            df[
                "hit_rate_at_10"
            ].astype(
                float
            ),

        "max_python_rss_mb":
            df[
                "max_observed_python_rss_mb"
            ].astype(
                float
            ),

        "total_experiment_time_sec":
            df[
                "total_experiment_time_sec"
            ].astype(
                float
            ),

        "timestamp":
            df[
                "timestamp"
            ],

        "corpus_sha256":
            df[
                "corpus_sha256"
            ],

        "manifest_sha256":
            df[
                "manifest_sha256"
            ],
    })

    return summary.sort_values(
        "documents"
    ).reset_index(
        drop=True
    )


# ============================================================
# Scaling calculations
# ============================================================

def calculate_scaling(
    summary: pd.DataFrame,
):

    if len(
        summary
    ) < 2:

        return {}

    smallest = summary.iloc[
        0
    ]

    largest = summary.iloc[
        -1
    ]

    corpus_growth = (
        largest[
            "documents"
        ]
        / smallest[
            "documents"
        ]
    )

    embedding_growth = (
        largest[
            "embedding_time_sec"
        ]
        / smallest[
            "embedding_time_sec"
        ]
    )

    index_growth = (
        largest[
            "index_size_mb"
        ]
        / smallest[
            "index_size_mb"
        ]
    )

    mean_latency_growth = (
        largest[
            "mean_search_latency_ms"
        ]
        / smallest[
            "mean_search_latency_ms"
        ]
    )

    p95_latency_growth = (
        largest[
            "p95_search_latency_ms"
        ]
        / smallest[
            "p95_search_latency_ms"
        ]
    )

    rss_growth = (
        largest[
            "max_python_rss_mb"
        ]
        / smallest[
            "max_python_rss_mb"
        ]
    )

    recall_1_change = (
        largest[
            "recall_at_1"
        ]
        - smallest[
            "recall_at_1"
        ]
    )

    recall_5_change = (
        largest[
            "recall_at_5"
        ]
        - smallest[
            "recall_at_5"
        ]
    )

    recall_10_change = (
        largest[
            "recall_at_10"
        ]
        - smallest[
            "recall_at_10"
        ]
    )

    return {

        "smallest_documents":
            int(
                smallest[
                    "documents"
                ]
            ),

        "largest_documents":
            int(
                largest[
                    "documents"
                ]
            ),

        "corpus_growth":
            corpus_growth,

        "embedding_time_growth":
            embedding_growth,

        "index_size_growth":
            index_growth,

        "mean_latency_growth":
            mean_latency_growth,

        "p95_latency_growth":
            p95_latency_growth,

        "rss_growth":
            rss_growth,

        "recall_at_1_change":
            recall_1_change,

        "recall_at_5_change":
            recall_5_change,

        "recall_at_10_change":
            recall_10_change,
    }


# ============================================================
# Plot helpers
# ============================================================

def save_plot(
    path: Path,
):

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_embedding_time(
    summary,
    path,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "embedding_time_sec"
        ],
        marker="o",
    )

    plt.xlabel(
        "Corpus Size (documents)"
    )

    plt.ylabel(
        "Document Embedding Time (seconds)"
    )

    plt.title(
        "V1 Exact Search - Embedding Time Scaling"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    save_plot(
        path
    )


def plot_embedding_throughput(
    summary,
    path,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "embedding_docs_per_sec"
        ],
        marker="o",
    )

    plt.xlabel(
        "Corpus Size (documents)"
    )

    plt.ylabel(
        "Embedding Throughput (documents/second)"
    )

    plt.title(
        "V1 Exact Search - Embedding Throughput"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    save_plot(
        path
    )


def plot_index_size(
    summary,
    path,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "index_size_mb"
        ],
        marker="o",
    )

    plt.xlabel(
        "Corpus Size (documents)"
    )

    plt.ylabel(
        "Serialized FAISS Index Size (MB)"
    )

    plt.title(
        "V1 Exact Search - Index Size Scaling"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    save_plot(
        path
    )


def plot_search_latency(
    summary,
    path,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "mean_search_latency_ms"
        ],
        marker="o",
        label="Mean",
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "median_search_latency_ms"
        ],
        marker="o",
        label="Median",
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "p95_search_latency_ms"
        ],
        marker="o",
        label="P95",
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "p99_search_latency_ms"
        ],
        marker="o",
        label="P99",
    )

    plt.xlabel(
        "Corpus Size (documents)"
    )

    plt.ylabel(
        "Exact Search Latency (ms/query)"
    )

    plt.title(
        "V1 Exact Search - Query Latency Scaling"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    save_plot(
        path
    )


def plot_memory(
    summary,
    path,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "max_python_rss_mb"
        ],
        marker="o",
    )

    plt.xlabel(
        "Corpus Size (documents)"
    )

    plt.ylabel(
        "Maximum Observed Python RSS (MB)"
    )

    plt.title(
        "V1 Exact Search - Process Memory Scaling"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    save_plot(
        path
    )


def plot_recall(
    summary,
    path,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "recall_at_1"
        ],
        marker="o",
        label="Recall@1",
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "recall_at_5"
        ],
        marker="o",
        label="Recall@5",
    )

    plt.plot(
        summary[
            "documents"
        ],
        summary[
            "recall_at_10"
        ],
        marker="o",
        label="Recall@10",
    )

    plt.xlabel(
        "Corpus Size (documents)"
    )

    plt.ylabel(
        "Relevance Recall"
    )

    plt.title(
        "V1 Exact Search - Retrieval Quality Scaling"
    )

    plt.ylim(
        0.0,
        1.02,
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    save_plot(
        path
    )


# ============================================================
# Console table
# ============================================================

def print_summary_table(
    summary,
):

    print()
    print(
        "V1 Exact Search Scaling Results"
    )

    print(
        "==============================="
    )

    print()

    header = (
        f"{'Docs':>10} "
        f"{'Embed(s)':>10} "
        f"{'Docs/s':>9} "
        f"{'IndexMB':>9} "
        f"{'Mean(ms)':>10} "
        f"{'P95(ms)':>9} "
        f"{'R@1':>7} "
        f"{'R@5':>7} "
        f"{'R@10':>7} "
        f"{'RSS(MB)':>9}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for _, row in summary.iterrows():

        print(
            f"{int(row['documents']):>10,} "
            f"{row['embedding_time_sec']:>10.2f} "
            f"{row['embedding_docs_per_sec']:>9.2f} "
            f"{row['index_size_mb']:>9.2f} "
            f"{row['mean_search_latency_ms']:>10.3f} "
            f"{row['p95_search_latency_ms']:>9.3f} "
            f"{row['recall_at_1']:>7.4f} "
            f"{row['recall_at_5']:>7.4f} "
            f"{row['recall_at_10']:>7.4f} "
            f"{row['max_python_rss_mb']:>9.1f}"
        )


# ============================================================
# Interpretation
# ============================================================

def print_interpretation(
    summary,
    scaling,
):

    if not scaling:
        return

    smallest = summary.iloc[
        0
    ]

    largest = summary.iloc[
        -1
    ]

    print()
    print(
        "Scaling Summary"
    )

    print(
        "==============="
    )

    print()

    print(
        f"Corpus growth:"
        f"              "
        f"{scaling['corpus_growth']:.2f}x"
    )

    print(
        f"Embedding-time growth:"
        f"      "
        f"{scaling['embedding_time_growth']:.2f}x"
    )

    print(
        f"Index-size growth:"
        f"          "
        f"{scaling['index_size_growth']:.2f}x"
    )

    print(
        f"Mean-latency growth:"
        f"        "
        f"{scaling['mean_latency_growth']:.2f}x"
    )

    print(
        f"P95-latency growth:"
        f"         "
        f"{scaling['p95_latency_growth']:.2f}x"
    )

    print(
        f"Python RSS growth:"
        f"           "
        f"{scaling['rss_growth']:.2f}x"
    )

    print()

    print(
        "Retrieval-quality change "
        f"({int(smallest['documents']):,} "
        "-> "
        f"{int(largest['documents']):,} docs):"
    )

    print(
        f"  Recall@1:  "
        f"{smallest['recall_at_1']:.4f} "
        "-> "
        f"{largest['recall_at_1']:.4f} "
        f"({scaling['recall_at_1_change']:+.4f})"
    )

    print(
        f"  Recall@5:  "
        f"{smallest['recall_at_5']:.4f} "
        "-> "
        f"{largest['recall_at_5']:.4f} "
        f"({scaling['recall_at_5_change']:+.4f})"
    )

    print(
        f"  Recall@10: "
        f"{smallest['recall_at_10']:.4f} "
        "-> "
        f"{largest['recall_at_10']:.4f} "
        f"({scaling['recall_at_10_change']:+.4f})"
    )

    print()
    print(
        "Interpretation"
    )

    print(
        "=============="
    )

    print()

    print(
        "1. Document embedding remains the dominant "
        "offline preprocessing cost."
    )

    print(
        "2. Embedding throughput remains relatively "
        "stable across the tested corpus sizes."
    )

    print(
        "3. IndexFlatIP storage grows approximately "
        "linearly because every float32 vector is retained."
    )

    print(
        "4. Exact-search latency increases substantially "
        "as the vector collection grows."
    )

    print(
        "5. Relevance Recall@K decreases as additional "
        "competing documents enter the collection."
    )

    print(
        "6. The Recall@K decline in V1 is NOT ANN "
        "approximation error: IndexFlatIP performs exact "
        "nearest-neighbor search."
    )

    print(
        "7. These exact-neighbor results provide the "
        "algorithmic ground truth required for V2 ANN "
        "evaluation."
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    experiment_path = Path(
        args.experiments
    )

    manifest_path = Path(
        args.manifest
    )

    results_dir = Path(
        args.results_dir
    )

    if not experiment_path.exists():

        raise FileNotFoundError(
            "Experiment CSV not found:\n"
            f"{experiment_path.resolve()}"
        )

    if not manifest_path.exists():

        raise FileNotFoundError(
            "Benchmark manifest not found:\n"
            f"{manifest_path.resolve()}"
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "V1 - Exact Vector Search Scaling Analysis"
    )

    print(
        "========================================="
    )

    print()

    print(
        f"Experiment CSV: "
        f"{experiment_path.resolve()}"
    )

    print(
        f"Manifest:       "
        f"{manifest_path.resolve()}"
    )

    # ========================================================
    # Current benchmark provenance
    # ========================================================

    current_manifest_sha256 = (
        sha256_file(
            manifest_path
        )
    )

    print()

    print(
        "Current manifest SHA-256:"
    )

    print(
        f"  {current_manifest_sha256}"
    )

    # ========================================================
    # Load experiments
    # ========================================================

    df = pd.read_csv(
        experiment_path
    )

    validate_columns(
        df
    )

    (
        selected,
        diagnostics,
    ) = select_current_runs(
        df,
        current_manifest_sha256,
    )

    validate_selected_provenance(
        selected
    )

    print()
    print(
        "Experiment Selection"
    )

    print(
        "===================="
    )

    print()

    print(
        f"CSV rows:                    "
        f"{diagnostics['total_rows']}"
    )

    print(
        f"Successful rows:             "
        f"{diagnostics['successful_rows']}"
    )

    print(
        f"Current-provenance rows:     "
        f"{diagnostics['current_provenance_rows']}"
    )

    print(
        f"Obsolete successful rows:    "
        f"{diagnostics['obsolete_success_rows']}"
    )

    print(
        f"Selected latest scale runs:  "
        f"{diagnostics['selected_rows']}"
    )

    # ========================================================
    # Scale-point check
    # ========================================================

    (
        actual_sizes,
        missing_sizes,
        extra_sizes,
    ) = validate_scale_points(
        selected
    )

    print()

    print(
        "Selected corpus sizes:"
    )

    print(
        "  "
        + ", ".join(
            f"{size:,}"
            for size
            in actual_sizes
        )
    )

    if missing_sizes:

        print()

        print(
            "WARNING: Missing expected V1 scale points:"
        )

        print(
            "  "
            + ", ".join(
                f"{size:,}"
                for size
                in missing_sizes
            )
        )

    if extra_sizes:

        print()

        print(
            "Additional scale points:"
        )

        print(
            "  "
            + ", ".join(
                f"{size:,}"
                for size
                in extra_sizes
            )
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = build_summary(
        selected
    )

    summary_path = (
        results_dir
        / "v1_scaling_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print_summary_table(
        summary
    )

    scaling = calculate_scaling(
        summary
    )

    print_interpretation(
        summary,
        scaling,
    )

    # ========================================================
    # Plots
    # ========================================================

    embedding_time_path = (
        results_dir
        / "v1_embedding_time.png"
    )

    embedding_throughput_path = (
        results_dir
        / "v1_embedding_throughput.png"
    )

    index_size_path = (
        results_dir
        / "v1_index_size.png"
    )

    search_latency_path = (
        results_dir
        / "v1_search_latency.png"
    )

    memory_path = (
        results_dir
        / "v1_memory.png"
    )

    recall_path = (
        results_dir
        / "v1_recall.png"
    )

    plot_embedding_time(
        summary,
        embedding_time_path,
    )

    plot_embedding_throughput(
        summary,
        embedding_throughput_path,
    )

    plot_index_size(
        summary,
        index_size_path,
    )

    plot_search_latency(
        summary,
        search_latency_path,
    )

    plot_memory(
        summary,
        memory_path,
    )

    plot_recall(
        summary,
        recall_path,
    )

    # ========================================================
    # Outputs
    # ========================================================

    print()
    print(
        "Outputs"
    )

    print(
        "======="
    )

    print()

    for path in (
        summary_path,
        embedding_time_path,
        embedding_throughput_path,
        index_size_path,
        search_latency_path,
        memory_path,
        recall_path,
    ):

        print(
            f"  {path.resolve()}"
        )

    print()

    print(
        "V1 exact-search analysis complete."
    )


if __name__ == "__main__":
    main()