"""
V1 - Prepare Controlled MS MARCO Vector Search Benchmark

Big Data Vector Search
======================

Purpose
-------
Create deterministic, nested MS MARCO subsets for controlled vector-search
experiments.

The benchmark preserves:

    - one fixed evaluation query set
    - all positively relevant documents for those queries
    - deterministic distractor sampling
    - nested document collections

Default target sizes:

    1K
    10K
    25K
    50K
    100K

Nesting:

    S_1K âŠ‚ S_10K âŠ‚ S_25K âŠ‚ S_50K âŠ‚ S_100K

This allows corpus size to change while keeping the evaluation workload
and required relevant documents fixed.

Dataset
-------
Hugging Face / BEIR MS MARCO:

    BeIR/msmarco

Configurations:

    corpus
    queries

Relevance judgments:

    BeIR/msmarco-qrels

Output
------
data/processed/

    benchmark_manifest.json
    queries.jsonl
    qrels.json

    msmarco_1k/
        corpus.jsonl

    msmarco_10k/
        corpus.jsonl

    msmarco_25k/
        corpus.jsonl

    msmarco_50k/
        corpus.jsonl

    msmarco_100k/
        corpus.jsonl

Examples
--------

Development validation:

    python src/prepare_msmarco_benchmark.py --max-size 1000

Complete V1 benchmark:

    python src/prepare_msmarco_benchmark.py --max-size 100000

Overwrite existing benchmark:

    python src/prepare_msmarco_benchmark.py \
        --max-size 1000 \
        --overwrite

Methodology
-----------
1. Load MS MARCO queries.
2. Load positive relevance judgments.
3. Select a deterministic set of queries with positive qrels.
4. Collect every positively relevant document required by those queries.
5. Scan the corpus once.
6. Preserve all required relevant documents.
7. Select deterministic distractors using reservoir sampling.
8. Construct one master corpus.
9. Create nested prefixes of that master corpus.
10. Write queries, qrels, corpora, and benchmark provenance.

The script deliberately does NOT create embeddings or FAISS indexes.
Those belong to the V1 exact-search evaluation stage.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
)


# ============================================================
# Dataset configuration
# ============================================================

CORPUS_DATASET = "BeIR/msmarco"
CORPUS_CONFIG = "corpus"

QUERY_DATASET = "BeIR/msmarco"
QUERY_CONFIG = "queries"

QRELS_DATASET = "BeIR/msmarco-qrels"

DEFAULT_SEED = 42
DEFAULT_QUERY_COUNT = 300

DEFAULT_SUBSET_SIZES = [
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
            "Prepare deterministic nested MS MARCO "
            "vector-search benchmark subsets."
        )
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=100_000,
        help=(
            "Largest corpus size to create. "
            "Supported values: "
            "1000, 10000, 25000, 50000, 100000. "
            "Default: 100000."
        ),
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help=(
            "Number of evaluation queries. "
            "Default: 300."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Random seed used for deterministic "
            "query and distractor selection. "
            "Default: 42."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing benchmark output files."
        ),
    )

    return parser.parse_args()


# ============================================================
# Argument validation
# ============================================================

def validate_args(args):

    if args.max_size not in DEFAULT_SUBSET_SIZES:

        supported = ", ".join(
            f"{size:,}"
            for size in DEFAULT_SUBSET_SIZES
        )

        raise ValueError(
            "--max-size must be one of: "
            f"{supported}"
        )

    if args.queries < 1:

        raise ValueError(
            "--queries must be >= 1"
        )


# ============================================================
# Hugging Face helpers
# ============================================================

def describe_dataset_object(
    dataset_object: Any,
    label: str,
):
    """
    Print concise schema information before expensive processing.
    """

    print()

    print(
        f"{label} object type: "
        f"{type(dataset_object).__name__}"
    )

    if isinstance(
        dataset_object,
        DatasetDict,
    ):

        print(
            f"{label} splits: "
            f"{list(dataset_object.keys())}"
        )

        for split_name, split in dataset_object.items():

            print(
                f"  {split_name}: "
                f"{len(split):,} rows, "
                f"columns={split.column_names}"
            )

    elif isinstance(
        dataset_object,
        Dataset,
    ):

        print(
            f"{label}: "
            f"{len(dataset_object):,} rows, "
            f"columns={dataset_object.column_names}"
        )


def select_split(
    dataset_object,
    preferred_splits=(),
    label="dataset",
):
    """
    Return a Dataset from either Dataset or DatasetDict.

    Split selection order:

        1. preferred_splits supplied by caller
        2. train
        3. validation
        4. test
        5. queries
        6. corpus
        7. only available split

    This avoids assuming that every Hugging Face dataset uses
    the conventional 'train' split.
    """

    if isinstance(
        dataset_object,
        Dataset,
    ):
        return dataset_object

    if not isinstance(
        dataset_object,
        DatasetDict,
    ):

        raise TypeError(
            "Unexpected Hugging Face dataset object "
            f"for {label}: "
            f"{type(dataset_object).__name__}"
        )

    split_names = list(
        dataset_object.keys()
    )

    candidates = (
        list(preferred_splits)
        + [
            "train",
            "validation",
            "test",
            "queries",
            "corpus",
        ]
    )

    seen = set()

    for split_name in candidates:

        if split_name in seen:
            continue

        seen.add(
            split_name
        )

        if split_name in dataset_object:

            print(
                f"Using {label} split: "
                f"{split_name}"
            )

            return dataset_object[
                split_name
            ]

    if len(split_names) == 1:

        split_name = (
            split_names[0]
        )

        print(
            f"Using only available "
            f"{label} split: {split_name}"
        )

        return dataset_object[
            split_name
        ]

    raise RuntimeError(
        f"Unable to determine {label} split. "
        f"Available splits: {split_names}"
    )


def normalize_id(
    value,
) -> str:

    if value is None:

        raise ValueError(
            "Encountered null identifier."
        )

    return str(
        value
    )


def first_existing_field(
    row,
    candidates,
):
    """
    Return the first candidate field found in a mapping-like row.
    """

    if not hasattr(
        row,
        "keys",
    ):

        raise TypeError(
            "Expected a dataset row mapping, "
            f"received {type(row).__name__}: "
            f"{row!r}"
        )

    for key in candidates:

        if key in row:
            return key

    return None


def extract_id(
    row,
    candidates,
    description,
):
    """
    Extract and normalize an identifier from a row.
    """

    key = first_existing_field(
        row,
        candidates,
    )

    if key is None:

        raise KeyError(
            f"Could not find {description}. "
            f"Available fields: "
            f"{list(row.keys())}"
        )

    return normalize_id(
        row[key]
    )


# ============================================================
# Query loading
# ============================================================

def load_queries():
    """
    Load the BEIR MS MARCO query collection.

    Returns:

        {
            query_id: query_text,
            ...
        }
    """

    print()
    print(
        "Loading MS MARCO queries..."
    )

    start = (
        time.perf_counter()
    )

    dataset_object = load_dataset(
        QUERY_DATASET,
        QUERY_CONFIG,
    )

    describe_dataset_object(
        dataset_object,
        "Query dataset",
    )

    dataset = select_split(
        dataset_object,
        preferred_splits=(
            "queries",
            "train",
        ),
        label="query",
    )

    print(
        f"Selected query columns: "
        f"{dataset.column_names}"
    )

    queries = {}

    for row in dataset:

        query_id = extract_id(
            row,
            candidates=(
                "_id",
                "id",
                "query-id",
                "query_id",
            ),
            description="query ID",
        )

        text_key = first_existing_field(
            row,
            (
                "text",
                "query",
                "question",
            ),
        )

        if text_key is None:

            raise KeyError(
                "Could not find query text field. "
                f"Available fields: "
                f"{list(row.keys())}"
            )

        text = str(
            row[text_key]
            or ""
        ).strip()

        if not text:
            continue

        queries[
            query_id
        ] = text

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print(
        f"Queries loaded: "
        f"{len(queries):,}"
    )

    print(
        f"Query load time: "
        f"{elapsed:.2f}s"
    )

    if not queries:

        raise RuntimeError(
            "No usable MS MARCO queries were loaded."
        )

    return queries


# ============================================================
# Qrels loading
# ============================================================

def load_qrels():
    """
    Load positive MS MARCO relevance judgments.

    Returns:

        {
            query_id: {
                doc_id: relevance_score,
                ...
            },
            ...
        }
    """

    print()
    print(
        "Loading MS MARCO qrels..."
    )

    start = (
        time.perf_counter()
    )

    dataset_object = load_dataset(
        QRELS_DATASET,
    )

    describe_dataset_object(
        dataset_object,
        "Qrels dataset",
    )

    # BEIR evaluation qrels are commonly exposed through
    # validation or test. Prefer those over training qrels.
    dataset = select_split(
        dataset_object,
        preferred_splits=(
            "validation",
	    "test",
            "train",
        ),
        label="qrels",
    )

    print(
        f"Selected qrels columns: "
        f"{dataset.column_names}"
    )

    qrels = {}

    for row in dataset:

        query_id = extract_id(
            row,
            candidates=(
                "query-id",
                "query_id",
                "queryid",
                "qid",
                "_id",
            ),
            description="qrels query ID",
        )

        document_id = extract_id(
            row,
            candidates=(
                "corpus-id",
                "corpus_id",
                "doc_id",
                "document_id",
                "docid",
                "pid",
            ),
            description="qrels document ID",
        )

        score_key = first_existing_field(
            row,
            (
                "score",
                "relevance",
                "rel",
                "label",
            ),
        )

        if score_key is None:

            raise KeyError(
                "Could not find qrels relevance field. "
                f"Available fields: "
                f"{list(row.keys())}"
            )

        score = float(
            row[
                score_key
            ]
        )

        # Only positive relevance judgments are required for
        # the controlled retrieval benchmark.
        if score <= 0:
            continue

        qrels.setdefault(
            query_id,
            {}
        )[document_id] = score

    elapsed = (
        time.perf_counter()
        - start
    )

    positive_pairs = sum(
        len(documents)
        for documents
        in qrels.values()
    )

    print()
    print(
        "Queries with positive qrels: "
        f"{len(qrels):,}"
    )

    print(
        "Positive query-document pairs: "
        f"{positive_pairs:,}"
    )

    print(
        f"Qrels load time: "
        f"{elapsed:.2f}s"
    )

    if not qrels:

        raise RuntimeError(
            "No positive MS MARCO qrels were loaded."
        )

    return qrels


# ============================================================
# Query selection
# ============================================================

def select_queries(
    queries,
    qrels,
    query_count,
    seed,
):
    """
    Deterministically select evaluation queries that have:

        - query text
        - at least one positive qrel
    """

    eligible = sorted(
        set(
            queries.keys()
        )
        & set(
            qrels.keys()
        )
    )

    print()
    print(
        f"Eligible queries with text + qrels: "
        f"{len(eligible):,}"
    )

    if len(eligible) < query_count:

        raise RuntimeError(
            "Not enough eligible evaluation queries. "
            f"Requested {query_count:,}, "
            f"available {len(eligible):,}."
        )

    rng = random.Random(
        seed
    )

    selected = rng.sample(
        eligible,
        query_count,
    )

    # Sort after sampling to produce stable output files while
    # preserving deterministic random selection.
    selected.sort()

    return selected


# ============================================================
# Required relevant documents
# ============================================================

def collect_required_documents(
    selected_query_ids,
    qrels,
):
    """
    Return every positively relevant document required by the
    selected evaluation queries.
    """

    required = set()

    for query_id in selected_query_ids:

        required.update(
            qrels[
                query_id
            ].keys()
        )

    return required


# ============================================================
# Corpus loading
# ============================================================

def load_corpus_dataset():
    """
    Load and return the MS MARCO corpus Dataset.
    """

    print()
    print(
        "Loading MS MARCO corpus metadata..."
    )

    dataset_object = load_dataset(
        CORPUS_DATASET,
        CORPUS_CONFIG,
    )

    describe_dataset_object(
        dataset_object,
        "Corpus dataset",
    )

    dataset = select_split(
        dataset_object,
        preferred_splits=(
            "corpus",
            "train",
        ),
        label="corpus",
    )

    print(
        f"Selected corpus columns: "
        f"{dataset.column_names}"
    )

    return dataset


# ============================================================
# Corpus row normalization
# ============================================================

def normalize_document(
    row,
):
    """
    Convert one BEIR corpus row into the canonical V1 document
    representation.
    """

    document_id = extract_id(
        row,
        candidates=(
            "_id",
            "id",
            "corpus-id",
            "corpus_id",
            "doc_id",
            "document_id",
        ),
        description="corpus document ID",
    )

    title_key = first_existing_field(
        row,
        (
            "title",
        ),
    )

    text_key = first_existing_field(
        row,
        (
            "text",
            "contents",
            "passage",
            "document",
        ),
    )

    if text_key is None:

        raise KeyError(
            "Could not find corpus text field. "
            f"Available fields: "
            f"{list(row.keys())}"
        )

    title = (
        str(
            row[
                title_key
            ]
            or ""
        )
        if title_key is not None
        else ""
    )

    text = str(
        row[
            text_key
        ]
        or ""
    )

    return {
        "_id": document_id,
        "title": title,
        "text": text,
    }


# ============================================================
# Master corpus construction
# ============================================================

def build_master_corpus(
    dataset,
    required_document_ids,
    max_size,
    seed,
):
    """
    Construct one deterministic master corpus.

    Every required relevant document is included.

    Remaining positions are filled using deterministic reservoir
    sampling from non-relevant corpus documents.

    Reservoir sampling prevents the entire multi-million-document
    corpus from being copied into a Python list.
    """

    if (
        len(
            required_document_ids
        )
        > max_size
    ):

        raise RuntimeError(
            "Required relevant-document set is larger "
            "than the requested maximum corpus size. "
            f"Required="
            f"{len(required_document_ids):,}, "
            f"max-size={max_size:,}."
        )

    distractor_target = (
        max_size
        - len(
            required_document_ids
        )
    )

    print()
    print(
        f"Source corpus documents: "
        f"{len(dataset):,}"
    )

    print(
        "Required relevant documents: "
        f"{len(required_document_ids):,}"
    )

    print(
        f"Distractors required: "
        f"{distractor_target:,}"
    )

    rng = random.Random(
        seed
    )

    relevant_documents = {}

    distractor_reservoir = []

    distractors_seen = 0

    print()
    print(
        "Scanning corpus..."
    )

    for row in tqdm(
        dataset,
        total=len(dataset),
        unit="docs",
    ):

        document = normalize_document(
            row
        )

        document_id = (
            document[
                "_id"
            ]
        )

        # ----------------------------------------------------
        # Required relevant document
        # ----------------------------------------------------

        if (
            document_id
            in required_document_ids
        ):

            relevant_documents[
                document_id
            ] = document

            continue

        # ----------------------------------------------------
        # No distractors required
        # ----------------------------------------------------

        if distractor_target == 0:
            continue

        # ----------------------------------------------------
        # Deterministic reservoir sampling
        # ----------------------------------------------------

        distractors_seen += 1

        if (
            len(
                distractor_reservoir
            )
            < distractor_target
        ):

            distractor_reservoir.append(
                document
            )

        else:

            replacement_index = (
                rng.randrange(
                    distractors_seen
                )
            )

            if (
                replacement_index
                < distractor_target
            ):

                distractor_reservoir[
                    replacement_index
                ] = document

    # --------------------------------------------------------
    # Validate relevant documents
    # --------------------------------------------------------

    missing_relevant = (
        required_document_ids
        - set(
            relevant_documents.keys()
        )
    )

    if missing_relevant:

        preview = sorted(
            missing_relevant
        )[:10]

        raise RuntimeError(
            "Some required relevant documents were not found "
            "in the corpus. "
            f"Missing={len(missing_relevant):,}. "
            f"Examples={preview}"
        )

    # --------------------------------------------------------
    # Validate distractor count
    # --------------------------------------------------------

    if (
        len(
            distractor_reservoir
        )
        != distractor_target
    ):

        raise RuntimeError(
            "Unable to collect enough distractors. "
            f"Needed={distractor_target:,}, "
            f"collected="
            f"{len(distractor_reservoir):,}."
        )

    # --------------------------------------------------------
    # Stable relevant-document ordering
    # --------------------------------------------------------

    relevant_list = [
        relevant_documents[
            document_id
        ]
        for document_id
        in sorted(
            relevant_documents.keys()
        )
    ]

    # Reservoir slot ordering is an implementation detail.
    # Shuffle deterministically before constructing prefixes.
    rng.shuffle(
        distractor_reservoir
    )

    master_corpus = (
        relevant_list
        + distractor_reservoir
    )

    if (
        len(
            master_corpus
        )
        != max_size
    ):

        raise RuntimeError(
            "Master corpus size mismatch. "
            f"Expected={max_size:,}, "
            f"actual={len(master_corpus):,}."
        )

    return master_corpus


# ============================================================
# File helpers
# ============================================================

def ensure_writable(
    path: Path,
    overwrite: bool,
):
    """
    Prevent accidental replacement of benchmark artifacts.
    """

    if (
        path.exists()
        and not overwrite
    ):

        raise FileExistsError(
            "Output already exists:\n"
            f"{path}\n"
            "Use --overwrite to replace existing output."
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
            sort_keys=False,
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

            f.write(
                "\n"
            )


# ============================================================
# Benchmark validation
# ============================================================

def validate_subset(
    subset,
    required_document_ids,
    expected_size,
    subset_name,
):
    """
    Validate one generated corpus subset.
    """

    if len(subset) != expected_size:

        raise RuntimeError(
            f"{subset_name} size mismatch. "
            f"Expected={expected_size:,}, "
            f"actual={len(subset):,}."
        )

    document_ids = [
        row[
            "_id"
        ]
        for row in subset
    ]

    unique_document_ids = set(
        document_ids
    )

    if (
        len(
            unique_document_ids
        )
        != len(
            document_ids
        )
    ):

        raise RuntimeError(
            f"{subset_name} contains duplicate "
            "document IDs."
        )

    missing_relevant = (
        required_document_ids
        - unique_document_ids
    )

    if missing_relevant:

        raise RuntimeError(
            f"{subset_name} is missing "
            f"{len(missing_relevant):,} required "
            "relevant documents."
        )


# ============================================================
# Benchmark output
# ============================================================

def write_benchmark(
    master_corpus,
    queries,
    qrels,
    selected_query_ids,
    required_document_ids,
    subset_sizes,
    seed,
    overwrite,
):
    """
    Write fixed queries, qrels, nested corpora, and manifest.
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Queries
    # --------------------------------------------------------

    query_rows = [
        {
            "_id": query_id,
            "text": queries[
                query_id
            ],
        }
        for query_id
        in selected_query_ids
    ]

    queries_path = (
        OUTPUT_ROOT
        / "queries.jsonl"
    )

    write_jsonl(
        queries_path,
        query_rows,
        overwrite,
    )

    # --------------------------------------------------------
    # Qrels
    # --------------------------------------------------------

    selected_qrels = {
        query_id: qrels[
            query_id
        ]
        for query_id
        in selected_query_ids
    }

    qrels_path = (
        OUTPUT_ROOT
        / "qrels.json"
    )

    write_json(
        qrels_path,
        selected_qrels,
        overwrite,
    )

    # --------------------------------------------------------
    # Nested corpora
    # --------------------------------------------------------

    subset_records = []

    previous_ids = None

    for size in subset_sizes:

        subset_name = (
            f"msmarco_"
            f"{size // 1000}k"
        )

        subset = (
            master_corpus[
                :size
            ]
        )

        validate_subset(
            subset,
            required_document_ids,
            size,
            subset_name,
        )

        subset_ids = {
            row[
                "_id"
            ]
            for row in subset
        }

        # Explicitly verify nesting.
        if (
            previous_ids is not None
            and not previous_ids.issubset(
                subset_ids
            )
        ):

            raise RuntimeError(
                "Nested-subset validation failed "
                f"for {subset_name}."
            )

        previous_ids = (
            subset_ids
        )

        subset_dir = (
            OUTPUT_ROOT
            / subset_name
        )

        corpus_path = (
            subset_dir
            / "corpus.jsonl"
        )

        write_jsonl(
            corpus_path,
            subset,
            overwrite,
        )

        subset_records.append({
            "name":
                subset_name,

            "documents":
                size,

            "corpus_file":
                str(
                    corpus_path.relative_to(
                        ROOT
                    )
                ),

            "unique_document_ids":
                len(
                    subset_ids
                ),

            "required_relevant_documents_present":
                True,

            "nested_with_previous_subset":
                True,
        })

        print(
            f"Wrote {subset_name}: "
            f"{size:,} documents"
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "benchmark_version":
            "v1",

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "dataset":
            CORPUS_DATASET,

        "corpus_config":
            CORPUS_CONFIG,

        "query_dataset":
            QUERY_DATASET,

        "query_config":
            QUERY_CONFIG,

        "qrels_dataset":
            QRELS_DATASET,

        "seed":
            seed,

        "evaluation_queries":
            len(
                selected_query_ids
            ),

        "required_relevant_documents":
            len(
                required_document_ids
            ),

        "subset_sizes":
            subset_sizes,

        "sampling":
            (
                "deterministic nested corpus; all required "
                "positive-qrel documents included; remaining "
                "documents selected using seeded reservoir "
                "sampling"
            ),

        "queries_file":
            str(
                queries_path.relative_to(
                    ROOT
                )
            ),

        "qrels_file":
            str(
                qrels_path.relative_to(
                    ROOT
                )
            ),

        "subsets":
            subset_records,

        "environment": {

            "python_version":
                platform.python_version(),

            "python_executable":
                sys.executable,

            "platform":
                platform.platform(),
        },
    }

    manifest_path = (
        OUTPUT_ROOT
        / "benchmark_manifest.json"
    )

    write_json(
        manifest_path,
        manifest,
        overwrite,
    )

    return manifest_path


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    validate_args(
        args
    )

    subset_sizes = [
        size
        for size
        in DEFAULT_SUBSET_SIZES
        if size <= args.max_size
    ]

    print()
    print(
        "V1 - MS MARCO Vector Search Benchmark Preparation"
    )

    print(
        "================================================"
    )

    print()

    print(
        f"Evaluation queries: "
        f"{args.queries:,}"
    )

    print(
        f"Random seed:        "
        f"{args.seed}"
    )

    print(
        f"Maximum corpus:     "
        f"{args.max_size:,}"
    )

    print(
        "Subset sizes:       "
        + ", ".join(
            f"{size:,}"
            for size
            in subset_sizes
        )
    )

    print(
        f"Output directory:   "
        f"{OUTPUT_ROOT}"
    )

    total_start = (
        time.perf_counter()
    )

    # ========================================================
    # Queries
    # ========================================================

    queries = load_queries()

    # ========================================================
    # Qrels
    # ========================================================

    qrels = load_qrels()

    # ========================================================
    # Select fixed evaluation workload
    # ========================================================

    selected_query_ids = (
        select_queries(
            queries,
            qrels,
            args.queries,
            args.seed,
        )
    )

    print()
    print(
        "Selected evaluation queries: "
        f"{len(selected_query_ids):,}"
    )

    # ========================================================
    # Relevant-document requirements
    # ========================================================

    required_document_ids = (
        collect_required_documents(
            selected_query_ids,
            qrels,
        )
    )

    print(
        "Required relevant documents: "
        f"{len(required_document_ids):,}"
    )

    smallest_subset = min(
        subset_sizes
    )

    if (
        len(
            required_document_ids
        )
        > smallest_subset
    ):

        raise RuntimeError(
            "The required relevant-document set does "
            "not fit inside the smallest benchmark subset. "
            f"Required="
            f"{len(required_document_ids):,}, "
            f"smallest subset="
            f"{smallest_subset:,}."
        )

    # ========================================================
    # Load corpus only after query/qrels validation succeeds
    # ========================================================

    corpus_dataset = (
        load_corpus_dataset()
    )

    # ========================================================
    # Build deterministic master corpus
    # ========================================================

    print()
    print(
        "Building deterministic master corpus..."
    )

    corpus_start = (
        time.perf_counter()
    )

    master_corpus = (
        build_master_corpus(
            corpus_dataset,
            required_document_ids,
            args.max_size,
            args.seed,
        )
    )

    corpus_time = (
        time.perf_counter()
        - corpus_start
    )

    print()
    print(
        "Master corpus created: "
        f"{len(master_corpus):,} documents"
    )

    print(
        f"Corpus preparation time: "
        f"{corpus_time:.2f}s"
    )

    # ========================================================
    # Write benchmark
    # ========================================================

    print()
    print(
        "Writing benchmark artifacts..."
    )

    manifest_path = (
        write_benchmark(
            master_corpus,
            queries,
            qrels,
            selected_query_ids,
            required_document_ids,
            subset_sizes,
            args.seed,
            args.overwrite,
        )
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 68)

    print(
        "V1 Benchmark Preparation Complete"
    )

    print("=" * 68)

    print(
        f"Queries:                    "
        f"{len(selected_query_ids):,}"
    )

    print(
        f"Required relevant documents:"
        f" {len(required_document_ids):,}"
    )

    print(
        f"Largest corpus:             "
        f"{args.max_size:,}"
    )

    print(
        f"Corpus preparation:         "
        f"{corpus_time:.2f}s"
    )

    print(
        f"Total preparation time:     "
        f"{total_time:.2f}s"
    )

    print()

    print(
        "Manifest:"
    )

    print(
        f"  {manifest_path}"
    )

    print()

    print(
        "Benchmark subsets:"
    )

    for size in subset_sizes:

        print(
            f"  msmarco_{size // 1000}k "
            f"({size:,} documents)"
        )

    print()

    print(
        "Benchmark invariants:"
    )

    print(
        "  Fixed evaluation queries"
    )

    print(
        "  All positive-qrel documents retained"
    )

    print(
        "  Unique document IDs"
    )

    print(
        "  Deterministic distractor sampling"
    )

    print(
        "  Nested corpus subsets"
    )

    print("=" * 68)


if __name__ == "__main__":
    main()
