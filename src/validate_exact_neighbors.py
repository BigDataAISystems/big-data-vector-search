"""
V1 - Validate Exact Nearest-Neighbor Ground Truth

Big Data Vector Search
======================

Independently validates exact-neighbor artifacts produced by:

    src/evaluate_exact_search.py

The exact-neighbor files become ground truth for future ANN
experiments.

This validator checks both:

    1. benchmark provenance using SHA-256 fingerprints
    2. structural and semantic integrity of exact neighbors

If the corpus, query set, qrels, or benchmark manifest changes after
the exact experiment was produced, validation fails before the
neighbor artifact is accepted.

Exit status:

    0 -> validation passed
    1 -> validation failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BENCHMARK_DIR = (
    ROOT
    / "data"
    / "processed"
)

DEFAULT_RESULTS_DIR = (
    ROOT
    / "results"
)

DEFAULT_QUERY_COUNT = 300
DEFAULT_K = 10


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Validate exact nearest-neighbor "
            "ground truth."
        )
    )

    parser.add_argument(
        "--subset",
        required=True,
        help="Example: msmarco_1k",
    )

    parser.add_argument(
        "--neighbors",
        default=None,
        help=(
            "Optional exact-neighbor JSONL path. "
            "Inferred from subset when omitted."
        ),
    )

    parser.add_argument(
        "--experiment",
        default=None,
        help=(
            "Optional exact-search experiment JSON. "
            "Inferred from subset when omitted."
        ),
    )

    parser.add_argument(
        "--benchmark-dir",
        default=str(
            DEFAULT_BENCHMARK_DIR
        ),
    )

    parser.add_argument(
        "--expected-queries",
        type=int,
        default=DEFAULT_QUERY_COUNT,
    )

    parser.add_argument(
        "--expected-k",
        type=int,
        default=DEFAULT_K,
    )

    return parser.parse_args()


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

    with path.open("rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# Helpers
# ============================================================

def subset_label(
    subset_name: str,
):

    prefix = "msmarco_"

    if subset_name.startswith(prefix):
        return subset_name[len(prefix):]

    return subset_name


def add_error(
    errors,
    message,
):

    errors.append(message)


# ============================================================
# Benchmark reference
# ============================================================

def load_benchmark_reference(
    benchmark_dir: Path,
    subset: str,
):

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
        / subset
        / "corpus.jsonl"
    )

    manifest_path = (
        benchmark_dir
        / "benchmark_manifest.json"
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

    query_ids = {
        str(row["_id"])
        for row in queries
    }

    document_ids = {
        str(row["_id"])
        for row in corpus
    }

    paths = {
        "queries": queries_path,
        "qrels": qrels_path,
        "corpus": corpus_path,
        "manifest": manifest_path,
    }

    return (
        queries,
        qrels,
        corpus,
        query_ids,
        document_ids,
        paths,
    )


# ============================================================
# Provenance validation
# ============================================================

def validate_provenance(
    experiment_path: Path,
    paths,
):

    errors = []

    experiment = read_json(
        experiment_path
    )

    provenance = experiment.get(
        "provenance"
    )

    if not isinstance(
        provenance,
        dict,
    ):

        return (
            [
                "Experiment result does not contain "
                "benchmark provenance metadata."
            ],
            {},
        )

    algorithm = provenance.get(
        "hash_algorithm"
    )

    if algorithm != "sha256":

        add_error(
            errors,
            (
                "Unsupported or missing provenance "
                f"hash algorithm: {algorithm!r}."
            ),
        )

    current = {

        "corpus_sha256":
            sha256_file(
                paths["corpus"]
            ),

        "queries_sha256":
            sha256_file(
                paths["queries"]
            ),

        "qrels_sha256":
            sha256_file(
                paths["qrels"]
            ),

        "manifest_sha256":
            sha256_file(
                paths["manifest"]
            ),
    }

    expected = {
        key: provenance.get(key)
        for key
        in current.keys()
    }

    matches = {}

    for key in current.keys():

        expected_hash = (
            expected[key]
        )

        current_hash = (
            current[key]
        )

        match = (
            expected_hash
            == current_hash
        )

        matches[key] = match

        if expected_hash is None:

            add_error(
                errors,
                (
                    "Experiment provenance is "
                    f"missing {key}."
                ),
            )

        elif not match:

            add_error(
                errors,
                (
                    "Benchmark provenance mismatch: "
                    f"{key}. "
                    f"Experiment={expected_hash}, "
                    f"current={current_hash}."
                ),
            )

    diagnostics = {
        "expected": expected,
        "current": current,
        "matches": matches,
    }

    return (
        errors,
        diagnostics,
    )


# ============================================================
# Qrels reference
# ============================================================

def expected_relevant_documents(
    qrels,
    query_id,
):

    if query_id not in qrels:
        return set()

    return {
        str(document_id)
        for document_id, score
        in qrels[
            query_id
        ].items()
        if float(score) > 0
    }


# ============================================================
# Neighbor validation
# ============================================================

def validate_neighbors(
    neighbor_rows,
    benchmark_query_ids,
    benchmark_document_ids,
    qrels,
    expected_queries,
    expected_k,
):

    errors = []

    if (
        len(neighbor_rows)
        != expected_queries
    ):

        add_error(
            errors,
            (
                "Query-record count mismatch: "
                f"expected={expected_queries:,}, "
                f"actual={len(neighbor_rows):,}."
            ),
        )

    observed_query_ids = []

    bad_neighbor_counts = 0
    bad_rank_sequences = 0
    duplicate_neighbor_queries = 0
    unknown_neighbor_queries = 0
    nonfinite_score_queries = 0
    unordered_score_queries = 0
    relevant_metadata_mismatches = 0

    for row_number, row in enumerate(
        neighbor_rows,
        start=1,
    ):

        # ----------------------------------------------------
        # Query
        # ----------------------------------------------------

        if "query_id" not in row:

            add_error(
                errors,
                (
                    f"Row {row_number} is "
                    "missing query_id."
                ),
            )

            continue

        query_id = str(
            row["query_id"]
        )

        observed_query_ids.append(
            query_id
        )

        if (
            query_id
            not in benchmark_query_ids
        ):

            add_error(
                errors,
                (
                    "Unknown query ID: "
                    f"{query_id}."
                ),
            )

        # ----------------------------------------------------
        # Relevant metadata
        # ----------------------------------------------------

        expected_relevant = (
            expected_relevant_documents(
                qrels,
                query_id,
            )
        )

        stored_relevant_raw = row.get(
            "relevant_document_ids"
        )

        if not isinstance(
            stored_relevant_raw,
            list,
        ):

            relevant_metadata_mismatches += 1

            add_error(
                errors,
                (
                    f"Query {query_id}: "
                    "relevant_document_ids "
                    "must be a list."
                ),
            )

        else:

            stored_relevant = {
                str(document_id)
                for document_id
                in stored_relevant_raw
            }

            if (
                stored_relevant
                != expected_relevant
            ):

                relevant_metadata_mismatches += 1

                add_error(
                    errors,
                    (
                        f"Query {query_id}: "
                        "relevant-document metadata "
                        "does not match qrels."
                    ),
                )

        # ----------------------------------------------------
        # Neighbor list
        # ----------------------------------------------------

        neighbors = row.get(
            "neighbors"
        )

        if not isinstance(
            neighbors,
            list,
        ):

            add_error(
                errors,
                (
                    f"Query {query_id}: "
                    "neighbors must be a list."
                ),
            )

            continue

        if len(neighbors) != expected_k:

            bad_neighbor_counts += 1

            add_error(
                errors,
                (
                    f"Query {query_id}: expected "
                    f"{expected_k} neighbors, "
                    f"found {len(neighbors)}."
                ),
            )

        ranks = []
        neighbor_ids = []
        scores = []

        malformed = False

        for position, neighbor in enumerate(
            neighbors,
            start=1,
        ):

            if not isinstance(
                neighbor,
                dict,
            ):

                malformed = True

                add_error(
                    errors,
                    (
                        f"Query {query_id}: "
                        f"neighbor {position} "
                        "is not an object."
                    ),
                )

                continue

            # Rank
            try:

                rank = int(
                    neighbor.get(
                        "rank"
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                malformed = True

                add_error(
                    errors,
                    (
                        f"Query {query_id}: "
                        f"invalid rank at "
                        f"neighbor {position}."
                    ),
                )

                continue

            ranks.append(rank)

            # Document ID
            if "doc_id" not in neighbor:

                malformed = True

                add_error(
                    errors,
                    (
                        f"Query {query_id}: "
                        f"rank {rank} missing "
                        "doc_id."
                    ),
                )

                continue

            document_id = str(
                neighbor["doc_id"]
            )

            neighbor_ids.append(
                document_id
            )

            # Score
            try:

                score = float(
                    neighbor.get(
                        "score"
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                malformed = True

                add_error(
                    errors,
                    (
                        f"Query {query_id}: "
                        f"invalid score at "
                        f"rank {rank}."
                    ),
                )

                continue

            scores.append(score)

        # ----------------------------------------------------
        # Rank sequence
        # ----------------------------------------------------

        expected_ranks = list(
            range(
                1,
                expected_k + 1,
            )
        )

        if (
            not malformed
            and ranks != expected_ranks
        ):

            bad_rank_sequences += 1

            add_error(
                errors,
                (
                    f"Query {query_id}: "
                    f"rank sequence={ranks}, "
                    f"expected={expected_ranks}."
                ),
            )

        # ----------------------------------------------------
        # Duplicate neighbors
        # ----------------------------------------------------

        if (
            len(neighbor_ids)
            != len(set(neighbor_ids))
        ):

            duplicate_neighbor_queries += 1

            add_error(
                errors,
                (
                    f"Query {query_id}: duplicate "
                    "neighbor IDs detected."
                ),
            )

        # ----------------------------------------------------
        # Corpus membership
        # ----------------------------------------------------

        unknown_ids = (
            set(neighbor_ids)
            - benchmark_document_ids
        )

        if unknown_ids:

            unknown_neighbor_queries += 1

            preview = sorted(
                unknown_ids
            )[:5]

            add_error(
                errors,
                (
                    f"Query {query_id}: "
                    f"{len(unknown_ids)} neighbor "
                    "IDs are not in the corpus. "
                    f"Examples={preview}"
                ),
            )

        # ----------------------------------------------------
        # Finite scores
        # ----------------------------------------------------

        if any(
            not math.isfinite(score)
            for score in scores
        ):

            nonfinite_score_queries += 1

            add_error(
                errors,
                (
                    f"Query {query_id}: "
                    "non-finite score detected."
                ),
            )

        # ----------------------------------------------------
        # Score ordering
        # ----------------------------------------------------

        tolerance = 1e-7

        ordered = all(
            scores[index]
            + tolerance
            >= scores[index + 1]
            for index
            in range(
                len(scores) - 1
            )
        )

        if scores and not ordered:

            unordered_score_queries += 1

            add_error(
                errors,
                (
                    f"Query {query_id}: "
                    "scores are not monotonically "
                    "non-increasing."
                ),
            )

    # ========================================================
    # Artifact query-set validation
    # ========================================================

    unique_observed = set(
        observed_query_ids
    )

    if (
        len(unique_observed)
        != len(observed_query_ids)
    ):

        duplicate_count = (
            len(observed_query_ids)
            - len(unique_observed)
        )

        add_error(
            errors,
            (
                "Duplicate query records: "
                f"{duplicate_count:,}."
            ),
        )

    missing_queries = (
        benchmark_query_ids
        - unique_observed
    )

    extra_queries = (
        unique_observed
        - benchmark_query_ids
    )

    if missing_queries:

        add_error(
            errors,
            (
                "Missing benchmark queries: "
                f"{len(missing_queries):,}."
            ),
        )

    if extra_queries:

        add_error(
            errors,
            (
                "Extra query records: "
                f"{len(extra_queries):,}."
            ),
        )

    diagnostics = {

        "query_records":
            len(neighbor_rows),

        "unique_query_ids":
            len(unique_observed),

        "bad_neighbor_counts":
            bad_neighbor_counts,

        "bad_rank_sequences":
            bad_rank_sequences,

        "duplicate_neighbor_queries":
            duplicate_neighbor_queries,

        "unknown_neighbor_queries":
            unknown_neighbor_queries,

        "nonfinite_score_queries":
            nonfinite_score_queries,

        "unordered_score_queries":
            unordered_score_queries,

        "relevant_metadata_mismatches":
            relevant_metadata_mismatches,

        "missing_queries":
            len(missing_queries),

        "extra_queries":
            len(extra_queries),
    }

    return (
        errors,
        diagnostics,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.expected_queries < 1:
        raise ValueError(
            "--expected-queries must be >= 1"
        )

    if args.expected_k < 1:
        raise ValueError(
            "--expected-k must be >= 1"
        )

    benchmark_dir = Path(
        args.benchmark_dir
    )

    if not benchmark_dir.exists():
        raise FileNotFoundError(
            "Benchmark directory not found:\n"
            f"{benchmark_dir.resolve()}"
        )

    label = subset_label(
        args.subset
    )

    if args.neighbors:

        neighbors_path = Path(
            args.neighbors
        )

    else:

        neighbors_path = (
            DEFAULT_RESULTS_DIR
            / (
                "v1_exact_neighbors_"
                f"{label}.jsonl"
            )
        )

    if args.experiment:

        experiment_path = Path(
            args.experiment
        )

    else:

        experiment_path = (
            DEFAULT_RESULTS_DIR
            / f"v1_exact_{label}.json"
        )

    print()
    print(
        "V1 - Exact Neighbor Ground-Truth Validation"
    )

    print(
        "==========================================="
    )

    print()

    print(
        f"Subset:            {args.subset}"
    )

    print(
        f"Benchmark:         "
        f"{benchmark_dir.resolve()}"
    )

    print(
        f"Neighbor artifact: "
        f"{neighbors_path.resolve()}"
    )

    print(
        f"Experiment record: "
        f"{experiment_path.resolve()}"
    )

    print(
        f"Expected queries:  "
        f"{args.expected_queries:,}"
    )

    print(
        f"Expected Top-K:    "
        f"{args.expected_k}"
    )

    start = time.perf_counter()

    # ========================================================
    # Benchmark
    # ========================================================

    print()
    print(
        "Loading benchmark reference..."
    )

    (
        queries,
        qrels,
        corpus,
        benchmark_query_ids,
        benchmark_document_ids,
        paths,
    ) = load_benchmark_reference(
        benchmark_dir,
        args.subset,
    )

    print(
        f"Benchmark queries:   "
        f"{len(queries):,}"
    )

    print(
        f"Benchmark documents: "
        f"{len(corpus):,}"
    )

    # ========================================================
    # Provenance
    # ========================================================

    print()
    print(
        "Validating benchmark provenance..."
    )

    (
        provenance_errors,
        provenance_diagnostics,
    ) = validate_provenance(
        experiment_path,
        paths,
    )

    matches = (
        provenance_diagnostics.get(
            "matches",
            {},
        )
    )

    for key in (
        "corpus_sha256",
        "queries_sha256",
        "qrels_sha256",
        "manifest_sha256",
    ):

        match = matches.get(
            key,
            False,
        )

        print(
            f"  {key:<18} "
            f"{'MATCH' if match else 'MISMATCH'}"
        )

    if provenance_errors:

        print()
        print("=" * 70)

        print(
            "PROVENANCE VALIDATION FAILED"
        )

        print("=" * 70)

        print()

        for index, message in enumerate(
            provenance_errors,
            start=1,
        ):

            print(
                f"  {index}. {message}"
            )

        print()

        print(
            "The exact-search result does not belong "
            "to the current benchmark artifacts."
        )

        print(
            "Regenerate evaluate_exact_search.py "
            "before using these neighbors as ANN "
            "ground truth."
        )

        print("=" * 70)

        sys.exit(1)

    # ========================================================
    # Neighbor artifact
    # ========================================================

    print()
    print(
        "Loading exact-neighbor artifact..."
    )

    neighbor_rows = read_jsonl(
        neighbors_path
    )

    print(
        f"Neighbor records:    "
        f"{len(neighbor_rows):,}"
    )

    print()
    print(
        "Validating exact neighbors..."
    )

    (
        errors,
        diagnostics,
    ) = validate_neighbors(
        neighbor_rows,
        benchmark_query_ids,
        benchmark_document_ids,
        qrels,
        args.expected_queries,
        args.expected_k,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print("=" * 70)

    print(
        "V1 Exact Neighbor Validation Result"
    )

    print("=" * 70)

    print(
        f"Query records:                "
        f"{diagnostics['query_records']:,}"
    )

    print(
        f"Unique query IDs:             "
        f"{diagnostics['unique_query_ids']:,}"
    )

    print(
        f"Expected neighbors/query:     "
        f"{args.expected_k}"
    )

    print(
        f"Bad neighbor counts:          "
        f"{diagnostics['bad_neighbor_counts']:,}"
    )

    print(
        f"Bad rank sequences:           "
        f"{diagnostics['bad_rank_sequences']:,}"
    )

    print(
        f"Duplicate-neighbor queries:   "
        f"{diagnostics['duplicate_neighbor_queries']:,}"
    )

    print(
        f"Unknown-neighbor queries:     "
        f"{diagnostics['unknown_neighbor_queries']:,}"
    )

    print(
        f"Non-finite-score queries:     "
        f"{diagnostics['nonfinite_score_queries']:,}"
    )

    print(
        f"Unordered-score queries:      "
        f"{diagnostics['unordered_score_queries']:,}"
    )

    print(
        f"Relevant metadata mismatches: "
        f"{diagnostics['relevant_metadata_mismatches']:,}"
    )

    print(
        f"Missing benchmark queries:    "
        f"{diagnostics['missing_queries']:,}"
    )

    print(
        f"Extra query records:          "
        f"{diagnostics['extra_queries']:,}"
    )

    print(
        f"Validation time:              "
        f"{elapsed:.4f}s"
    )

    print()

    if errors:

        print(
            "VALIDATION FAILED"
        )

        print()

        print(
            f"Errors ({len(errors)}):"
        )

        max_display = 25

        for index, message in enumerate(
            errors[:max_display],
            start=1,
        ):

            print(
                f"  {index}. {message}"
            )

        if len(errors) > max_display:

            print(
                "  ... "
                f"{len(errors) - max_display:,} "
                "additional errors omitted."
            )

        print("=" * 70)

        sys.exit(1)

    print(
        "VALIDATION PASSED"
    )

    print()

    print(
        "Verified provenance:"
    )

    print(
        "  Corpus SHA-256 matches"
    )

    print(
        "  Query-set SHA-256 matches"
    )

    print(
        "  Qrels SHA-256 matches"
    )

    print(
        "  Benchmark-manifest SHA-256 matches"
    )

    print()

    print(
        "Verified exact-neighbor invariants:"
    )

    print(
        "  Complete benchmark query set"
    )

    print(
        "  Unique query records"
    )

    print(
        f"  Exactly {args.expected_k} neighbors/query"
    )

    print(
        "  Exact rank sequence"
    )

    print(
        "  Unique neighbors within each query"
    )

    print(
        "  Every neighbor belongs to the corpus"
    )

    print(
        "  All similarity scores are finite"
    )

    print(
        "  Similarity scores are non-increasing"
    )

    print(
        "  Relevant-document metadata agrees with qrels"
    )

    print()

    print(
        "This artifact is valid exact-search "
        "ground truth for future ANN evaluation."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()