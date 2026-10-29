#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "/root/autodl-tmp/work2-algorithm-innovation"
)

PARQUET = Path(
    "/root/autodl-tmp/datasets/CodeSearchNet/"
    "code_search_net/python/"
    "train-00000-of-00001.parquet"
)

OLD_SANDBOX = (
    ROOT
    / "fhex_v1/sandbox/"
      "python_trainonly_qwen3_top500_v1.npz"
)

RCPS = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a4_rcps.npz"
)

OUT_NPZ = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_manifest_v1.npz"
)

OUT_JSON = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_manifest_v1.json"
)


N = 12000

SEED_STRING = (
    "work2-rcps-prospective-p1-v1"
)

FROZEN_RCPS_SHA256 = (
    "fb61f7a1b30c69698b9ee7365cb92cd39c4931b9849be5d25c4e4a314943beb5"
)


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def row_hash(row):

    s = (
        SEED_STRING
        +
        ":"
        +
        str(
            int(row)
        )
    )

    return hashlib.sha256(
        s.encode(
            "utf-8"
        )
    ).digest()


def main():

    OUT_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Verify frozen RCPS artifact BEFORE choosing P1.
    # --------------------------------------------------------

    actual_rcps_sha = sha256_file(
        RCPS
    )

    if (
        actual_rcps_sha
        !=
        FROZEN_RCPS_SHA256
    ):

        raise AssertionError(
            (
                "Frozen RCPS artifact SHA mismatch.",
                actual_rcps_sha,
                FROZEN_RCPS_SHA256,
            )
        )

    # --------------------------------------------------------
    # Existing 12k queries are forbidden.
    # --------------------------------------------------------

    old = np.load(
        OLD_SANDBOX,
        allow_pickle=True,
    )

    old_rows = old[
        "source_row_indices"
    ].astype(
        np.int64
    )

    old_set = set(
        old_rows.tolist()
    )

    assert len(old_set) == 12000

    # --------------------------------------------------------
    # Load TRAIN metadata only.
    # No embeddings / ranks / AST / retrieval outcomes.
    # --------------------------------------------------------

    df = pd.read_parquet(
        PARQUET,
        columns=[
            "func_documentation_string",
        ],
    )

    train_n = len(df)

    assert train_n == 412178

    eligible = []

    for row in range(
        train_n
    ):

        if row in old_set:
            continue

        text = df.iloc[
            row
        ][
            "func_documentation_string"
        ]

        if text is None:
            continue

        if not str(
            text
        ).strip():
            continue

        eligible.append(
            row
        )

    eligible = np.asarray(
        eligible,
        dtype=np.int64,
    )

    if len(
        eligible
    ) < N:

        raise AssertionError(
            "Not enough eligible unseen TRAIN queries."
        )

    # --------------------------------------------------------
    # Deterministic outcome-blind sampling:
    # sort solely by SHA256(seed,row_id).
    # --------------------------------------------------------

    keyed = [
        (
            row_hash(
                row
            ),
            int(
                row
            ),
        )
        for row in eligible
    ]

    keyed.sort(
        key=lambda x: x[0]
    )

    selected = np.asarray(
        [
            row
            for _, row in keyed[
                :N
            ]
        ],
        dtype=np.int64,
    )

    assert len(
        selected
    ) == N

    assert len(
        np.unique(
            selected
        )
    ) == N

    assert len(
        set(
            selected.tolist()
        )
        &
        old_set
    ) == 0

    # In TRAIN-only retrieval, GT code row is its own
    # CodeSearchNet TRAIN row.
    gt = selected.copy()

    # --------------------------------------------------------
    # Deterministic five prospective shards.
    #
    # These are NOT fitted folds.
    # Used only for stability reporting after full freeze.
    # --------------------------------------------------------

    shard_ids = np.arange(
        N,
        dtype=np.int64,
    ) % 5

    # --------------------------------------------------------
    # Save immutable manifest.
    # --------------------------------------------------------

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "work2_rcps_prospective_p1_manifest_v1"
        ),

        source_row_indices=
            selected.astype(
                np.int64
            ),

        ground_truth_code_indices=
            gt.astype(
                np.int64
            ),

        shard_ids=
            shard_ids.astype(
                np.int8
            ),

        selection_seed=np.asarray(
            SEED_STRING
        ),

        frozen_rcps_sha256=np.asarray(
            FROZEN_RCPS_SHA256
        ),

        old_query_overlap=np.asarray(
            0,
            dtype=np.int32,
        ),

        retrieval_accessed=np.asarray(
            False
        ),

        outcome_accessed=np.asarray(
            False
        ),

        valid_accessed=np.asarray(
            False
        ),

        test_accessed=np.asarray(
            False
        ),
    )

    manifest_sha = sha256_file(
        OUT_NPZ
    )

    info = {
        "name":
            "RCPS Prospective Confirmation P1",

        "version":
            1,

        "query_count":
            N,

        "universe":
            "CodeSearchNet Python TRAIN only",

        "train_rows":
            train_n,

        "selection":
            (
                "Outcome-blind SHA256 ordering over TRAIN "
                "row IDs after excluding every source row "
                "from the previous 12k development set."
            ),

        "selection_seed":
            SEED_STRING,

        "old_query_overlap":
            0,

        "frozen_method":
            "RCPS-A0",

        "frozen_method_artifact":
            str(
                RCPS
            ),

        "frozen_method_sha256":
            FROZEN_RCPS_SHA256,

        "protocol": {
            "protected_k":
                100,

            "local_candidate_region":
                "ranks101..180",

            "local_candidate_k":
                80,

            "final_budget":
                20,

            "max_questions":
                2,

            "question_library":
                (
                    "44 atomic AST-presence predicates "
                    "+ all pairwise AND "
                    "+ all pairwise OR"
                ),

            "question_library_size":
                1936,

            "policy":
                "Frozen BCCT objective",

            "no_not":
                True,

            "no_triples":
                True,

            "no_feature_tuning":
                True,
        },

        "prospective_decision_rule": {
            "minimum_local80_recoverable_for_decision":
                60,

            "primary_metric":
                "R20 success rate among Local80-recoverable failures",

            "rcps_success_rate_gate":
                0.90,

            "paired_absolute_gain_over_atomic_bcct_gate":
                0.03,

            "gt_retention_gate":
                1.0,

            "shard_nonworse_gate":
                "at least 4/5",

            "shard_strict_better_gate":
                "at least 2/5",

            "interpretation_if_n_below_60":
                (
                    "INCONCLUSIVE, not negative; "
                    "do not alter RCPS."
                ),
        },

        "secondary_metrics": [
            "R1",
            "R2",
            "R5",
            "R10",
            "MRR",
            "questions per query",
            "unique predicate count",
        ],

        "important_secondary_note":
            (
                "RCPS is a fixed-budget repair algorithm. "
                "R1/R2/MRR are reported transparently but "
                "are not optimization targets or continuation gates."
            ),

        "manifest_sha256":
            manifest_sha,
    }

    OUT_JSON.write_text(
        json.dumps(
            info,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 120
    )

    print(
        "RCPS PROSPECTIVE-P1 MANIFEST FROZEN"
    )

    print(
        "=" * 120
    )

    print(
        "selected queries =",
        len(
            selected
        ),
    )

    print(
        "old-query overlap =",
        len(
            set(
                selected.tolist()
            )
            &
            old_set
        ),
    )

    print(
        "min row =",
        int(
            selected.min()
        ),
    )

    print(
        "max row =",
        int(
            selected.max()
        ),
    )

    print(
        "shards =",
        np.bincount(
            shard_ids
        ).tolist(),
    )

    print(
        "frozen RCPS SHA =",
        actual_rcps_sha,
    )

    print(
        "manifest =",
        OUT_NPZ,
    )

    print(
        "manifest SHA256 =",
        manifest_sha,
    )

    print(
        "metadata =",
        OUT_JSON,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No retrieval outcome has been accessed."
    )

    print(
        "Do not regenerate this manifest after the next stage."
    )


if __name__ == "__main__":
    main()
