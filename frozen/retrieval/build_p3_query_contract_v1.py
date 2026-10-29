#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(
    "/root/autodl-tmp/work2-algorithm-innovation"
)

P3DIR = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2"
)

MANIFEST = (
    P3DIR
    / "jters_p3_manifest_v1.npz"
)

PARQUET = Path(
    "/root/autodl-tmp/datasets/CodeSearchNet/"
    "code_search_net/python/train-00000-of-00001.parquet"
)

CONSUMED = (
    ROOT
    / "frozen_artifacts/"
      "work2_repairability_frontier_v1/"
      "benchmark_train/"
      "python_train_retrievals_v1.npz"
)

OUT = (
    P3DIR
    / "retrieval_v1/"
      "jters_p3_queries_historical_contract_v1.jsonl"
)

EXPECTED_MANIFEST_SHA = (
    "07650ed49593985ce506a728597e83fd84f76c6ad78d82d4f731d122133db7d0"
)

EXPECTED_PROTOCOL_SHA = (
    "ce22ddf88da8a1c15727346d2679248d102eb11a65a54b641c99095394b95601"
)

EXPECTED_FREEZE_SHA = (
    "7ec59b91d1803688144b914cf1727054e1a85099255594acf32c0470bc5d77be"
)

EXPECTED_SEED = 5630306543764808130
TRAIN_ROWS = 412178
P3_ROWS = 6000


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            x = f.read(1 << 20)

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def normalize_tokens(x):

    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, (list, tuple)):

        return " ".join(
            str(t)
            for t in x
        )

    raise TypeError(
        f"Unexpected token container: {type(x)}"
    )


def main():

    print("=" * 150)
    print("P3 QUERY CONTRACT BUILD v1")
    print("=" * 150)

    # ------------------------------------------------------------
    # PRE-P3 invariant:
    # TRAIN source-row id == code-catalog GT id.
    # Verify only on already-consumed historical 12k.
    # ------------------------------------------------------------

    with np.load(
        CONSUMED,
        allow_pickle=False,
    ) as z:

        consumed_rows = np.asarray(
            z["source_row_indices"],
            dtype=np.int64,
        )

        consumed_gt = np.asarray(
            z["ground_truth_code_indices"],
            dtype=np.int64,
        )

    identity_gt = bool(
        np.array_equal(
            consumed_rows,
            consumed_gt,
        )
    )

    print(
        "consumed TRAIN row == GT code index =",
        identity_gt,
    )

    if not identity_gt:

        raise RuntimeError(
            "TRAIN-to-codebase identity mapping failed. "
            "Abort before opening P3 identities."
        )

    # ------------------------------------------------------------
    # P3 IS NOW OPENED.
    # ------------------------------------------------------------

    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:

        raise RuntimeError(
            "P3 manifest SHA mismatch."
        )

    with np.load(
        MANIFEST,
        allow_pickle=False,
    ) as z:

        rows = np.asarray(
            z["source_row_indices"],
            dtype=np.int64,
        )

        shards = np.asarray(
            z["shard_ids"],
            dtype=np.int16,
        )

        seed = int(
            np.asarray(z["seed"]).item()
        )

        train_row_count = int(
            np.asarray(
                z["train_row_count"]
            ).item()
        )

        protocol_sha = str(
            np.asarray(
                z["protocol_sha256"]
            ).item()
        )

        freeze_sha = str(
            np.asarray(
                z["frozen_v2_manifest_sha256"]
            ).item()
        )

    assert rows.shape == (P3_ROWS,)
    assert shards.shape == (P3_ROWS,)
    assert len(np.unique(rows)) == P3_ROWS
    assert rows.min() >= 0
    assert rows.max() < TRAIN_ROWS
    assert seed == EXPECTED_SEED
    assert train_row_count == TRAIN_ROWS
    assert protocol_sha == EXPECTED_PROTOCOL_SHA
    assert freeze_sha == EXPECTED_FREEZE_SHA

    unique_shards, shard_counts = np.unique(
        shards,
        return_counts=True,
    )

    expected_shards = np.arange(
        5,
        dtype=np.int16,
    )

    assert np.array_equal(
        unique_shards,
        expected_shards,
    )

    assert np.array_equal(
        shard_counts,
        np.full(
            5,
            1200,
            dtype=shard_counts.dtype,
        ),
    )

    print(
        "P3 identities =",
        len(rows),
    )

    print(
        "P3 unique identities =",
        len(np.unique(rows)),
    )

    print(
        "shard counts =",
        dict(
            zip(
                unique_shards.tolist(),
                shard_counts.tolist(),
            )
        ),
    )

    # ------------------------------------------------------------
    # Exact query extraction contract previously verified 12k/12k.
    # ------------------------------------------------------------

    df = pd.read_parquet(
        PARQUET,
        columns=[
            "func_documentation_tokens",
        ],
    )

    assert len(df) == TRAIN_ROWS

    tmp = OUT.with_suffix(
        ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        for pos in range(P3_ROWS):

            row_id = int(
                rows[pos]
            )

            raw = normalize_tokens(
                df.iloc[
                    row_id
                ][
                    "func_documentation_tokens"
                ]
            )

            rec = {
                "p3_pos":
                    pos,

                "source_row_index":
                    row_id,

                "ground_truth_code_index":
                    row_id,

                "shard_id":
                    int(
                        shards[pos]
                    ),

                "raw_query":
                    raw,
            }

            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

        f.flush()
        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        OUT,
    )

    out_sha = sha256(
        OUT
    )

    print()
    print(
        "P3 query contract =",
        OUT,
    )

    print(
        "P3 query contract SHA256 =",
        out_sha,
    )

    print(
        "P3_QUERY_TEXT_ACCESSED = True"
    )

    print(
        "P3_QUERY_EMBEDDINGS_GENERATED = False"
    )

    print(
        "P3_RETRIEVAL_GENERATED = False"
    )

    print(
        "P3_ALGORITHM_OUTCOMES_OBSERVED = False"
    )

    # Avoid a known pyarrow/pandas interpreter-shutdown issue
    # observed in the preceding consumed-only audit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
