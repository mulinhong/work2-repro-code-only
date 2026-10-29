#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch


ROOT = Path(
    "/root/autodl-tmp/work2-algorithm-innovation"
)

CONFIG = (
    ROOT
    / "fcqr_mvp_v2/configs/"
      "fcqr_python_qwen3_mvp_v2.json"
)

QWEN_FULL = (
    ROOT
    / "frozen_artifacts/"
      "work2_repairability_frontier_v1/"
      "comparable_retriever_probe_train/"
      "python_train_qwen3_top500_v1.npz"
)

CCE = (
    ROOT
    / "fhex_v1/results/"
      "cce_v2_slot1_oof_fit_5fold.npz"
)

AST_CACHE = (
    ROOT
    / "fhex_v1/cache/"
      "python_train_ast_signature_v0.npz"
)

OUT_NPZ = (
    ROOT
    / "fhex_v1/sandbox/"
      "python_trainonly_qwen3_top500_v1.npz"
)

OUT_JSON = (
    ROOT
    / "fhex_v1/sandbox/"
      "python_trainonly_qwen3_top500_v1_metrics.json"
)


TRAIN_CODE_ROWS = 412178

TOPK = 500

PROTECTED_K = 100

LOCAL80_START = 100
LOCAL80_STOP = 180

B = 20

QUERY_BATCH = 32

GPU_CODE_CHUNK = 8192


def resolve_path(value):

    p = Path(value)

    return (
        p
        if p.is_absolute()
        else ROOT / p
    )


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def normalize_rows(x):

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    return (
        x
        /
        np.maximum(
            np.linalg.norm(
                x,
                axis=-1,
                keepdims=True,
            ),
            1e-12,
        )
    )


def load_code_gpu(code_cpu):

    print()
    print(
        "Loading TRAIN-only code catalog to GPU..."
    )

    d = int(
        code_cpu.shape[
            1
        ]
    )

    code_gpu = torch.empty(
        (
            TRAIN_CODE_ROWS,
            d,
        ),
        dtype=torch.float32,
        device="cuda",
    )

    for start in range(
        0,
        TRAIN_CODE_ROWS,
        GPU_CODE_CHUNK,
    ):

        stop = min(
            start
            +
            GPU_CODE_CHUNK,
            TRAIN_CODE_ROWS,
        )

        x = np.asarray(
            code_cpu[
                start:stop
            ],
            dtype=np.float32,
        )

        code_gpu[
            start:stop
        ].copy_(
            torch.from_numpy(
                x.copy()
            ).cuda()
        )

    gib = (
        code_gpu.numel()
        *
        code_gpu.element_size()
        /
        1024**3
    )

    print(
        "TRAIN catalog shape =",
        tuple(
            code_gpu.shape
        ),
    )

    print(
        "GPU GiB =",
        gib,
    )

    return code_gpu


def metrics_for_mask(
    rank501,
    mask,
):

    r = rank501[
        mask
    ]

    return {
        "n":
            int(
                len(
                    r
                )
            ),

        "top1":
            int(
                np.sum(
                    r
                    <=
                    1
                )
            ),

        "top5":
            int(
                np.sum(
                    r
                    <=
                    5
                )
            ),

        "top10":
            int(
                np.sum(
                    r
                    <=
                    10
                )
            ),

        "top20":
            int(
                np.sum(
                    r
                    <=
                    20
                )
            ),

        "top100":
            int(
                np.sum(
                    r
                    <=
                    100
                )
            ),

        "failure_gt100":
            int(
                np.sum(
                    r
                    >
                    100
                )
            ),

        "local80_recoverable_101_180":
            int(
                np.sum(
                    (
                        r
                        >=
                        101
                    )
                    &
                    (
                        r
                        <=
                        180
                    )
                )
            ),

        "top500_recoverable_101_500":
            int(
                np.sum(
                    (
                        r
                        >=
                        101
                    )
                    &
                    (
                        r
                        <=
                        500
                    )
                )
            ),

        "beyond500":
            int(
                np.sum(
                    r
                    ==
                    501
                )
            ),
    }


def main():

    t0 = time.perf_counter()

    torch.backends.cuda.matmul.allow_tf32 = False

    print(
        "=" * 132
    )

    print(
        "TRAIN-ONLY FULLY OBSERVABLE REPAIR SANDBOX"
    )

    print(
        "=" * 132
    )

    print(
        "Development infrastructure only."
    )

    print(
        "Raw code universe = CodeSearchNet Python TRAIN only."
    )

    print(
        "CAL / HOLDOUT / TEST are NOT accessed."
    )

    print(
        "Catalog size =",
        TRAIN_CODE_ROWS,
    )

    print(
        "Base failure = Qwen rank > 100."
    )

    print(
        "Final repair budget remains B =",
        B,
    )

    # ========================================================
    # Load frozen embeddings.
    # ========================================================

    cfg = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    query_path = resolve_path(
        cfg[
            "train_query_embeddings"
        ]
    )

    code_path = resolve_path(
        cfg[
            "code_embeddings"
        ]
    )

    query_all = np.load(
        query_path,
        mmap_mode="r",
    )

    code_all = np.load(
        code_path,
        mmap_mode="r",
    )

    assert query_all.shape[
        0
    ] == 12000

    assert code_all.shape[
        0
    ] >= TRAIN_CODE_ROWS

    with np.load(
        QWEN_FULL,
        allow_pickle=True,
    ) as z:

        source_rows = z[
            "source_row_indices"
        ].astype(
            np.int64
        )

        gt = z[
            "ground_truth_code_indices"
        ].astype(
            np.int64
        )

        role_codes = z[
            "role_codes"
        ].astype(
            np.int8
        )

        old_full_top500 = z[
            "qwen3_indices"
        ].astype(
            np.int32
        )

        assert not bool(
            np.asarray(
                z[
                    "valid_accessed"
                ]
            ).item()
        )

        assert not bool(
            np.asarray(
                z[
                    "test_accessed"
                ]
            ).item()
        )

    assert len(
        source_rows
    ) == 12000

    assert len(
        gt
    ) == 12000

    # Every sandbox query/GT must belong to raw TRAIN.
    if np.any(
        source_rows
        >=
        TRAIN_CODE_ROWS
    ):

        raise AssertionError(
            "TRAIN sandbox query source outside raw TRAIN."
        )

    if np.any(
        gt
        >=
        TRAIN_CODE_ROWS
    ):

        raise AssertionError(
            "TRAIN sandbox GT outside raw TRAIN."
        )

    # ========================================================
    # Recover frozen FIT fold IDs.
    # ========================================================

    cce = np.load(
        CCE,
        allow_pickle=True,
    )

    cce_rows = cce[
        "manifest_rows"
    ].astype(
        np.int64
    )

    cce_folds = cce[
        "fold_ids"
    ].astype(
        np.int8
    )

    fold_ids = np.full(
        12000,
        -1,
        dtype=np.int8,
    )

    fold_ids[
        cce_rows
    ] = cce_folds

    fit_mask = (
        role_codes
        ==
        0
    )

    if int(
        fit_mask.sum()
    ) != 7200:

        raise AssertionError(
            (
                "Unexpected FIT count",
                int(
                    fit_mask.sum()
                ),
            )
        )

    if np.any(
        fold_ids[
            fit_mask
        ]
        <
        0
    ):

        raise AssertionError(
            "Missing fold ID for FIT row."
        )

    # ========================================================
    # Exact Qwen TRAIN-only Top500.
    # ========================================================

    q = normalize_rows(
        np.asarray(
            query_all,
            dtype=np.float32,
        )
    )

    code_gpu = load_code_gpu(
        code_all
    )

    top_indices = np.empty(
        (
            12000,
            TOPK,
        ),
        dtype=np.int32,
    )

    top_scores = np.empty(
        (
            12000,
            TOPK,
        ),
        dtype=np.float32,
    )

    print()
    print(
        "=" * 132
    )

    print(
        "EXACT TRAIN-ONLY QWEN TOP500"
    )

    print(
        "=" * 132
    )

    search_start = time.perf_counter()

    for start in range(
        0,
        12000,
        QUERY_BATCH,
    ):

        stop = min(
            start
            +
            QUERY_BATCH,
            12000,
        )

        qgpu = torch.from_numpy(
            q[
                start:stop
            ].copy()
        ).cuda()

        scores = (
            qgpu
            @
            code_gpu.T
        )

        values, indices = torch.topk(
            scores,
            k=TOPK,
            dim=1,
            largest=True,
            sorted=True,
        )

        top_indices[
            start:stop
        ] = (
            indices
            .cpu()
            .numpy()
            .astype(
                np.int32
            )
        )

        top_scores[
            start:stop
        ] = (
            values
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        del (
            qgpu,
            scores,
            values,
            indices,
        )

        if (
            stop % 1024 == 0
            or
            stop
            ==
            12000
        ):

            elapsed = (
                time.perf_counter()
                -
                search_start
            )

            print(
                f"[QWEN] {stop}/12000",
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )

    # ========================================================
    # Exact rank within top500; >500 is censored to 501.
    # ========================================================

    rank501 = np.full(
        12000,
        501,
        dtype=np.int16,
    )

    for i in range(
        12000
    ):

        p = np.flatnonzero(
            top_indices[
                i
            ]
            ==
            gt[
                i
            ]
        )

        if len(
            p
        ):

            rank501[
                i
            ] = int(
                p[
                    0
                ]
                +
                1
            )

    failure_mask = (
        rank501
        >
        PROTECTED_K
    )

    local80_recoverable = (
        (
            rank501
            >=
            LOCAL80_START
            +
            1
        )
        &
        (
            rank501
            <=
            LOCAL80_STOP
        )
    )

    top500_recoverable = (
        (
            rank501
            >=
            101
        )
        &
        (
            rank501
            <=
            500
        )
    )

    # ========================================================
    # AST accessibility audit.
    # ========================================================

    ast_valid = None

    if AST_CACHE.exists():

        az = np.load(
            AST_CACHE,
            allow_pickle=True,
        )

        ast_valid = az[
            "valid"
        ].astype(
            bool
        )

        assert len(
            ast_valid
        ) == TRAIN_CODE_ROWS

    # ========================================================
    # Report.
    # ========================================================

    all_metrics = metrics_for_mask(
        rank501,
        np.ones(
            12000,
            dtype=bool,
        ),
    )

    fit_metrics = metrics_for_mask(
        rank501,
        fit_mask,
    )

    print()
    print(
        "=" * 132
    )

    print(
        "TRAIN-ONLY SANDBOX SUMMARY"
    )

    print(
        "=" * 132
    )

    print(
        "ALL 12000:"
    )

    print(
        json.dumps(
            all_metrics,
            indent=2,
        )
    )

    print()
    print(
        "FIT 7200:"
    )

    print(
        json.dumps(
            fit_metrics,
            indent=2,
        )
    )

    print()
    print(
        "FIT failure count =",
        int(
            np.sum(
                fit_mask
                &
                failure_mask
            )
        ),
    )

    print(
        "FIT Local80 recoverable =",
        int(
            np.sum(
                fit_mask
                &
                local80_recoverable
            )
        ),
    )

    print(
        "FIT Top500 recoverable =",
        int(
            np.sum(
                fit_mask
                &
                top500_recoverable
            )
        ),
    )

    if ast_valid is not None:

        local80_ids = top_indices[
            :,
            LOCAL80_START:LOCAL80_STOP
        ]

        fit_failure_rows = np.flatnonzero(
            fit_mask
            &
            failure_mask
        )

        candidate_ast_valid_fraction = float(
            ast_valid[
                local80_ids[
                    fit_failure_rows
                ]
            ].mean()
        )

        gt_ast_valid_fraction = float(
            ast_valid[
                gt[
                    fit_failure_rows
                ]
            ].mean()
        )

        print()
        print(
            "FIT failure Local80 candidate AST-valid fraction =",
            candidate_ast_valid_fraction,
        )

        print(
            "FIT failure GT AST-valid fraction =",
            gt_ast_valid_fraction,
        )

    print()
    print(
        "=" * 132
    )

    print(
        "FIT PER-FOLD FAILURE FRONTIER"
    )

    print(
        "=" * 132
    )

    fold_summary = {}

    for fold in range(
        5
    ):

        fm = (
            fit_mask
            &
            (
                fold_ids
                ==
                fold
            )
        )

        f_fail = (
            fm
            &
            failure_mask
        )

        f_local = (
            fm
            &
            local80_recoverable
        )

        f_500 = (
            fm
            &
            top500_recoverable
        )

        fold_summary[
            str(
                fold
            )
        ] = {
            "n":
                int(
                    fm.sum()
                ),

            "base_failures":
                int(
                    f_fail.sum()
                ),

            "local80_recoverable":
                int(
                    f_local.sum()
                ),

            "top500_recoverable":
                int(
                    f_500.sum()
                ),
        }

        print(
            f"fold={fold}",
            f"n={int(fm.sum())}",
            f"fail={int(f_fail.sum())}",
            f"local80={int(f_local.sum())}",
            f"top500={int(f_500.sum())}",
        )

    # ========================================================
    # Difference from old full catalog -- diagnostic only.
    # ========================================================

    old_rank501 = np.full(
        12000,
        501,
        dtype=np.int16,
    )

    for i in range(
        12000
    ):

        p = np.flatnonzero(
            old_full_top500[
                i
            ]
            ==
            gt[
                i
            ]
        )

        if len(
            p
        ):

            old_rank501[
                i
            ] = int(
                p[
                    0
                ]
                +
                1
            )

    both_fit = fit_mask

    train_improved = int(
        np.sum(
            both_fit
            &
            (
                rank501
                <
                old_rank501
            )
        )
    )

    same_rank = int(
        np.sum(
            both_fit
            &
            (
                rank501
                ==
                old_rank501
            )
        )
    )

    train_worse = int(
        np.sum(
            both_fit
            &
            (
                rank501
                >
                old_rank501
            )
        )
    )

    print()
    print(
        "FIT rank relation vs original full catalog:"
    )

    print(
        "TRAIN-only improved =",
        train_improved,
    )

    print(
        "same =",
        same_rank,
    )

    print(
        "worse =",
        train_worse,
    )

    # ========================================================
    # Save.
    # ========================================================

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "work2_trainonly_qwen3_sandbox_v1"
        ),

        catalog_split=np.asarray(
            "CodeSearchNet Python TRAIN only"
        ),

        catalog_size=np.asarray(
            TRAIN_CODE_ROWS,
            dtype=np.int32,
        ),

        source_row_indices=
            source_rows.astype(
                np.int64
            ),

        ground_truth_code_indices=
            gt.astype(
                np.int32
            ),

        role_codes=
            role_codes.astype(
                np.int8
            ),

        fold_ids=
            fold_ids.astype(
                np.int8
            ),

        qwen3_indices=
            top_indices.astype(
                np.int32
            ),

        qwen3_scores=
            top_scores.astype(
                np.float32
            ),

        base_rank_1indexed_capped501=
            rank501.astype(
                np.int16
            ),

        base_failure_gt100=
            failure_mask.astype(
                bool
            ),

        local80_recoverable=
            local80_recoverable.astype(
                bool
            ),

        top500_recoverable=
            top500_recoverable.astype(
                bool
            ),

        protected_k=np.asarray(
            PROTECTED_K,
            dtype=np.int16,
        ),

        local_candidate_k=np.asarray(
            80,
            dtype=np.int16,
        ),

        final_budget=np.asarray(
            B,
            dtype=np.int16,
        ),

        valid_accessed=np.asarray(
            False
        ),

        test_accessed=np.asarray(
            False
        ),
    )

    sha = sha256_file(
        OUT_NPZ
    )

    runtime = (
        time.perf_counter()
        -
        t0
    )

    metrics = {
        "artifact":
            "TRAIN-only fully observable repair sandbox",

        "catalog_size":
            TRAIN_CODE_ROWS,

        "queries":
            12000,

        "fit_queries":
            int(
                fit_mask.sum()
            ),

        "all":
            all_metrics,

        "fit":
            fit_metrics,

        "fit_folds":
            fold_summary,

        "catalog_relation_to_original": {
            "fit_trainonly_rank_improved":
                train_improved,

            "fit_rank_same":
                same_rank,

            "fit_trainonly_rank_worse":
                train_worse,
        },

        "candidate_regions": {
            "protected":
                "ranks1..100",

            "local80":
                "ranks101..180",

            "diagnostic_top500_supplementary":
                "ranks101..500",

            "final_budget":
                20,
        },

        "runtime_seconds":
            float(
                runtime
            ),

        "artifact_sha256":
            sha,

        "protocol": [
            "Development sandbox only.",
            "Catalog contains only the 412178 CodeSearchNet Python TRAIN code rows.",
            "All query source rows and GT rows are inside TRAIN.",
            "No CAL/HOLDOUT/TEST raw data is accessed.",
            "Qwen query embeddings are the previously frozen TRAIN query embeddings.",
            "Code embeddings are the previously frozen Qwen code embeddings restricted to TRAIN rows.",
            "Top500 is recomputed exactly against the TRAIN-only catalog.",
            "Base failure remains Qwen rank >100.",
            "Final repair budget remains B=20.",
            "Ranks101..180 define a protocol-aligned local E80 diagnostic pool.",
            "Ranks101..500 are diagnostic only and do not redefine the final repair budget.",
            "No content-aware repair algorithm is evaluated in this step.",
        ],
    }

    OUT_JSON.write_text(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 132
    )

    print(
        "SANDBOX BUILD COMPLETE"
    )

    print(
        "=" * 132
    )

    print(
        "runtime =",
        runtime,
    )

    print(
        "artifact =",
        OUT_NPZ,
    )

    print(
        "sha256 =",
        sha,
    )

    print(
        "metrics =",
        OUT_JSON,
    )


if __name__ == "__main__":
    main()
