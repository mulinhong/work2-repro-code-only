#!/usr/bin/env python3

from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import sys
import time

import numpy as np
import torch


ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")

ORIGINAL_KERNEL = (
    ROOT
    / "fhex_v1/scripts/"
      "build_trainonly_repair_sandbox.py"
)

DEV_SANDBOX = (
    ROOT
    / "fhex_v1/sandbox/"
      "python_trainonly_qwen3_top500_v1.npz"
)

CODE_CACHE = (
    ROOT
    / "fcqr_source_cache/"
      "python_qwen3_code_embeddings_f32.npy"
)

HIST_QUERY = Path(
    "/root/autodl-tmp/qwen_historical_transfer_check/"
    "parity_reference/"
    "python_train_sampled_qwen3_queries_f32.npy"
)

P3_QUERY = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "retrieval_v1/"
      "jters_p3_qwen3_queries_f32_v1.npy"
)

P3_MANIFEST = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "jters_p3_manifest_v1.npz"
)

SUPERSEDED_FULL = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "retrieval_v1/"
      "jters_p3_qwen3_top500_historical_v1.npz"
)

FAILED_V2 = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "retrieval_v2_trainonly/"
      "build_p3_trainonly_retrieval_v1.py"
)

FAILED_V2_LOG = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "retrieval_v2_trainonly/logs/"
      "build_p3_trainonly_retrieval_v1.log"
)

OUTDIR = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "retrieval_v3_trainonly_exactkernel"
)

OUT = (
    OUTDIR
    / "jters_p3_qwen3_top500_trainonly_exactkernel_v1.npz"
)

MEMO = (
    OUTDIR
    / "p3_trainonly_exactkernel_freeze_v1.json"
)


EXPECTED = {
    "kernel_script":
        "1a8167cd4774935782e30f7347941bd7e43795e43a16ad3e08c49c47c3e4331f",

    "dev_sandbox":
        "ef7cbf7404e863bc12b717e5573b62eb7cbb27733f2283b3eeb367864c64dd97",

    "code_cache":
        "f47355e8e5317a5b058ac997269ed8c9c70e2c1be344d37b7596a38065666482",

    "historical_query":
        "9668aefd78d7ac6aa082a41ca6d515fb6921dd03da57aef4b1a34ec7d5bb5f80",

    "p3_query":
        "3255b5c4143df30f2ab4935a9f8f57dffc85709ff3ee1185605750441652edb9",

    "p3_manifest":
        "07650ed49593985ce506a728597e83fd84f76c6ad78d82d4f731d122133db7d0",

    "superseded_full":
        "47c5d0cf06172ae7560a05b4bd0cb6c101024d30b09f0f42dae40539e7c6e408",
}


TRAIN_ROWS = 412178
FULL_ROWS = 457461
DIM = 2560
TOPK = 500
QUERY_BATCH = 32
GPU_CODE_CHUNK = 8192
PREFLIGHT_N = 32
P3_N = 6000


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            b = f.read(1 << 20)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def require_sha(name, path):

    actual = sha256(path)
    expected = EXPECTED[name]

    ok = actual == expected

    print(
        f"{name} SHA PASS =",
        ok,
    )

    if not ok:

        print(" expected =", expected)
        print(" actual   =", actual)

        raise RuntimeError(
            f"{name}: SHA mismatch"
        )


def load_original_kernel():

    spec = importlib.util.spec_from_file_location(
        "frozen_trainonly_kernel",
        ORIGINAL_KERNEL,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot construct original kernel module."
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def retrieve_exact(
    q_np,
    code_gpu,
    label,
):

    n = len(q_np)

    indices = np.empty(
        (n, TOPK),
        dtype=np.int32,
    )

    scores_out = np.empty(
        (n, TOPK),
        dtype=np.float32,
    )

    begin = time.perf_counter()

    with torch.no_grad():

        for start in range(
            0,
            n,
            QUERY_BATCH,
        ):

            stop = min(
                start + QUERY_BATCH,
                n,
            )

            qgpu = torch.from_numpy(
                q_np[
                    start:stop
                ].copy()
            ).cuda()

            # --------------------------------------------------
            # EXACT historical TRAIN-only retrieval kernel.
            # Same shape/operation as frozen sandbox:
            #
            #     qgpu @ code_gpu.T
            #     torch.topk(k=500, sorted=True)
            # --------------------------------------------------

            score = (
                qgpu
                @
                code_gpu.T
            )

            values, ids = torch.topk(
                score,
                k=TOPK,
                dim=1,
                largest=True,
                sorted=True,
            )

            indices[
                start:stop
            ] = (
                ids
                .cpu()
                .numpy()
                .astype(
                    np.int32,
                    copy=False,
                )
            )

            scores_out[
                start:stop
            ] = (
                values
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            if (
                stop % 320 == 0
                or
                stop == n
            ):

                print(
                    f"{label}:",
                    stop,
                    "/",
                    n,
                )

            del (
                qgpu,
                score,
                values,
                ids,
            )

    elapsed = (
        time.perf_counter()
        -
        begin
    )

    print(
        f"{label} seconds =",
        elapsed,
    )

    return (
        scores_out,
        indices,
    )


def main():

    print("=" * 150)
    print("P3 TRAIN-ONLY EXACT HISTORICAL KERNEL v1")
    print("=" * 150)

    print(
        "NO Atomic / RCPS / TECS / RBJS / APQC "
        "outcome is computed."
    )

    print()

    # ========================================================
    # Frozen provenance.
    # ========================================================

    require_sha(
        "kernel_script",
        ORIGINAL_KERNEL,
    )

    require_sha(
        "dev_sandbox",
        DEV_SANDBOX,
    )

    require_sha(
        "code_cache",
        CODE_CACHE,
    )

    require_sha(
        "historical_query",
        HIST_QUERY,
    )

    require_sha(
        "p3_query",
        P3_QUERY,
    )

    require_sha(
        "p3_manifest",
        P3_MANIFEST,
    )

    require_sha(
        "superseded_full",
        SUPERSEDED_FULL,
    )

    # ========================================================
    # Import original TRAIN-only implementation.
    # We directly reuse its normalization + GPU code loader.
    # ========================================================

    frozen = load_original_kernel()

    print()
    print("=" * 150)
    print("ORIGINAL KERNEL CONTRACT")
    print("=" * 150)

    print(
        "TRAIN_CODE_ROWS =",
        frozen.TRAIN_CODE_ROWS,
    )

    print(
        "TOPK =",
        frozen.TOPK,
    )

    print(
        "QUERY_BATCH =",
        frozen.QUERY_BATCH,
    )

    print(
        "GPU_CODE_CHUNK =",
        frozen.GPU_CODE_CHUNK,
    )

    print(
        "normalize_rows signature =",
        inspect.signature(
            frozen.normalize_rows
        ),
    )

    print(
        "load_code_gpu signature =",
        inspect.signature(
            frozen.load_code_gpu
        ),
    )

    assert frozen.TRAIN_CODE_ROWS == TRAIN_ROWS
    assert frozen.TOPK == TOPK
    assert frozen.QUERY_BATCH == QUERY_BATCH
    assert frozen.GPU_CODE_CHUNK == GPU_CODE_CHUNK

    # This is explicitly present in original sandbox main().
    torch.backends.cuda.matmul.allow_tf32 = False

    print(
        "torch version =",
        torch.__version__,
    )

    print(
        "CUDA version =",
        torch.version.cuda,
    )

    print(
        "GPU =",
        torch.cuda.get_device_name(0),
    )

    print(
        "allow_tf32 =",
        torch.backends.cuda.matmul.allow_tf32,
    )

    # ========================================================
    # Historical code cache.
    # Original train-only loader normalizes code chunks and
    # materializes the full 412178 x 2560 matrix on GPU.
    # ========================================================

    code_all = np.load(
        CODE_CACHE,
        mmap_mode="r",
        allow_pickle=False,
    )

    assert code_all.shape == (
        FULL_ROWS,
        DIM,
    )

    assert code_all.dtype == np.float32

    print()
    print(
        "Loading exact historical TRAIN-only "
        "normalized code matrix to GPU..."
    )

    code_gpu = frozen.load_code_gpu(
        code_all
    )

    assert tuple(
        code_gpu.shape
    ) == (
        TRAIN_ROWS,
        DIM,
    )

    assert code_gpu.dtype == torch.float32

    # ========================================================
    # CONSUMED-ONLY HARD PREFLIGHT.
    #
    # Crucial difference vs failed v2:
    # 1. reuse original normalize_rows()
    # 2. reuse original load_code_gpu()
    # 3. batch=32
    # 4. full TRAIN matrix matmul
    # 5. TF32=False
    # ========================================================

    old_query = np.load(
        HIST_QUERY,
        mmap_mode="r",
        allow_pickle=False,
    )

    assert old_query.shape == (
        12000,
        DIM,
    )

    pre_q = frozen.normalize_rows(
        np.asarray(
            old_query[
                :PREFLIGHT_N
            ],
            dtype=np.float32,
        )
    )

    with np.load(
        DEV_SANDBOX,
        allow_pickle=False,
    ) as z:

        old_idx = np.asarray(
            z["qwen3_indices"][
                :PREFLIGHT_N
            ],
            dtype=np.int32,
        )

        old_scores = np.asarray(
            z["qwen3_scores"][
                :PREFLIGHT_N
            ],
            dtype=np.float32,
        )

    print()
    print("=" * 150)
    print("CONSUMED-ONLY EXACT-KERNEL PREFLIGHT")
    print("=" * 150)

    pre_scores, pre_idx = retrieve_exact(
        pre_q,
        code_gpu,
        "historical exact-kernel parity",
    )

    row_exact = np.all(
        pre_idx
        ==
        old_idx,
        axis=1,
    )

    index_exact = bool(
        np.array_equal(
            pre_idx,
            old_idx,
        )
    )

    exact_rows = int(
        row_exact.sum()
    )

    score_max_abs = float(
        np.max(
            np.abs(
                pre_scores
                -
                old_scores
            )
        )
    )

    score_exact = bool(
        np.array_equal(
            pre_scores,
            old_scores,
        )
    )

    print(
        "Top500 index exact equality =",
        index_exact,
    )

    print(
        "exact Top500 rows =",
        exact_rows,
        "/",
        PREFLIGHT_N,
    )

    print(
        "score exact equality =",
        score_exact,
    )

    print(
        "score maximum absolute delta =",
        score_max_abs,
    )

    # Ranking/candidate identity is the algorithmic contract.
    if (
        not index_exact
        or
        exact_rows != PREFLIGHT_N
    ):

        raise RuntimeError(
            "EXACT historical TRAIN-only kernel still "
            "does not reproduce frozen sandbox. "
            "Abort before P3 retrieval."
        )

    print(
        "TRAINONLY_EXACT_KERNEL_PREFLIGHT_PASS = True"
    )

    # ========================================================
    # P3 retrieval — only after exact consumed parity.
    # ========================================================

    p3_query = np.load(
        P3_QUERY,
        mmap_mode="r",
        allow_pickle=False,
    )

    assert p3_query.shape == (
        P3_N,
        DIM,
    )

    # Historical TRAIN-only sandbox explicitly re-normalized
    # cached query embeddings prior to retrieval.
    p3_q = frozen.normalize_rows(
        np.asarray(
            p3_query,
            dtype=np.float32,
        )
    )

    with np.load(
        P3_MANIFEST,
        allow_pickle=False,
    ) as m:

        rows = np.asarray(
            m["source_row_indices"],
            dtype=np.int64,
        )

        shards = np.asarray(
            m["shard_ids"],
            dtype=np.int16,
        )

    assert rows.shape == (P3_N,)
    assert shards.shape == (P3_N,)

    assert rows.min() >= 0
    assert rows.max() < TRAIN_ROWS

    print()
    print("=" * 150)
    print("P3 TRAIN-ONLY EXACT-KERNEL TOP500")
    print("=" * 150)

    p3_scores, p3_idx = retrieve_exact(
        p3_q,
        code_gpu,
        "P3 TRAIN-only exact kernel",
    )

    assert p3_idx.shape == (
        P3_N,
        TOPK,
    )

    assert p3_scores.shape == (
        P3_N,
        TOPK,
    )

    assert p3_idx.min() >= 0
    assert p3_idx.max() < TRAIN_ROWS

    duplicate_rows = int(
        sum(
            len(
                np.unique(
                    p3_idx[i]
                )
            )
            != TOPK
            for i in range(
                P3_N
            )
        )
    )

    if duplicate_rows != 0:

        raise RuntimeError(
            "Duplicate candidate IDs within P3 Top500."
        )

    # ========================================================
    # Freeze prospective retrieval.
    #
    # Deliberately no R@k / GT rank / Local80 / repair score.
    # ========================================================

    np.savez_compressed(
        OUT,

        format=np.asarray(
            "jters_p3_qwen3_top500_trainonly_exactkernel_v1"
        ),

        source_row_indices=
            rows,

        shard_ids=
            shards,

        ground_truth_code_indices=
            rows.astype(
                np.int32
            ),

        qwen3_indices=
            p3_idx,

        qwen3_scores=
            p3_scores,

        catalog_scope=np.asarray(
            "CodeSearchNet Python TRAIN only"
        ),

        catalog_rows=np.asarray(
            TRAIN_ROWS,
            dtype=np.int32,
        ),

        top_k=np.asarray(
            TOPK,
            dtype=np.int32,
        ),

        retrieval_query_batch=np.asarray(
            QUERY_BATCH,
            dtype=np.int32,
        ),

        gpu_code_chunk=np.asarray(
            GPU_CODE_CHUNK,
            dtype=np.int32,
        ),

        tf32_allowed=np.asarray(
            False
        ),

        original_kernel_sha256=np.asarray(
            EXPECTED[
                "kernel_script"
            ]
        ),

        frozen_dev_sandbox_sha256=np.asarray(
            EXPECTED[
                "dev_sandbox"
            ]
        ),

        p3_manifest_sha256=np.asarray(
            EXPECTED[
                "p3_manifest"
            ]
        ),

        p3_query_embeddings_sha256=np.asarray(
            EXPECTED[
                "p3_query"
            ]
        ),

        code_embeddings_sha256=np.asarray(
            EXPECTED[
                "code_cache"
            ]
        ),

        superseded_full_catalog_sha256=np.asarray(
            EXPECTED[
                "superseded_full"
            ]
        ),

        valid_accessed=np.asarray(
            False
        ),

        test_accessed=np.asarray(
            False
        ),
    )

    out_sha = sha256(
        OUT
    )

    failed_v2_script_sha = (
        sha256(
            FAILED_V2
        )
        if FAILED_V2.exists()
        else None
    )

    failed_v2_log_sha = (
        sha256(
            FAILED_V2_LOG
        )
        if FAILED_V2_LOG.exists()
        else None
    )

    memo = {
        "format":
            "p3_trainonly_exactkernel_freeze_v1",

        "prospective_status":
            "before any P3 algorithm evaluation",

        "algorithm_outcomes_observed":
            False,

        "catalog_rows":
            TRAIN_ROWS,

        "kernel": {
            "source":
                str(
                    ORIGINAL_KERNEL
                ),

            "sha256":
                EXPECTED[
                    "kernel_script"
                ],

            "query_normalization":
                "original frozen normalize_rows",

            "code_normalization_and_gpu_loader":
                "original frozen load_code_gpu",

            "query_batch":
                QUERY_BATCH,

            "gpu_code_chunk":
                GPU_CODE_CHUNK,

            "matmul":
                "qgpu @ code_gpu.T",

            "topk":
                TOPK,

            "tf32":
                False,
        },

        "consumed_preflight": {
            "n":
                PREFLIGHT_N,

            "exact_top500_rows":
                exact_rows,

            "top500_index_exact":
                index_exact,

            "score_exact":
                score_exact,

            "score_max_abs_delta":
                score_max_abs,

            "pass":
                True,
        },

        "superseded_full_catalog_artifact": {
            "path":
                str(
                    SUPERSEDED_FULL
                ),

            "sha256":
                EXPECTED[
                    "superseded_full"
                ],

            "used_for_algorithm_evaluation":
                False,
        },

        "failed_v2_preflight_attempt": {
            "script":
                str(
                    FAILED_V2
                ),

            "script_sha256":
                failed_v2_script_sha,

            "log":
                str(
                    FAILED_V2_LOG
                ),

            "log_sha256":
                failed_v2_log_sha,

            "generated_corrected_p3_artifact":
                False,

            "algorithm_outcomes_observed":
                False,
        },

        "frozen_p3_artifact": {
            "path":
                str(
                    OUT
                ),

            "sha256":
                out_sha,
        },
    }

    MEMO.write_text(
        json.dumps(
            memo,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    memo_sha = sha256(
        MEMO
    )

    print()
    print("=" * 150)
    print("P3 TRAIN-ONLY EXACT-KERNEL RETRIEVAL FROZEN")
    print("=" * 150)

    print(
        "catalog rows =",
        TRAIN_ROWS,
    )

    print(
        "shape =",
        p3_idx.shape,
    )

    print(
        "maximum candidate id =",
        int(
            p3_idx.max()
        ),
    )

    print(
        "duplicate Top500 rows =",
        duplicate_rows,
    )

    print(
        "artifact =",
        OUT,
    )

    print(
        "artifact SHA256 =",
        out_sha,
    )

    print(
        "freeze memo SHA256 =",
        memo_sha,
    )

    print()

    print(
        "SUPERSEDED_FULL_CATALOG_USED_FOR_ALGORITHM = False"
    )

    print(
        "FAILED_V2_USED_FOR_ALGORITHM = False"
    )

    print(
        "P3_ALGORITHM_OUTCOMES_OBSERVED = False"
    )

    print(
        "STATUS = P3 TRAIN-ONLY EXACT-KERNEL "
        "RETRIEVAL FROZEN"
    )


if __name__ == "__main__":
    main()
