#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


ROOT = Path(
    "/root/autodl-tmp/work2-algorithm-innovation"
)

PARQUET = Path(
    "/root/autodl-tmp/datasets/CodeSearchNet/"
    "code_search_net/python/"
    "train-00000-of-00001.parquet"
)

MODEL_PATH = Path(
    "/root/autodl-tmp/hf_models/Qwen3-Embedding-4B"
)

CONFIG = (
    ROOT
    / "fcqr_mvp_v2/configs/"
      "fcqr_python_qwen3_mvp_v2.json"
)

MANIFEST = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_manifest_v1.npz"
)

OLD_SANDBOX = (
    ROOT
    / "fhex_v1/sandbox/"
      "python_trainonly_qwen3_top500_v1.npz"
)

OUT_QUERY_EMB = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_qwen3_queries_f32.npy"
)

OUT_NPZ = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_qwen3_top500_v1.npz"
)

OUT_JSON = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_qwen3_top500_v1_metrics.json"
)


EXPECTED_MANIFEST_SHA = (
    "fa349f76dfb73808a7203eee4f4be4ef"
    "3feec71f3bc98b671869a4a165233592"
)

EXPECTED_RCPS_SHA = (
    "fb61f7a1b30c69698b9ee7365cb92cd"
    "39c4931b9849be5d25c4e4a314943beb5"
)

TRAIN_N = 412178

DIM = 2560

TOPK = 500

PROTECTED_K = 100

LOCAL_START = 100
LOCAL_STOP = 180

ENCODE_BATCH = 16

RETRIEVAL_BATCH = 32

CODE_GPU_CHUNK = 8192

MAX_LENGTH = 512

QWEN_TASK = (
    "Given a natural language query, retrieve the most "
    "relevant code snippet that implements the described "
    "functionality"
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


def resolve_path(value):

    p = Path(value)

    return (
        p
        if p.is_absolute()
        else ROOT / p
    )


def normalize_np(x):

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    n = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )

    return (
        x
        /
        np.maximum(
            n,
            1e-12,
        )
    )


def token_text(tokens):

    if tokens is None:

        raise ValueError(
            "documentation tokens are None"
        )

    if isinstance(
        tokens,
        str,
    ):

        # This should not normally happen for CSN token column.
        text = tokens.strip()

    else:

        text = " ".join(
            str(x)
            for x in tokens
        ).strip()

    if not text:

        raise ValueError(
            "empty documentation token query"
        )

    return text


def make_prompt(query):

    return (
        f"Instruct: {QWEN_TASK}\n"
        f"Query:{query}"
    )


@torch.inference_mode()
def encode_texts(
    model,
    tokenizer,
    texts,
):

    batch = tokenizer(
        [
            make_prompt(x)
            for x in texts
        ],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    batch = {
        k: v.cuda(
            non_blocking=True
        )
        for k, v in batch.items()
    }

    out = model(
        **batch,
        return_dict=True,
    )

    hidden = out.last_hidden_state

    # Left padding means the final position is always the
    # final real token, matching the frozen reconstruction.
    emb = hidden[
        :,
        -1,
        :
    ].float()

    emb = F.normalize(
        emb,
        p=2,
        dim=1,
    )

    return (
        emb
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


def summarize_rank(rank501):

    r = np.asarray(
        rank501
    )

    return {
        "n":
            int(
                len(r)
            ),

        "top1":
            int(
                np.sum(
                    r <= 1
                )
            ),

        "top5":
            int(
                np.sum(
                    r <= 5
                )
            ),

        "top10":
            int(
                np.sum(
                    r <= 10
                )
            ),

        "top20":
            int(
                np.sum(
                    r <= 20
                )
            ),

        "top100":
            int(
                np.sum(
                    r <= 100
                )
            ),

        "failure_gt100":
            int(
                np.sum(
                    r > 100
                )
            ),

        "local80_recoverable_101_180":
            int(
                np.sum(
                    (r >= 101)
                    &
                    (r <= 180)
                )
            ),

        "top500_recoverable_101_500":
            int(
                np.sum(
                    (r >= 101)
                    &
                    (r <= 500)
                )
            ),

        "beyond500":
            int(
                np.sum(
                    r == 501
                )
            ),
    }


def main():

    t0 = time.perf_counter()

    torch.backends.cuda.matmul.allow_tf32 = False

    print(
        "=" * 140
    )

    print(
        "RCPS PROSPECTIVE-P1 — STAGE 1"
    )

    print(
        "=" * 140
    )

    print(
        "Encode frozen P1 queries and build exact TRAIN-only Top500."
    )

    print(
        "NO RCPS / BCCT evaluation occurs in this script."
    )

    # ========================================================
    # Immutable manifest verification.
    # ========================================================

    actual_manifest_sha = sha256_file(
        MANIFEST
    )

    if (
        actual_manifest_sha
        !=
        EXPECTED_MANIFEST_SHA
    ):

        raise AssertionError(
            (
                "P1 manifest SHA changed.",
                actual_manifest_sha,
                EXPECTED_MANIFEST_SHA,
            )
        )

    manifest = np.load(
        MANIFEST,
        allow_pickle=True,
    )

    rows = manifest[
        "source_row_indices"
    ].astype(
        np.int64
    )

    gt = manifest[
        "ground_truth_code_indices"
    ].astype(
        np.int64
    )

    shards = manifest[
        "shard_ids"
    ].astype(
        np.int8
    )

    frozen_rcps_sha = str(
        np.asarray(
            manifest[
                "frozen_rcps_sha256"
            ]
        ).item()
    )

    assert frozen_rcps_sha == EXPECTED_RCPS_SHA

    assert len(rows) == 12000

    assert np.array_equal(
        rows,
        gt,
    )

    assert len(
        np.unique(rows)
    ) == 12000

    assert rows.min() >= 0
    assert rows.max() < TRAIN_N

    assert not bool(
        np.asarray(
            manifest[
                "retrieval_accessed"
            ]
        ).item()
    )

    assert not bool(
        np.asarray(
            manifest[
                "outcome_accessed"
            ]
        ).item()
    )

    assert not bool(
        np.asarray(
            manifest[
                "valid_accessed"
            ]
        ).item()
    )

    assert not bool(
        np.asarray(
            manifest[
                "test_accessed"
            ]
        ).item()
    )

    print(
        "manifest SHA verified =",
        actual_manifest_sha,
    )

    print(
        "frozen RCPS SHA verified =",
        frozen_rcps_sha,
    )

    # ========================================================
    # Load frozen config.
    # ========================================================

    cfg = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    old_query_emb_path = resolve_path(
        cfg[
            "train_query_embeddings"
        ]
    )

    code_emb_path = resolve_path(
        cfg[
            "code_embeddings"
        ]
    )

    code_all = np.load(
        code_emb_path,
        mmap_mode="r",
    )

    assert code_all.shape[
        0
    ] >= TRAIN_N

    assert code_all.shape[
        1
    ] == DIM

    # ========================================================
    # Load query text.
    # ========================================================

    print()
    print(
        "Loading CodeSearchNet query tokens..."
    )

    df = pd.read_parquet(
        PARQUET,
        columns=[
            "func_documentation_tokens",
        ],
    )

    assert len(df) == TRAIN_N

    p1_texts = []

    for row in rows:

        p1_texts.append(
            token_text(
                df.iloc[
                    int(row)
                ][
                    "func_documentation_tokens"
                ]
            )
        )

    print(
        "P1 non-empty token queries =",
        len(p1_texts),
    )

    # ========================================================
    # Load encoder.
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "QWEN ENCODER PARITY PREFLIGHT"
    )

    print(
        "=" * 140
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = model.cuda()
    model.eval()

    # --------------------------------------------------------
    # Re-encode 32 old development queries before touching
    # P1 embeddings and compare against frozen embeddings.
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

    old_emb = np.load(
        old_query_emb_path,
        mmap_mode="r",
    )

    assert old_emb.shape == (
        12000,
        DIM,
    )

    parity_ids = np.linspace(
        0,
        11999,
        32,
        dtype=np.int64,
    )

    parity_text = [
        token_text(
            df.iloc[
                int(
                    old_rows[
                        j
                    ]
                )
            ][
                "func_documentation_tokens"
            ]
        )
        for j in parity_ids
    ]

    parity_new = encode_texts(
        model,
        tokenizer,
        parity_text,
    )

    parity_old = normalize_np(
        old_emb[
            parity_ids
        ]
    )

    parity_cos = np.sum(
        parity_new
        *
        parity_old,
        axis=1,
    )

    print(
        "parity cosine mean =",
        float(
            parity_cos.mean()
        ),
    )

    print(
        "parity cosine median =",
        float(
            np.median(
                parity_cos
            )
        ),
    )

    print(
        "parity cosine min =",
        float(
            parity_cos.min()
        ),
    )

    # Implementation gate only.
    if (
        float(
            parity_cos.mean()
        ) < 0.9990
        or
        float(
            parity_cos.min()
        ) < 0.9950
    ):

        raise AssertionError(
            (
                "Qwen encoder parity failed. "
                "Do NOT encode/evaluate P1.",
                float(
                    parity_cos.mean()
                ),
                float(
                    parity_cos.min()
                ),
            )
        )

    print(
        "QWEN ENCODER PARITY PASS."
    )

    # ========================================================
    # Encode all frozen P1 queries.
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "ENCODING FROZEN P1 QUERIES"
    )

    print(
        "=" * 140
    )

    q_mm = np.lib.format.open_memmap(
        OUT_QUERY_EMB,
        mode="w+",
        dtype=np.float32,
        shape=(
            12000,
            DIM,
        ),
    )

    enc_start = time.perf_counter()

    for start in range(
        0,
        12000,
        ENCODE_BATCH,
    ):

        stop = min(
            start
            +
            ENCODE_BATCH,
            12000,
        )

        emb = encode_texts(
            model,
            tokenizer,
            p1_texts[
                start:stop
            ],
        )

        if emb.shape != (
            stop - start,
            DIM,
        ):

            raise AssertionError(
                (
                    "Unexpected embedding shape",
                    emb.shape,
                )
            )

        q_mm[
            start:stop
        ] = emb

        if (
            stop % 512 == 0
            or
            stop == 12000
        ):

            print(
                f"[ENCODE] {stop}/12000",
                f"elapsed={time.perf_counter()-enc_start:.1f}s",
                flush=True,
            )

    q_mm.flush()

    del q_mm
    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    # ========================================================
    # Exact TRAIN-only retrieval.
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "EXACT TRAIN-ONLY QWEN TOP500"
    )

    print(
        "=" * 140
    )

    query_emb = np.load(
        OUT_QUERY_EMB,
        mmap_mode="r",
    )

    assert query_emb.shape == (
        12000,
        DIM,
    )

    # Load normalized TRAIN code catalog to GPU.
    code_gpu = torch.empty(
        (
            TRAIN_N,
            DIM,
        ),
        dtype=torch.float32,
        device="cuda",
    )

    for start in range(
        0,
        TRAIN_N,
        CODE_GPU_CHUNK,
    ):

        stop = min(
            start
            +
            CODE_GPU_CHUNK,
            TRAIN_N,
        )

        x = normalize_np(
            code_all[
                start:stop
            ]
        )

        code_gpu[
            start:stop
        ].copy_(
            torch.from_numpy(
                x.copy()
            ).cuda()
        )

    print(
        "TRAIN code catalog GPU shape =",
        tuple(
            code_gpu.shape
        ),
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

    search_start = time.perf_counter()

    for start in range(
        0,
        12000,
        RETRIEVAL_BATCH,
    ):

        stop = min(
            start
            +
            RETRIEVAL_BATCH,
            12000,
        )

        q = torch.from_numpy(
            np.asarray(
                query_emb[
                    start:stop
                ],
                dtype=np.float32,
            ).copy()
        ).cuda()

        scores = (
            q
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
            q,
            scores,
            values,
            indices,
        )

        if (
            stop % 1024 == 0
            or
            stop == 12000
        ):

            print(
                f"[RETRIEVE] {stop}/12000",
                f"elapsed={time.perf_counter()-search_start:.2f}s",
                flush=True,
            )

    # ========================================================
    # Freeze prospective failure/frontier pool.
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

        if len(p):

            rank501[i] = int(
                p[0] + 1
            )

    failure = (
        rank501 > PROTECTED_K
    )

    local80 = (
        (rank501 >= 101)
        &
        (rank501 <= 180)
    )

    top500_recoverable = (
        (rank501 >= 101)
        &
        (rank501 <= 500)
    )

    overall = summarize_rank(
        rank501
    )

    print()
    print(
        "=" * 140
    )

    print(
        "P1 PROSPECTIVE FRONTIER"
    )

    print(
        "=" * 140
    )

    print(
        json.dumps(
            overall,
            indent=2,
        )
    )

    print()
    print(
        "PER SHARD"
    )

    shard_metrics = {}

    for shard in range(
        5
    ):

        m = (
            shards == shard
        )

        x = summarize_rank(
            rank501[m]
        )

        shard_metrics[
            str(shard)
        ] = x

        print(
            f"shard={shard}",
            f"n={x['n']}",
            f"fail={x['failure_gt100']}",
            f"local80={x['local80_recoverable_101_180']}",
            f"top500={x['top500_recoverable_101_500']}",
        )

    # ========================================================
    # Save immutable prospective retrieval artifact.
    # ========================================================

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "work2_rcps_prospective_p1_qwen3_top500_v1"
        ),

        source_row_indices=
            rows.astype(
                np.int64
            ),

        ground_truth_code_indices=
            gt.astype(
                np.int64
            ),

        shard_ids=
            shards.astype(
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
            failure.astype(
                bool
            ),

        local80_recoverable=
            local80.astype(
                bool
            ),

        top500_recoverable=
            top500_recoverable.astype(
                bool
            ),

        manifest_sha256=np.asarray(
            EXPECTED_MANIFEST_SHA
        ),

        frozen_rcps_sha256=np.asarray(
            EXPECTED_RCPS_SHA
        ),

        encoder_parity_cosine=
            parity_cos.astype(
                np.float32
            ),

        rcps_evaluated=np.asarray(
            False
        ),

        valid_accessed=np.asarray(
            False
        ),

        test_accessed=np.asarray(
            False
        ),
    )

    artifact_sha = sha256_file(
        OUT_NPZ
    )

    runtime = (
        time.perf_counter()
        -
        t0
    )

    result = {
        "name":
            "RCPS Prospective P1 Retrieval Pool",

        "manifest_sha256":
            EXPECTED_MANIFEST_SHA,

        "frozen_rcps_sha256":
            EXPECTED_RCPS_SHA,

        "encoder_parity": {
            "mean":
                float(
                    parity_cos.mean()
                ),

            "median":
                float(
                    np.median(
                        parity_cos
                    )
                ),

            "min":
                float(
                    parity_cos.min()
                ),
        },

        "overall":
            overall,

        "shards":
            shard_metrics,

        "prospective_decision_status":
            (
                "READY"
                if overall[
                    "local80_recoverable_101_180"
                ] >= 60
                else
                "INSUFFICIENT_LOCAL80_SAMPLE__INCONCLUSIVE"
            ),

        "method_evaluated":
            False,

        "runtime_seconds":
            float(
                runtime
            ),

        "artifact_sha256":
            artifact_sha,

        "protocol": [
            "Frozen prospective P1 manifest only.",
            "No development query reused.",
            "No CAL/HOLDOUT/TEST access.",
            "Exact Qwen encoder implementation parity checked before P1 encoding.",
            "Natural-language query reconstructed as space-joined func_documentation_tokens.",
            "Prompt is frozen Qwen retrieval instruction.",
            "Maximum token length 512.",
            "Final hidden token pooled and L2-normalized in FP32.",
            "Code catalog restricted to 412178 Python TRAIN rows.",
            "Exact cosine Top500 retrieval.",
            "Base failure is rank >100.",
            "Local80 recoverability is rank101..180.",
            "RCPS and BCCT are not evaluated in this stage.",
            "This retrieval/frontier artifact must not be regenerated after method evaluation.",
        ],
    }

    OUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 140
    )

    print(
        "P1 STAGE-1 FREEZE COMPLETE"
    )

    print(
        "=" * 140
    )

    print(
        "prospective status =",
        result[
            "prospective_decision_status"
        ],
    )

    print(
        "artifact =",
        OUT_NPZ,
    )

    print(
        "artifact SHA256 =",
        artifact_sha,
    )

    print(
        "metrics =",
        OUT_JSON,
    )

    print(
        "runtime =",
        runtime,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT regenerate P1 retrieval after RCPS evaluation."
    )


if __name__ == "__main__":
    main()
