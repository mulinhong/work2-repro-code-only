#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import hashlib
import json
import os
import sys
import gc

import numpy as np
import torch


ROOT = Path("/root/autodl-tmp")
WORK2 = ROOT / "work2-algorithm-innovation"
RUNTIME = ROOT / "qwen_historical_runtime"

P3DIR = (
    WORK2
    / "fhex_v1/prospective/p3_jters_v2"
)

OUTDIR = (
    P3DIR
    / "retrieval_v1"
)

CODE_DIR = (
    RUNTIME
    / "code/text2code_minimal"
)

BUILDER = (
    CODE_DIR
    / "build_comparable_retriever_probe.py"
)

MODEL = (
    RUNTIME
    / "model/Qwen3-Embedding-4B"
)

CODE_CACHE = (
    WORK2
    / "fcqr_source_cache/"
      "python_qwen3_code_embeddings_f32.npy"
)

QUERY_JSONL = (
    OUTDIR
    / "jters_p3_queries_historical_contract_v1.jsonl"
)

QUERY_EMB = (
    OUTDIR
    / "jters_p3_qwen3_queries_f32_v1.npy"
)

OUT_NPZ = (
    OUTDIR
    / "jters_p3_qwen3_top500_historical_v1.npz"
)

PROTOCOL = (
    OUTDIR
    / "p3_qwen_historical_retrieval_protocol_v1.json"
)

P3_MANIFEST = (
    P3DIR
    / "jters_p3_manifest_v1.npz"
)

EXPECTED = {
    "builder":
        "a08d9168d05a2cbf65841394d7220a560147523c69da4168381b69120356be17",

    "code_cache":
        "f47355e8e5317a5b058ac997269ed8c9c70e2c1be344d37b7596a38065666482",

    "manifest":
        "07650ed49593985ce506a728597e83fd84f76c6ad78d82d4f731d122133db7d0",
}

BATCH = 8
TOPK = 500
QUERY_CHUNK = 16
CODE_CHUNK = 8192
DIM = 2560
CODE_ROWS = 457461
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


def load_builder():

    sys.path.insert(
        0,
        str(CODE_DIR),
    )

    spec = importlib.util.spec_from_file_location(
        "historical_builder",
        BUILDER,
    )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def atomic_npy(path, array):

    tmp = path.with_name(
        path.stem
        +
        ".tmp.npy"
    )

    np.save(
        tmp,
        array,
        allow_pickle=False,
    )

    os.replace(
        tmp,
        path,
    )


def main():

    print("=" * 150)
    print("P3 HISTORICAL-EXACT QWEN TOP500 BUILD v1")
    print("=" * 150)

    for name, path in [
        ("builder", BUILDER),
        ("code_cache", CODE_CACHE),
        ("manifest", P3_MANIFEST),
    ]:

        got = sha256(
            path
        )

        if got != EXPECTED[name]:

            raise RuntimeError(
                f"{name} SHA mismatch"
            )

        print(
            name,
            "SHA PASS = True",
        )

    if not QUERY_JSONL.exists():

        raise FileNotFoundError(
            QUERY_JSONL
        )

    protocol_sha = sha256(
        PROTOCOL
    )

    query_jsonl_sha = sha256(
        QUERY_JSONL
    )

    records = []

    with QUERY_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                records.append(
                    json.loads(line)
                )

    assert len(records) == P3_ROWS

    for i, rec in enumerate(
        records
    ):

        assert int(
            rec["p3_pos"]
        ) == i

    rows = np.asarray(
        [
            int(
                x["source_row_index"]
            )
            for x in records
        ],
        dtype=np.int64,
    )

    gt = np.asarray(
        [
            int(
                x["ground_truth_code_index"]
            )
            for x in records
        ],
        dtype=np.int32,
    )

    shards = np.asarray(
        [
            int(
                x["shard_id"]
            )
            for x in records
        ],
        dtype=np.int16,
    )

    queries = [
        x["raw_query"]
        for x in records
    ]

    assert np.array_equal(
        rows,
        gt.astype(np.int64),
    )

    b = load_builder()

    assert b.QWEN3_MAX_LENGTH == 512
    assert b.QWEN3_EMBED_DIM == DIM
    assert b.QWEN3_DTYPE == "bfloat16"
    assert b.QWEN3_POOLING == "last_token_l2"

    device = torch.device(
        "cuda"
    )

    print()
    print(
        "Loading historical Qwen model..."
    )

    model, tokenizer = b.load_qwen3(
        MODEL,
        device,
    )

    print(
        "model dtype =",
        next(
            model.parameters()
        ).dtype,
    )

    print(
        "padding side =",
        tokenizer.padding_side,
    )

    print(
        "attention implementation =",
        getattr(
            model.config,
            "_attn_implementation",
            None,
        ),
    )

    blocks = []

    print()
    print(
        "Encoding frozen P3 queries..."
    )

    for start in range(
        0,
        P3_ROWS,
        BATCH,
    ):

        end = min(
            start + BATCH,
            P3_ROWS,
        )

        emb = b.encode_qwen3_adaptive(
            model,
            tokenizer,
            queries[
                start:end
            ],
            device=device,
            is_query=True,
        )

        blocks.append(
            emb
        )

        if (
            end % 400 == 0
            or
            end == P3_ROWS
        ):

            print(
                "P3 query encoding:",
                end,
                "/",
                P3_ROWS,
            )

    q = np.concatenate(
        blocks,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    assert q.shape == (
        P3_ROWS,
        DIM,
    )

    assert np.isfinite(
        q
    ).all()

    atomic_npy(
        QUERY_EMB,
        q,
    )

    query_emb_sha = sha256(
        QUERY_EMB
    )

    print()
    print(
        "query embeddings =",
        QUERY_EMB,
    )

    print(
        "query embeddings SHA256 =",
        query_emb_sha,
    )

    del model
    del tokenizer
    del blocks

    gc.collect()
    torch.cuda.empty_cache()

    code = np.load(
        CODE_CACHE,
        mmap_mode="r",
        allow_pickle=False,
    )

    assert code.shape == (
        CODE_ROWS,
        DIM,
    )

    assert code.dtype == np.float32

    print()
    print(
        "Retrieving frozen P3 Top500..."
    )

    scores, indices = b.retrieve_embedding_topk(
        query_embeddings=q,
        code_embeddings=code,
        device=device,
        top_k=TOPK,
        query_chunk_size=QUERY_CHUNK,
        code_chunk_size=CODE_CHUNK,
        description="P3 historical-exact Qwen Top500",
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    indices = np.asarray(
        indices,
        dtype=np.int32,
    )

    assert indices.shape == (
        P3_ROWS,
        TOPK,
    )

    assert scores.shape == (
        P3_ROWS,
        TOPK,
    )

    assert indices.min() >= 0
    assert indices.max() < CODE_ROWS
    assert np.isfinite(
        scores
    ).all()

    # Structural checks only — NO R@k / GT-rank outcome is computed here.
    duplicate_rows = int(
        sum(
            len(
                np.unique(
                    indices[i]
                )
            )
            != TOPK
            for i in range(
                P3_ROWS
            )
        )
    )

    if duplicate_rows != 0:

        raise RuntimeError(
            "Duplicate candidate ids detected within Top500."
        )

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "jters_p3_qwen3_top500_historical_v1"
        ),

        source_row_indices=
            rows,

        shard_ids=
            shards,

        ground_truth_code_indices=
            gt,

        qwen3_indices=
            indices,

        qwen3_scores=
            scores,

        p3_manifest_sha256=np.asarray(
            EXPECTED[
                "manifest"
            ]
        ),

        retrieval_protocol_sha256=np.asarray(
            protocol_sha
        ),

        query_jsonl_sha256=np.asarray(
            query_jsonl_sha
        ),

        query_embeddings_sha256=np.asarray(
            query_emb_sha
        ),

        code_embeddings_sha256=np.asarray(
            EXPECTED[
                "code_cache"
            ]
        ),

        builder_sha256=np.asarray(
            EXPECTED[
                "builder"
            ]
        ),

        encoder_batch_size=np.asarray(
            BATCH,
            dtype=np.int32,
        ),

        query_chunk_size=np.asarray(
            QUERY_CHUNK,
            dtype=np.int32,
        ),

        code_chunk_size=np.asarray(
            CODE_CHUNK,
            dtype=np.int32,
        ),

        top_k=np.asarray(
            TOPK,
            dtype=np.int32,
        ),

        valid_accessed=np.asarray(
            False
        ),

        test_accessed=np.asarray(
            False
        ),
    )

    retrieval_sha = sha256(
        OUT_NPZ
    )

    print()
    print("=" * 150)
    print("P3 RETRIEVAL FROZEN")
    print("=" * 150)

    print(
        "query JSONL SHA256 =",
        query_jsonl_sha,
    )

    print(
        "query embeddings SHA256 =",
        query_emb_sha,
    )

    print(
        "Top500 artifact =",
        OUT_NPZ,
    )

    print(
        "Top500 artifact SHA256 =",
        retrieval_sha,
    )

    print(
        "shape =",
        indices.shape,
    )

    print(
        "duplicate Top500 rows =",
        duplicate_rows,
    )

    print()
    print(
        "P3_QUERY_TEXT_ACCESSED = True"
    )

    print(
        "P3_QUERY_EMBEDDINGS_GENERATED = True"
    )

    print(
        "P3_RETRIEVAL_GENERATED = True"
    )

    print(
        "P3_ALGORITHM_OUTCOMES_OBSERVED = False"
    )

    print(
        "STATUS = P3 RETRIEVAL FROZEN; "
        "READY FOR FROZEN ALGORITHM EVALUATION"
    )


if __name__ == "__main__":
    main()
