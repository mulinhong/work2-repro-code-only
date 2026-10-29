#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import time

import numpy as np


ROOT = Path(
    "/root/autodl-tmp/work2-algorithm-innovation"
)

RCPS_SCRIPT = (
    ROOT
    / "fhex_v1/scripts/"
      "audit_bcr_a4_rcps.py"
)

A2 = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a2_exact_program_fact_bcct.npz"
)

A4 = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a4_rcps.npz"
)

AST_CACHE = (
    ROOT
    / "fhex_v1/cache/"
      "python_train_ast_signature_v0.npz"
)

MANIFEST = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_manifest_v1.npz"
)

P1 = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_qwen3_top500_v1.npz"
)

OUT_NPZ = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_confirmation_v1.npz"
)

OUT_JSON = (
    ROOT
    / "fhex_v1/prospective/"
      "rcps_p1_confirmation_v1.json"
)


# ============================================================
# Frozen hashes
# ============================================================

EXPECTED_A2_SHA = (
    "4ce63e6e5eabf2662a291ebf9064b923"
    "880df124d5afe617e14ee357c0248d85"
)

EXPECTED_A4_SHA = (
    "fb61f7a1b30c69698b9ee7365cb92cd"
    "39c4931b9849be5d25c4e4a314943beb5"
)

EXPECTED_MANIFEST_SHA = (
    "fa349f76dfb73808a7203eee4f4be4ef"
    "3feec71f3bc98b671869a4a165233592"
)

EXPECTED_P1_SHA = (
    "d829cba534a2089e1e680432a3ef0ca"
    "40497722ed98c0b0216018b3ef10f61f5"
)


# ============================================================
# Frozen prospective gates
# ============================================================

MIN_N = 60

MIN_RCPS_SUCCESS_RATE = 0.90

MIN_ABSOLUTE_GAIN_OVER_ATOMIC = 0.03

REQUIRED_RETENTION_RATE = 1.0

MIN_NONWORSE_SHARDS = 4

MIN_STRICT_BETTER_SHARDS = 2


E80 = 80
B = 20

UNKNOWN = -1


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


def require_sha(
    path,
    expected,
    name,
):

    actual = sha256_file(
        path
    )

    if actual != expected:

        raise AssertionError(
            (
                f"{name} SHA mismatch",
                actual,
                expected,
            )
        )

    print(
        f"{name} SHA verified =",
        actual,
    )


def load_rcps_module():

    spec = (
        importlib.util
        .spec_from_file_location(
            "frozen_rcps_a4_impl",
            RCPS_SCRIPT,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):

        raise RuntimeError(
            "Cannot import frozen RCPS script."
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def build_atomic_facts(
    candidates,
    candidate_valid,
    signature,
    allowed_ids,
):

    n = candidates.shape[0]

    facts = np.full(
        (
            n,
            E80,
            len(allowed_ids),
        ),
        UNKNOWN,
        dtype=np.int8,
    )

    presence = (
        signature[
            candidates
        ][
            :,
            :,
            allowed_ids
        ]
        >
        0.0
    )

    facts[
        candidate_valid
    ] = (
        presence[
            candidate_valid
        ]
        .astype(
            np.int8
        )
    )

    return facts


def find_gt_positions(
    candidates,
    gt,
):

    n = len(gt)

    pos = np.full(
        n,
        -1,
        dtype=np.int16,
    )

    for i in range(n):

        p = np.flatnonzero(
            candidates[
                i
            ]
            ==
            gt[
                i
            ]
        )

        if len(p) != 1:

            raise AssertionError(
                (
                    "GT must appear exactly once "
                    "in prospective Local80.",
                    i,
                    int(gt[i]),
                    len(p),
                )
            )

        pos[i] = int(
            p[0]
        )

    return pos


def run_atomic(
    mod,
    atomic_facts,
    candidate_valid,
    gt_pos,
    gt_valid,
):

    n = len(gt_pos)

    ranks = np.empty(
        n,
        dtype=np.int16,
    )

    retained = np.zeros(
        n,
        dtype=bool,
    )

    questions = np.zeros(
        n,
        dtype=np.int8,
    )

    for i in range(n):

        facts = atomic_facts[i]

        policy, s0, s1 = (
            mod.choose_bcct(
                facts,
                candidate_valid[i],
            )
        )

        (
            ranks[i],
            retained[i],
            questions[i],
        ) = mod.execute_policy(
            policy,
            facts,
            s0,
            s1,
            int(
                gt_pos[i]
            ),
            bool(
                gt_valid[i]
            ),
        )

    return (
        ranks,
        retained,
        questions,
    )


def run_rcps(
    mod,
    atomic_facts,
    candidate_valid,
    gt_pos,
    gt_valid,
    progress_label,
):

    n = len(gt_pos)

    ranks = np.empty(
        n,
        dtype=np.int16,
    )

    retained = np.zeros(
        n,
        dtype=bool,
    )

    questions = np.zeros(
        n,
        dtype=np.int8,
    )

    unique_p = np.zeros(
        n,
        dtype=np.int16,
    )

    first_global = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    second0_global = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    second1_global = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    t0 = time.perf_counter()

    for i in range(n):

        full_facts, _, _, _ = (
            mod.build_synth_library(
                atomic_facts[i]
            )
        )

        if full_facts.shape != (
            E80,
            1936,
        ):

            raise AssertionError(
                full_facts.shape
            )

        facts, keep = (
            mod.deduplicate_predicates(
                full_facts,
                candidate_valid[i],
            )
        )

        unique_p[i] = int(
            facts.shape[1]
        )

        policy, s0, s1 = (
            mod.choose_bcct(
                facts,
                candidate_valid[i],
            )
        )

        if policy is not None:

            f1 = int(
                policy[
                    "first_feature"
                ]
            )

            first_global[i] = int(
                keep[
                    f1
                ]
            )

            f20 = int(
                policy[
                    "second_zero"
                ]
            )

            f21 = int(
                policy[
                    "second_one"
                ]
            )

            if f20 >= 0:

                second0_global[i] = int(
                    keep[
                        f20
                    ]
                )

            if f21 >= 0:

                second1_global[i] = int(
                    keep[
                        f21
                    ]
                )

        (
            ranks[i],
            retained[i],
            questions[i],
        ) = mod.execute_policy(
            policy,
            facts,
            s0,
            s1,
            int(
                gt_pos[i]
            ),
            bool(
                gt_valid[i]
            ),
        )

        if (
            (i + 1) % 10 == 0
            or
            i == n - 1
        ):

            print(
                f"[{progress_label}] "
                f"{i+1}/{n} "
                f"R20={int(np.sum(ranks[:i+1] <= B))} "
                f"elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    return (
        ranks,
        retained,
        questions,
        unique_p,
        first_global,
        second0_global,
        second1_global,
    )


def main():

    t_all = time.perf_counter()

    print(
        "=" * 144
    )

    print(
        "RCPS PROSPECTIVE-P1 — FINAL CONFIRMATION"
    )

    print(
        "=" * 144
    )

    print(
        "Frozen RCPS-A0 method."
    )

    print(
        "No tuning, no new predicates, no gate changes."
    )

    # ========================================================
    # Verify immutable artifacts.
    # ========================================================

    require_sha(
        A2,
        EXPECTED_A2_SHA,
        "A2",
    )

    require_sha(
        A4,
        EXPECTED_A4_SHA,
        "A4/RCPS",
    )

    require_sha(
        MANIFEST,
        EXPECTED_MANIFEST_SHA,
        "P1 manifest",
    )

    # ========================================================
    # Load frozen implementation.
    # ========================================================

    mod = load_rcps_module()

    # ========================================================
    # AST cache / frozen feature family.
    # ========================================================

    az = np.load(
        AST_CACHE,
        allow_pickle=True,
    )

    all_names = az[
        "feature_names"
    ].astype(
        str
    )

    signature = az[
        "signature"
    ].astype(
        np.float32
    )

    ast_valid = az[
        "valid"
    ].astype(
        bool
    )

    allowed = np.asarray(
        [
            name
            !=
            "max_ast_depth"
            for name in all_names
        ],
        dtype=bool,
    )

    allowed_ids = np.flatnonzero(
        allowed
    )

    assert len(
        allowed_ids
    ) == 44

    # ========================================================
    # METHOD-INTEGRITY CHECK ON OLD DEVELOPMENT DATA.
    #
    # This occurs BEFORE P1 result artifact is opened.
    # ========================================================

    print()
    print(
        "=" * 144
    )

    print(
        "FROZEN METHOD INTEGRITY CHECK"
    )

    print(
        "=" * 144
    )

    a2 = np.load(
        A2,
        allow_pickle=True,
    )

    a4 = np.load(
        A4,
        allow_pickle=True,
    )

    dev_gt = a2[
        "gt"
    ].astype(
        np.int32
    )

    dev_candidates = a2[
        "candidates"
    ].astype(
        np.int32
    )

    dev_candidate_valid = a2[
        "candidate_ast_valid"
    ].astype(
        bool
    )

    dev_gt_valid = a2[
        "gt_ast_valid"
    ].astype(
        bool
    )

    dev_raw = a2[
        "raw_rank"
    ].astype(
        np.int16
    )

    dev_atomic_saved = a2[
        "bcct_rank"
    ].astype(
        np.int16
    )

    dev_rcps_saved = a4[
        "rcps_rank"
    ].astype(
        np.int16
    )

    dev_mask = (
        dev_raw <= E80
    )

    dev_ids = np.flatnonzero(
        dev_mask
    )

    assert len(
        dev_ids
    ) == 121

    dev_gt_pos = find_gt_positions(
        dev_candidates[
            dev_ids
        ],
        dev_gt[
            dev_ids
        ],
    )

    dev_atomic_facts = (
        build_atomic_facts(
            dev_candidates[
                dev_ids
            ],
            dev_candidate_valid[
                dev_ids
            ],
            signature,
            allowed_ids,
        )
    )

    (
        dev_atomic_now,
        _,
        _,
    ) = run_atomic(
        mod,
        dev_atomic_facts,
        dev_candidate_valid[
            dev_ids
        ],
        dev_gt_pos,
        dev_gt_valid[
            dev_ids
        ],
    )

    atomic_mismatch = int(
        np.sum(
            dev_atomic_now
            !=
            dev_atomic_saved[
                dev_ids
            ]
        )
    )

    print(
        "atomic rank mismatches =",
        atomic_mismatch,
    )

    if atomic_mismatch != 0:

        raise AssertionError(
            "Frozen atomic implementation drift."
        )

    (
        dev_rcps_now,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = run_rcps(
        mod,
        dev_atomic_facts,
        dev_candidate_valid[
            dev_ids
        ],
        dev_gt_pos,
        dev_gt_valid[
            dev_ids
        ],
        "DEV-INTEGRITY",
    )

    rcps_mismatch = int(
        np.sum(
            dev_rcps_now
            !=
            dev_rcps_saved[
                dev_ids
            ]
        )
    )

    print(
        "RCPS rank mismatches =",
        rcps_mismatch,
    )

    print(
        "stored RCPS R20 =",
        int(
            np.sum(
                dev_rcps_saved[
                    dev_ids
                ]
                <= B
            )
        ),
    )

    print(
        "recomputed RCPS R20 =",
        int(
            np.sum(
                dev_rcps_now <= B
            )
        ),
    )

    if rcps_mismatch != 0:

        raise AssertionError(
            "Frozen RCPS implementation drift."
        )

    print(
        "FROZEN METHOD INTEGRITY PASS."
    )

    # Free development arrays before P1.
    del (
        a2,
        a4,
        dev_atomic_facts,
        dev_atomic_now,
        dev_rcps_now,
    )

    # ========================================================
    # NOW open prospective P1 retrieval artifact.
    # ========================================================

    print()
    print(
        "=" * 144
    )

    print(
        "OPENING FROZEN PROSPECTIVE P1"
    )

    print(
        "=" * 144
    )

    require_sha(
        P1,
        EXPECTED_P1_SHA,
        "P1 retrieval",
    )

    p1 = np.load(
        P1,
        allow_pickle=True,
    )

    if bool(
        np.asarray(
            p1[
                "rcps_evaluated"
            ]
        ).item()
    ):

        raise AssertionError(
            "P1 says RCPS was already evaluated."
        )

    assert not bool(
        np.asarray(
            p1[
                "valid_accessed"
            ]
        ).item()
    )

    assert not bool(
        np.asarray(
            p1[
                "test_accessed"
            ]
        ).item()
    )

    manifest_sha = str(
        np.asarray(
            p1[
                "manifest_sha256"
            ]
        ).item()
    )

    frozen_method_sha = str(
        np.asarray(
            p1[
                "frozen_rcps_sha256"
            ]
        ).item()
    )

    assert (
        manifest_sha
        ==
        EXPECTED_MANIFEST_SHA
    )

    assert (
        frozen_method_sha
        ==
        EXPECTED_A4_SHA
    )

    p1_gt = p1[
        "ground_truth_code_indices"
    ].astype(
        np.int32
    )

    top500 = p1[
        "qwen3_indices"
    ].astype(
        np.int32
    )

    base_rank = p1[
        "base_rank_1indexed_capped501"
    ].astype(
        np.int16
    )

    shards_all = p1[
        "shard_ids"
    ].astype(
        np.int8
    )

    failure_mask = (
        base_rank > 100
    )

    local80_mask = (
        (base_rank >= 101)
        &
        (base_rank <= 180)
    )

    failure_n = int(
        failure_mask.sum()
    )

    local_n = int(
        local80_mask.sum()
    )

    print(
        "P1 Top100 failures =",
        failure_n,
    )

    print(
        "P1 Local80-recoverable =",
        local_n,
    )

    if local_n < MIN_N:

        raise AssertionError(
            (
                "Prospective sample below frozen "
                "minimum; result is INCONCLUSIVE.",
                local_n,
            )
        )

    ids = np.flatnonzero(
        local80_mask
    )

    candidates = top500[
        ids,
        100:180
    ]

    gt = p1_gt[
        ids
    ]

    shards = shards_all[
        ids
    ]

    raw_rank = (
        base_rank[
            ids
        ]
        -
        100
    ).astype(
        np.int16
    )

    assert np.all(
        (raw_rank >= 1)
        &
        (raw_rank <= 80)
    )

    gt_pos = find_gt_positions(
        candidates,
        gt,
    )

    assert np.array_equal(
        gt_pos.astype(
            np.int16
        )
        +
        1,
        raw_rank,
    )

    candidate_valid = ast_valid[
        candidates
    ]

    gt_valid = ast_valid[
        gt
    ]

    atomic_facts = (
        build_atomic_facts(
            candidates,
            candidate_valid,
            signature,
            allowed_ids,
        )
    )

    print(
        "candidate AST-valid fraction =",
        float(
            candidate_valid.mean()
        ),
    )

    print(
        "GT AST-valid fraction =",
        float(
            gt_valid.mean()
        ),
    )

    # ========================================================
    # Frozen atomic BCCT.
    # ========================================================

    print()
    print(
        "=" * 144
    )

    print(
        "P1 FROZEN ATOMIC BCCT"
    )

    print(
        "=" * 144
    )

    (
        atomic_rank,
        atomic_retained,
        atomic_questions,
    ) = run_atomic(
        mod,
        atomic_facts,
        candidate_valid,
        gt_pos,
        gt_valid,
    )

    print(
        "atomic R20 =",
        int(
            np.sum(
                atomic_rank <= B
            )
        ),
        "/",
        local_n,
    )

    # ========================================================
    # Frozen RCPS.
    # ========================================================

    print()
    print(
        "=" * 144
    )

    print(
        "P1 FROZEN RCPS"
    )

    print(
        "=" * 144
    )

    (
        rcps_rank,
        rcps_retained,
        rcps_questions,
        unique_p,
        first_global,
        second0_global,
        second1_global,
    ) = run_rcps(
        mod,
        atomic_facts,
        candidate_valid,
        gt_pos,
        gt_valid,
        "P1-RCPS",
    )

    # ========================================================
    # Metrics.
    # ========================================================

    raw_metrics = mod.summarize(
        raw_rank
    )

    atomic_metrics = mod.summarize(
        atomic_rank
    )

    rcps_metrics = mod.summarize(
        rcps_rank
    )

    raw20 = int(
        np.sum(
            raw_rank <= B
        )
    )

    atomic20 = int(
        np.sum(
            atomic_rank <= B
        )
    )

    rcps20 = int(
        np.sum(
            rcps_rank <= B
        )
    )

    atomic_rate = (
        atomic20
        /
        local_n
    )

    rcps_rate = (
        rcps20
        /
        local_n
    )

    abs_gain = (
        rcps_rate
        -
        atomic_rate
    )

    retention_n = int(
        rcps_retained.sum()
    )

    retention_rate = (
        retention_n
        /
        local_n
    )

    atomic_hit = (
        atomic_rank <= B
    )

    rcps_hit = (
        rcps_rank <= B
    )

    both = int(
        np.sum(
            atomic_hit
            &
            rcps_hit
        )
    )

    rcps_only = int(
        np.sum(
            rcps_hit
            &
            ~atomic_hit
        )
    )

    atomic_only = int(
        np.sum(
            atomic_hit
            &
            ~rcps_hit
        )
    )

    neither = int(
        np.sum(
            ~atomic_hit
            &
            ~rcps_hit
        )
    )

    print()
    print(
        "=" * 144
    )

    print(
        "P1 PROSPECTIVE RESULTS"
    )

    print(
        "=" * 144
    )

    print(
        "RAW         =",
        raw_metrics,
    )

    print(
        "ATOMIC BCCT =",
        atomic_metrics,
    )

    print(
        "RCPS        =",
        rcps_metrics,
    )

    print()

    print(
        "Local80 recoverable =",
        local_n,
    )

    print(
        "Atomic R20 rate =",
        atomic_rate,
        f"({atomic20}/{local_n})",
    )

    print(
        "RCPS R20 rate =",
        rcps_rate,
        f"({rcps20}/{local_n})",
    )

    print(
        "Absolute gain =",
        abs_gain,
    )

    print(
        "RCPS GT retained =",
        retention_n,
        "/",
        local_n,
    )

    print()

    print(
        "paired both =",
        both,
    )

    print(
        "RCPS-only wins =",
        rcps_only,
    )

    print(
        "Atomic-only wins =",
        atomic_only,
    )

    print(
        "neither =",
        neither,
    )

    print()

    print(
        "mean unique predicates =",
        float(
            unique_p.mean()
        ),
    )

    print(
        "median unique predicates =",
        float(
            np.median(
                unique_p
            )
        ),
    )

    print(
        "mean RCPS questions =",
        float(
            rcps_questions.mean()
        ),
    )

    # ========================================================
    # Prospective shard stability.
    # ========================================================

    print()
    print(
        "=" * 144
    )

    print(
        "P1 PER-SHARD CONFIRMATION"
    )

    print(
        "=" * 144
    )

    shard_result = {}

    nonworse_shards = 0
    strict_better_shards = 0

    for shard in range(5):

        m = (
            shards == shard
        )

        n_s = int(
            m.sum()
        )

        raw_s = int(
            np.sum(
                m
                &
                (
                    raw_rank <= B
                )
            )
        )

        atomic_s = int(
            np.sum(
                m
                &
                (
                    atomic_rank <= B
                )
            )
        )

        rcps_s = int(
            np.sum(
                m
                &
                (
                    rcps_rank <= B
                )
            )
        )

        delta_s = (
            rcps_s
            -
            atomic_s
        )

        if rcps_s >= atomic_s:

            nonworse_shards += 1

        if rcps_s > atomic_s:

            strict_better_shards += 1

        shard_result[
            str(shard)
        ] = {
            "n":
                n_s,

            "raw_R20":
                raw_s,

            "atomic_R20":
                atomic_s,

            "rcps_R20":
                rcps_s,

            "atomic_rate":
                (
                    atomic_s / n_s
                    if n_s
                    else None
                ),

            "rcps_rate":
                (
                    rcps_s / n_s
                    if n_s
                    else None
                ),

            "delta_count":
                delta_s,
        }

        print(
            f"shard={shard}",
            f"n={n_s}",
            f"RAW={raw_s}",
            f"ATOMIC={atomic_s}",
            f"RCPS={rcps_s}",
            f"delta={delta_s}",
        )

    # ========================================================
    # Frozen prospective decision.
    # ========================================================

    sample_gate = (
        local_n >= MIN_N
    )

    success_gate = (
        rcps_rate
        >=
        MIN_RCPS_SUCCESS_RATE
    )

    gain_gate = (
        abs_gain
        >=
        MIN_ABSOLUTE_GAIN_OVER_ATOMIC
    )

    retention_gate = (
        retention_rate
        >=
        REQUIRED_RETENTION_RATE
    )

    nonworse_gate = (
        nonworse_shards
        >=
        MIN_NONWORSE_SHARDS
    )

    better_gate = (
        strict_better_shards
        >=
        MIN_STRICT_BETTER_SHARDS
    )

    confirmed = (
        sample_gate
        and
        success_gate
        and
        gain_gate
        and
        retention_gate
        and
        nonworse_gate
        and
        better_gate
    )

    print()
    print(
        "=" * 144
    )

    print(
        "FROZEN PROSPECTIVE DECISION"
    )

    print(
        "=" * 144
    )

    print(
        "sample-size gate:",
        sample_gate,
        "|",
        local_n,
        ">=",
        MIN_N,
    )

    print(
        "RCPS success-rate gate:",
        success_gate,
        "|",
        rcps_rate,
        ">=",
        MIN_RCPS_SUCCESS_RATE,
    )

    print(
        "paired absolute-gain gate:",
        gain_gate,
        "|",
        abs_gain,
        ">=",
        MIN_ABSOLUTE_GAIN_OVER_ATOMIC,
    )

    print(
        "retention gate:",
        retention_gate,
        "|",
        retention_rate,
        ">=",
        REQUIRED_RETENTION_RATE,
    )

    print(
        "shard nonworse gate:",
        nonworse_gate,
        "|",
        nonworse_shards,
        ">=",
        MIN_NONWORSE_SHARDS,
    )

    print(
        "shard strict-better gate:",
        better_gate,
        "|",
        strict_better_shards,
        ">=",
        MIN_STRICT_BETTER_SHARDS,
    )

    print()

    if confirmed:

        diagnosis = (
            "RCPS PROSPECTIVELY CONFIRMED. THE FROZEN "
            "PAIRWISE LOGICAL PREDICATE-SYNTHESIS METHOD "
            "REPLICATES ITS FIXED-BUDGET REPAIR ADVANTAGE "
            "ON 12,000 PRE-REGISTERED, PREVIOUSLY UNSEEN "
            "TRAIN QUERIES. FREEZE P1 AS CONFIRMATORY "
            "EVIDENCE. DO NOT MODIFY RCPS USING P1."
        )

    else:

        diagnosis = (
            "RCPS NOT PROSPECTIVELY CONFIRMED UNDER THE "
            "PRE-REGISTERED P1 GATES. ACCEPT THIS AS THE "
            "CONFIRMATORY RESULT. DO NOT CHANGE GATES, "
            "QUESTION LIBRARY, BCCT OBJECTIVE, OR REGENERATE "
            "P1. THE DEVELOPMENT MECHANISM RESULTS REMAIN "
            "VALID AS DEVELOPMENT EVIDENCE, BUT RCPS CANNOT "
            "BE CLAIMED AS PROSPECTIVELY CONFIRMED."
        )

    print(
        diagnosis
    )

    # ========================================================
    # Save once.
    # ========================================================

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "work2_rcps_prospective_p1_confirmation_v1"
        ),

        p1_local80_indices=
            ids.astype(
                np.int32
            ),

        shard_ids=
            shards.astype(
                np.int8
            ),

        gt=
            gt.astype(
                np.int32
            ),

        candidates=
            candidates.astype(
                np.int32
            ),

        raw_rank=
            raw_rank.astype(
                np.int16
            ),

        atomic_rank=
            atomic_rank.astype(
                np.int16
            ),

        rcps_rank=
            rcps_rank.astype(
                np.int16
            ),

        atomic_retained=
            atomic_retained.astype(
                bool
            ),

        rcps_retained=
            rcps_retained.astype(
                bool
            ),

        atomic_questions=
            atomic_questions.astype(
                np.int8
            ),

        rcps_questions=
            rcps_questions.astype(
                np.int8
            ),

        unique_predicate_count=
            unique_p.astype(
                np.int16
            ),

        first_predicate_global=
            first_global.astype(
                np.int32
            ),

        second_if_absent_global=
            second0_global.astype(
                np.int32
            ),

        second_if_present_global=
            second1_global.astype(
                np.int32
            ),

        frozen_rcps_sha256=np.asarray(
            EXPECTED_A4_SHA
        ),

        frozen_manifest_sha256=np.asarray(
            EXPECTED_MANIFEST_SHA
        ),

        frozen_p1_retrieval_sha256=np.asarray(
            EXPECTED_P1_SHA
        ),
    )

    out_sha = sha256_file(
        OUT_NPZ
    )

    runtime = (
        time.perf_counter()
        -
        t_all
    )

    result = {
        "name":
            "RCPS Prospective P1 Confirmation",

        "frozen_artifacts": {
            "A2_sha256":
                EXPECTED_A2_SHA,

            "RCPS_A4_sha256":
                EXPECTED_A4_SHA,

            "manifest_sha256":
                EXPECTED_MANIFEST_SHA,

            "P1_retrieval_sha256":
                EXPECTED_P1_SHA,

            "confirmation_sha256":
                out_sha,
        },

        "population": {
            "P1_queries":
                12000,

            "top100_failures":
                failure_n,

            "local80_recoverable":
                local_n,

            "candidate_ast_valid_fraction":
                float(
                    candidate_valid.mean()
                ),

            "gt_ast_valid_fraction":
                float(
                    gt_valid.mean()
                ),
        },

        "metrics": {
            "raw":
                raw_metrics,

            "atomic_bcct":
                atomic_metrics,

            "rcps":
                rcps_metrics,

            "atomic_R20_rate":
                atomic_rate,

            "rcps_R20_rate":
                rcps_rate,

            "absolute_gain":
                abs_gain,

            "retention_rate":
                retention_rate,
        },

        "paired": {
            "both":
                both,

            "rcps_only":
                rcps_only,

            "atomic_only":
                atomic_only,

            "neither":
                neither,
        },

        "shards":
            shard_result,

        "frozen_gate": {
            "minimum_n":
                MIN_N,

            "minimum_rcps_success_rate":
                MIN_RCPS_SUCCESS_RATE,

            "minimum_absolute_gain":
                MIN_ABSOLUTE_GAIN_OVER_ATOMIC,

            "required_retention_rate":
                REQUIRED_RETENTION_RATE,

            "minimum_nonworse_shards":
                MIN_NONWORSE_SHARDS,

            "minimum_strict_better_shards":
                MIN_STRICT_BETTER_SHARDS,
        },

        "decision": {
            "sample_gate":
                bool(
                    sample_gate
                ),

            "success_gate":
                bool(
                    success_gate
                ),

            "gain_gate":
                bool(
                    gain_gate
                ),

            "retention_gate":
                bool(
                    retention_gate
                ),

            "nonworse_shards_gate":
                bool(
                    nonworse_gate
                ),

            "strict_better_shards_gate":
                bool(
                    better_gate
                ),

            "prospectively_confirmed":
                bool(
                    confirmed
                ),
        },

        "secondary": {
            "mean_questions":
                float(
                    rcps_questions.mean()
                ),

            "mean_unique_predicates":
                float(
                    unique_p.mean()
                ),

            "median_unique_predicates":
                float(
                    np.median(
                        unique_p
                    )
                ),
        },

        "diagnosis":
            diagnosis,

        "runtime_seconds":
            float(
                runtime
            ),

        "protocol": [
            "One-shot prospective confirmation.",
            "12,000 P1 queries were frozen before retrieval outcome access.",
            "Zero overlap with the 12,000 development queries.",
            "P1 retrieval artifact frozen before method evaluation.",
            "Frozen RCPS-A0 implementation integrity was rechecked on the development artifact before P1 was opened.",
            "No RCPS modification after P1 manifest freeze.",
            "No threshold modification.",
            "No question-family modification.",
            "No NOT predicates.",
            "No triple predicates.",
            "No score fusion.",
            "No reranking training.",
            "Primary endpoint is fixed-budget B20 repair among Local80-recoverable Top100 failures.",
            "R1/R2/R5/R10/MRR are secondary and reported transparently.",
            "P1 must not be regenerated or reused for method development after this evaluation.",
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
        "=" * 144
    )

    print(
        "P1 CONFIRMATION ARTIFACT FROZEN"
    )

    print(
        "=" * 144
    )

    print(
        "prospectively confirmed =",
        confirmed,
    )

    print(
        "artifact =",
        OUT_NPZ,
    )

    print(
        "artifact SHA256 =",
        out_sha,
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
        "P1 IS NOW CONSUMED."
    )

    print(
        "Do not modify RCPS using this result."
    )


if __name__ == "__main__":
    main()
