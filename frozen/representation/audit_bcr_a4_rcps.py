#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")

A2 = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a2_exact_program_fact_bcct.npz"
)

AST_CACHE = (
    ROOT
    / "fhex_v1/cache/"
      "python_train_ast_signature_v0.npz"
)

OUT_NPZ = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a4_rcps.npz"
)

OUT_JSON = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a4_rcps_metrics.json"
)


E80 = 80
B = 20

UNKNOWN = -1
ABSENT = 0
PRESENT = 1

# ============================================================
# Frozen gates
# ============================================================

MIN_RCPS_R20 = 115
MIN_GAIN_OVER_ATOMIC_BCCT = 3
MIN_RETENTION = 121
MIN_NONWORSE_FOLDS = 4
MIN_STRICT_BETTER_FOLDS = 2


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:
            x = f.read(1024 * 1024)

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def summarize(ranks):

    ranks = np.asarray(ranks)

    out = {}

    for k in [1, 2, 5, 10, 20, 40, 80]:

        out[f"R{k}"] = int(
            np.sum(ranks <= k)
        )

    rr = np.zeros(
        len(ranks),
        dtype=np.float64,
    )

    valid = ranks <= E80

    rr[valid] = (
        1.0
        /
        ranks[valid].astype(np.float64)
    )

    out["MRR"] = float(rr.mean())

    return out


# ============================================================
# Three-valued program-fact logic
# ============================================================

def tri_and(a, b):

    out = np.full(
        a.shape,
        UNKNOWN,
        dtype=np.int8,
    )

    out[
        (a == ABSENT)
        |
        (b == ABSENT)
    ] = ABSENT

    out[
        (a == PRESENT)
        &
        (b == PRESENT)
    ] = PRESENT

    return out


def tri_or(a, b):

    out = np.full(
        a.shape,
        UNKNOWN,
        dtype=np.int8,
    )

    out[
        (a == PRESENT)
        |
        (b == PRESENT)
    ] = PRESENT

    out[
        (a == ABSENT)
        &
        (b == ABSENT)
    ] = ABSENT

    return out


def build_synth_library(base_facts):

    """
    Frozen library:

        44 atomic
        C(44,2) AND
        C(44,2) OR

    Total = 1936.
    """

    n, f = base_facts.shape

    assert f == 44

    total = (
        f
        +
        f * (f - 1) // 2
        +
        f * (f - 1) // 2
    )

    facts = np.empty(
        (n, total),
        dtype=np.int8,
    )

    op = np.empty(
        total,
        dtype=np.int8,
    )

    left = np.empty(
        total,
        dtype=np.int16,
    )

    right = np.full(
        total,
        -1,
        dtype=np.int16,
    )

    p = 0

    # atomic
    for i in range(f):

        facts[:, p] = base_facts[:, i]

        op[p] = 0
        left[p] = i

        p += 1

    # conjunction
    for i in range(f):

        for j in range(i + 1, f):

            facts[:, p] = tri_and(
                base_facts[:, i],
                base_facts[:, j],
            )

            op[p] = 1
            left[p] = i
            right[p] = j

            p += 1

    # disjunction
    for i in range(f):

        for j in range(i + 1, f):

            facts[:, p] = tri_or(
                base_facts[:, i],
                base_facts[:, j],
            )

            op[p] = 2
            left[p] = i
            right[p] = j

            p += 1

    assert p == 1936

    return (
        facts,
        op,
        left,
        right,
    )


def deduplicate_predicates(
    facts,
    valid_candidate,
):

    """
    Exact query-local partition deduplication.

    Two predicates are duplicates only if their complete
    ternary vectors over all 80 candidates are identical.

    This does NOT inspect GT or retrieval outcome.
    """

    keep = []

    seen = set()

    valid_ids = np.flatnonzero(
        valid_candidate
    )

    for p in range(
        facts.shape[1]
    ):

        col = facts[:, p]

        # Predicate must actually have both answers among
        # valid candidate hypotheses.
        v = col[valid_ids]

        if not (
            np.any(v == ABSENT)
            and
            np.any(v == PRESENT)
        ):

            continue

        key = col.tobytes()

        if key in seen:
            continue

        seen.add(key)

        keep.append(p)

    keep = np.asarray(
        keep,
        dtype=np.int32,
    )

    return (
        facts[:, keep],
        keep,
    )


# ============================================================
# Exact BCCT solver
# ============================================================

def build_survival_masks(facts):

    # [P,80]
    x = facts.T

    survive0 = (
        (x == ABSENT)
        |
        (x == UNKNOWN)
    )

    survive1 = (
        (x == PRESENT)
        |
        (x == UNKNOWN)
    )

    return (
        survive0,
        survive1,
    )


def evaluate_stop(
    current_mask,
    ids,
):

    prefix = np.cumsum(
        current_mask,
        dtype=np.int16,
    )

    ranks = prefix[ids]

    hit = ranks <= B

    coverage = int(
        hit.sum()
    )

    gain = int(
        np.sum(
            (
                ids[hit]
                +
                1
            )
            -
            ranks[hit]
        )
    )

    return {
        "feature":
            -1,

        "coverage":
            coverage,

        "gain":
            gain,

        "worst_leaf":
            int(
                current_mask.sum()
            ),
    }


def choose_second(
    facts,
    valid_candidate,
    survive0,
    survive1,
    current_mask,
):

    """
    Exact vectorized second-question BCCT solver.

    Same objective as A2:

      maximize certified coverage
      then rank reduction
      then minimize worst leaf
      then deterministic predicate id
    """

    valid_ids = np.flatnonzero(
        valid_candidate
        &
        current_mask
    )

    best = evaluate_stop(
        current_mask,
        valid_ids,
    )

    if (
        len(valid_ids) == 0
        or
        int(current_mask.sum()) <= B
    ):

        return best

    # [P,H]
    answers = facts[
        valid_ids
    ].T

    has0 = np.any(
        answers == ABSENT,
        axis=1,
    )

    has1 = np.any(
        answers == PRESENT,
        axis=1,
    )

    valid_pred = (
        has0
        &
        has1
    )

    if not np.any(valid_pred):

        return best

    # [P,80]
    masks0 = (
        survive0
        &
        current_mask[
            None,
            :
        ]
    )

    masks1 = (
        survive1
        &
        current_mask[
            None,
            :
        ]
    )

    prefix0 = np.cumsum(
        masks0,
        axis=1,
        dtype=np.int16,
    )

    prefix1 = np.cumsum(
        masks1,
        axis=1,
        dtype=np.int16,
    )

    # [P,H]
    rank0 = prefix0[
        :,
        valid_ids
    ]

    rank1 = prefix1[
        :,
        valid_ids
    ]

    ranks = np.where(
        answers == ABSENT,
        rank0,
        rank1,
    )

    hit = ranks <= B

    coverage = np.sum(
        hit,
        axis=1,
    ).astype(
        np.int32
    )

    original_rank = (
        valid_ids[
            None,
            :
        ]
        +
        1
    )

    gain = np.sum(
        np.where(
            hit,
            original_rank
            -
            ranks,
            0,
        ),
        axis=1,
    ).astype(
        np.int32
    )

    worst_leaf = np.maximum(
        masks0.sum(
            axis=1
        ),
        masks1.sum(
            axis=1
        ),
    ).astype(
        np.int32
    )

    candidates = np.flatnonzero(
        valid_pred
    )

    if len(candidates) == 0:

        return best

    # maximize:
    #   coverage
    #   gain
    #   -worst_leaf
    #   -feature id
    #
    order = np.lexsort(
        (
            candidates,
            worst_leaf[
                candidates
            ],
            -gain[
                candidates
            ],
            -coverage[
                candidates
            ],
        )
    )

    p = int(
        candidates[
            order[0]
        ]
    )

    cand = {
        "feature":
            p,

        "coverage":
            int(
                coverage[p]
            ),

        "gain":
            int(
                gain[p]
            ),

        "worst_leaf":
            int(
                worst_leaf[p]
            ),
    }

    key_c = (
        cand["coverage"],
        cand["gain"],
        -cand["worst_leaf"],
        -cand["feature"],
    )

    key_b = (
        best["coverage"],
        best["gain"],
        -best["worst_leaf"],
        -best["feature"],
    )

    if key_c > key_b:

        best = cand

    return best


def choose_bcct(
    facts,
    valid_candidate,
):

    survive0, survive1 = (
        build_survival_masks(
            facts
        )
    )

    P = facts.shape[1]

    full = np.ones(
        E80,
        dtype=bool,
    )

    valid_ids = np.flatnonzero(
        valid_candidate
    )

    best = None

    # Cache second-stage result by actual branch mask.
    cache = {}

    def get_branch(mask):

        key = np.packbits(
            mask
        ).tobytes()

        if key not in cache:

            cache[key] = choose_second(
                facts,
                valid_candidate,
                survive0,
                survive1,
                mask,
            )

        return cache[key]

    for f1 in range(P):

        a = facts[
            valid_ids,
            f1
        ]

        if not (
            np.any(a == ABSENT)
            and
            np.any(a == PRESENT)
        ):

            continue

        mask0 = survive0[f1]
        mask1 = survive1[f1]

        p0 = get_branch(
            mask0
        )

        p1 = get_branch(
            mask1
        )

        cand = {
            "first_feature":
                f1,

            "second_zero":
                int(
                    p0["feature"]
                ),

            "second_one":
                int(
                    p1["feature"]
                ),

            "coverage":
                int(
                    p0["coverage"]
                    +
                    p1["coverage"]
                ),

            "gain":
                int(
                    p0["gain"]
                    +
                    p1["gain"]
                ),

            "worst_leaf":
                max(
                    int(
                        p0["worst_leaf"]
                    ),
                    int(
                        p1["worst_leaf"]
                    ),
                ),
        }

        if best is None:

            best = cand

            continue

        key_c = (
            cand["coverage"],
            cand["gain"],
            -cand["worst_leaf"],
            -cand["first_feature"],
        )

        key_b = (
            best["coverage"],
            best["gain"],
            -best["worst_leaf"],
            -best["first_feature"],
        )

        if key_c > key_b:

            best = cand

    return (
        best,
        survive0,
        survive1,
    )


def execute_policy(
    policy,
    facts,
    survive0,
    survive1,
    gt_pos,
    gt_valid,
):

    # No exact user-answer facts available.
    if (
        not gt_valid
        or
        policy is None
    ):

        return (
            gt_pos + 1,
            True,
            0,
        )

    f1 = int(
        policy[
            "first_feature"
        ]
    )

    a1 = int(
        facts[
            gt_pos,
            f1
        ]
    )

    if a1 == ABSENT:

        mask = survive0[
            f1
        ].copy()

        f2 = int(
            policy[
                "second_zero"
            ]
        )

    else:

        mask = survive1[
            f1
        ].copy()

        f2 = int(
            policy[
                "second_one"
            ]
        )

    questions = 1

    if (
        int(mask.sum()) > B
        and
        f2 >= 0
    ):

        a2 = int(
            facts[
                gt_pos,
                f2
            ]
        )

        if a2 == ABSENT:

            mask &= survive0[
                f2
            ]

        else:

            mask &= survive1[
                f2
            ]

        questions = 2

    retained = bool(
        mask[
            gt_pos
        ]
    )

    if not retained:

        return (
            E80 + 1,
            False,
            questions,
        )

    rank = int(
        np.sum(
            mask[
                :gt_pos + 1
            ]
        )
    )

    return (
        rank,
        True,
        questions,
    )


# ============================================================
# Main
# ============================================================

def main():

    t0 = time.perf_counter()

    print(
        "=" * 140
    )

    print(
        "BCR-A4 / RCPS-A0:"
        " REPAIR-CERTIFICATE PREDICATE SYNTHESIS AUDIT"
    )

    print(
        "=" * 140
    )

    print(
        "Only question representation changes."
    )

    print(
        "BCCT objective / B20 / two-question budget are frozen."
    )

    print(
        "Predicate library = atomic + pairwise AND + pairwise OR."
    )

    print(
        "No NOT, no triple predicates, no learned weights."
    )

    a2 = np.load(
        A2,
        allow_pickle=True,
    )

    folds = a2[
        "fold_ids"
    ].astype(
        np.int8
    )

    gt = a2[
        "gt"
    ].astype(
        np.int32
    )

    candidates = a2[
        "candidates"
    ].astype(
        np.int32
    )

    feature_names = a2[
        "feature_names"
    ].astype(
        str
    )

    candidate_valid = a2[
        "candidate_ast_valid"
    ].astype(
        bool
    )

    gt_valid = a2[
        "gt_ast_valid"
    ].astype(
        bool
    )

    raw_rank = a2[
        "raw_rank"
    ].astype(
        np.int16
    )

    gbs_rank = a2[
        "gbs_rank"
    ].astype(
        np.int16
    )

    atomic_bcct_rank = a2[
        "bcct_rank"
    ].astype(
        np.int16
    )

    assert len(feature_names) == 44

    assert int(
        np.sum(
            raw_rank <= B
        )
    ) == 40

    assert int(
        np.sum(
            gbs_rank <= B
        )
    ) == 110

    assert int(
        np.sum(
            atomic_bcct_rank <= B
        )
    ) == 112

    # --------------------------------------------------------
    # Exact AST facts
    # --------------------------------------------------------

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
            name != "max_ast_depth"
            for name in all_names
        ],
        dtype=bool,
    )

    allowed_ids = np.flatnonzero(
        allowed
    )

    assert np.array_equal(
        all_names[
            allowed_ids
        ],
        feature_names,
    )

    n = len(gt)

    atomic_facts = np.full(
        (
            n,
            E80,
            44,
        ),
        UNKNOWN,
        dtype=np.int8,
    )

    exact_presence = (
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

    valid_check = ast_valid[
        candidates
    ]

    assert np.array_equal(
        valid_check,
        candidate_valid,
    )

    atomic_facts[
        candidate_valid
    ] = (
        exact_presence[
            candidate_valid
        ]
        .astype(
            np.int8
        )
    )

    # --------------------------------------------------------
    # GT positions
    # --------------------------------------------------------

    recoverable = (
        raw_rank <= E80
    )

    gt_pos = np.full(
        n,
        -1,
        dtype=np.int16,
    )

    for i in np.flatnonzero(
        recoverable
    ):

        p = np.flatnonzero(
            candidates[i]
            ==
            gt[i]
        )

        assert len(p) == 1

        gt_pos[i] = int(
            p[0]
        )

    assert int(
        recoverable.sum()
    ) == 121

    # ========================================================
    # First verify exact atomic BCCT reproduction.
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "ATOMIC BCCT SOLVER EQUIVALENCE CHECK"
    )

    print(
        "=" * 140
    )

    atomic_recomputed = raw_rank.copy()

    selected = np.flatnonzero(
        recoverable
    )

    for count, i in enumerate(
        selected,
        start=1,
    ):

        facts = atomic_facts[i]

        policy, s0, s1 = choose_bcct(
            facts,
            candidate_valid[i],
        )

        (
            atomic_recomputed[i],
            _,
            _,
        ) = execute_policy(
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
            count % 25 == 0
            or
            count == len(selected)
        ):

            print(
                f"[atomic {count:>3}/121]",
                flush=True,
            )

    mismatch = np.flatnonzero(
        atomic_recomputed[
            recoverable
        ]
        !=
        atomic_bcct_rank[
            recoverable
        ]
    )

    atomic_r20 = int(
        np.sum(
            atomic_recomputed <= B
        )
    )

    print()
    print(
        "stored atomic BCCT R20 =",
        int(
            np.sum(
                atomic_bcct_rank <= B
            )
        ),
    )

    print(
        "recomputed atomic BCCT R20 =",
        atomic_r20,
    )

    print(
        "exact-rank mismatches on 121 =",
        len(mismatch),
    )

    # Rank tie differences are not acceptable here:
    # RCPS must use exactly the same BCCT solver semantics.
    if len(mismatch) != 0:

        raise AssertionError(
            (
                "Atomic BCCT solver does not exactly reproduce A2. "
                "Abort RCPS before observing synthesized results.",
                len(mismatch),
            )
        )

    print(
        "ATOMIC SOLVER EQUIVALENCE PASS."
    )

    # ========================================================
    # Synthesized predicate BCCT
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "SYNTHESIZED-PREDICATE BCCT"
    )

    print(
        "=" * 140
    )

    rcps_rank = raw_rank.copy()

    rcps_retained = np.zeros(
        n,
        dtype=bool,
    )

    rcps_questions = np.zeros(
        n,
        dtype=np.int8,
    )

    first_global = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    second_zero_global = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    second_one_global = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    unique_predicate_count = np.zeros(
        n,
        dtype=np.int16,
    )

    for count, i in enumerate(
        selected,
        start=1,
    ):

        full_facts, _, _, _ = (
            build_synth_library(
                atomic_facts[i]
            )
        )

        assert full_facts.shape == (
            E80,
            1936,
        )

        facts, keep = (
            deduplicate_predicates(
                full_facts,
                candidate_valid[i],
            )
        )

        unique_predicate_count[i] = int(
            facts.shape[1]
        )

        policy, s0, s1 = choose_bcct(
            facts,
            candidate_valid[i],
        )

        if policy is not None:

            f1 = int(
                policy[
                    "first_feature"
                ]
            )

            first_global[i] = int(
                keep[f1]
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

                second_zero_global[i] = int(
                    keep[f20]
                )

            if f21 >= 0:

                second_one_global[i] = int(
                    keep[f21]
                )

        (
            rcps_rank[i],
            rcps_retained[i],
            rcps_questions[i],
        ) = execute_policy(
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
            count % 5 == 0
            or
            count == len(selected)
        ):

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            current20 = int(
                np.sum(
                    rcps_rank[
                        selected[
                            :count
                        ]
                    ]
                    <=
                    B
                )
            )

            print(
                f"[synth {count:>3}/121]",
                f"uniqueP={facts.shape[1]}",
                f"local_R20={current20}",
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    # Nonrecoverable queries do not have a Local80 GT;
    # they are conservatively untouched.
    #
    # Every recoverable GT must survive its own exact answers.
    retained_n = int(
        np.sum(
            recoverable
            &
            rcps_retained
        )
    )

    raw_metrics = summarize(
        raw_rank
    )

    gbs_metrics = summarize(
        gbs_rank
    )

    atomic_metrics = summarize(
        atomic_bcct_rank
    )

    rcps_metrics = summarize(
        rcps_rank
    )

    raw20 = raw_metrics["R20"]
    gbs20 = gbs_metrics["R20"]
    atomic20 = atomic_metrics["R20"]
    rcps20 = rcps_metrics["R20"]

    print()
    print(
        "=" * 140
    )

    print(
        "RCPS-A0 RESULTS"
    )

    print(
        "=" * 140
    )

    print(
        "RAW         =",
        raw_metrics,
    )

    print(
        "GBS         =",
        gbs_metrics,
    )

    print(
        "ATOMIC BCCT =",
        atomic_metrics,
    )

    print(
        "RCPS BCCT   =",
        rcps_metrics,
    )

    print()
    print(
        "RCPS GT retained =",
        retained_n,
        "/121",
    )

    up = unique_predicate_count[
        recoverable
    ]

    print(
        "unique synthesized predicates mean =",
        float(
            up.mean()
        ),
    )

    print(
        "unique synthesized predicates median =",
        float(
            np.median(up)
        ),
    )

    print(
        "unique synthesized predicates min/max =",
        int(
            up.min()
        ),
        "/",
        int(
            up.max()
        ),
    )

    print(
        "mean questions =",
        float(
            rcps_questions[
                recoverable
            ].mean()
        ),
    )

    # ========================================================
    # Fold stability
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "PER-FOLD RCPS vs ATOMIC BCCT"
    )

    print(
        "=" * 140
    )

    folds_out = {}

    nonworse = 0
    strict_better = 0

    for fold in range(5):

        mask = (
            folds == fold
        )

        rw = int(
            np.sum(
                mask
                &
                (
                    raw_rank <= B
                )
            )
        )

        gb = int(
            np.sum(
                mask
                &
                (
                    gbs_rank <= B
                )
            )
        )

        at = int(
            np.sum(
                mask
                &
                (
                    atomic_bcct_rank <= B
                )
            )
        )

        rc = int(
            np.sum(
                mask
                &
                (
                    rcps_rank <= B
                )
            )
        )

        if rc >= at:
            nonworse += 1

        if rc > at:
            strict_better += 1

        folds_out[
            str(fold)
        ] = {
            "raw_R20": rw,
            "gbs_R20": gb,
            "atomic_bcct_R20": at,
            "rcps_R20": rc,
            "delta_vs_atomic":
                rc - at,
        }

        print(
            f"fold={fold}",
            f"RAW={rw}",
            f"GBS={gb}",
            f"ATOMIC={at}",
            f"RCPS={rc}",
            f"delta={rc-at}",
        )

    # ========================================================
    # Frozen decision
    # ========================================================

    action_gate = (
        rcps20
        >=
        MIN_RCPS_R20
    )

    gain_gate = (
        rcps20
        -
        atomic20
        >=
        MIN_GAIN_OVER_ATOMIC_BCCT
    )

    retention_gate = (
        retained_n
        >=
        MIN_RETENTION
    )

    nonworse_gate = (
        nonworse
        >=
        MIN_NONWORSE_FOLDS
    )

    better_gate = (
        strict_better
        >=
        MIN_STRICT_BETTER_FOLDS
    )

    specialization_vs_gbs = (
        rcps20
        >=
        gbs20
        +
        5
    )

    positive = (
        action_gate
        and
        gain_gate
        and
        retention_gate
        and
        nonworse_gate
        and
        better_gate
        and
        specialization_vs_gbs
    )

    print()
    print(
        "=" * 140
    )

    print(
        "PRE-REGISTERED RCPS-A0 DECISION"
    )

    print(
        "=" * 140
    )

    print(
        "R20 gate:",
        action_gate,
        "|",
        rcps20,
        ">=",
        MIN_RCPS_R20,
    )

    print(
        "gain over atomic BCCT:",
        gain_gate,
        "|",
        rcps20 - atomic20,
        ">=",
        MIN_GAIN_OVER_ATOMIC_BCCT,
    )

    print(
        "retention:",
        retention_gate,
        "|",
        retained_n,
        ">=",
        MIN_RETENTION,
    )

    print(
        "fold nonworse:",
        nonworse_gate,
        "|",
        nonworse,
        ">=",
        MIN_NONWORSE_FOLDS,
    )

    print(
        "fold strict-better:",
        better_gate,
        "|",
        strict_better,
        ">=",
        MIN_STRICT_BETTER_FOLDS,
    )

    print(
        "specialization vs GBS:",
        specialization_vs_gbs,
        "| RCPS",
        rcps20,
        "| GBS",
        gbs20,
        "| delta=",
        rcps20 - gbs20,
    )

    print()

    if positive:

        diagnosis = (
            "RCPS-A0 POSITIVE. PAIRWISE LOGICAL "
            "PREDICATE SYNTHESIS CLOSES ENOUGH OF THE "
            "ATOMIC-QUESTION GAP TO CLEAR THE FROZEN "
            "SPECIALIZATION GATE. THIS SUPPORTS QUESTION-"
            "REPRESENTATION EXPRESSIVITY AS THE REMAINING "
            "BCR BOTTLENECK. FREEZE THIS PREDICATE FAMILY; "
            "DO NOT ADD NOT/TRIPLES OR TUNE FEATURES ON FIT."
        )

    else:

        diagnosis = (
            "RCPS-A0 NEGATIVE UNDER THE PRE-REGISTERED "
            "GATES. PAIRWISE AND/OR QUESTION SYNTHESIS "
            "DOES NOT PROVIDE ENOUGH ADDITIONAL GT-BLIND "
            "REPAIR OVER FROZEN ATOMIC BCCT. CLOSE THIS "
            "PREDICATE-EXPRESSIVITY BRANCH WITHOUT ADDING "
            "NOT, TRIPLE PREDICATES, FEATURE SUBSETS, OR "
            "POST-HOC LOGIC SEARCH. BCR-A0 AND A2 "
            "MECHANISM RESULTS REMAIN FROZEN POSITIVE."
        )

    print(
        diagnosis
    )

    # ========================================================
    # Save
    # ========================================================

    runtime = (
        time.perf_counter()
        -
        t0
    )

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "bcr_a4_rcps_a0"
        ),

        fold_ids=
            folds.astype(
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

        gbs_rank=
            gbs_rank.astype(
                np.int16
            ),

        atomic_bcct_rank=
            atomic_bcct_rank.astype(
                np.int16
            ),

        rcps_rank=
            rcps_rank.astype(
                np.int16
            ),

        rcps_retained=
            rcps_retained.astype(
                bool
            ),

        rcps_questions=
            rcps_questions.astype(
                np.int8
            ),

        first_predicate_global=
            first_global.astype(
                np.int32
            ),

        second_if_absent_global=
            second_zero_global.astype(
                np.int32
            ),

        second_if_present_global=
            second_one_global.astype(
                np.int32
            ),

        unique_predicate_count=
            unique_predicate_count.astype(
                np.int16
            ),
    )

    sha = sha256_file(
        OUT_NPZ
    )

    result = {
        "method":
            "BCR-A4 / RCPS-A0 Predicate Synthesis Audit",

        "question_library":
            {
                "atomic": 44,
                "pairwise_and": 946,
                "pairwise_or": 946,
                "total": 1936,
            },

        "ranking":
            {
                "raw":
                    raw_metrics,

                "gbs":
                    gbs_metrics,

                "atomic_bcct":
                    atomic_metrics,

                "rcps_bcct":
                    rcps_metrics,
            },

        "retention":
            retained_n,

        "folds":
            folds_out,

        "gate":
            {
                "R20":
                    bool(
                        action_gate
                    ),

                "gain_over_atomic":
                    bool(
                        gain_gate
                    ),

                "retention":
                    bool(
                        retention_gate
                    ),

                "fold_nonworse":
                    bool(
                        nonworse_gate
                    ),

                "fold_strict_better":
                    bool(
                        better_gate
                    ),

                "specialization_vs_gbs":
                    bool(
                        specialization_vs_gbs
                    ),

                "positive":
                    bool(
                        positive
                    ),
            },

        "runtime_seconds":
            float(
                runtime
            ),

        "artifact_sha256":
            sha,

        "diagnosis":
            diagnosis,

        "protocol":
            [
                "TRAIN-only fully observable sandbox.",
                "No CAL/HOLDOUT/TEST access.",
                "Frozen Local80 ranks101..180.",
                "Frozen B=20.",
                "Frozen maximum two clarification questions.",
                "Frozen BCCT objective.",
                "Only question representation changes.",
                "Question library fixed before observing A4 outcomes.",
                "Library contains 44 atomic presence predicates, all pairwise conjunctions, and all pairwise disjunctions.",
                "Three-valued UNKNOWN semantics conservatively preserve AST-invalid candidates.",
                "Query-local exact duplicate partitions are removed only as an equivalent computational optimization.",
                "GT is never used to synthesize, deduplicate, or select a question.",
                "Atomic BCCT must exactly reproduce the frozen A2 ranks before synthesized-predicate evaluation proceeds.",
                "No NOT predicates.",
                "No triple predicates.",
                "No feature weighting.",
                "No learned reranker.",
                "No score fusion.",
                "No post-hoc logic expansion if this branch fails.",
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
