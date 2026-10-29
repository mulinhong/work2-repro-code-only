#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np


ROOT = Path(
    "/root/autodl-tmp/work2-algorithm-innovation"
)

SANDBOX = (
    ROOT
    / "fhex_v1/sandbox/"
      "python_trainonly_qwen3_top500_v1.npz"
)

AST_CACHE = (
    ROOT
    / "fhex_v1/cache/"
      "python_train_ast_signature_v0.npz"
)

OUT_NPZ = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a2_exact_program_fact_bcct.npz"
)

OUT_JSON = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a2_exact_program_fact_bcct_metrics.json"
)


# ============================================================
# Frozen research protocol
# ============================================================

E80 = 80

B = 20

MAX_DEPTH = 2

LOCAL_START = 100
LOCAL_STOP = 180


# ============================================================
# Pre-registered continuation gates
# ============================================================

MIN_ORACLE_RECOVERY_RATE = 0.90

MIN_BCCT_RECOVERY_RATE = 0.70

MIN_BCCT_GAIN = 20

MIN_RETENTION_RATE = 0.95

MIN_POSITIVE_FOLDS = 4

MIN_ADVANTAGE_OVER_GBS = 5


UNKNOWN = -1
ABSENT = 0
PRESENT = 1


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


def summarize(ranks):

    ranks = np.asarray(
        ranks
    )

    out = {}

    for k in [
        1,
        2,
        5,
        10,
        20,
        40,
        80,
    ]:

        out[
            f"R{k}"
        ] = int(
            np.sum(
                ranks <= k
            )
        )

    valid = (
        ranks <= E80
    )

    rr = np.zeros(
        len(ranks),
        dtype=np.float64,
    )

    rr[
        valid
    ] = (
        1.0
        /
        ranks[
            valid
        ].astype(
            np.float64
        )
    )

    out[
        "MRR"
    ] = float(
        rr.mean()
    )

    return out


# ============================================================
# Question semantics
#
# Exact raw-AST presence predicates.
#
# Candidate UNKNOWN survives either answer.
# This is deliberately conservative.
# ============================================================

def answer_filter(
    facts,
    current_mask,
    feature,
    answer,
):

    v = facts[
        :,
        feature
    ]

    return (
        current_mask
        &
        (
            (v == answer)
            |
            (v == UNKNOWN)
        )
    )


def rank_under_mask(
    target_pos,
    mask,
):

    if not bool(
        mask[
            target_pos
        ]
    ):

        return E80 + 1

    return int(
        np.sum(
            mask[
                :target_pos + 1
            ]
        )
    )


# ============================================================
# BCCT objective
# ============================================================

def evaluate_leaf_for_hypotheses(
    target_ids,
    mask,
):

    """
    target_ids:
        candidate positions whose exact facts define
        the hypothetical user answer path.

    Returns fixed-budget certified coverage and rank gain.
    """

    coverage = 0

    gain = 0

    for pos in target_ids:

        pos = int(
            pos
        )

        r = rank_under_mask(
            pos,
            mask,
        )

        if r <= B:

            coverage += 1

            gain += max(
                0,
                (
                    pos + 1
                )
                -
                r,
            )

    return (
        coverage,
        gain,
    )


def branch_bcct(
    facts,
    current_mask,
    target_ids,
    first_feature,
):
    """
    Exact second-question optimization for one first-answer branch.

    GT-blind:
      target_ids are candidate hypotheses, NOT the real GT.
    """

    current_target_ids = np.asarray(
        target_ids,
        dtype=np.int32,
    )

    stop_cov, stop_gain = (
        evaluate_leaf_for_hypotheses(
            current_target_ids,
            current_mask,
        )
    )

    best = {
        "feature": -1,
        "coverage": stop_cov,
        "gain": stop_gain,
        "worst_leaf":
            int(
                current_mask.sum()
            ),
    }

    # If every survivor already fits B, a second question
    # cannot improve budget certification.
    if int(
        current_mask.sum()
    ) <= B:

        return best

    F = facts.shape[
        1
    ]

    for f in range(
        F
    ):

        if f == first_feature:
            continue

        # Only hypotheses with known exact facts can define
        # an answer to this question.
        ids0 = current_target_ids[
            facts[
                current_target_ids,
                f
            ]
            ==
            ABSENT
        ]

        ids1 = current_target_ids[
            facts[
                current_target_ids,
                f
            ]
            ==
            PRESENT
        ]

        # Need both possible answers represented.
        if (
            len(ids0) == 0
            or
            len(ids1) == 0
        ):

            continue

        mask0 = answer_filter(
            facts,
            current_mask,
            f,
            ABSENT,
        )

        mask1 = answer_filter(
            facts,
            current_mask,
            f,
            PRESENT,
        )

        cov0, gain0 = (
            evaluate_leaf_for_hypotheses(
                ids0,
                mask0,
            )
        )

        cov1, gain1 = (
            evaluate_leaf_for_hypotheses(
                ids1,
                mask1,
            )
        )

        cand = {
            "feature":
                f,

            "coverage":
                cov0 + cov1,

            "gain":
                gain0 + gain1,

            "worst_leaf":
                max(
                    int(
                        mask0.sum()
                    ),
                    int(
                        mask1.sum()
                    ),
                ),
        }

        key_c = (
            cand[
                "coverage"
            ],
            cand[
                "gain"
            ],
            -cand[
                "worst_leaf"
            ],
            -f,
        )

        key_b = (
            best[
                "coverage"
            ],
            best[
                "gain"
            ],
            -best[
                "worst_leaf"
            ],
            -best[
                "feature"
            ],
        )

        if key_c > key_b:

            best = cand

    return best


def choose_bcct(
    facts,
    valid_hypotheses,
):

    full_mask = np.ones(
        E80,
        dtype=bool,
    )

    target_ids = np.flatnonzero(
        valid_hypotheses
    )

    best = None

    F = facts.shape[
        1
    ]

    for f1 in range(
        F
    ):

        ids0 = target_ids[
            facts[
                target_ids,
                f1
            ]
            ==
            ABSENT
        ]

        ids1 = target_ids[
            facts[
                target_ids,
                f1
            ]
            ==
            PRESENT
        ]

        if (
            len(ids0) == 0
            or
            len(ids1) == 0
        ):

            continue

        mask0 = answer_filter(
            facts,
            full_mask,
            f1,
            ABSENT,
        )

        mask1 = answer_filter(
            facts,
            full_mask,
            f1,
            PRESENT,
        )

        p0 = branch_bcct(
            facts,
            mask0,
            ids0,
            f1,
        )

        p1 = branch_bcct(
            facts,
            mask1,
            ids1,
            f1,
        )

        cand = {
            "first_feature":
                f1,

            "second_zero":
                int(
                    p0[
                        "feature"
                    ]
                ),

            "second_one":
                int(
                    p1[
                        "feature"
                    ]
                ),

            "coverage":
                int(
                    p0[
                        "coverage"
                    ]
                    +
                    p1[
                        "coverage"
                    ]
                ),

            "gain":
                int(
                    p0[
                        "gain"
                    ]
                    +
                    p1[
                        "gain"
                    ]
                ),

            "worst_leaf":
                max(
                    int(
                        p0[
                            "worst_leaf"
                        ]
                    ),
                    int(
                        p1[
                            "worst_leaf"
                        ]
                    ),
                ),
        }

        if best is None:

            best = cand

            continue

        key_c = (
            cand[
                "coverage"
            ],
            cand[
                "gain"
            ],
            -cand[
                "worst_leaf"
            ],
            -f1,
        )

        key_b = (
            best[
                "coverage"
            ],
            best[
                "gain"
            ],
            -best[
                "worst_leaf"
            ],
            -best[
                "first_feature"
            ],
        )

        if key_c > key_b:

            best = cand

    return best


# ============================================================
# Generic balanced-search baseline
# ============================================================

def choose_gbs_second(
    facts,
    current_mask,
    first_feature,
):

    if int(
        current_mask.sum()
    ) <= B:

        return -1

    best = None

    F = facts.shape[
        1
    ]

    for f in range(
        F
    ):

        if f == first_feature:
            continue

        mask0 = answer_filter(
            facts,
            current_mask,
            f,
            ABSENT,
        )

        mask1 = answer_filter(
            facts,
            current_mask,
            f,
            PRESENT,
        )

        # Question must actually reduce both possible worlds.
        if (
            np.array_equal(
                mask0,
                current_mask,
            )
            or
            np.array_equal(
                mask1,
                current_mask,
            )
        ):

            continue

        worst = max(
            int(
                mask0.sum()
            ),
            int(
                mask1.sum()
            ),
        )

        cand = (
            -worst,
            -f,
        )

        if (
            best is None
            or
            cand > best[
                0
            ]
        ):

            best = (
                cand,
                f,
            )

    if best is None:

        return -1

    return int(
        best[
            1
        ]
    )


def choose_gbs(
    facts,
):

    full_mask = np.ones(
        E80,
        dtype=bool,
    )

    best = None

    F = facts.shape[
        1
    ]

    for f in range(
        F
    ):

        mask0 = answer_filter(
            facts,
            full_mask,
            f,
            ABSENT,
        )

        mask1 = answer_filter(
            facts,
            full_mask,
            f,
            PRESENT,
        )

        if (
            np.array_equal(
                mask0,
                full_mask,
            )
            or
            np.array_equal(
                mask1,
                full_mask,
            )
        ):

            continue

        worst = max(
            int(
                mask0.sum()
            ),
            int(
                mask1.sum()
            ),
        )

        cand = (
            -worst,
            -f,
        )

        if (
            best is None
            or
            cand > best[
                0
            ]
        ):

            best = (
                cand,
                f,
                mask0,
                mask1,
            )

    if best is None:

        return None

    f1 = int(
        best[
            1
        ]
    )

    mask0 = best[
        2
    ]

    mask1 = best[
        3
    ]

    return {
        "first_feature":
            f1,

        "second_zero":
            choose_gbs_second(
                facts,
                mask0,
                f1,
            ),

        "second_one":
            choose_gbs_second(
                facts,
                mask1,
                f1,
            ),
    }


# ============================================================
# Execute GT-blind policy using GT only as user answer
# ============================================================

def execute_policy(
    policy,
    facts,
    gt_pos,
    gt_facts,
):

    if policy is None:

        mask = np.ones(
            E80,
            dtype=bool,
        )

        return (
            rank_under_mask(
                gt_pos,
                mask,
            ),
            True,
            0,
        )

    # If target AST itself is unknown, conservatively ask nothing.
    if np.any(
        gt_facts
        ==
        UNKNOWN
    ):

        mask = np.ones(
            E80,
            dtype=bool,
        )

        return (
            rank_under_mask(
                gt_pos,
                mask,
            ),
            True,
            0,
        )

    full = np.ones(
        E80,
        dtype=bool,
    )

    f1 = int(
        policy[
            "first_feature"
        ]
    )

    a1 = int(
        gt_facts[
            f1
        ]
    )

    mask = answer_filter(
        facts,
        full,
        f1,
        a1,
    )

    questions = 1

    if int(
        mask.sum()
    ) > B:

        f2 = (
            int(
                policy[
                    "second_one"
                ]
            )
            if a1
            ==
            PRESENT
            else
            int(
                policy[
                    "second_zero"
                ]
            )
        )

        if f2 >= 0:

            a2 = int(
                gt_facts[
                    f2
                ]
            )

            mask = answer_filter(
                facts,
                mask,
                f2,
                a2,
            )

            questions = 2

    retained = bool(
        mask[
            gt_pos
        ]
    )

    rank = rank_under_mask(
        gt_pos,
        mask,
    )

    return (
        rank,
        retained,
        questions,
    )


# ============================================================
# Exact-predicate oracle upper bound
# ============================================================

def exact_oracle_rank(
    facts,
    gt_pos,
    gt_facts,
):

    if np.any(
        gt_facts
        ==
        UNKNOWN
    ):

        return gt_pos + 1

    full = np.ones(
        E80,
        dtype=bool,
    )

    best = (
        gt_pos + 1
    )

    F = facts.shape[
        1
    ]

    for f1 in range(
        F
    ):

        a1 = int(
            gt_facts[
                f1
            ]
        )

        mask1 = answer_filter(
            facts,
            full,
            f1,
            a1,
        )

        r1 = rank_under_mask(
            gt_pos,
            mask1,
        )

        best = min(
            best,
            r1,
        )

        if r1 <= B:
            continue

        for f2 in range(
            F
        ):

            if f2 == f1:
                continue

            a2 = int(
                gt_facts[
                    f2
                ]
            )

            mask2 = answer_filter(
                facts,
                mask1,
                f2,
                a2,
            )

            r2 = rank_under_mask(
                gt_pos,
                mask2,
            )

            best = min(
                best,
                r2,
            )

    return best


def main():

    t0 = time.perf_counter()

    print(
        "=" * 140
    )

    print(
        "BCR-A2: EXACT-PROGRAM-FACT BCCT"
    )

    print(
        "=" * 140
    )

    print(
        "TRAIN-only fully observable sandbox."
    )

    print(
        "Raw AST facts replace distilled candidate facts."
    )

    print(
        "Question selection remains completely GT-blind."
    )

    print(
        "GT is used only to simulate user answers and evaluate."
    )

    print(
        "AST-parse-invalid candidates are UNKNOWN and survive both answers."
    )

    # ========================================================
    # Load sandbox.
    # ========================================================

    s = np.load(
        SANDBOX,
        allow_pickle=True,
    )

    role_codes = s[
        "role_codes"
    ].astype(
        np.int8
    )

    folds_all = s[
        "fold_ids"
    ].astype(
        np.int8
    )

    gt_all = s[
        "ground_truth_code_indices"
    ].astype(
        np.int64
    )

    top500 = s[
        "qwen3_indices"
    ].astype(
        np.int32
    )

    base_failure = s[
        "base_failure_gt100"
    ].astype(
        bool
    )

    local_recoverable_saved = s[
        "local80_recoverable"
    ].astype(
        bool
    )

    assert not bool(
        np.asarray(
            s[
                "valid_accessed"
            ]
        ).item()
    )

    assert not bool(
        np.asarray(
            s[
                "test_accessed"
            ]
        ).item()
    )

    fit_failure_mask = (
        (
            role_codes
            ==
            0
        )
        &
        base_failure
    )

    rows = np.flatnonzero(
        fit_failure_mask
    )

    assert len(
        rows
    ) == 568

    folds = folds_all[
        rows
    ]

    gt = gt_all[
        rows
    ]

    candidates = top500[
        rows,
        LOCAL_START:LOCAL_STOP
    ].astype(
        np.int32
    )

    assert candidates.shape == (
        568,
        80,
    )

    # ========================================================
    # Exact raw AST facts.
    # ========================================================

    az = np.load(
        AST_CACHE,
        allow_pickle=True,
    )

    feature_names_all = az[
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

    # Same realizable question family as A1R:
    # presence/absence of every feature except max_ast_depth.
    allowed = np.asarray(
        [
            name
            !=
            "max_ast_depth"
            for name in feature_names_all
        ],
        dtype=bool,
    )

    allowed_ids = np.flatnonzero(
        allowed
    )

    feature_names = feature_names_all[
        allowed_ids
    ]

    F = len(
        feature_names
    )

    print()
    print(
        "exact presence-predicate count =",
        F,
    )

    # ========================================================
    # Build facts for every Local80 candidate.
    #
    # valid AST:
    #     signature > 0 -> PRESENT
    #     signature = 0 -> ABSENT
    #
    # invalid AST:
    #     UNKNOWN
    # ========================================================

    facts = np.full(
        (
            len(
                rows
            ),
            E80,
            F,
        ),
        UNKNOWN,
        dtype=np.int8,
    )

    candidate_valid = ast_valid[
        candidates
    ]

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

    facts[
        candidate_valid
    ] = (
        exact_presence[
            candidate_valid
        ]
        .astype(
            np.int8
        )
    )

    gt_valid = ast_valid[
        gt
    ]

    gt_facts = np.full(
        (
            len(
                rows
            ),
            F,
        ),
        UNKNOWN,
        dtype=np.int8,
    )

    gt_presence = (
        signature[
            gt
        ][
            :,
            allowed_ids
        ]
        >
        0.0
    )

    gt_facts[
        gt_valid
    ] = (
        gt_presence[
            gt_valid
        ]
        .astype(
            np.int8
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
    # Correct Local80 baseline.
    # ========================================================

    raw_rank = np.full(
        len(
            rows
        ),
        E80 + 1,
        dtype=np.int16,
    )

    gt_pos = np.full(
        len(
            rows
        ),
        -1,
        dtype=np.int16,
    )

    recoverable = np.zeros(
        len(
            rows
        ),
        dtype=bool,
    )

    for i in range(
        len(
            rows
        )
    ):

        p = np.flatnonzero(
            candidates[
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

            pos = int(
                p[
                    0
                ]
            )

            gt_pos[
                i
            ] = pos

            recoverable[
                i
            ] = True

            raw_rank[
                i
            ] = (
                pos + 1
            )

    assert int(
        recoverable.sum()
    ) == 121

    # Cross-check saved sandbox flag.
    assert int(
        local_recoverable_saved[
            rows
        ].sum()
    ) == 121

    raw_metrics = summarize(
        raw_rank
    )

    raw_r20 = int(
        raw_metrics[
            "R20"
        ]
    )

    print()
    print(
        "=" * 140
    )

    print(
        "TRAIN-ONLY LOCAL80 BASELINE"
    )

    print(
        "=" * 140
    )

    print(
        "FIT failures =",
        len(
            rows
        ),
    )

    print(
        "Local80 recoverable =",
        int(
            recoverable.sum()
        ),
    )

    print(
        "RAW =",
        raw_metrics,
    )

    # ========================================================
    # Exact predicate oracle + GT-blind policies.
    # ========================================================

    oracle_rank = raw_rank.copy()

    bcct_rank = raw_rank.copy()

    gbs_rank = raw_rank.copy()

    bcct_retained = np.zeros(
        len(
            rows
        ),
        dtype=bool,
    )

    gbs_retained = np.zeros(
        len(
            rows
        ),
        dtype=bool,
    )

    bcct_questions = np.zeros(
        len(
            rows
        ),
        dtype=np.int8,
    )

    gbs_questions = np.zeros(
        len(
            rows
        ),
        dtype=np.int8,
    )

    bcct_first = np.full(
        len(
            rows
        ),
        -1,
        dtype=np.int16,
    )

    bcct_second_zero = np.full(
        len(
            rows
        ),
        -1,
        dtype=np.int16,
    )

    bcct_second_one = np.full(
        len(
            rows
        ),
        -1,
        dtype=np.int16,
    )

    gbs_first = np.full(
        len(
            rows
        ),
        -1,
        dtype=np.int16,
    )

    print()
    print(
        "=" * 140
    )

    print(
        "EXACT-FACT ORACLE + GT-BLIND POLICIES"
    )

    print(
        "=" * 140
    )

    for i in range(
        len(
            rows
        )
    ):

        if not recoverable[
            i
        ]:

            continue

        f = facts[
            i
        ]

        valid_hypotheses = candidate_valid[
            i
        ]

        # ----------------------------------------------
        # Exact presence-predicate oracle.
        # ----------------------------------------------

        oracle_rank[
            i
        ] = exact_oracle_rank(
            f,
            int(
                gt_pos[
                    i
                ]
            ),
            gt_facts[
                i
            ],
        )

        # ----------------------------------------------
        # BCCT.
        # ----------------------------------------------

        bcct = choose_bcct(
            f,
            valid_hypotheses,
        )

        if bcct is not None:

            bcct_first[
                i
            ] = int(
                bcct[
                    "first_feature"
                ]
            )

            bcct_second_zero[
                i
            ] = int(
                bcct[
                    "second_zero"
                ]
            )

            bcct_second_one[
                i
            ] = int(
                bcct[
                    "second_one"
                ]
            )

            (
                bcct_rank[
                    i
                ],
                bcct_retained[
                    i
                ],
                bcct_questions[
                    i
                ],
            ) = execute_policy(
                bcct,
                f,
                int(
                    gt_pos[
                        i
                    ]
                ),
                gt_facts[
                    i
                ],
            )

        else:

            bcct_rank[
                i
            ] = raw_rank[
                i
            ]

            bcct_retained[
                i
            ] = True

        # ----------------------------------------------
        # GBS.
        # ----------------------------------------------

        gbs = choose_gbs(
            f
        )

        if gbs is not None:

            gbs_first[
                i
            ] = int(
                gbs[
                    "first_feature"
                ]
            )

            (
                gbs_rank[
                    i
                ],
                gbs_retained[
                    i
                ],
                gbs_questions[
                    i
                ],
            ) = execute_policy(
                gbs,
                f,
                int(
                    gt_pos[
                        i
                    ]
                ),
                gt_facts[
                    i
                ],
            )

        else:

            gbs_rank[
                i
            ] = raw_rank[
                i
            ]

            gbs_retained[
                i
            ] = True

        if (
            (i + 1) % 100 == 0
            or
            i
            ==
            len(
                rows
            )
            -
            1
        ):

            print(
                f"[{i+1:>3}/568]",
                "RAW20=",
                int(
                    np.sum(
                        raw_rank[
                            :i+1
                        ]
                        <=
                        B
                    )
                ),
                "ORACLE20=",
                int(
                    np.sum(
                        oracle_rank[
                            :i+1
                        ]
                        <=
                        B
                    )
                ),
                "GBS20=",
                int(
                    np.sum(
                        gbs_rank[
                            :i+1
                        ]
                        <=
                        B
                    )
                ),
                "BCCT20=",
                int(
                    np.sum(
                        bcct_rank[
                            :i+1
                        ]
                        <=
                        B
                    )
                ),
                flush=True,
            )

    oracle_metrics = summarize(
        oracle_rank
    )

    gbs_metrics = summarize(
        gbs_rank
    )

    bcct_metrics = summarize(
        bcct_rank
    )

    oracle_r20 = int(
        oracle_metrics[
            "R20"
        ]
    )

    gbs_r20 = int(
        gbs_metrics[
            "R20"
        ]
    )

    bcct_r20 = int(
        bcct_metrics[
            "R20"
        ]
    )

    bcct_retained_n = int(
        np.sum(
            recoverable
            &
            bcct_retained
        )
    )

    gbs_retained_n = int(
        np.sum(
            recoverable
            &
            gbs_retained
        )
    )

    n_recoverable = int(
        recoverable.sum()
    )

    print()
    print(
        "=" * 140
    )

    print(
        "BCR-A2 RESULTS"
    )

    print(
        "=" * 140
    )

    print(
        "RAW          =",
        raw_metrics,
    )

    print(
        "EXACT ORACLE =",
        oracle_metrics,
    )

    print(
        "GBS          =",
        gbs_metrics,
    )

    print(
        "BCCT         =",
        bcct_metrics,
    )

    print()

    print(
        "recoverable =",
        n_recoverable,
    )

    print(
        "oracle recovery rate =",
        oracle_r20
        /
        n_recoverable,
    )

    print(
        "GBS recovery rate =",
        gbs_r20
        /
        n_recoverable,
    )

    print(
        "BCCT recovery rate =",
        bcct_r20
        /
        n_recoverable,
    )

    print(
        "GBS GT retained =",
        gbs_retained_n,
        "/",
        n_recoverable,
    )

    print(
        "BCCT GT retained =",
        bcct_retained_n,
        "/",
        n_recoverable,
    )

    print(
        "GBS mean questions =",
        float(
            gbs_questions[
                recoverable
            ].mean()
        ),
    )

    print(
        "BCCT mean questions =",
        float(
            bcct_questions[
                recoverable
            ].mean()
        ),
    )

    # ========================================================
    # Most common BCCT first questions.
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "BCCT FIRST-QUESTION USAGE"
    )

    print(
        "=" * 140
    )

    valid_first = bcct_first[
        bcct_first >= 0
    ]

    if len(
        valid_first
    ):

        vals, counts = np.unique(
            valid_first,
            return_counts=True,
        )

        order = np.argsort(
            counts
        )[::-1]

        for j in order[
            :15
        ]:

            f = int(
                vals[
                    j
                ]
            )

            print(
                f"{feature_names[f]:<28}",
                int(
                    counts[
                        j
                    ]
                ),
            )

    # ========================================================
    # Fold stability.
    # ========================================================

    print()
    print(
        "=" * 140
    )

    print(
        "PER-FOLD BCR-A2"
    )

    print(
        "=" * 140
    )

    fold_results = {}

    positive_folds = 0

    for fold in range(
        5
    ):

        mask = (
            folds
            ==
            fold
        )

        recoverable_fold = (
            mask
            &
            recoverable
        )

        raw20 = int(
            np.sum(
                mask
                &
                (
                    raw_rank
                    <=
                    B
                )
            )
        )

        oracle20 = int(
            np.sum(
                mask
                &
                (
                    oracle_rank
                    <=
                    B
                )
            )
        )

        gbs20 = int(
            np.sum(
                mask
                &
                (
                    gbs_rank
                    <=
                    B
                )
            )
        )

        bcct20 = int(
            np.sum(
                mask
                &
                (
                    bcct_rank
                    <=
                    B
                )
            )
        )

        gain = (
            bcct20
            -
            raw20
        )

        if gain > 0:
            positive_folds += 1

        fold_results[
            str(
                fold
            )
        ] = {
            "failure_n":
                int(
                    mask.sum()
                ),

            "recoverable_n":
                int(
                    recoverable_fold.sum()
                ),

            "raw_R20":
                raw20,

            "oracle_R20":
                oracle20,

            "gbs_R20":
                gbs20,

            "bcct_R20":
                bcct20,

            "bcct_gain":
                gain,
        }

        print(
            f"fold={fold}",
            f"recoverable={int(recoverable_fold.sum())}",
            f"RAW={raw20}",
            f"ORACLE={oracle20}",
            f"GBS={gbs20}",
            f"BCCT={bcct20}",
            f"gain={gain}",
        )

    # ========================================================
    # Frozen gates.
    # ========================================================

    oracle_rate = (
        oracle_r20
        /
        n_recoverable
    )

    bcct_rate = (
        bcct_r20
        /
        n_recoverable
    )

    retention_rate = (
        bcct_retained_n
        /
        n_recoverable
    )

    oracle_gate = (
        oracle_rate
        >=
        MIN_ORACLE_RECOVERY_RATE
    )

    bcct_gate = (
        bcct_rate
        >=
        MIN_BCCT_RECOVERY_RATE
    )

    gain_gate = (
        bcct_r20
        -
        raw_r20
        >=
        MIN_BCCT_GAIN
    )

    retention_gate = (
        retention_rate
        >=
        MIN_RETENTION_RATE
    )

    fold_gate = (
        positive_folds
        >=
        MIN_POSITIVE_FOLDS
    )

    specialization_gate = (
        bcct_r20
        >=
        gbs_r20
        +
        MIN_ADVANTAGE_OVER_GBS
    )

    algorithm_positive = (
        oracle_gate
        and
        bcct_gate
        and
        gain_gate
        and
        retention_gate
        and
        fold_gate
        and
        specialization_gate
    )

    print()
    print(
        "=" * 140
    )

    print(
        "PRE-REGISTERED BCR-A2 DECISION"
    )

    print(
        "=" * 140
    )

    print(
        "exact-oracle gate:",
        oracle_gate,
        "| rate",
        oracle_rate,
        ">=",
        MIN_ORACLE_RECOVERY_RATE,
    )

    print(
        "BCCT recovery gate:",
        bcct_gate,
        "| rate",
        bcct_rate,
        ">=",
        MIN_BCCT_RECOVERY_RATE,
    )

    print(
        "BCCT gain gate:",
        gain_gate,
        "|",
        bcct_r20
        -
        raw_r20,
        ">=",
        MIN_BCCT_GAIN,
    )

    print(
        "BCCT retention gate:",
        retention_gate,
        "| rate",
        retention_rate,
        ">=",
        MIN_RETENTION_RATE,
    )

    print(
        "fold gate:",
        fold_gate,
        "| positive folds",
        positive_folds,
        ">=",
        MIN_POSITIVE_FOLDS,
    )

    print(
        "specialization vs GBS:",
        specialization_gate,
        "| BCCT",
        bcct_r20,
        "| GBS",
        gbs_r20,
        "| advantage",
        bcct_r20
        -
        gbs_r20,
        ">=",
        MIN_ADVANTAGE_OVER_GBS,
    )

    print()

    if algorithm_positive:

        diagnosis = (
            "BCR-A2 STRONGLY POSITIVE. EXACT PROGRAM "
            "FACTS RESTORE A GT-BLIND BUDGET-CERTIFYING "
            "CLARIFICATION POLICY. THIS SUPPORTS THE "
            "HYPOTHESIS THAT BCR-A1R FAILED PRIMARILY "
            "BECAUSE DISTILLED CANDIDATE ATTRIBUTES WERE "
            "NOISY, NOT BECAUSE THE TWO-BIT REPAIR "
            "MECHANISM OR BCCT OBJECTIVE WAS INVALID. "
            "BCR REMAINS THE PRIMARY ALGORITHM DIRECTION."
        )

    elif oracle_gate and not bcct_gate:

        diagnosis = (
            "BCR MECHANISM REMAINS POSITIVE: EXACT "
            "PROGRAM FACTS PROVIDE STRONG TWO-BIT "
            "HEADROOM, BUT GT-BLIND BCCT DOES NOT "
            "CAPTURE IT. ATTRIBUTE NOISE IS NOT THE "
            "ONLY BOTTLENECK. KEEP THE BCR-A0 MECHANISM "
            "RESULT, CLOSE BCCT-v0, AND DESIGN A NEW "
            "GT-BLIND POLICY SOLVER WITHOUT CHANGING "
            "THE TWO-BIT INFORMATION BUDGET."
        )

    else:

        diagnosis = (
            "BCR-A2 DOES NOT SUPPORT THE CURRENT "
            "REALIZABLE-PREDICATE / BCCT IMPLEMENTATION. "
            "THIS DOES NOT ERASE THE FROZEN BCR-A0 "
            "STRONG-POSITIVE INFORMATION-DEFICIT RESULT. "
            "DO NOT TUNE QUESTION COUNT OR PREDICATE "
            "THRESHOLDS POST HOC."
        )

    print(
        diagnosis
    )

    # ========================================================
    # Save.
    # ========================================================

    runtime = (
        time.perf_counter()
        -
        t0
    )

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "bcr_a2_exact_program_fact_bcct"
        ),

        sandbox_rows=
            rows.astype(
                np.int32
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

        feature_names=
            feature_names,

        candidate_ast_valid=
            candidate_valid.astype(
                bool
            ),

        gt_ast_valid=
            gt_valid.astype(
                bool
            ),

        raw_rank=
            raw_rank.astype(
                np.int16
            ),

        exact_oracle_rank=
            oracle_rank.astype(
                np.int16
            ),

        gbs_rank=
            gbs_rank.astype(
                np.int16
            ),

        bcct_rank=
            bcct_rank.astype(
                np.int16
            ),

        gbs_retained=
            gbs_retained.astype(
                bool
            ),

        bcct_retained=
            bcct_retained.astype(
                bool
            ),

        gbs_questions=
            gbs_questions.astype(
                np.int8
            ),

        bcct_questions=
            bcct_questions.astype(
                np.int8
            ),

        bcct_first_question=
            bcct_first.astype(
                np.int16
            ),

        bcct_second_if_absent=
            bcct_second_zero.astype(
                np.int16
            ),

        bcct_second_if_present=
            bcct_second_one.astype(
                np.int16
            ),
    )

    sha = sha256_file(
        OUT_NPZ
    )

    metrics = {
        "method":
            "BCR-A2 Exact-Program-Fact BCCT",

        "purpose":
            (
                "Causal test of whether BCR-A1R failure "
                "was caused primarily by noisy distilled "
                "candidate structural facts."
            ),

        "sandbox": {
            "fit_failures":
                int(
                    len(
                        rows
                    )
                ),

            "local80_recoverable":
                n_recoverable,

            "candidate_ast_valid_fraction":
                float(
                    candidate_valid.mean()
                ),

            "gt_ast_valid_fraction":
                float(
                    gt_valid.mean()
                ),
        },

        "ranking": {
            "raw":
                raw_metrics,

            "exact_oracle":
                oracle_metrics,

            "gbs":
                gbs_metrics,

            "bcct":
                bcct_metrics,
        },

        "rates": {
            "oracle_recovery":
                oracle_rate,

            "gbs_recovery":
                (
                    gbs_r20
                    /
                    n_recoverable
                ),

            "bcct_recovery":
                bcct_rate,

            "bcct_retention":
                retention_rate,
        },

        "folds":
            fold_results,

        "gate": {
            "oracle":
                bool(
                    oracle_gate
                ),

            "bcct_recovery":
                bool(
                    bcct_gate
                ),

            "bcct_gain":
                bool(
                    gain_gate
                ),

            "bcct_retention":
                bool(
                    retention_gate
                ),

            "fold":
                bool(
                    fold_gate
                ),

            "specialization_vs_gbs":
                bool(
                    specialization_gate
                ),

            "algorithm_positive":
                bool(
                    algorithm_positive
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

        "protocol": [
            "TRAIN-only fully observable development sandbox.",
            "No CAL/HOLDOUT/TEST raw code is accessed.",
            "FIT failures are Qwen TRAIN-only rank >100.",
            "Local80 is frozen ranks101..180.",
            "Final budget remains B=20.",
            "At most two binary clarification questions are allowed.",
            "Two-question budget is unchanged from BCR-A0/A1R.",
            "Question family is unchanged from BCR-A1R: exact AST-feature presence/absence, with max_ast_depth excluded.",
            "Candidate attributes come directly from raw TRAIN AST facts, not embedding distillation.",
            "AST-parse-invalid candidates are treated as UNKNOWN and survive either answer.",
            "Question selection is GT-blind.",
            "GT is used only to simulate user answers and evaluate final rank.",
            "Original Local80 rank order is preserved.",
            "BCCT exactly optimizes fixed-budget candidate certification over candidate hypotheses.",
            "GBS is the generic balanced-search baseline.",
            "No relevance reranker is trained.",
            "No semantic/structural score fusion is used.",
            "No feature weighting is tuned.",
            "No question threshold is tuned.",
            "BCR-A0 strong-positive mechanism result remains frozen regardless of A2 outcome.",
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
