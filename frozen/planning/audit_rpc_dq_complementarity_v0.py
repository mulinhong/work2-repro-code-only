from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import importlib.util
import json
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")
D = ROOT / "fhex_v1/obstruction_audit_v0"

UNION_SCRIPT = D / "audit_tecs_rcps_union_v0.py"
RPC_SCRIPT = D / "audit_rpc_capacity_bound_value_v0.py"
DQ_SCRIPT = D / "audit_rpc_depth_quota_bound_v0.py"

RPC_FULL_JSON = D / "rpc_capacity_bound_value_v0_full82.json"

EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_RPC_SHA = (
    "4da6e382b26b35506f8de2ccd6d107aed4a3e5918b8f26fa16d254ac73aa380a"
)

EXPECTED_DQ_SHA = (
    "162cc018992d014cd56ab46e69c957144589f2264ad0c3f94f7bd441e1b3cdc2"
)


E = 80
B = 20

UNKNOWN = -1
ABSENT = 0
PRESENT = 1


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    mod = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(mod)

    return mod


def bool_to_bits(x):
    out = 0

    for i in np.flatnonzero(x):
        out |= 1 << int(i)

    return out


# ==================================================================
# Tiny construction checks.
#
# These are NOT empirical evidence.
# They instantiate the two directions of theoretical
# incomparability in the same rank-repair semantics.
# ==================================================================

def tiny_rpc_capacity(
    valid,
    target,
    B0,
    L,
):
    n = len(valid)

    slack = [B0] * L
    reward = 0

    for i in range(n):

        if not valid[i]:

            for j in range(L):
                slack[j] = max(
                    0,
                    slack[j] - 1,
                )

            continue

        if target[i]:

            chosen = None

            for j, s in enumerate(slack):
                if s > 0:
                    chosen = j
                    break

            if chosen is not None:

                reward += 1
                slack[chosen] -= 1

            continue

        if slack[0] > 0:
            slack[0] -= 1

    return int(reward)


def tiny_dq_bound(
    facts,
    valid,
    target,
    B0,
    depth,
    leaves,
):
    n, P = facts.shape

    k = min(
        int(depth),
        int(leaves) - 1,
    )

    feasible = 0

    for t in range(n):

        if not target[t]:
            continue

        rank = t + 1

        if rank <= B0:
            feasible += 1
            continue

        delta = rank - B0

        sigmas = []

        for p in range(P):

            ans = int(
                facts[t, p]
            )

            sigma = 0

            for i in range(
                t + 1
            ):

                if not valid[i]:
                    continue

                if int(
                    facts[i, p]
                ) == 1 - ans:

                    sigma += 1

            sigmas.append(
                sigma
            )

        sigmas.sort(
            reverse=True
        )

        optimistic = sum(
            sigmas[:k]
        )

        if optimistic >= delta:
            feasible += 1

    return int(feasible)


def check_incomparability_constructions():

    # --------------------------------------------------------------
    # Construction A: RPC strictly tighter than DQ.
    #
    # B=1, L=2, d=1.
    #
    # Three targets:
    # t1, t2, t3.
    #
    # Individually:
    # - t1 is already top-1.
    # - p0 separates t2 from t1.
    # - p1 separates t3 from both earlier targets.
    #
    # Hence DQ says all 3 are individually reachable.
    #
    # But two leaves with B=1 can repair at most 2 targets.
    # Therefore C=2 < Q=3.
    # --------------------------------------------------------------

    valid_a = np.array(
        [True, True, True],
        dtype=bool,
    )

    target_a = np.array(
        [True, True, True],
        dtype=bool,
    )

    facts_a = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.int8,
    )

    C_a = tiny_rpc_capacity(
        valid_a,
        target_a,
        B0=1,
        L=2,
    )

    Q_a = tiny_dq_bound(
        facts_a,
        valid_a,
        target_a,
        B0=1,
        depth=1,
        leaves=2,
    )

    assert C_a == 2
    assert Q_a == 3
    assert C_a < Q_a

    # --------------------------------------------------------------
    # Construction B: DQ strictly tighter than RPC.
    #
    # B=1, L=2, d=1.
    #
    # t1, t2 followed by a valid non-target n.
    #
    # Arbitrary two-leaf assignment can place t1/t2 separately:
    # C=2.
    #
    # Predicate p is legal globally because n has answer 1,
    # but t1 and t2 both answer 0.  Therefore no remaining
    # question can separate t2 from its earlier blocker t1:
    # Q=1.
    #
    # Hence Q=1 < C=2.
    # --------------------------------------------------------------

    valid_b = np.array(
        [True, True, True],
        dtype=bool,
    )

    target_b = np.array(
        [True, True, False],
        dtype=bool,
    )

    facts_b = np.array(
        [
            [0],
            [0],
            [1],
        ],
        dtype=np.int8,
    )

    C_b = tiny_rpc_capacity(
        valid_b,
        target_b,
        B0=1,
        L=2,
    )

    Q_b = tiny_dq_bound(
        facts_b,
        valid_b,
        target_b,
        B0=1,
        depth=1,
        leaves=2,
    )

    assert C_b == 2
    assert Q_b == 1
    assert Q_b < C_b

    return {
        "C_lt_Q":
            {
                "C": C_a,
                "Q": Q_a,
            },

        "Q_lt_C":
            {
                "C": C_b,
                "Q": Q_b,
            },
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-hard-queries",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="smoke10",
    )

    args = parser.parse_args()

    print("=" * 162)
    print(
        "RPC–DQ COMPLEMENTARITY / INCOMPARABILITY AUDIT v0"
    )
    print(
        "COMMON SEARCH SPACE = FROZEN RPC-ONLY EXACT SEARCH"
    )
    print(
        "DQ IS SHADOW-EVALUATED AND CANNOT ALTER SEARCH"
    )
    print("=" * 162)

    assert (
        sha256_file(
            UNION_SCRIPT
        )
        ==
        EXPECTED_UNION_SHA
    )

    assert (
        sha256_file(
            RPC_SCRIPT
        )
        ==
        EXPECTED_RPC_SHA
    )

    assert (
        sha256_file(
            DQ_SCRIPT
        )
        ==
        EXPECTED_DQ_SHA
    )

    assert RPC_FULL_JSON.exists()

    print(
        "FROZEN SCRIPT INTEGRITY = PASS"
    )

    constructions = (
        check_incomparability_constructions()
    )

    print(
        "THEORETICAL CONSTRUCTION C<Q =",
        constructions[
            "C_lt_Q"
        ],
    )

    print(
        "THEORETICAL CONSTRUCTION Q<C =",
        constructions[
            "Q_lt_C"
        ],
    )

    print(
        "TWO-DIRECTION CONSTRUCTION CHECK = PASS"
    )

    # ==============================================================
    # Frozen objects.
    # ==============================================================

    unionmod = load_module(
        UNION_SCRIPT,
        "comp_union",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "comp_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "comp_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "comp_tecs",
    )

    a2 = np.load(
        unionmod.A2_ART,
        allow_pickle=True,
    )

    az = np.load(
        unionmod.AST_CACHE,
        allow_pickle=True,
    )

    rpc_full = json.loads(
        RPC_FULL_JSON.read_text(
            encoding="utf-8"
        )
    )

    frozen_records = {
        int(r["query"]): r
        for r in rpc_full["records"]
    }

    query_ids = sorted(
        frozen_records
    )

    assert len(query_ids) == 82

    candidates_all = (
        a2[
            "candidates"
        ].astype(
            np.int32
        )
    )

    valid_all = (
        a2[
            "candidate_ast_valid"
        ].astype(
            bool
        )
    )

    names_all = (
        az[
            "feature_names"
        ].astype(
            str
        )
    )

    signature = (
        az[
            "signature"
        ].astype(
            np.float32
        )
    )

    allowed_ids = np.flatnonzero(
        names_all
        !=
        "max_ast_depth"
    )

    assert len(
        allowed_ids
    ) == 44

    # ==============================================================
    # Global counters.
    # ==============================================================

    agg = {
        # Unique bound states:
        "unique_bound_states":
            0,

        "unique_C_lt_Q":
            0,

        "unique_Q_lt_C":
            0,

        "unique_equal":
            0,

        "unique_C_lt_Q_margin_sum":
            0,

        "unique_Q_lt_C_margin_sum":
            0,

        "unique_C_lt_Q_margin_max":
            0,

        "unique_Q_lt_C_margin_max":
            0,

        # Actual RPC recursion states:
        "recursive_states":
            0,

        "recursive_C_lt_Q":
            0,

        "recursive_Q_lt_C":
            0,

        "recursive_equal":
            0,

        # Child-bound evaluation events.
        # These use multiplicity because these are actual
        # branch-allocation decisions encountered by RPC.
        "child_bound_events":
            0,

        "child_C_lt_Q":
            0,

        "child_Q_lt_C":
            0,

        "child_equal":
            0,

        # Composite two-child branch bounds.
        "branch_events":
            0,

        "branch_Csum_lt_Qsum":
            0,

        "branch_Qsum_lt_Csum":
            0,

        "branch_equal_sum":
            0,

        # Shadow opportunity:
        # RPC would recurse, but min(C,Q) would prune now.
        "shadow_extra_prunable":
            0,

        "queries_with_C_lt_Q":
            0,

        "queries_with_Q_lt_C":
            0,

        "queries_with_both_directions":
            0,
    }

    records = []

    processed = 0
    t_all = time.perf_counter()

    for qi in query_ids:

        if (
            args.max_hard_queries
            >
            0
            and
            processed
            >=
            args.max_hard_queries
        ):

            break

        q0 = time.perf_counter()

        frozen = frozen_records[
            qi
        ]

        expected_joint = int(
            frozen[
                "joint_K5"
            ]
        )

        expected_rpc_states = int(
            frozen[
                "rpc"
            ][
                "states"
            ]
        )

        candidates = (
            candidates_all[
                qi
            ]
        )

        valid = (
            valid_all[
                qi
            ]
        )

        # ----------------------------------------------------------
        # Exact UNION library.
        # ----------------------------------------------------------

        atomic = np.full(
            (
                E,
                44,
            ),
            UNKNOWN,
            dtype=np.int8,
        )

        presence = (
            signature[
                candidates
            ][
                :,
                allowed_ids
            ]
            >
            0.0
        ).astype(
            np.int8
        )

        atomic[
            valid
        ] = (
            presence[
                valid
            ]
        )

        built = (
            tecs.build_tecs_library(
                atomic,
                valid,
                a4,
            )
        )

        lib_tecs = (
            built[
                "library"
            ]
        )

        (
            full_rcps,
            _,
            _,
            _,
        ) = (
            a4.build_synth_library(
                atomic
            )
        )

        (
            lib_rcps,
            _,
        ) = (
            a4.deduplicate_predicates(
                full_rcps,
                valid,
            )
        )

        (
            facts,
            _,
        ) = (
            unionmod.exact_column_union(
                lib_tecs,
                lib_rcps,
            )
        )

        P = int(
            facts.shape[
                1
            ]
        )

        ind = (
            gap.individual_b20_vector(
                facts,
                valid,
                a4,
            )
        )

        # ----------------------------------------------------------
        # Bitsets.
        # ----------------------------------------------------------

        full_mask = (
            (1 << E)
            -
            1
        )

        valid_bits = (
            bool_to_bits(
                valid
            )
        )

        target_bits = (
            bool_to_bits(
                ind
            )
        )

        target_ids = [
            int(x)
            for x in np.flatnonzero(
                ind
            )
        ]

        prefix_bits = [
            (1 << (t + 1))
            -
            1
            for t in range(E)
        ]

        (
            survive0_np,
            survive1_np,
        ) = (
            a4.build_survival_masks(
                facts
            )
        )

        survive0 = [
            bool_to_bits(
                survive0_np[
                    p
                ]
            )
            for p in range(P)
        ]

        survive1 = [
            bool_to_bits(
                survive1_np[
                    p
                ]
            )
            for p in range(P)
        ]

        valid0 = [
            bool_to_bits(
                valid
                &
                (
                    facts[
                        :,
                        p
                    ]
                    ==
                    ABSENT
                )
            )
            for p in range(P)
        ]

        valid1 = [
            bool_to_bits(
                valid
                &
                (
                    facts[
                        :,
                        p
                    ]
                    ==
                    PRESENT
                )
            )
            for p in range(P)
        ]

        # ----------------------------------------------------------
        # STOP.
        # ----------------------------------------------------------

        @lru_cache(
            maxsize=None
        )
        def stop_cov(mask):

            alive = (
                mask
                &
                target_bits
            )

            reward = 0

            for t in target_ids:

                bit = (
                    1 << t
                )

                if not (
                    alive
                    &
                    bit
                ):
                    continue

                rank = (
                    mask
                    &
                    prefix_bits[
                        t
                    ]
                ).bit_count()

                if rank <= B:
                    reward += 1

            return int(
                reward
            )

        # ----------------------------------------------------------
        # RPC C_L(M).
        # ----------------------------------------------------------

        @lru_cache(
            maxsize=None
        )
        def rpc_capacity(
            mask,
            leaves,
        ):

            slack = [
                B
                for _ in range(
                    int(leaves)
                )
            ]

            reward = 0

            for i in range(E):

                bit = (
                    1 << i
                )

                if not (
                    mask
                    &
                    bit
                ):
                    continue

                if not (
                    valid_bits
                    &
                    bit
                ):

                    for j in range(
                        len(
                            slack
                        )
                    ):

                        if slack[
                            j
                        ] > 0:

                            slack[
                                j
                            ] -= 1

                    continue

                if (
                    target_bits
                    &
                    bit
                ):

                    chosen = -1

                    for j, s in enumerate(
                        slack
                    ):

                        if s > 0:

                            chosen = j
                            break

                    if chosen >= 0:

                        reward += 1

                        slack[
                            chosen
                        ] -= 1

                    continue

                if slack[
                    0
                ] > 0:

                    slack[
                        0
                    ] -= 1

            return int(
                reward
            )

        # ----------------------------------------------------------
        # DQ Q_{d,L}(M).
        # ----------------------------------------------------------

        @lru_cache(
            maxsize=None
        )
        def dq_bound(
            mask,
            depth,
            leaves,
        ):

            k = min(
                int(depth),
                int(leaves)
                -
                1,
            )

            if k <= 0:

                return stop_cov(
                    mask
                )

            alive = (
                mask
                &
                target_bits
            )

            feasible = 0

            for t in target_ids:

                bit = (
                    1 << t
                )

                if not (
                    alive
                    &
                    bit
                ):
                    continue

                rank = (
                    mask
                    &
                    prefix_bits[
                        t
                    ]
                ).bit_count()

                if rank <= B:

                    feasible += 1
                    continue

                delta = int(
                    rank
                    -
                    B
                )

                blockers = (
                    mask
                    &
                    prefix_bits[
                        t
                    ]
                )

                top = [
                    0
                    for _ in range(
                        k
                    )
                ]

                top_sum = 0
                possible = False

                for p in range(P):

                    ans = int(
                        facts[
                            t,
                            p
                        ]
                    )

                    if ans == ABSENT:

                        opposite = (
                            valid1[
                                p
                            ]
                        )

                    else:

                        opposite = (
                            valid0[
                                p
                            ]
                        )

                    sigma = (
                        blockers
                        &
                        opposite
                    ).bit_count()

                    if sigma <= top[
                        0
                    ]:

                        continue

                    top_sum += (
                        sigma
                        -
                        top[
                            0
                        ]
                    )

                    top[
                        0
                    ] = sigma

                    top.sort()

                    if (
                        top_sum
                        >=
                        delta
                    ):

                        possible = True
                        break

                if possible:
                    feasible += 1

            return int(
                feasible
            )

        # ----------------------------------------------------------
        # Unique bound-state comparison.
        #
        # This function is cached, so unique counters refer to
        # distinct (mask, depth, leaves) states.
        # ----------------------------------------------------------

        unique_local = {
            "total": 0,
            "C_lt_Q": 0,
            "Q_lt_C": 0,
            "equal": 0,
            "C_margin_sum": 0,
            "Q_margin_sum": 0,
            "C_margin_max": 0,
            "Q_margin_max": 0,
        }

        example_C = None
        example_Q = None

        @lru_cache(
            maxsize=None
        )
        def bound_pair(
            mask,
            depth,
            leaves,
        ):

            nonlocal example_C
            nonlocal example_Q

            C = rpc_capacity(
                mask,
                leaves,
            )

            Q = dq_bound(
                mask,
                depth,
                leaves,
            )

            unique_local[
                "total"
            ] += 1

            if C < Q:

                margin = int(
                    Q - C
                )

                unique_local[
                    "C_lt_Q"
                ] += 1

                unique_local[
                    "C_margin_sum"
                ] += margin

                unique_local[
                    "C_margin_max"
                ] = max(
                    unique_local[
                        "C_margin_max"
                    ],
                    margin,
                )

                if example_C is None:

                    example_C = {
                        "mask_size":
                            int(
                                mask.bit_count()
                            ),

                        "alive_targets":
                            int(
                                (
                                    mask
                                    &
                                    target_bits
                                ).bit_count()
                            ),

                        "depth":
                            int(
                                depth
                            ),

                        "leaves":
                            int(
                                leaves
                            ),

                        "C":
                            int(
                                C
                            ),

                        "Q":
                            int(
                                Q
                            ),

                        "STOP":
                            int(
                                stop_cov(
                                    mask
                                )
                            ),
                    }

            elif Q < C:

                margin = int(
                    C - Q
                )

                unique_local[
                    "Q_lt_C"
                ] += 1

                unique_local[
                    "Q_margin_sum"
                ] += margin

                unique_local[
                    "Q_margin_max"
                ] = max(
                    unique_local[
                        "Q_margin_max"
                    ],
                    margin,
                )

                if example_Q is None:

                    example_Q = {
                        "mask_size":
                            int(
                                mask.bit_count()
                            ),

                        "alive_targets":
                            int(
                                (
                                    mask
                                    &
                                    target_bits
                                ).bit_count()
                            ),

                        "depth":
                            int(
                                depth
                            ),

                        "leaves":
                            int(
                                leaves
                            ),

                        "C":
                            int(
                                C
                            ),

                        "Q":
                            int(
                                Q
                            ),

                        "STOP":
                            int(
                                stop_cov(
                                    mask
                                )
                            ),
                    }

            else:

                unique_local[
                    "equal"
                ] += 1

            return (
                int(
                    C
                ),
                int(
                    Q
                ),
            )

        # ----------------------------------------------------------
        # RPC-only exact solver.
        #
        # Q is observed but NEVER used for:
        # - pruning
        # - incumbent
        # - early stop
        # - search order.
        #
        # Thus the audited search space is exactly RPC-only.
        # ----------------------------------------------------------

        recursion = {
            "total": 0,
            "C_lt_Q": 0,
            "Q_lt_C": 0,
            "equal": 0,
        }

        child_events = {
            "total": 0,
            "C_lt_Q": 0,
            "Q_lt_C": 0,
            "equal": 0,
        }

        branch_events = {
            "total": 0,
            "Csum_lt_Qsum": 0,
            "Qsum_lt_Csum": 0,
            "equal_sum": 0,
            "shadow_extra_prunable": 0,
        }

        @lru_cache(
            maxsize=None
        )
        def V(
            mask,
            depth,
            leaves,
        ):

            recursion[
                "total"
            ] += 1

            base = stop_cov(
                mask
            )

            C_state, Q_state = (
                bound_pair(
                    mask,
                    depth,
                    leaves,
                )
            )

            if C_state < Q_state:

                recursion[
                    "C_lt_Q"
                ] += 1

            elif Q_state < C_state:

                recursion[
                    "Q_lt_C"
                ] += 1

            else:

                recursion[
                    "equal"
                ] += 1

            # IMPORTANT:
            # only RPC C_state is used.
            ub_state = C_state

            assert (
                base
                <=
                ub_state
            )

            if (
                ub_state
                ==
                base
            ):

                return int(
                    base
                )

            if (
                depth <= 0
                or
                leaves <= 1
                or
                mask.bit_count()
                <=
                B
            ):

                return int(
                    base
                )

            best = int(
                base
            )

            for p in range(P):

                if not (
                    mask
                    &
                    valid0[
                        p
                    ]
                ):

                    continue

                if not (
                    mask
                    &
                    valid1[
                        p
                    ]
                ):

                    continue

                m0 = (
                    mask
                    &
                    survive0[
                        p
                    ]
                )

                m1 = (
                    mask
                    &
                    survive1[
                        p
                    ]
                )

                if (
                    m0 == mask
                    or
                    m1 == mask
                ):

                    continue

                for l0 in range(
                    1,
                    leaves
                ):

                    l1 = (
                        leaves
                        -
                        l0
                    )

                    C0, Q0 = (
                        bound_pair(
                            m0,
                            depth - 1,
                            l0,
                        )
                    )

                    C1, Q1 = (
                        bound_pair(
                            m1,
                            depth - 1,
                            l1,
                        )
                    )

                    for Cx, Qx in (
                        (C0, Q0),
                        (C1, Q1),
                    ):

                        child_events[
                            "total"
                        ] += 1

                        if Cx < Qx:

                            child_events[
                                "C_lt_Q"
                            ] += 1

                        elif Qx < Cx:

                            child_events[
                                "Q_lt_C"
                            ] += 1

                        else:

                            child_events[
                                "equal"
                            ] += 1

                    Csum = (
                        C0 + C1
                    )

                    Qsum = (
                        Q0 + Q1
                    )

                    branch_events[
                        "total"
                    ] += 1

                    if Csum < Qsum:

                        branch_events[
                            "Csum_lt_Qsum"
                        ] += 1

                    elif Qsum < Csum:

                        branch_events[
                            "Qsum_lt_Csum"
                        ] += 1

                    else:

                        branch_events[
                            "equal_sum"
                        ] += 1

                    # Shadow combined bound.
                    min_sum = (
                        min(
                            C0,
                            Q0,
                        )
                        +
                        min(
                            C1,
                            Q1,
                        )
                    )

                    # RPC alone would NOT prune,
                    # but the combined bound would.
                    if (
                        Csum > best
                        and
                        min_sum <= best
                    ):

                        branch_events[
                            "shadow_extra_prunable"
                        ] += 1

                    # ----------------------------------------------
                    # FROZEN RPC-ONLY pruning.
                    # DQ is NOT used below this line.
                    # ----------------------------------------------

                    if (
                        Csum
                        <=
                        best
                    ):

                        continue

                    cand = (
                        V(
                            m0,
                            depth - 1,
                            l0,
                        )
                        +
                        V(
                            m1,
                            depth - 1,
                            l1,
                        )
                    )

                    if cand > best:

                        best = int(
                            cand
                        )

                    if (
                        best
                        ==
                        ub_state
                    ):

                        break

                if (
                    best
                    ==
                    ub_state
                ):

                    break

            return int(
                best
            )

        start = time.perf_counter()

        value = V(
            full_mask,
            3,
            5,
        )

        elapsed = (
            time.perf_counter()
            -
            start
        )

        assert (
            value
            ==
            expected_joint
        ), (
            qi,
            value,
            expected_joint,
        )

        assert (
            recursion[
                "total"
            ]
            ==
            expected_rpc_states
        ), (
            qi,
            recursion[
                "total"
            ],
            expected_rpc_states,
        )

        has_C = (
            unique_local[
                "C_lt_Q"
            ] > 0
        )

        has_Q = (
            unique_local[
                "Q_lt_C"
            ] > 0
        )

        # ----------------------------------------------------------
        # Aggregate.
        # ----------------------------------------------------------

        agg[
            "unique_bound_states"
        ] += unique_local[
            "total"
        ]

        agg[
            "unique_C_lt_Q"
        ] += unique_local[
            "C_lt_Q"
        ]

        agg[
            "unique_Q_lt_C"
        ] += unique_local[
            "Q_lt_C"
        ]

        agg[
            "unique_equal"
        ] += unique_local[
            "equal"
        ]

        agg[
            "unique_C_lt_Q_margin_sum"
        ] += unique_local[
            "C_margin_sum"
        ]

        agg[
            "unique_Q_lt_C_margin_sum"
        ] += unique_local[
            "Q_margin_sum"
        ]

        agg[
            "unique_C_lt_Q_margin_max"
        ] = max(
            agg[
                "unique_C_lt_Q_margin_max"
            ],
            unique_local[
                "C_margin_max"
            ],
        )

        agg[
            "unique_Q_lt_C_margin_max"
        ] = max(
            agg[
                "unique_Q_lt_C_margin_max"
            ],
            unique_local[
                "Q_margin_max"
            ],
        )

        for key in (
            "total",
            "C_lt_Q",
            "Q_lt_C",
            "equal",
        ):

            map_key = {
                "total":
                    "recursive_states",

                "C_lt_Q":
                    "recursive_C_lt_Q",

                "Q_lt_C":
                    "recursive_Q_lt_C",

                "equal":
                    "recursive_equal",
            }[
                key
            ]

            agg[
                map_key
            ] += recursion[
                key
            ]

        agg[
            "child_bound_events"
        ] += child_events[
            "total"
        ]

        agg[
            "child_C_lt_Q"
        ] += child_events[
            "C_lt_Q"
        ]

        agg[
            "child_Q_lt_C"
        ] += child_events[
            "Q_lt_C"
        ]

        agg[
            "child_equal"
        ] += child_events[
            "equal"
        ]

        agg[
            "branch_events"
        ] += branch_events[
            "total"
        ]

        agg[
            "branch_Csum_lt_Qsum"
        ] += branch_events[
            "Csum_lt_Qsum"
        ]

        agg[
            "branch_Qsum_lt_Csum"
        ] += branch_events[
            "Qsum_lt_Csum"
        ]

        agg[
            "branch_equal_sum"
        ] += branch_events[
            "equal_sum"
        ]

        agg[
            "shadow_extra_prunable"
        ] += branch_events[
            "shadow_extra_prunable"
        ]

        if has_C:

            agg[
                "queries_with_C_lt_Q"
            ] += 1

        if has_Q:

            agg[
                "queries_with_Q_lt_C"
            ] += 1

        if (
            has_C
            and
            has_Q
        ):

            agg[
                "queries_with_both_directions"
            ] += 1

        records.append(
            {
                "query":
                    int(
                        qi
                    ),

                "joint_K5":
                    int(
                        value
                    ),

                "rpc_states":
                    int(
                        recursion[
                            "total"
                        ]
                    ),

                "unique":
                    dict(
                        unique_local
                    ),

                "recursion":
                    dict(
                        recursion
                    ),

                "child_events":
                    dict(
                        child_events
                    ),

                "branch_events":
                    dict(
                        branch_events
                    ),

                "has_C_lt_Q":
                    bool(
                        has_C
                    ),

                "has_Q_lt_C":
                    bool(
                        has_Q
                    ),

                "example_C_lt_Q":
                    example_C,

                "example_Q_lt_C":
                    example_Q,

                "runtime_seconds":
                    float(
                        elapsed
                    ),
            }
        )

        processed += 1

        print(
            f"q={qi:03d} "
            f"states={recursion['total']} "
            f"unique={unique_local['total']} "
            f"C<Q={unique_local['C_lt_Q']} "
            f"Q<C={unique_local['Q_lt_C']} "
            f"eq={unique_local['equal']} "
            f"shadowExtraPrune="
            f"{branch_events['shadow_extra_prunable']} "
            f"both={int(has_C and has_Q)} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    # ==============================================================
    # Summary.
    # ==============================================================

    total_time = (
        time.perf_counter()
        -
        t_all
    )

    print()
    print("=" * 162)
    print(
        "RPC–DQ COMPLEMENTARITY SUMMARY"
    )
    print("=" * 162)

    print(
        "hard queries processed =",
        processed,
    )

    print()
    print(
        "UNIQUE BOUND STATES"
    )

    print(
        "total =",
        agg[
            "unique_bound_states"
        ],
    )

    print(
        "RPC strictly tighter (C<Q) =",
        agg[
            "unique_C_lt_Q"
        ],
    )

    print(
        "DQ strictly tighter (Q<C) =",
        agg[
            "unique_Q_lt_C"
        ],
    )

    print(
        "equal =",
        agg[
            "unique_equal"
        ],
    )

    print(
        "max RPC-strict margin Q-C =",
        agg[
            "unique_C_lt_Q_margin_max"
        ],
    )

    print(
        "max DQ-strict margin C-Q =",
        agg[
            "unique_Q_lt_C_margin_max"
        ],
    )

    print()
    print(
        "RPC RECURSION STATES"
    )

    print(
        "total =",
        agg[
            "recursive_states"
        ],
    )

    print(
        "C<Q =",
        agg[
            "recursive_C_lt_Q"
        ],
    )

    print(
        "Q<C =",
        agg[
            "recursive_Q_lt_C"
        ],
    )

    print(
        "equal =",
        agg[
            "recursive_equal"
        ],
    )

    print()
    print(
        "CHILD-BOUND EVENTS"
    )

    print(
        "total =",
        agg[
            "child_bound_events"
        ],
    )

    print(
        "C<Q =",
        agg[
            "child_C_lt_Q"
        ],
    )

    print(
        "Q<C =",
        agg[
            "child_Q_lt_C"
        ],
    )

    print(
        "equal =",
        agg[
            "child_equal"
        ],
    )

    print()
    print(
        "TWO-CHILD BRANCH-BOUND EVENTS"
    )

    print(
        "total =",
        agg[
            "branch_events"
        ],
    )

    print(
        "RPC sum tighter =",
        agg[
            "branch_Csum_lt_Qsum"
        ],
    )

    print(
        "DQ sum tighter =",
        agg[
            "branch_Qsum_lt_Csum"
        ],
    )

    print(
        "equal sum =",
        agg[
            "branch_equal_sum"
        ],
    )

    print(
        "shadow extra-prunable under min(C,Q) =",
        agg[
            "shadow_extra_prunable"
        ],
    )

    print()
    print(
        "QUERY-LEVEL COMPLEMENTARITY"
    )

    print(
        "queries with some C<Q state =",
        agg[
            "queries_with_C_lt_Q"
        ],
    )

    print(
        "queries with some Q<C state =",
        agg[
            "queries_with_Q_lt_C"
        ],
    )

    print(
        "queries exhibiting BOTH directions =",
        agg[
            "queries_with_both_directions"
        ],
    )

    assert (
        agg[
            "unique_C_lt_Q"
        ]
        >
        0
    )

    assert (
        agg[
            "unique_Q_lt_C"
        ]
        >
        0
    )

    print()
    print(
        "EMPIRICAL TWO-DIRECTION STRICTNESS = PASS"
    )

    print(
        "CONSTRUCTION-LEVEL INCOMPARABILITY = PASS"
    )

    print(
        "RPC-ONLY EXACT VALUE / STATE PARITY = PASS"
    )

    print(
        "runtime seconds =",
        total_time,
    )

    OUT = (
        D
        /
        f"rpc_dq_complementarity_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "name":
                    "RPC-DQ Complementarity Audit v0",

                "processed":
                    int(
                        processed
                    ),

                "constructions":
                    constructions,

                "aggregate":
                    agg,

                "records":
                    records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "JSON =",
        OUT,
    )

    print(
        "SCRIPT SHA256 =",
        sha256_file(
            Path(
                __file__
            )
        ),
    )

    print("=" * 162)


if __name__ == "__main__":
    main()
