from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import importlib.util
import json
import time

import numpy as np


ROOT = Path(
    "/root/autodl-tmp/work2-algorithm-innovation"
)

D = ROOT / "fhex_v1/obstruction_audit_v0"

UNION_SCRIPT = (
    D / "audit_tecs_rcps_union_v0.py"
)

JOINT_SCRIPT = (
    D / "audit_joint_k5_union_exact_v0.py"
)

JOINT_JSON = (
    D / "joint_k5_union_exact_v0_full82.json"
)

EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_JOINT_SHA = (
    "33d8a037da621e3213f94e8578a7bdb4d9e87483d5e9ef06f1eab885c0136d55"
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

            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def load_module(path, name):

    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    mod = (
        importlib.util.module_from_spec(
            spec
        )
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        mod
    )

    return mod


def bool_to_bits(x):

    out = 0

    for i in np.flatnonzero(x):

        out |= (
            1 << int(i)
        )

    return out


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

    print("=" * 154)
    print(
        "RANK-PREFIX CAPACITY UPPER-BOUND VALUE AUDIT v0"
    )
    print(
        "GENERIC TARGET-COUNT UB vs EXACT RPC CAPACITY UB"
    )
    print(
        "SAME JOINT K<=5 UNION-PREDICATE DP / SAME SEARCH ORDER"
    )
    print("=" * 154)

    # ==============================================================
    # Frozen integrity
    # ==============================================================

    assert (
        sha256_file(
            UNION_SCRIPT
        )
        ==
        EXPECTED_UNION_SHA
    )

    assert (
        sha256_file(
            JOINT_SCRIPT
        )
        ==
        EXPECTED_JOINT_SHA
    )

    assert JOINT_JSON.exists()

    print(
        "FROZEN SCRIPT INTEGRITY = PASS"
    )

    # ==============================================================
    # Load frozen objects
    # ==============================================================

    unionmod = load_module(
        UNION_SCRIPT,
        "rpc_union",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "rpc_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "rpc_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "rpc_tecs",
    )

    a2 = np.load(
        unionmod.A2_ART,
        allow_pickle=True,
    )

    az = np.load(
        unionmod.AST_CACHE,
        allow_pickle=True,
    )

    joint = json.loads(
        JOINT_JSON.read_text(
            encoding="utf-8"
        )
    )

    records_by_query = {
        int(r["query"]): r
        for r in joint[
            "records"
        ]
    }

    query_ids = sorted(
        records_by_query
    )

    assert len(
        query_ids
    ) == 82

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
    # Aggregates
    # ==============================================================

    processed = 0

    aggregate = {
        "generic_runtime": 0.0,
        "rpc_runtime": 0.0,

        "generic_states": 0,
        "rpc_states": 0,

        "generic_branch_options": 0,
        "rpc_branch_options": 0,

        "generic_pruned": 0,
        "rpc_pruned": 0,

        "generic_state_bound_stops": 0,
        "rpc_state_bound_stops": 0,
    }

    out_records = []

    t_all = time.perf_counter()

    # ==============================================================
    # Query loop
    # ==============================================================

    for qi in query_ids:

        if (
            args.max_hard_queries > 0
            and
            processed
            >=
            args.max_hard_queries
        ):
            break

        q_start = time.perf_counter()

        frozen = (
            records_by_query[
                qi
            ]
        )

        expected_joint = int(
            frozen[
                "joint_union_K5"
            ]
        )

        expected_O = int(
            frozen[
                "O"
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
        # Reconstruct exact UNION library.
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

        O = int(
            ind.sum()
        )

        assert O == expected_O

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

        target_ids = [
            int(x)
            for x in np.flatnonzero(
                ind
            )
        ]

        # ----------------------------------------------------------
        # Exact stop coverage shared by both solvers.
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
        # Rank-Prefix Capacity oracle C_L(M).
        #
        # This deliberately ignores predicate realizability
        # and depth. Every valid candidate can be assigned to
        # any one of L leaves; invalid candidates survive all
        # leaves.
        # ----------------------------------------------------------

        @lru_cache(
            maxsize=None
        )
        def rpc_capacity(
            mask,
            leaves,
        ):

            leaves = int(
                leaves
            )

            assert leaves >= 1

            slack = [
                B
                for _ in range(
                    leaves
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

                # UNKNOWN / invalid:
                # common erosion on every leaf.
                if not (
                    valid_bits
                    &
                    bit
                ):

                    for j in range(
                        leaves
                    ):

                        if slack[j] > 0:
                            slack[j] -= 1

                    continue

                # Valid target:
                # smallest positive slack.
                if (
                    target_bits
                    &
                    bit
                ):

                    chosen = -1

                    for j in range(
                        leaves
                    ):

                        if slack[j] > 0:

                            chosen = j
                            break

                    if chosen >= 0:

                        reward += 1

                        slack[
                            chosen
                        ] -= 1

                    # If no positive slack remains,
                    # assign target to a dead leaf.
                    continue

                # Valid non-target:
                # smallest slack.
                if slack[0] > 0:

                    slack[0] -= 1

                # Sortedness is preserved by these operations.

            return int(
                reward
            )

        # Architecture relaxation sanity.
        #
        # Previous exact/constructive audits showed K=5
        # arbitrary policy reaches O on this hard-core set.
        assert (
            rpc_capacity(
                full_mask,
                5,
            )
            ==
            O
        ), (
            qi,
            rpc_capacity(
                full_mask,
                5,
            ),
            O,
        )

        # ==========================================================
        # Exact solver factory.
        # Only the admissible UB differs.
        # ==========================================================

        def run_solver(
            mode,
        ):

            assert mode in (
                "generic",
                "rpc",
            )

            stats = {
                "states": 0,
                "branch_options": 0,
                "pruned": 0,
                "state_bound_stops": 0,
                "legal_predicate_visits": 0,
            }

            def state_upper_bound(
                mask,
                leaves,
            ):

                if mode == "generic":

                    return int(
                        (
                            mask
                            &
                            target_bits
                        ).bit_count()
                    )

                return int(
                    rpc_capacity(
                        mask,
                        leaves,
                    )
                )

            @lru_cache(
                maxsize=None
            )
            def V(
                mask,
                depth,
                leaves,
            ):

                stats[
                    "states"
                ] += 1

                base = (
                    stop_cov(
                        mask
                    )
                )

                ub_state = (
                    state_upper_bound(
                        mask,
                        leaves,
                    )
                )

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

                    stats[
                        "state_bound_stops"
                    ] += 1

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

                for p in range(
                    P
                ):

                    # Frozen A4 legality:
                    # both answers among valid
                    # survivors of this state.
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

                    stats[
                        "legal_predicate_visits"
                    ] += 1

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

                        stats[
                            "branch_options"
                        ] += 1

                        ub = (
                            state_upper_bound(
                                m0,
                                l0,
                            )
                            +
                            state_upper_bound(
                                m1,
                                l1,
                            )
                        )

                        if ub <= best:

                            stats[
                                "pruned"
                            ] += 1

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

                        # Exact local ceiling reached.
                        if best == ub_state:

                            break

                    if best == ub_state:

                        break

                return int(
                    best
                )

            t0 = (
                time.perf_counter()
            )

            value = V(
                full_mask,
                3,
                5,
            )

            runtime = (
                time.perf_counter()
                -
                t0
            )

            info = (
                V.cache_info()
            )

            stats[
                "cache_hits"
            ] = int(
                info.hits
            )

            stats[
                "cache_misses"
            ] = int(
                info.misses
            )

            stats[
                "runtime"
            ] = float(
                runtime
            )

            stats[
                "value"
            ] = int(
                value
            )

            return stats

        # ==========================================================
        # Same search twice: only UB differs.
        # ==========================================================

        generic = (
            run_solver(
                "generic"
            )
        )

        rpc = (
            run_solver(
                "rpc"
            )
        )

        assert (
            generic[
                "value"
            ]
            ==
            expected_joint
        ), (
            qi,
            generic[
                "value"
            ],
            expected_joint,
        )

        assert (
            rpc[
                "value"
            ]
            ==
            expected_joint
        ), (
            qi,
            rpc[
                "value"
            ],
            expected_joint,
        )

        assert (
            generic[
                "value"
            ]
            ==
            rpc[
                "value"
            ]
        )

        state_reduction = (
            1.0
            -
            (
                rpc[
                    "states"
                ]
                /
                generic[
                    "states"
                ]
            )
        )

        runtime_speedup = (
            generic[
                "runtime"
            ]
            /
            max(
                rpc[
                    "runtime"
                ],
                1e-12,
            )
        )

        prune_generic = (
            generic[
                "pruned"
            ]
        )

        prune_rpc = (
            rpc[
                "pruned"
            ]
        )

        print(
            f"q={qi:03d} "
            f"J5={expected_joint:02d} "
            f"states "
            f"{generic['states']}->{rpc['states']} "
            f"reduction={100.0*state_reduction:.2f}% "
            f"pruned "
            f"{prune_generic}->{prune_rpc} "
            f"time "
            f"{generic['runtime']:.3f}s->"
            f"{rpc['runtime']:.3f}s "
            f"speedup={runtime_speedup:.2f}x",
            flush=True,
        )

        aggregate[
            "generic_runtime"
        ] += float(
            generic[
                "runtime"
            ]
        )

        aggregate[
            "rpc_runtime"
        ] += float(
            rpc[
                "runtime"
            ]
        )

        aggregate[
            "generic_states"
        ] += int(
            generic[
                "states"
            ]
        )

        aggregate[
            "rpc_states"
        ] += int(
            rpc[
                "states"
            ]
        )

        aggregate[
            "generic_branch_options"
        ] += int(
            generic[
                "branch_options"
            ]
        )

        aggregate[
            "rpc_branch_options"
        ] += int(
            rpc[
                "branch_options"
            ]
        )

        aggregate[
            "generic_pruned"
        ] += int(
            generic[
                "pruned"
            ]
        )

        aggregate[
            "rpc_pruned"
        ] += int(
            rpc[
                "pruned"
            ]
        )

        aggregate[
            "generic_state_bound_stops"
        ] += int(
            generic[
                "state_bound_stops"
            ]
        )

        aggregate[
            "rpc_state_bound_stops"
        ] += int(
            rpc[
                "state_bound_stops"
            ]
        )

        out_records.append(
            {
                "query":
                    int(
                        qi
                    ),

                "joint_K5":
                    int(
                        expected_joint
                    ),

                "O":
                    int(
                        O
                    ),

                "generic":
                    generic,

                "rpc":
                    rpc,

                "state_reduction_fraction":
                    float(
                        state_reduction
                    ),

                "runtime_speedup":
                    float(
                        runtime_speedup
                    ),
            }
        )

        processed += 1

        del stop_cov
        del rpc_capacity

    # ==============================================================
    # Aggregate
    # ==============================================================

    elapsed_all = (
        time.perf_counter()
        -
        t_all
    )

    generic_states = (
        aggregate[
            "generic_states"
        ]
    )

    rpc_states = (
        aggregate[
            "rpc_states"
        ]
    )

    generic_runtime = (
        aggregate[
            "generic_runtime"
        ]
    )

    rpc_runtime = (
        aggregate[
            "rpc_runtime"
        ]
    )

    state_reduction = (
        1.0
        -
        (
            rpc_states
            /
            generic_states
        )
    )

    speedup = (
        generic_runtime
        /
        max(
            rpc_runtime,
            1e-12,
        )
    )

    print()
    print("=" * 154)
    print(
        "RPC CAPACITY BOUND SUMMARY"
    )
    print("=" * 154)

    print(
        "hard queries processed =",
        processed,
    )

    print(
        "generic states =",
        generic_states,
    )

    print(
        "RPC states =",
        rpc_states,
    )

    print(
        "state reduction fraction =",
        state_reduction,
    )

    print(
        "generic branch options =",
        aggregate[
            "generic_branch_options"
        ],
    )

    print(
        "RPC branch options =",
        aggregate[
            "rpc_branch_options"
        ],
    )

    print(
        "generic pruned branch options =",
        aggregate[
            "generic_pruned"
        ],
    )

    print(
        "RPC pruned branch options =",
        aggregate[
            "rpc_pruned"
        ],
    )

    print(
        "generic states solved directly by bound =",
        aggregate[
            "generic_state_bound_stops"
        ],
    )

    print(
        "RPC states solved directly by bound =",
        aggregate[
            "rpc_state_bound_stops"
        ],
    )

    print(
        "generic runtime seconds =",
        generic_runtime,
    )

    print(
        "RPC runtime seconds =",
        rpc_runtime,
    )

    print(
        "runtime speedup =",
        speedup,
    )

    print(
        "wall runtime seconds =",
        elapsed_all,
    )

    print(
        "EXACT VALUE PARITY = PASS"
    )

    OUT = (
        D
        /
        f"rpc_capacity_bound_value_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "name":
                    "Rank-Prefix Capacity Bound Value Audit v0",

                "processed":
                    int(
                        processed
                    ),

                "aggregate":
                    aggregate,

                "state_reduction_fraction":
                    float(
                        state_reduction
                    ),

                "runtime_speedup":
                    float(
                        speedup
                    ),

                "records":
                    out_records,
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

    print("=" * 154)


if __name__ == "__main__":
    main()
