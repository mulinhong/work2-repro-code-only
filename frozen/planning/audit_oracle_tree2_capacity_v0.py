from __future__ import annotations

from pathlib import Path
from itertools import combinations_with_replacement, product
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

UNION_ART = (
    D / "tecs_rcps_union_v0.npz"
)

V0_SCRIPT = (
    D / "audit_depth2_obstruction_v0.py"
)

ORACLE2_SCRIPT = (
    D / "audit_oracle_continuation_capacity_v0.py"
)

OUT_NPZ = (
    D / "oracle_tree2_capacity_v0.npz"
)

OUT_JSON = (
    D / "oracle_tree2_capacity_v0.json"
)


EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_V0_SHA = (
    "aa4e1d359ff9dfe801a1ddfd23cc91c35c2f05e36dd08113f5d7af776b3f46e8"
)

EXPECTED_ORACLE2_SHA = (
    "791efbebbe03f02f63c8a2b65f545736b46b9ddcbbd3e6a934dd942a0465059a"
)


E = 80
B = 20
UNKNOWN = -1

NEG = -10000


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

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    mod = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        mod
    )

    return mod


# =====================================================================
# Exact symmetry-quotiented four-leaf automaton
# =====================================================================

def build_four_leaf_automaton(budget):

    cap = budget + 1

    # Count values:
    # 0 ... B, B+1
    #
    # B+1 is the saturated "already too large" state.
    values = range(
        cap + 1
    )

    states = np.asarray(
        list(
            combinations_with_replacement(
                values,
                4,
            )
        ),
        dtype=np.int16,
    )

    index = {
        tuple(
            int(x)
            for x in s
        ):
        i

        for i, s in enumerate(
            states
        )
    }

    n = len(states)

    next_valid = np.zeros(
        (
            4,
            n,
        ),
        dtype=np.int32,
    )

    reward_valid = np.zeros(
        (
            4,
            n,
        ),
        dtype=np.int8,
    )

    next_unknown = np.zeros(
        n,
        dtype=np.int32,
    )

    for i, s0 in enumerate(
        states
    ):

        s0 = [
            int(x)
            for x in s0
        ]

        # --------------------------------------------------------------
        # Invalid / UNKNOWN candidate:
        # survives all four leaves.
        # --------------------------------------------------------------

        u = [
            min(
                cap,
                x + 1,
            )
            for x in s0
        ]

        # Same increment on every component preserves sorting.
        next_unknown[i] = index[
            tuple(u)
        ]

        # --------------------------------------------------------------
        # Valid candidate:
        # choose exactly one of the four leaves.
        # --------------------------------------------------------------

        for leaf in range(
            4
        ):

            before = int(
                s0[
                    leaf
                ]
            )

            x = list(
                s0
            )

            x[
                leaf
            ] = min(
                cap,
                before + 1,
            )

            x.sort()

            next_valid[
                leaf,
                i,
            ] = index[
                tuple(x)
            ]

            # A target assigned to this leaf is repaired iff
            # its post-insertion leaf rank <= B.
            reward_valid[
                leaf,
                i,
            ] = int(
                before + 1
                <=
                budget
            )

    return {
        "budget":
            int(
                budget
            ),

        "cap":
            int(
                cap
            ),

        "states":
            states,

        "initial":
            int(
                index[
                    (
                        0,
                        0,
                        0,
                        0,
                    )
                ]
            ),

        "next_valid":
            next_valid,

        "reward_valid":
            reward_valid,

        "next_unknown":
            next_unknown,
    }


def oracle_four_leaf_dp(
    target_mask,
    valid,
    automaton,
):

    """
    Exact maximum target coverage for an arbitrary depth<=2
    binary policy.

    Equivalent view:
      every VALID candidate may be assigned to one of four leaves;
      every INVALID candidate is UNKNOWN and survives all four leaves.

    Leaf counts are quotiented by permutation symmetry.
    """

    target_mask = np.asarray(
        target_mask,
        dtype=bool,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    assert len(target_mask) == len(valid)

    assert not np.any(
        target_mask
        &
        ~valid
    )

    states = automaton[
        "states"
    ]

    next_valid = automaton[
        "next_valid"
    ]

    reward_valid = automaton[
        "reward_valid"
    ]

    next_unknown = automaton[
        "next_unknown"
    ]

    dp = np.full(
        len(states),
        NEG,
        dtype=np.int16,
    )

    dp[
        automaton[
            "initial"
        ]
    ] = 0

    for pos in range(
        len(valid)
    ):

        ndp = np.full_like(
            dp,
            NEG,
        )

        if not bool(
            valid[
                pos
            ]
        ):

            np.maximum.at(
                ndp,
                next_unknown,
                dp,
            )

        else:

            is_target = int(
                target_mask[
                    pos
                ]
            )

            for leaf in range(
                4
            ):

                val = (
                    dp
                    +
                    is_target
                    *
                    reward_valid[
                        leaf
                    ]
                )

                np.maximum.at(
                    ndp,
                    next_valid[
                        leaf
                    ],
                    val,
                )

        dp = ndp

    return int(
        dp.max()
    )


# =====================================================================
# Independent small brute-force verification
# =====================================================================

def brute_force_four_leaf(
    target_mask,
    valid,
    budget,
):

    target_mask = np.asarray(
        target_mask,
        dtype=bool,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    valid_pos = np.flatnonzero(
        valid
    )

    best = -1

    for assignment in product(
        range(
            4
        ),
        repeat=len(
            valid_pos
        ),
    ):

        counts = [
            0,
            0,
            0,
            0,
        ]

        score = 0
        k = 0

        for pos in range(
            len(valid)
        ):

            if not bool(
                valid[
                    pos
                ]
            ):

                # UNKNOWN survives every leaf.
                for j in range(
                    4
                ):
                    counts[j] += 1

                continue

            leaf = int(
                assignment[
                    k
                ]
            )

            k += 1

            counts[
                leaf
            ] += 1

            if (
                bool(
                    target_mask[
                        pos
                    ]
                )
                and
                counts[
                    leaf
                ]
                <=
                budget
            ):

                score += 1

        best = max(
            best,
            score,
        )

    return int(
        best
    )


def self_test():

    rng = np.random.default_rng(
        20261005
    )

    small_budget = 2

    auto = (
        build_four_leaf_automaton(
            small_budget
        )
    )

    for trial in range(
        30
    ):

        n = int(
            rng.integers(
                4,
                8,
            )
        )

        valid = (
            rng.random(
                n
            )
            <
            0.75
        )

        # Keep brute force bounded and nontrivial.
        if int(
            valid.sum()
        ) == 0:

            valid[
                0
            ] = True

        while int(
            valid.sum()
        ) > 7:

            j = int(
                np.flatnonzero(
                    valid
                )[
                    -1
                ]
            )

            valid[
                j
            ] = False

        target = (
            valid
            &
            (
                rng.random(
                    n
                )
                <
                0.65
            )
        )

        exact_dp = (
            oracle_four_leaf_dp(
                target,
                valid,
                auto,
            )
        )

        brute = (
            brute_force_four_leaf(
                target,
                valid,
                small_budget,
            )
        )

        assert (
            exact_dp
            ==
            brute
        ), (
            trial,
            exact_dp,
            brute,
            valid,
            target,
        )

    print(
        "SMALL BRUTE-FORCE DP SELF-TEST = PASS"
    )


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-gap-queries",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    print("=" * 132)
    print(
        "ARBITRARY DEPTH-2 ORACLE TREE CAPACITY AUDIT v0"
    )
    print(
        "EXACT FOUR-LEAF ASSIGNMENT / UNKNOWN SEMANTICS PRESERVED"
    )
    print("=" * 132)

    # -----------------------------------------------------------------
    # Integrity
    # -----------------------------------------------------------------

    assert (
        sha256_file(
            UNION_SCRIPT
        )
        ==
        EXPECTED_UNION_SHA
    )

    assert (
        sha256_file(
            V0_SCRIPT
        )
        ==
        EXPECTED_V0_SHA
    )

    assert (
        sha256_file(
            ORACLE2_SCRIPT
        )
        ==
        EXPECTED_ORACLE2_SHA
    )

    print(
        "FROZEN SCRIPT INTEGRITY = PASS"
    )

    # Independent DP correctness test.
    self_test()

    # Production automaton.
    auto4 = (
        build_four_leaf_automaton(
            B
        )
    )

    print(
        "FOUR-LEAF QUOTIENT STATES =",
        len(
            auto4[
                "states"
            ]
        ),
    )

    assert (
        len(
            auto4[
                "states"
            ]
        )
        ==
        12650
    )

    # -----------------------------------------------------------------
    # Load frozen modules/data
    # -----------------------------------------------------------------

    unionmod = load_module(
        UNION_SCRIPT,
        "tree2_union",
    )

    v0 = load_module(
        V0_SCRIPT,
        "tree2_v0",
    )

    oracle2mod = load_module(
        ORACLE2_SCRIPT,
        "tree2_oracle2",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "tree2_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "tree2_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "tree2_tecs",
    )

    a2 = np.load(
        unionmod.A2_ART,
        allow_pickle=True,
    )

    az = np.load(
        unionmod.AST_CACHE,
        allow_pickle=True,
    )

    ua = np.load(
        UNION_ART,
        allow_pickle=False,
    )

    candidates_all = (
        a2[
            "candidates"
        ]
        .astype(
            np.int32
        )
    )

    valid_all = (
        a2[
            "candidate_ast_valid"
        ]
        .astype(
            bool
        )
    )

    names_all = (
        az[
            "feature_names"
        ]
        .astype(
            str
        )
    )

    signature = (
        az[
            "signature"
        ]
        .astype(
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

    O_saved = (
        ua[
            "union_O"
        ]
        .astype(
            np.int16
        )
    )

    S_saved = (
        ua[
            "union_S"
        ]
        .astype(
            np.int16
        )
    )

    G_saved = (
        ua[
            "union_Gamma"
        ]
        .astype(
            np.int16
        )
    )

    assert int(
        G_saved.sum()
    ) == 1381

    assert int(
        np.sum(
            G_saved
            >
            0
        )
    ) == 329

    print(
        "UNION ARTIFACT ANCHOR = PASS"
    )

    # -----------------------------------------------------------------
    # Result buffers
    # -----------------------------------------------------------------

    oracle2_arr = np.full(
        568,
        -1,
        dtype=np.int16,
    )

    oracle4_arr = np.full(
        568,
        -1,
        dtype=np.int16,
    )

    processed = 0

    sum_O = 0
    sum_S = 0
    sum_oracle2 = 0
    sum_oracle4 = 0

    child_headroom_total = 0
    root_headroom_total = 0
    depth2_residual_total = 0

    tree2_improves_oracle2 = 0
    tree2_closes_gap = 0

    t_all = time.perf_counter()

    # =================================================================
    # Queries
    # =================================================================

    for qi in range(
        568
    ):

        if int(
            G_saved[
                qi
            ]
        ) <= 0:

            continue

        if (
            args.max_gap_queries
            >
            0
            and
            processed
            >=
            args.max_gap_queries
        ):

            break

        q_t0 = (
            time.perf_counter()
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

        # -------------------------------------------------------------
        # UNION reconstruction
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # O / S parity
        # -------------------------------------------------------------

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

        (
            policy,
            s0,
            s1,
        ) = (
            a4.choose_bcct(
                facts,
                valid,
            )
        )

        shared = (
            gap.eval_compound_policy_all(
                policy,
                facts,
                s0,
                s1,
                valid,
                a4,
            )
        )

        S = int(
            shared.sum()
        )

        assert O == int(
            O_saved[
                qi
            ]
        )

        assert S == int(
            S_saved[
                qi
            ]
        )

        # -------------------------------------------------------------
        # Fixed-root / arbitrary-child oracle2
        # -------------------------------------------------------------

        char = (
            v0.build_root_characterization(
                facts,
                valid,
                ind,
                a4,
            )
        )

        assert (
            int(
                char[
                    "characterization_S"
                ]
            )
            ==
            S
        )

        (
            survive0,
            survive1,
        ) = (
            a4.build_survival_masks(
                facts
            )
        )

        target_ids = (
            np.flatnonzero(
                ind
            )
        )

        # Raw STOP baseline.
        oracle2 = int(
            np.sum(
                target_ids
                <
                B
            )
        )

        branch_cache = {}

        def branch_oracle(mask):

            key = (
                np.packbits(
                    mask
                )
                .tobytes()
            )

            if key not in branch_cache:

                branch_target = (
                    ind
                    &
                    mask
                )

                branch_cache[
                    key
                ] = (
                    oracle2mod.arbitrary_binary_branch_optimum(
                        mask,
                        valid,
                        branch_target,
                    )
                )

            return int(
                branch_cache[
                    key
                ]
            )

        for rec in char[
            "roots"
        ]:

            p = int(
                rec[
                    "root"
                ]
            )

            candidate_oracle2 = (
                branch_oracle(
                    survive0[
                        p
                    ]
                )
                +
                branch_oracle(
                    survive1[
                        p
                    ]
                )
            )

            oracle2 = max(
                oracle2,
                int(
                    candidate_oracle2
                ),
            )

        # -------------------------------------------------------------
        # Arbitrary-root + arbitrary-child depth-2 oracle.
        #
        # Every valid candidate chooses one of four leaves.
        # Invalid candidate survives all four.
        # -------------------------------------------------------------

        oracle4 = (
            oracle_four_leaf_dp(
                ind,
                valid,
                auto4,
            )
        )

        assert (
            S
            <=
            oracle2
            <=
            oracle4
            <=
            O
        ), (
            qi,
            S,
            oracle2,
            oracle4,
            O,
        )

        child_headroom = (
            oracle2
            -
            S
        )

        root_headroom = (
            oracle4
            -
            oracle2
        )

        depth2_residual = (
            O
            -
            oracle4
        )

        assert (
            child_headroom
            +
            root_headroom
            +
            depth2_residual
            ==
            O
            -
            S
        )

        oracle2_arr[
            qi
        ] = int(
            oracle2
        )

        oracle4_arr[
            qi
        ] = int(
            oracle4
        )

        sum_O += O
        sum_S += S
        sum_oracle2 += oracle2
        sum_oracle4 += oracle4

        child_headroom_total += (
            child_headroom
        )

        root_headroom_total += (
            root_headroom
        )

        depth2_residual_total += (
            depth2_residual
        )

        if (
            oracle4
            >
            oracle2
        ):

            tree2_improves_oracle2 += 1

        if (
            oracle4
            ==
            O
        ):

            tree2_closes_gap += 1

        elapsed = (
            time.perf_counter()
            -
            q_t0
        )

        print(
            f"q={qi:03d} "
            f"O={O:02d} "
            f"S={S:02d} "
            f"Oracle2={oracle2:02d} "
            f"OracleTree2={oracle4:02d} "
            f"G={O-S:02d} "
            f"child={child_headroom:02d} "
            f"root={root_headroom:02d} "
            f"depth2={depth2_residual:02d} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

        processed += 1

    # =================================================================
    # Summary
    # =================================================================

    runtime = (
        time.perf_counter()
        -
        t_all
    )

    print()
    print("=" * 132)
    print(
        "ARBITRARY DEPTH-2 ORACLE TREE SUMMARY"
    )
    print("=" * 132)

    print(
        "residual queries processed =",
        processed,
    )

    print(
        "sum O / UNION S / Oracle2 / OracleTree2 =",
        sum_O,
        sum_S,
        sum_oracle2,
        sum_oracle4,
    )

    print(
        "UNION residual gap =",
        sum_O
        -
        sum_S,
    )

    print(
        "child-family headroom =",
        child_headroom_total,
    )

    print(
        "root-family headroom after child relaxation =",
        root_headroom_total,
    )

    print(
        "genuine arbitrary-depth2 residual =",
        depth2_residual_total,
    )

    print(
        "queries arbitrary root improves Oracle2 =",
        tree2_improves_oracle2,
    )

    print(
        "queries arbitrary depth2 closes full gap =",
        tree2_closes_gap,
    )

    print(
        "decomposition check =",
        child_headroom_total
        +
        root_headroom_total
        +
        depth2_residual_total,
    )

    assert (
        child_headroom_total
        +
        root_headroom_total
        +
        depth2_residual_total
        ==
        sum_O
        -
        sum_S
    )

    # Smoke10 parity with frozen Oracle2 full script.
    if (
        args.max_gap_queries
        ==
        10
    ):

        assert sum_O == 782
        assert sum_S == 748
        assert sum_oracle2 == 757

        print(
            "ORACLE2 SMOKE10 CROSS-PARITY = PASS"
        )

    # Full329 parity with frozen Oracle2 run.
    if (
        args.max_gap_queries
        ==
        0
    ):

        assert processed == 329
        assert sum_O == 25640
        assert sum_S == 24259
        assert sum_oracle2 == 24574

        print(
            "ORACLE2 FULL329 CROSS-PARITY = PASS"
        )

    print(
        "runtime seconds =",
        runtime,
    )

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "ARBITRARY_DEPTH2_ORACLE_TREE_V0"
        ),

        union_O=O_saved,
        union_S=S_saved,
        union_Gamma=G_saved,

        fixed_root_oracle2=
            oracle2_arr,

        arbitrary_tree2=
            oracle4_arr,
    )

    result = {
        "name":
            "Arbitrary Depth-2 Oracle Tree Capacity Audit v0",

        "processed":
            int(
                processed
            ),

        "sum_O":
            int(
                sum_O
            ),

        "sum_union_S":
            int(
                sum_S
            ),

        "sum_fixed_root_oracle2":
            int(
                sum_oracle2
            ),

        "sum_arbitrary_tree2":
            int(
                sum_oracle4
            ),

        "child_family_headroom":
            int(
                child_headroom_total
            ),

        "root_family_headroom_after_child_relaxation":
            int(
                root_headroom_total
            ),

        "depth2_residual":
            int(
                depth2_residual_total
            ),

        "queries_tree2_improves_oracle2":
            int(
                tree2_improves_oracle2
            ),

        "queries_tree2_closes_gap":
            int(
                tree2_closes_gap
            ),

        "runtime_seconds":
            float(
                runtime
            ),
    }

    OUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "NPZ =",
        OUT_NPZ,
    )

    print(
        "JSON =",
        OUT_JSON,
    )

    print(
        "ORACLE TREE2 DECOMPOSITION = PASS"
    )

    print(
        "SCRIPT SHA256 =",
        sha256_file(
            Path(
                __file__
            )
        ),
    )

    print("=" * 132)


if __name__ == "__main__":
    main()
