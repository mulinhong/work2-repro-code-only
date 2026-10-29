from __future__ import annotations

from pathlib import Path
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

FIXED_SCRIPT = (
    D / "audit_fixed_tree_one_leaf_existing_split_v0.py"
)

ARBITRARY_FIXED_SCRIPT = (
    D / "audit_fixed_tree_arbitrary_leaf_split_v0.py"
)

ORACLE2_SCRIPT = (
    D / "audit_oracle_continuation_capacity_v0.py"
)

TREE2_ART = (
    D / "oracle_tree2_capacity_v0.npz"
)


EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_FIXED_SHA = (
    "016d465edf8d70ae56970e155070c23e2c6559abd00693fa5bee5ea2dc520620"
)

EXPECTED_ARB_FIXED_SHA = (
    "6d78e1c068fb9c324943299516b48aa8bca53578bd91a2b1f5b7fa590ee40b12"
)

EXPECTED_ORACLE2_SHA = (
    "791efbebbe03f02f63c8a2b65f545736b46b9ddcbbd3e6a934dd942a0465059a"
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


def branch_profile(
    mask,
    facts,
    valid,
    target_mask,
    survive0,
    survive1,
):
    """
    Exact coverage profile for ONE root branch.

    Returns all second-stage choices tied for maximum COVERAGE.

    Choice -1 = STOP.

    Secondary frozen objectives (gain/worst-leaf/id) are deliberately
    ignored here because we want ALL coverage-optimal depth-2
    backbones.
    """

    target_ids = np.flatnonzero(
        target_mask
        &
        mask
    )

    if len(target_ids) == 0:

        return {
            "max_coverage": 0,
            "optimal_children": [-1],
        }

    prefix = np.cumsum(
        mask,
        dtype=np.int16,
    )

    stop_hit = (
        prefix[
            target_ids
        ]
        <=
        B
    )

    stop_cov = int(
        stop_hit.sum()
    )

    # Frozen A4 canonical STOP.
    if (
        int(
            mask.sum()
        )
        <=
        B
    ):

        assert (
            stop_cov
            ==
            len(
                target_ids
            )
        )

        return {
            "max_coverage":
                stop_cov,

            "optimal_children":
                [-1],
        }

    valid_ids = np.flatnonzero(
        valid
        &
        mask
    )

    if len(valid_ids) == 0:

        return {
            "max_coverage":
                stop_cov,

            "optimal_children":
                [-1],
        }

    # --------------------------------------------------------------
    # Frozen A4 second-predicate legality:
    # both 0 and 1 among valid candidates surviving this branch.
    # --------------------------------------------------------------

    answers_valid = (
        facts[
            valid_ids
        ].T
    )

    has0 = np.any(
        answers_valid
        ==
        ABSENT,
        axis=1,
    )

    has1 = np.any(
        answers_valid
        ==
        PRESENT,
        axis=1,
    )

    legal = (
        has0
        &
        has1
    )

    if not np.any(
        legal
    ):

        return {
            "max_coverage":
                stop_cov,

            "optimal_children":
                [-1],
        }

    masks0 = (
        survive0
        &
        mask[
            None,
            :
        ]
    )

    masks1 = (
        survive1
        &
        mask[
            None,
            :
        ]
    )

    rank0 = (
        np.cumsum(
            masks0,
            axis=1,
            dtype=np.int16,
        )[
            :,
            target_ids
        ]
    )

    rank1 = (
        np.cumsum(
            masks1,
            axis=1,
            dtype=np.int16,
        )[
            :,
            target_ids
        ]
    )

    target_answers = (
        facts[
            target_ids
        ].T
    )

    assert np.all(
        (
            target_answers
            ==
            ABSENT
        )
        |
        (
            target_answers
            ==
            PRESENT
        )
    )

    ranks = np.where(
        target_answers
        ==
        ABSENT,
        rank0,
        rank1,
    )

    coverage = np.sum(
        ranks
        <=
        B,
        axis=1,
    ).astype(
        np.int32
    )

    coverage[
        ~legal
    ] = -1

    pred_max = int(
        coverage.max()
    )

    max_cov = max(
        stop_cov,
        pred_max,
    )

    choices = []

    if (
        stop_cov
        ==
        max_cov
    ):

        choices.append(
            -1
        )

    pred_choices = np.flatnonzero(
        legal
        &
        (
            coverage
            ==
            max_cov
        )
    )

    choices.extend(
        int(x)
        for x in pred_choices
    )

    assert len(
        choices
    ) > 0

    return {
        "max_coverage":
            int(
                max_cov
            ),

        "optimal_children":
            choices,
    }


def leaf_arbitrary_gain(
    leaf_mask,
    valid,
    target_mask,
    oracle2,
):
    """
    Exact arbitrary-binary one-question gain on one depth-2 leaf.
    """

    ids = np.flatnonzero(
        target_mask
        &
        leaf_mask
    )

    if len(ids) == 0:
        return 0

    prefix = np.cumsum(
        leaf_mask,
        dtype=np.int16,
    )

    current_cov = int(
        np.sum(
            prefix[
                ids
            ]
            <=
            B
        )
    )

    arbitrary_cov = int(
        oracle2.arbitrary_binary_branch_optimum(
            leaf_mask,
            valid,
            target_mask
            &
            leaf_mask,
        )
    )

    assert (
        arbitrary_cov
        >=
        current_cov
    )

    return int(
        arbitrary_cov
        -
        current_cov
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-gap-queries",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="smoke10",
    )

    args = parser.parse_args()

    print("=" * 144)
    print(
        "ALL COVERAGE-OPTIMAL DEPTH-2 BACKBONES + ONE ARBITRARY LEAF AUDIT v0"
    )
    print(
        "EXACT COVERAGE-TIE ENUMERATION / NO LEXICOGRAPHIC BACKBONE COMMITMENT"
    )
    print("=" * 144)

    # ==============================================================
    # Integrity
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
            FIXED_SCRIPT
        )
        ==
        EXPECTED_FIXED_SHA
    )

    assert (
        sha256_file(
            ARBITRARY_FIXED_SCRIPT
        )
        ==
        EXPECTED_ARB_FIXED_SHA
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

    # ==============================================================
    # Modules / data
    # ==============================================================

    unionmod = load_module(
        UNION_SCRIPT,
        "ob1_union",
    )

    fixed = load_module(
        FIXED_SCRIPT,
        "ob1_fixed",
    )

    oracle2 = load_module(
        ORACLE2_SCRIPT,
        "ob1_oracle2",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "ob1_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "ob1_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "ob1_tecs",
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

    ta = np.load(
        TREE2_ART,
        allow_pickle=False,
    )

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

    O_saved = (
        ua[
            "union_O"
        ].astype(
            np.int16
        )
    )

    S_saved = (
        ua[
            "union_S"
        ].astype(
            np.int16
        )
    )

    G_saved = (
        ua[
            "union_Gamma"
        ].astype(
            np.int16
        )
    )

    tree2_saved = (
        ta[
            "arbitrary_tree2"
        ].astype(
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
        "UNION / TREE2 ANCHOR = PASS"
    )

    # ==============================================================
    # Totals
    # ==============================================================

    processed = 0

    sum_O = 0
    sum_S = 0

    sum_selected_arb = 0
    sum_optbackbone_arb = 0

    tie_headroom_total = 0
    optbackbone_residual_total = 0

    tie_improved_queries = 0
    optbackbone_closed_queries = 0

    arch_queries = 0
    arch_tie_headroom = 0
    arch_residual = 0
    arch_tie_improved = 0
    arch_closed = 0

    optimal_root_counts = []
    tied_child_counts = []

    records = []

    t_all = time.perf_counter()

    # ==============================================================
    # Query loop
    # ==============================================================

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

        q_t0 = time.perf_counter()

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
        # UNION reconstruction
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

        # ----------------------------------------------------------
        # Frozen H_ind
        # ----------------------------------------------------------

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

        assert (
            O
            ==
            int(
                O_saved[
                    qi
                ]
            )
        )

        # ----------------------------------------------------------
        # Frozen selected depth-2 optimum
        # ----------------------------------------------------------

        (
            policy,
            survive0,
            survive1,
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
                survive0,
                survive1,
                valid,
                a4,
            )
        )

        assert not np.any(
            shared
            &
            ~ind
        )

        S = int(
            (
                shared
                &
                ind
            ).sum()
        )

        assert (
            S
            ==
            int(
                S_saved[
                    qi
                ]
            )
        )

        # ----------------------------------------------------------
        # Selected-backbone arbitrary one-leaf expansion parity
        # ----------------------------------------------------------

        selected_gain = 0

        selected_leaves = (
            fixed.build_terminal_leaves(
                policy,
                survive0,
                survive1,
            )
        )

        leaf_oracle_cache = {}

        def get_leaf_gain(mask):

            key = (
                np.packbits(
                    mask
                ).tobytes()
            )

            if key not in leaf_oracle_cache:

                leaf_oracle_cache[
                    key
                ] = (
                    leaf_arbitrary_gain(
                        mask,
                        valid,
                        ind,
                        oracle2,
                    )
                )

            return int(
                leaf_oracle_cache[
                    key
                ]
            )

        for leaf in selected_leaves:

            if int(
                leaf[
                    "depth"
                ]
            ) != 2:

                continue

            selected_gain = max(
                selected_gain,
                get_leaf_gain(
                    leaf[
                        "mask"
                    ]
                ),
            )

        selected_arb = int(
            S
            +
            selected_gain
        )

        # ----------------------------------------------------------
        # First pass:
        # exact branch coverage profiles for every legal root.
        # ----------------------------------------------------------

        valid_ids = np.flatnonzero(
            valid
        )

        profile_cache = {}

        def get_profile(mask):

            key = (
                np.packbits(
                    mask
                ).tobytes()
            )

            if key not in profile_cache:

                profile_cache[
                    key
                ] = (
                    branch_profile(
                        mask,
                        facts,
                        valid,
                        ind,
                        survive0,
                        survive1,
                    )
                )

            return profile_cache[
                key
            ]

        root_records = []

        best_depth2_coverage = -1

        for root in range(
            P
        ):

            ans = facts[
                valid_ids,
                root
            ]

            if not (
                np.any(
                    ans
                    ==
                    ABSENT
                )
                and
                np.any(
                    ans
                    ==
                    PRESENT
                )
            ):

                continue

            mask0 = (
                survive0[
                    root
                ]
            )

            mask1 = (
                survive1[
                    root
                ]
            )

            prof0 = get_profile(
                mask0
            )

            prof1 = get_profile(
                mask1
            )

            cov = int(
                prof0[
                    "max_coverage"
                ]
                +
                prof1[
                    "max_coverage"
                ]
            )

            best_depth2_coverage = max(
                best_depth2_coverage,
                cov,
            )

            root_records.append(
                (
                    int(root),
                    mask0,
                    mask1,
                    prof0,
                    prof1,
                    cov,
                )
            )

        assert (
            best_depth2_coverage
            ==
            S
        ), (
            qi,
            best_depth2_coverage,
            S,
        )

        optimal_roots = [
            r
            for r in root_records
            if int(
                r[
                    5
                ]
            )
            ==
            S
        ]

        assert len(
            optimal_roots
        ) > 0

        optimal_root_counts.append(
            len(
                optimal_roots
            )
        )

        # ----------------------------------------------------------
        # For each branch, inspect EVERY coverage-optimal child.
        # Only one final depth-2 leaf can be expanded.
        # ----------------------------------------------------------

        branch_extension_cache = {}

        def best_branch_extension(
            branch_mask,
            profile,
        ):

            key = (
                np.packbits(
                    branch_mask
                ).tobytes()
            )

            if key in branch_extension_cache:

                return branch_extension_cache[
                    key
                ]

            best_gain = 0
            best_child = -1
            best_side = -1

            choices = profile[
                "optimal_children"
            ]

            tied_child_counts.append(
                len(
                    choices
                )
            )

            for child in choices:

                child = int(
                    child
                )

                if child < 0:

                    # STOP leaves this branch at depth 1.
                    # It does not create a depth-2 terminal leaf
                    # eligible for the single third question.
                    continue

                leaf0 = (
                    branch_mask
                    &
                    survive0[
                        child
                    ]
                )

                leaf1 = (
                    branch_mask
                    &
                    survive1[
                        child
                    ]
                )

                g0 = get_leaf_gain(
                    leaf0
                )

                g1 = get_leaf_gain(
                    leaf1
                )

                if g0 > best_gain:

                    best_gain = int(
                        g0
                    )

                    best_child = child
                    best_side = 0

                if g1 > best_gain:

                    best_gain = int(
                        g1
                    )

                    best_child = child
                    best_side = 1

            out = {
                "gain":
                    int(
                        best_gain
                    ),

                "child":
                    int(
                        best_child
                    ),

                "side":
                    int(
                        best_side
                    ),
            }

            branch_extension_cache[
                key
            ] = out

            return out

        optbackbone_gain = 0
        best_root = -1
        best_branch = -1
        best_child = -1
        best_side = -1

        for (
            root,
            mask0,
            mask1,
            prof0,
            prof1,
            cov,
        ) in optimal_roots:

            assert int(
                cov
            ) == S

            ext0 = (
                best_branch_extension(
                    mask0,
                    prof0,
                )
            )

            ext1 = (
                best_branch_extension(
                    mask1,
                    prof1,
                )
            )

            if int(
                ext0[
                    "gain"
                ]
            ) > optbackbone_gain:

                optbackbone_gain = int(
                    ext0[
                        "gain"
                    ]
                )

                best_root = int(
                    root
                )

                best_branch = 0

                best_child = int(
                    ext0[
                        "child"
                    ]
                )

                best_side = int(
                    ext0[
                        "side"
                    ]
                )

            if int(
                ext1[
                    "gain"
                ]
            ) > optbackbone_gain:

                optbackbone_gain = int(
                    ext1[
                        "gain"
                    ]
                )

                best_root = int(
                    root
                )

                best_branch = 1

                best_child = int(
                    ext1[
                        "child"
                    ]
                )

                best_side = int(
                    ext1[
                        "side"
                    ]
                )

        optbackbone_arb = int(
            S
            +
            optbackbone_gain
        )

        assert (
            selected_arb
            <=
            optbackbone_arb
            <=
            O
        ), (
            qi,
            selected_arb,
            optbackbone_arb,
            O,
        )

        tie_headroom = int(
            optbackbone_arb
            -
            selected_arb
        )

        residual = int(
            O
            -
            optbackbone_arb
        )

        # ----------------------------------------------------------
        # Aggregate
        # ----------------------------------------------------------

        sum_O += O
        sum_S += S

        sum_selected_arb += (
            selected_arb
        )

        sum_optbackbone_arb += (
            optbackbone_arb
        )

        tie_headroom_total += (
            tie_headroom
        )

        optbackbone_residual_total += (
            residual
        )

        if tie_headroom > 0:
            tie_improved_queries += 1

        if optbackbone_arb == O:
            optbackbone_closed_queries += 1

        arch = bool(
            int(
                tree2_saved[
                    qi
                ]
            )
            <
            O
        )

        if arch:

            arch_queries += 1
            arch_tie_headroom += (
                tie_headroom
            )
            arch_residual += (
                residual
            )

            if tie_headroom > 0:
                arch_tie_improved += 1

            if optbackbone_arb == O:
                arch_closed += 1

        elapsed = (
            time.perf_counter()
            -
            q_t0
        )

        records.append(
            {
                "query":
                    int(
                        qi
                    ),

                "O":
                    O,

                "S":
                    S,

                "selected_arbitrary_plus1":
                    int(
                        selected_arb
                    ),

                "optimal_backbone_arbitrary_plus1":
                    int(
                        optbackbone_arb
                    ),

                "tie_headroom":
                    int(
                        tie_headroom
                    ),

                "residual_after_all_optimal_backbones":
                    int(
                        residual
                    ),

                "number_coverage_optimal_roots":
                    int(
                        len(
                            optimal_roots
                        )
                    ),

                "best_root":
                    int(
                        best_root
                    ),

                "best_branch":
                    int(
                        best_branch
                    ),

                "best_child":
                    int(
                        best_child
                    ),

                "best_leaf_side":
                    int(
                        best_side
                    ),

                "architecture_limited":
                    arch,

                "runtime_seconds":
                    float(
                        elapsed
                    ),
            }
        )

        processed += 1

        print(
            f"q={qi:03d} "
            f"O={O:02d} "
            f"S={S:02d} "
            f"SelArb={selected_arb:02d} "
            f"OptBack+1={optbackbone_arb:02d} "
            f"tie={tie_headroom:02d} "
            f"res={residual:02d} "
            f"optRoots={len(optimal_roots):03d} "
            f"arch={int(arch)} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    # ==============================================================
    # Summary
    # ==============================================================

    runtime = (
        time.perf_counter()
        -
        t_all
    )

    print()
    print("=" * 144)
    print(
        "ALL OPTIMAL BACKBONES + ONE-LEAF SUMMARY"
    )
    print("=" * 144)

    print(
        "residual queries processed =",
        processed,
    )

    print(
        "sum O / S / SelectedArb+1 / OptBackboneArb+1 =",
        sum_O,
        sum_S,
        sum_selected_arb,
        sum_optbackbone_arb,
    )

    print(
        "selected-backbone residual before tie search =",
        sum_O
        -
        sum_selected_arb,
    )

    print(
        "coverage-optimal backbone tie headroom =",
        tie_headroom_total,
    )

    print(
        "residual after ALL coverage-optimal backbones =",
        optbackbone_residual_total,
    )

    print(
        "queries tie search improves selected backbone =",
        tie_improved_queries,
    )

    print(
        "queries all-optimal-backbone +1 closes full gap =",
        optbackbone_closed_queries,
    )

    print()

    print(
        "architecture-limited queries =",
        arch_queries,
    )

    print(
        "architecture-limited tie headroom =",
        arch_tie_headroom,
    )

    print(
        "architecture-limited residual after all optimal backbones =",
        arch_residual,
    )

    print(
        "architecture-limited queries tie search improves =",
        arch_tie_improved,
    )

    print(
        "architecture-limited queries closed =",
        arch_closed,
    )

    print()

    print(
        "coverage-optimal roots min / median / mean / max =",
        int(
            np.min(
                optimal_root_counts
            )
        ),
        float(
            np.median(
                optimal_root_counts
            )
        ),
        float(
            np.mean(
                optimal_root_counts
            )
        ),
        int(
            np.max(
                optimal_root_counts
            )
        ),
    )

    print(
        "coverage-optimal branch-child tie count min / median / mean / max =",
        int(
            np.min(
                tied_child_counts
            )
        ),
        float(
            np.median(
                tied_child_counts
            )
        ),
        float(
            np.mean(
                tied_child_counts
            )
        ),
        int(
            np.max(
                tied_child_counts
            )
        ),
    )

    assert (
        tie_headroom_total
        +
        optbackbone_residual_total
        ==
        sum_O
        -
        sum_selected_arb
    )

    print(
        "decomposition check =",
        tie_headroom_total
        +
        optbackbone_residual_total,
    )

    # --------------------------------------------------------------
    # Cross-parity with frozen selected-backbone arbitrary audit.
    # --------------------------------------------------------------

    if (
        args.max_gap_queries
        ==
        10
    ):

        assert sum_O == 782
        assert sum_S == 748
        assert sum_selected_arb == 776

        print(
            "SELECTED-ARBITRARY SMOKE10 CROSS-PARITY = PASS"
        )

    if (
        args.max_gap_queries
        ==
        0
    ):

        assert processed == 329
        assert sum_O == 25640
        assert sum_S == 24259
        assert sum_selected_arb == 25373

        print(
            "SELECTED-ARBITRARY FULL329 CROSS-PARITY = PASS"
        )

    print(
        "runtime seconds =",
        runtime,
    )

    # ==============================================================
    # Save
    # ==============================================================

    OUT = (
        D
        /
        f"all_optimal_backbones_one_leaf_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "name":
                    "All Coverage-Optimal Depth-2 Backbones + One Arbitrary Leaf Audit v0",

                "processed":
                    int(
                        processed
                    ),

                "sum_O":
                    int(
                        sum_O
                    ),

                "sum_S":
                    int(
                        sum_S
                    ),

                "sum_selected_arbitrary_plus1":
                    int(
                        sum_selected_arb
                    ),

                "sum_optimal_backbone_arbitrary_plus1":
                    int(
                        sum_optbackbone_arb
                    ),

                "tie_headroom":
                    int(
                        tie_headroom_total
                    ),

                "residual_after_all_optimal_backbones":
                    int(
                        optbackbone_residual_total
                    ),

                "architecture_limited_tie_headroom":
                    int(
                        arch_tie_headroom
                    ),

                "architecture_limited_residual":
                    int(
                        arch_residual
                    ),

                "records":
                    records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "JSON =",
        OUT,
    )

    print(
        "ALL-OPTIMAL-BACKBONE DECOMPOSITION = PASS"
    )

    print(
        "SCRIPT SHA256 =",
        sha256_file(
            Path(
                __file__
            )
        ),
    )

    print("=" * 144)


if __name__ == "__main__":
    main()
