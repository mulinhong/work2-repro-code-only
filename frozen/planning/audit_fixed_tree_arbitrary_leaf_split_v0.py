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

EXPECTED_ORACLE2_SHA = (
    "791efbebbe03f02f63c8a2b65f545736b46b9ddcbbd3e6a934dd942a0465059a"
)


E = 80
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

    print("=" * 142)
    print(
        "FIXED DEPTH-2 TREE + ARBITRARY ONE-LEAF SPLIT ORACLE v0"
    )
    print(
        "SELECTED BACKBONE FROZEN / ONE DEPTH-2 TERMINAL LEAF ONLY"
    )
    print("=" * 142)

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
            ORACLE2_SCRIPT
        )
        ==
        EXPECTED_ORACLE2_SHA
    )

    print(
        "FROZEN SCRIPT INTEGRITY = PASS"
    )

    # ==============================================================
    # Modules
    # ==============================================================

    unionmod = load_module(
        UNION_SCRIPT,
        "fta_union",
    )

    fixed = load_module(
        FIXED_SCRIPT,
        "fta_fixed",
    )

    oracle2 = load_module(
        ORACLE2_SCRIPT,
        "fta_oracle2",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "fta_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "fta_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "fta_tecs",
    )

    # ==============================================================
    # Data
    # ==============================================================

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
            G_saved > 0
        )
    ) == 329

    print(
        "UNION / ORACLETREE2 ANCHOR = PASS"
    )

    # ==============================================================
    # Totals
    # ==============================================================

    processed = 0

    sum_O = 0
    sum_S = 0
    sum_existing = 0
    sum_arbitrary = 0

    existing_gain_total = 0
    arbitrary_gain_total = 0

    q3_expressivity_headroom = 0
    selected_backbone_residual = 0

    arbitrary_improves_existing = 0
    arbitrary_closes = 0

    arch_queries = 0

    arch_O = 0
    arch_S = 0
    arch_existing = 0
    arch_arbitrary = 0

    arch_q3_headroom = 0
    arch_backbone_residual = 0

    arch_arbitrary_improves = 0
    arch_arbitrary_closes = 0

    nonarch_queries = 0

    nonarch_gap = 0
    nonarch_existing_gain = 0
    nonarch_arbitrary_gain = 0
    nonarch_remaining = 0

    records = []

    t_all = time.perf_counter()

    # ==============================================================
    # Queries
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
        # Frozen selected depth-2 UNION optimum
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

        base_success = (
            shared
            &
            ind
        )

        S = int(
            base_success.sum()
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
        # Exact terminal leaves of selected depth-2 policy
        # ----------------------------------------------------------

        leaves = (
            fixed.build_terminal_leaves(
                policy,
                survive0,
                survive1,
            )
        )

        reconstructed = np.zeros(
            E,
            dtype=bool,
        )

        target_membership = np.zeros(
            E,
            dtype=np.int16,
        )

        existing_best_gain = 0
        existing_best_leaf = -1
        existing_best_predicate = -1

        arbitrary_best_gain = 0
        arbitrary_best_leaf = -1

        leaf_records = []

        for li, leaf in enumerate(
            leaves
        ):

            leaf_mask = (
                leaf[
                    "mask"
                ]
            )

            (
                target_ids,
                current_hit,
            ) = (
                fixed.current_leaf_success(
                    leaf_mask,
                    ind,
                )
            )

            target_membership[
                target_ids
            ] += 1

            reconstructed[
                target_ids
            ] = current_hit

            current_cov = int(
                current_hit.sum()
            )

            # ------------------------------------------------------
            # Existing-predicate Local+1 parity
            # ------------------------------------------------------

            existing = (
                fixed.best_existing_predicate_split(
                    facts,
                    valid,
                    ind,
                    leaf_mask,
                    survive0,
                    survive1,
                )
            )

            existing_gain = int(
                existing[
                    "gain"
                ]
            )

            if (
                int(
                    leaf[
                        "depth"
                    ]
                )
                ==
                2
                and
                existing_gain
                >
                existing_best_gain
            ):

                existing_best_gain = (
                    existing_gain
                )

                existing_best_leaf = int(
                    li
                )

                existing_best_predicate = int(
                    existing[
                        "predicate"
                    ]
                )

            # ------------------------------------------------------
            # Arbitrary third question.
            #
            # IMPORTANT:
            # only true depth-2 terminal leaves may be expanded.
            # A shallower leaf would test missing root/child
            # expressivity, not one-extra-leaf depth expansion.
            # ------------------------------------------------------

            arbitrary_cov = (
                current_cov
            )

            arbitrary_gain = 0

            if int(
                leaf[
                    "depth"
                ]
            ) == 2:

                leaf_target_mask = (
                    ind
                    &
                    leaf_mask
                )

                arbitrary_cov = int(
                    oracle2.arbitrary_binary_branch_optimum(
                        leaf_mask,
                        valid,
                        leaf_target_mask,
                    )
                )

                arbitrary_gain = int(
                    arbitrary_cov
                    -
                    current_cov
                )

                assert (
                    arbitrary_gain
                    >=
                    existing_gain
                ), (
                    qi,
                    li,
                    current_cov,
                    existing_gain,
                    arbitrary_gain,
                )

                if (
                    arbitrary_gain
                    >
                    arbitrary_best_gain
                ):

                    arbitrary_best_gain = (
                        arbitrary_gain
                    )

                    arbitrary_best_leaf = int(
                        li
                    )

            else:

                # Existing-predicate positive gain at depth<2 was
                # already proved impossible in frozen Local+1 audit.
                assert (
                    existing_gain
                    ==
                    0
                )

            leaf_records.append(
                {
                    "leaf_index":
                        int(
                            li
                        ),

                    "label":
                        leaf[
                            "label"
                        ],

                    "depth":
                        int(
                            leaf[
                                "depth"
                            ]
                        ),

                    "current_coverage":
                        current_cov,

                    "existing_gain":
                        int(
                            existing_gain
                        ),

                    "arbitrary_gain":
                        int(
                            arbitrary_gain
                        ),

                    "arbitrary_coverage":
                        int(
                            arbitrary_cov
                        ),
                }
            )

        assert np.all(
            target_membership[
                ind
            ]
            ==
            1
        )

        assert np.array_equal(
            reconstructed,
            base_success,
        )

        existing_S = int(
            S
            +
            existing_best_gain
        )

        arbitrary_S = int(
            S
            +
            arbitrary_best_gain
        )

        assert (
            S
            <=
            existing_S
            <=
            arbitrary_S
            <=
            O
        )

        # ----------------------------------------------------------
        # Known frozen Local+1 aggregate parity is checked below.
        # ----------------------------------------------------------

        q3_headroom = int(
            arbitrary_S
            -
            existing_S
        )

        backbone_residual = int(
            O
            -
            arbitrary_S
        )

        arch = bool(
            int(
                tree2_saved[
                    qi
                ]
            )
            <
            O
        )

        # ----------------------------------------------------------
        # Totals
        # ----------------------------------------------------------

        sum_O += O
        sum_S += S
        sum_existing += existing_S
        sum_arbitrary += arbitrary_S

        existing_gain_total += (
            existing_best_gain
        )

        arbitrary_gain_total += (
            arbitrary_best_gain
        )

        q3_expressivity_headroom += (
            q3_headroom
        )

        selected_backbone_residual += (
            backbone_residual
        )

        if (
            arbitrary_S
            >
            existing_S
        ):
            arbitrary_improves_existing += 1

        if (
            arbitrary_S
            ==
            O
        ):
            arbitrary_closes += 1

        if arch:

            arch_queries += 1

            arch_O += O
            arch_S += S
            arch_existing += existing_S
            arch_arbitrary += arbitrary_S

            arch_q3_headroom += (
                q3_headroom
            )

            arch_backbone_residual += (
                backbone_residual
            )

            if (
                arbitrary_S
                >
                existing_S
            ):
                arch_arbitrary_improves += 1

            if (
                arbitrary_S
                ==
                O
            ):
                arch_arbitrary_closes += 1

        else:

            nonarch_queries += 1

            nonarch_gap += (
                O
                -
                S
            )

            nonarch_existing_gain += (
                existing_S
                -
                S
            )

            nonarch_arbitrary_gain += (
                arbitrary_S
                -
                S
            )

            nonarch_remaining += (
                O
                -
                arbitrary_S
            )

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

                "LocalExisting":
                    int(
                        existing_S
                    ),

                "LocalArbitrary":
                    int(
                        arbitrary_S
                    ),

                "existing_gain":
                    int(
                        existing_best_gain
                    ),

                "arbitrary_gain":
                    int(
                        arbitrary_best_gain
                    ),

                "q3_expressivity_headroom":
                    int(
                        q3_headroom
                    ),

                "selected_backbone_residual":
                    int(
                        backbone_residual
                    ),

                "architecture_limited":
                    arch,

                "best_existing_leaf":
                    int(
                        existing_best_leaf
                    ),

                "best_existing_predicate":
                    int(
                        existing_best_predicate
                    ),

                "best_arbitrary_leaf":
                    int(
                        arbitrary_best_leaf
                    ),

                "leaves":
                    leaf_records,

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
            f"Existing={existing_S:02d} "
            f"Arb={arbitrary_S:02d} "
            f"q3={q3_headroom:02d} "
            f"backbone={backbone_residual:02d} "
            f"arch={int(arch)} "
            f"leafE={existing_best_leaf} "
            f"leafA={arbitrary_best_leaf} "
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
    print("=" * 142)
    print(
        "FIXED-TREE ARBITRARY-LEAF ORACLE SUMMARY"
    )
    print("=" * 142)

    print(
        "residual queries processed =",
        processed,
    )

    print(
        "sum O / UNION S / LocalExisting / LocalArbitrary =",
        sum_O,
        sum_S,
        sum_existing,
        sum_arbitrary,
    )

    print(
        "UNION residual gap =",
        sum_O
        -
        sum_S,
    )

    print(
        "existing Local+1 gain =",
        existing_gain_total,
    )

    print(
        "arbitrary Local+1 gain =",
        arbitrary_gain_total,
    )

    print(
        "third-question predicate expressivity headroom =",
        q3_expressivity_headroom,
    )

    print(
        "selected-backbone residual =",
        selected_backbone_residual,
    )

    print(
        "queries arbitrary q3 improves existing q3 =",
        arbitrary_improves_existing,
    )

    print(
        "queries arbitrary q3 closes full gap =",
        arbitrary_closes,
    )

    print()

    print(
        "architecture-limited queries =",
        arch_queries,
    )

    print(
        "architecture-limited O / S / Existing / Arbitrary =",
        arch_O,
        arch_S,
        arch_existing,
        arch_arbitrary,
    )

    print(
        "architecture-limited q3 expressivity headroom =",
        arch_q3_headroom,
    )

    print(
        "architecture-limited selected-backbone residual =",
        arch_backbone_residual,
    )

    print(
        "architecture-limited queries arbitrary q3 improves =",
        arch_arbitrary_improves,
    )

    print(
        "architecture-limited queries arbitrary q3 closes =",
        arch_arbitrary_closes,
    )

    print()

    print(
        "non-architecture-limited queries =",
        nonarch_queries,
    )

    print(
        "non-architecture-limited UNION gap =",
        nonarch_gap,
    )

    print(
        "non-architecture-limited existing gain =",
        nonarch_existing_gain,
    )

    print(
        "non-architecture-limited arbitrary gain =",
        nonarch_arbitrary_gain,
    )

    print(
        "non-architecture-limited residual after arbitrary q3 =",
        nonarch_remaining,
    )

    # --------------------------------------------------------------
    # Exact decomposition.
    # --------------------------------------------------------------

    assert (
        existing_gain_total
        +
        q3_expressivity_headroom
        +
        selected_backbone_residual
        ==
        sum_O
        -
        sum_S
    )

    print()
    print(
        "decomposition check =",
        existing_gain_total
        +
        q3_expressivity_headroom
        +
        selected_backbone_residual,
    )

    # --------------------------------------------------------------
    # Cross-parity against frozen Local+1 result.
    # --------------------------------------------------------------

    if (
        args.max_gap_queries
        ==
        10
    ):

        assert sum_O == 782
        assert sum_S == 748
        assert sum_existing == 776

        print(
            "LOCAL+1 SMOKE10 CROSS-PARITY = PASS"
        )

    if (
        args.max_gap_queries
        ==
        0
    ):

        assert processed == 329
        assert sum_O == 25640
        assert sum_S == 24259
        assert sum_existing == 25368

        assert arch_queries == 263
        assert (
            arch_O
            -
            arch_S
        ) == 1276

        assert (
            arch_existing
            -
            arch_S
        ) == 1012

        print(
            "LOCAL+1 FULL329 CROSS-PARITY = PASS"
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
        f"fixed_tree_arbitrary_leaf_split_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "name":
                    "Fixed Depth-2 Tree + Arbitrary One-Leaf Split Oracle v0",

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

                "sum_local_existing":
                    int(
                        sum_existing
                    ),

                "sum_local_arbitrary":
                    int(
                        sum_arbitrary
                    ),

                "existing_gain":
                    int(
                        existing_gain_total
                    ),

                "arbitrary_gain":
                    int(
                        arbitrary_gain_total
                    ),

                "q3_expressivity_headroom":
                    int(
                        q3_expressivity_headroom
                    ),

                "selected_backbone_residual":
                    int(
                        selected_backbone_residual
                    ),

                "architecture_limited_queries":
                    int(
                        arch_queries
                    ),

                "architecture_limited_q3_headroom":
                    int(
                        arch_q3_headroom
                    ),

                "architecture_limited_selected_backbone_residual":
                    int(
                        arch_backbone_residual
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
        "FIXED-TREE ARBITRARY-LEAF DECOMPOSITION = PASS"
    )

    print(
        "SCRIPT SHA256 =",
        sha256_file(
            Path(
                __file__
            )
        ),
    )

    print("=" * 142)


if __name__ == "__main__":
    main()
