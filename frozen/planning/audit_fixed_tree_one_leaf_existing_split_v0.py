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

TREE2_SCRIPT = (
    D / "audit_oracle_tree2_capacity_v0.py"
)

TREE2_ART = (
    D / "oracle_tree2_capacity_v0.npz"
)

EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_TREE2_SHA = (
    "e23d9f72588d314d15685e2692fae192e317cebc391ca56dfa59efb2347c5d0b"
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

    mod = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None
    spec.loader.exec_module(mod)

    return mod


def build_terminal_leaves(
    policy,
    survive0,
    survive1,
):
    """
    Reconstruct terminal survivor masks of the FROZEN exact
    depth<=2 policy.

    No root / child restructuring is allowed.
    """

    full = np.ones(
        E,
        dtype=bool,
    )

    if policy is None:
        return [
            {
                "label": "STOP",
                "depth": 0,
                "mask": full,
                "root_answer": -1,
                "second_feature": -1,
                "second_answer": -1,
            }
        ]

    root = int(
        policy[
            "first_feature"
        ]
    )

    leaves = []

    branch_specs = [
        (
            ABSENT,
            survive0[root],
            int(
                policy[
                    "second_zero"
                ]
            ),
        ),
        (
            PRESENT,
            survive1[root],
            int(
                policy[
                    "second_one"
                ]
            ),
        ),
    ]

    for (
        root_answer,
        branch_mask,
        second,
    ) in branch_specs:

        if second < 0:

            leaves.append(
                {
                    "label":
                        str(root_answer),

                    "depth":
                        1,

                    "mask":
                        branch_mask.copy(),

                    "root_answer":
                        int(root_answer),

                    "second_feature":
                        -1,

                    "second_answer":
                        -1,
                }
            )

            continue

        leaves.append(
            {
                "label":
                    f"{root_answer}0",

                "depth":
                    2,

                "mask":
                    (
                        branch_mask
                        &
                        survive0[
                            second
                        ]
                    ),

                "root_answer":
                    int(root_answer),

                "second_feature":
                    int(second),

                "second_answer":
                    ABSENT,
            }
        )

        leaves.append(
            {
                "label":
                    f"{root_answer}1",

                "depth":
                    2,

                "mask":
                    (
                        branch_mask
                        &
                        survive1[
                            second
                        ]
                    ),

                "root_answer":
                    int(root_answer),

                "second_feature":
                    int(second),

                "second_answer":
                    PRESENT,
            }
        )

    return leaves


def current_leaf_success(
    leaf_mask,
    target_mask,
):
    target_ids = np.flatnonzero(
        target_mask
        &
        leaf_mask
    )

    if len(target_ids) == 0:
        return (
            target_ids,
            np.zeros(
                0,
                dtype=bool,
            ),
        )

    prefix = np.cumsum(
        leaf_mask,
        dtype=np.int16,
    )

    ranks = prefix[
        target_ids
    ]

    hit = (
        ranks
        <=
        B
    )

    return (
        target_ids,
        hit,
    )


def best_existing_predicate_split(
    facts,
    valid,
    target_mask,
    leaf_mask,
    survive0,
    survive1,
):
    """
    Exact best split of ONE FIXED terminal leaf by ONE predicate
    already present in the UNION library.

    Root and all previous decisions are frozen.

    Predicate legality matches frozen A4:
        among AST-valid candidates surviving this leaf,
        both binary answers must occur.
    """

    target_ids, current_hit = (
        current_leaf_success(
            leaf_mask,
            target_mask,
        )
    )

    current_cov = int(
        current_hit.sum()
    )

    result = {
        "current_coverage":
            current_cov,

        "best_coverage":
            current_cov,

        "gain":
            0,

        "predicate":
            -1,

        "target_ids":
            target_ids,

        "new_hit":
            current_hit.copy(),

        "newly_repaired":
            np.zeros(
                0,
                dtype=np.int32,
            ),

        "legal_predicates":
            0,
    }

    if len(target_ids) == 0:
        return result

    # If whole terminal leaf is already <= B, all targets there
    # are already repaired.
    if int(
        leaf_mask.sum()
    ) <= B:
        return result

    valid_ids = np.flatnonzero(
        valid
        &
        leaf_mask
    )

    if len(valid_ids) == 0:
        return result

    # [P, Hleaf]
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

    legal_ids = np.flatnonzero(
        legal
    )

    result[
        "legal_predicates"
    ] = int(
        len(
            legal_ids
        )
    )

    if len(legal_ids) == 0:
        return result

    # ------------------------------------------------------------
    # Evaluate every existing UNION predicate as a third question.
    # ------------------------------------------------------------

    masks0 = (
        survive0
        &
        leaf_mask[
            None,
            :
        ]
    )

    masks1 = (
        survive1
        &
        leaf_mask[
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

    hit = (
        ranks
        <=
        B
    )

    coverage = np.sum(
        hit,
        axis=1,
    ).astype(
        np.int32
    )

    coverage[
        ~legal
    ] = -1

    best_q = int(
        np.argmax(
            coverage
        )
    )

    best_cov = int(
        coverage[
            best_q
        ]
    )

    # Asking an additional question can only remove survivors from
    # the target's current leaf. Existing repaired targets must not
    # be lost.
    best_hit = (
        hit[
            best_q
        ]
    )

    assert not np.any(
        current_hit
        &
        ~best_hit
    )

    assert (
        best_cov
        >=
        current_cov
    )

    newly = target_ids[
        best_hit
        &
        ~current_hit
    ]

    result.update(
        {
            "best_coverage":
                best_cov,

            "gain":
                int(
                    best_cov
                    -
                    current_cov
                ),

            "predicate":
                int(
                    best_q
                ),

            "new_hit":
                best_hit.copy(),

            "newly_repaired":
                newly.astype(
                    np.int32
                ),
        }
    )

    return result


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

    print("=" * 140)
    print(
        "FIXED DEPTH-2 OPTIMUM + ONE EXISTING-PREDICATE LEAF SPLIT AUDIT v0"
    )
    print(
        "POST-HOC SELECTIVE EXPANSION BASELINE / ROOT+DEPTH2 TREE FROZEN"
    )
    print("=" * 140)

    # ============================================================
    # Integrity
    # ============================================================

    assert (
        sha256_file(
            UNION_SCRIPT
        )
        ==
        EXPECTED_UNION_SHA
    )

    assert (
        sha256_file(
            TREE2_SCRIPT
        )
        ==
        EXPECTED_TREE2_SHA
    )

    assert UNION_ART.exists()
    assert TREE2_ART.exists()

    print(
        "FROZEN SCRIPT INTEGRITY = PASS"
    )

    # ============================================================
    # Frozen environment
    # ============================================================

    unionmod = load_module(
        UNION_SCRIPT,
        "local1_union",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "local1_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "local1_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "local1_tecs",
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
        O_saved.sum()
    ) == 44711

    assert int(
        S_saved.sum()
    ) == 43330

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
        "UNION / ORACLETREE2 ANCHOR = PASS"
    )

    # ============================================================
    # Aggregate counters
    # ============================================================

    processed = 0

    sum_O = 0
    sum_S = 0
    sum_local1 = 0

    total_gain = 0

    improved_queries = 0
    closed_queries = 0

    positive_gain_depth0 = 0
    positive_gain_depth1 = 0
    positive_gain_depth2 = 0

    architecture_limited_queries = 0
    architecture_limited_gap = 0

    architecture_limited_gain = 0
    architecture_limited_improved = 0
    architecture_limited_closed = 0

    records = []

    t_all = time.perf_counter()

    # ============================================================
    # Queries
    # ============================================================

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

        # --------------------------------------------------------
        # Exact UNION reconstruction
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Frozen individual ground set
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Frozen exact UNION depth-2 optimum
        # --------------------------------------------------------

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

        shared_all_valid = (
            gap.eval_compound_policy_all(
                policy,
                facts,
                survive0,
                survive1,
                valid,
                a4,
            )
        )

        # A shared depth-2 success must also be individually
        # depth-2 repairable.
        assert not np.any(
            shared_all_valid
            &
            ~ind
        )

        base_success = (
            shared_all_valid
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

        # --------------------------------------------------------
        # Reconstruct terminal leaves exactly
        # --------------------------------------------------------

        leaves = (
            build_terminal_leaves(
                policy,
                survive0,
                survive1,
            )
        )

        membership = np.zeros(
            E,
            dtype=np.int16,
        )

        reconstructed_success = np.zeros(
            E,
            dtype=bool,
        )

        leaf_results = []

        for li, leaf in enumerate(
            leaves
        ):

            leaf_mask = (
                leaf[
                    "mask"
                ]
            )

            target_ids, current_hit = (
                current_leaf_success(
                    leaf_mask,
                    ind,
                )
            )

            membership[
                target_ids
            ] += 1

            reconstructed_success[
                target_ids
            ] = current_hit

            split = (
                best_existing_predicate_split(
                    facts,
                    valid,
                    ind,
                    leaf_mask,
                    survive0,
                    survive1,
                )
            )

            leaf_results.append(
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

                    "survivors":
                        int(
                            leaf_mask.sum()
                        ),

                    "valid_survivors":
                        int(
                            np.sum(
                                valid
                                &
                                leaf_mask
                            )
                        ),

                    "target_count":
                        int(
                            len(
                                target_ids
                            )
                        ),

                    "current_coverage":
                        int(
                            split[
                                "current_coverage"
                            ]
                        ),

                    "best_split_coverage":
                        int(
                            split[
                                "best_coverage"
                            ]
                        ),

                    "gain":
                        int(
                            split[
                                "gain"
                            ]
                        ),

                    "predicate":
                        int(
                            split[
                                "predicate"
                            ]
                        ),

                    "legal_predicates":
                        int(
                            split[
                                "legal_predicates"
                            ]
                        ),

                    "newly_repaired":
                        [
                            int(x)
                            for x in split[
                                "newly_repaired"
                            ]
                        ],
                }
            )

        # Every individually repairable valid target has one unique
        # path through a deterministic binary policy.
        assert np.all(
            membership[
                ind
            ]
            ==
            1
        )

        assert np.array_equal(
            reconstructed_success,
            base_success,
        )

        assert int(
            reconstructed_success.sum()
        ) == S

        # --------------------------------------------------------
        # Choose exactly one terminal leaf for optional expansion.
        # Root / depth-2 tree itself is NOT changed.
        # --------------------------------------------------------

        gains = np.asarray(
            [
                r[
                    "gain"
                ]
                for r in leaf_results
            ],
            dtype=np.int16,
        )

        best_leaf_index = int(
            np.argmax(
                gains
            )
        )

        best_leaf = (
            leaf_results[
                best_leaf_index
            ]
        )

        best_gain = int(
            best_leaf[
                "gain"
            ]
        )

        local1_S = (
            S
            +
            best_gain
        )

        assert (
            S
            <=
            local1_S
            <=
            O
        )

        # Important structural check:
        # if a depth<2 terminal leaf could obtain positive coverage
        # gain from one existing predicate, frozen choose_second
        # should already have taken that predicate.
        for r in leaf_results:

            if (
                int(
                    r[
                        "gain"
                    ]
                )
                >
                0
            ):

                if int(
                    r[
                        "depth"
                    ]
                ) == 0:
                    positive_gain_depth0 += 1

                elif int(
                    r[
                        "depth"
                    ]
                ) == 1:
                    positive_gain_depth1 += 1

                elif int(
                    r[
                        "depth"
                    ]
                ) == 2:
                    positive_gain_depth2 += 1

        if best_gain > 0:
            improved_queries += 1

        if local1_S == O:
            closed_queries += 1

        architecture_limited = bool(
            int(
                tree2_saved[
                    qi
                ]
            )
            <
            O
        )

        if architecture_limited:

            architecture_limited_queries += 1

            architecture_limited_gap += (
                O
                -
                S
            )

            architecture_limited_gain += (
                best_gain
            )

            if best_gain > 0:
                architecture_limited_improved += 1

            if local1_S == O:
                architecture_limited_closed += 1

        sum_O += O
        sum_S += S
        sum_local1 += local1_S

        total_gain += best_gain

        elapsed = (
            time.perf_counter()
            -
            q_t0
        )

        root = (
            int(
                policy[
                    "first_feature"
                ]
            )
            if policy is not None
            else
            -1
        )

        rec = {
            "query":
                int(
                    qi
                ),

            "O":
                O,

            "S":
                S,

            "Gamma":
                O - S,

            "union_predicates":
                P,

            "root":
                root,

            "terminal_leaf_count":
                int(
                    len(
                        leaves
                    )
                ),

            "local1_S":
                int(
                    local1_S
                ),

            "local1_gain":
                int(
                    best_gain
                ),

            "remaining_gap":
                int(
                    O
                    -
                    local1_S
                ),

            "best_leaf_index":
                int(
                    best_leaf_index
                ),

            "best_leaf_label":
                best_leaf[
                    "label"
                ],

            "best_leaf_depth":
                int(
                    best_leaf[
                        "depth"
                    ]
                ),

            "best_third_predicate":
                int(
                    best_leaf[
                        "predicate"
                    ]
                ),

            "architecture_limited_K4":
                architecture_limited,

            "oracle_tree2":
                int(
                    tree2_saved[
                        qi
                    ]
                ),

            "leaves":
                leaf_results,

            "runtime_seconds":
                float(
                    elapsed
                ),
        }

        records.append(
            rec
        )

        processed += 1

        print(
            f"q={qi:03d} "
            f"O={O:02d} "
            f"S={S:02d} "
            f"Local+1={local1_S:02d} "
            f"gain={best_gain:+02d} "
            f"G={O-S:02d}->{O-local1_S:02d} "
            f"leaves={len(leaves)} "
            f"bestLeaf={best_leaf['label']} "
            f"depth={best_leaf['depth']} "
            f"q3={best_leaf['predicate']:04d} "
            f"K4lim={int(architecture_limited)} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    # ============================================================
    # Summary
    # ============================================================

    runtime = (
        time.perf_counter()
        -
        t_all
    )

    print()
    print("=" * 140)
    print(
        "FIXED-TREE ONE-LEAF EXISTING-SPLIT SUMMARY"
    )
    print("=" * 140)

    print(
        "residual queries processed =",
        processed,
    )

    print(
        "sum O / UNION S / Local+1 S =",
        sum_O,
        sum_S,
        sum_local1,
    )

    print(
        "UNION residual gap =",
        sum_O
        -
        sum_S,
    )

    print(
        "Local+1 total gain =",
        total_gain,
    )

    print(
        "Local+1 remaining gap =",
        sum_O
        -
        sum_local1,
    )

    if (
        sum_O
        -
        sum_S
    ) > 0:

        print(
            "fraction of UNION gap recovered =",
            float(
                total_gain
                /
                (
                    sum_O
                    -
                    sum_S
                )
            ),
        )

    print(
        "queries improved =",
        improved_queries,
    )

    print(
        "queries full gap closed =",
        closed_queries,
    )

    print()
    print(
        "positive-gain leaf candidates by original terminal depth:"
    )

    print(
        "depth0 =",
        positive_gain_depth0,
    )

    print(
        "depth1 =",
        positive_gain_depth1,
    )

    print(
        "depth2 =",
        positive_gain_depth2,
    )

    # A positive gain below depth 2 would contradict exact frozen
    # second-stage optimization with the same predicate family.
    assert (
        positive_gain_depth0
        ==
        0
    )

    assert (
        positive_gain_depth1
        ==
        0
    )

    print()
    print(
        "architecture-limited (exact K4 < O) queries =",
        architecture_limited_queries,
    )

    print(
        "architecture-limited UNION gap =",
        architecture_limited_gap,
    )

    print(
        "Local+1 gain on architecture-limited queries =",
        architecture_limited_gain,
    )

    print(
        "architecture-limited queries improved =",
        architecture_limited_improved,
    )

    print(
        "architecture-limited queries fully closed =",
        architecture_limited_closed,
    )

    if architecture_limited_gap > 0:

        print(
            "architecture-limited gap recovery fraction =",
            float(
                architecture_limited_gain
                /
                architecture_limited_gap
            ),
        )

    print(
        "runtime seconds =",
        runtime,
    )

    # ============================================================
    # Save
    # ============================================================

    OUT = (
        D
        /
        f"fixed_tree_one_leaf_existing_split_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "name":
                    "Fixed Depth-2 Optimum + One Existing-Predicate Leaf Split Audit v0",

                "scope":
                    "FIT structural baseline; fixed UNION H_ind",

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

                "sum_local1_S":
                    int(
                        sum_local1
                    ),

                "total_gain":
                    int(
                        total_gain
                    ),

                "remaining_gap":
                    int(
                        sum_O
                        -
                        sum_local1
                    ),

                "queries_improved":
                    int(
                        improved_queries
                    ),

                "queries_closed":
                    int(
                        closed_queries
                    ),

                "positive_gain_depth0":
                    int(
                        positive_gain_depth0
                    ),

                "positive_gain_depth1":
                    int(
                        positive_gain_depth1
                    ),

                "positive_gain_depth2":
                    int(
                        positive_gain_depth2
                    ),

                "architecture_limited_queries":
                    int(
                        architecture_limited_queries
                    ),

                "architecture_limited_gap":
                    int(
                        architecture_limited_gap
                    ),

                "architecture_limited_gain":
                    int(
                        architecture_limited_gain
                    ),

                "architecture_limited_queries_improved":
                    int(
                        architecture_limited_improved
                    ),

                "architecture_limited_queries_closed":
                    int(
                        architecture_limited_closed
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
        "SCRIPT SHA256 =",
        sha256_file(
            Path(
                __file__
            )
        ),
    )

    print("=" * 140)


if __name__ == "__main__":
    main()
