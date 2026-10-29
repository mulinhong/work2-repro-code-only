from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.util
import itertools
import json
import time

import numpy as np


# ======================================================================
# Frozen problem definition
# ======================================================================

ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")

E = 80
B = 20

UNKNOWN = -1
ABSENT = 0
PRESENT = 1


A2_ART = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a2_exact_program_fact_bcct.npz"
)

AST_CACHE = (
    ROOT
    / "fhex_v1/cache/"
      "python_train_ast_signature_v0.npz"
)

A4_SCRIPT = (
    ROOT
    / "fhex_v1/scripts/"
      "audit_bcr_a4_rcps.py"
)

GAP_SCRIPT = (
    ROOT
    / "fhex_v1/scripts/"
      "audit_rcps_all_hypothesis_shared_gap_v0.py"
)

TECS_SCRIPT = (
    ROOT
    / "fhex_v1/frozen_v2/"
      "audit_tree_embeddable_certificate_synthesis_v0.py"
)

RPS_MERGED = (
    ROOT
    / "fhex_v1/tecs_plus_exploratory_v1/"
      "rps_tecs_plus_v1_full_fit_merged.npz"
)

OUT_DIR = (
    ROOT
    / "fhex_v1/obstruction_audit_v0"
)


EXPECTED_A4_SHA = (
    "580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9"
)

EXPECTED_GAP_SHA = (
    "df2d8f33869ed623122f480bcd7a1d4f79a48a083a791b8ac43ac05746712e15"
)

EXPECTED_TECS_SHA = (
    "12c31549062575e68b4cd0daa57659ad4572d372c16d11cf366a8c5ba7221d98"
)

EXPECTED_AST_SHA = (
    "4316bf9327754a47a0ad97e868087cd8c4ced236800b11346a3d073f733f0e36"
)

EXPECTED_RPS_SHA = (
    "a398fd1c67fd51d15dff68518fe48b98c14521ace4b1496a49b95eea6462de87"
)


# ======================================================================
# Utilities
# ======================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def bool_vector_to_int(x: np.ndarray) -> int:
    """
    Bit i corresponds exactly to predicate i.
    Arbitrary precision Python int allows P > 64.
    """
    x = np.asarray(
        x,
        dtype=np.uint8,
    )

    packed = np.packbits(
        x,
        bitorder="little",
    )

    return int.from_bytes(
        packed.tobytes(),
        byteorder="little",
        signed=False,
    )


def witness_json(
    local_indices,
    target_ids,
    candidates,
    shared_vec,
):
    tids = [
        int(target_ids[j])
        for j in local_indices
    ]

    return {
        "target_positions_zero_based":
            tids,

        "target_ranks_one_based":
            [
                int(t + 1)
                for t in tids
            ],

        "candidate_program_ids":
            [
                int(candidates[t])
                for t in tids
            ],

        "residual_under_frozen_optimum":
            [
                bool(
                    not shared_vec[t]
                )
                for t in tids
            ],
    }


# ======================================================================
# Exact root-conditioned continuation characterization
# ======================================================================

def build_root_characterization(
    facts: np.ndarray,
    valid: np.ndarray,
    individually_repairable: np.ndarray,
    a4,
):
    """
    For every eligible first question p and every individually
    repairable target t, construct the exact set of legal second-stage
    choices that repair t after taking t's true branch at p.

    Second-stage choice universe:
        q = 0 ... P-1
        STOP = P

    A bit mask represents this set.

    Important:
    - root eligibility exactly follows frozen A4:
      root must have both 0 and 1 among all valid candidates.
    - second-predicate eligibility exactly follows frozen A4:
      within the actual root branch it must have both 0 and 1
      among valid candidate hypotheses.
    - STOP is feasible for t iff root alone already puts t at rank <= B.
    - UNKNOWN candidates survive either answer through frozen masks.
    """

    P = int(
        facts.shape[1]
    )

    stop_bit = (
        1 << P
    )

    survive0, survive1 = (
        a4.build_survival_masks(
            facts
        )
    )

    target_ids = np.flatnonzero(
        individually_repairable
    ).astype(
        np.int16
    )

    valid_ids = np.flatnonzero(
        valid
    )

    assert len(target_ids) > 0

    # Every valid target must have exact 0/1 answers.
    assert np.all(
        (
            facts[
                target_ids
            ]
            ==
            ABSENT
        )
        |
        (
            facts[
                target_ids
            ]
            ==
            PRESENT
        )
    )

    root_records = []

    # Rootless STOP baseline.
    raw_stop_coverage = int(
        np.sum(
            target_ids
            <
            B
        )
    )

    characterization_S = (
        raw_stop_coverage
    )

    for p in range(P):

        root_answers_valid = (
            facts[
                valid_ids,
                p
            ]
        )

        # Frozen A4 root eligibility.
        if not (
            np.any(
                root_answers_valid
                ==
                ABSENT
            )
            and
            np.any(
                root_answers_valid
                ==
                PRESENT
            )
        ):
            continue

        root_answers = (
            facts[
                target_ids,
                p
            ]
            .astype(
                np.int8,
                copy=True,
            )
        )

        qmask = [
            0
            for _ in range(
                len(
                    target_ids
                )
            )
        ]

        root_coverage = 0

        for answer, current_mask in (
            (
                ABSENT,
                survive0[p],
            ),
            (
                PRESENT,
                survive1[p],
            ),
        ):

            local = np.flatnonzero(
                root_answers
                ==
                answer
            )

            if len(local) == 0:
                continue

            branch_targets = (
                target_ids[
                    local
                ]
            )

            prefix_root = np.cumsum(
                current_mask,
                dtype=np.int16,
            )

            root_ranks = (
                prefix_root[
                    branch_targets
                ]
            )

            stop_hit = (
                root_ranks
                <=
                B
            )

            # Frozen choose_second returns STOP immediately
            # whenever total branch size <= B.
            if int(
                current_mask.sum()
            ) <= B:

                assert np.all(
                    stop_hit
                )

                for j in local:
                    qmask[
                        int(j)
                    ] = (
                        stop_bit
                    )

                root_coverage += int(
                    len(local)
                )

                continue

            branch_valid_ids = (
                np.flatnonzero(
                    valid
                    &
                    current_mask
                )
            )

            assert (
                len(
                    branch_valid_ids
                )
                >
                0
            )

            second_answers_valid = (
                facts[
                    branch_valid_ids
                ].T
            )

            has0 = np.any(
                second_answers_valid
                ==
                ABSENT,
                axis=1,
            )

            has1 = np.any(
                second_answers_valid
                ==
                PRESENT,
                axis=1,
            )

            valid_second = (
                has0
                &
                has1
            )

            branch_best = int(
                stop_hit.sum()
            )

            if np.any(
                valid_second
            ):

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

                # [P, Hbranch]
                second_answers_target = (
                    facts[
                        branch_targets
                    ].T
                )

                ranks = np.where(
                    second_answers_target
                    ==
                    ABSENT,

                    prefix0[
                        :,
                        branch_targets
                    ],

                    prefix1[
                        :,
                        branch_targets
                    ],
                )

                hit = (
                    (
                        ranks
                        <=
                        B
                    )
                    &
                    valid_second[
                        :,
                        None
                    ]
                )

                second_best = int(
                    np.max(
                        hit.sum(
                            axis=1
                        )
                    )
                )

                branch_best = max(
                    branch_best,
                    second_best,
                )

                for h, j0 in enumerate(
                    local
                ):
                    bits = (
                        bool_vector_to_int(
                            hit[
                                :,
                                h
                            ]
                        )
                    )

                    if bool(
                        stop_hit[h]
                    ):
                        bits |= (
                            stop_bit
                        )

                    qmask[
                        int(j0)
                    ] = (
                        bits
                    )

            else:

                # No legal second predicate.
                # Only STOP may repair a target.
                for h, j0 in enumerate(
                    local
                ):
                    if bool(
                        stop_hit[h]
                    ):
                        qmask[
                            int(j0)
                        ] = (
                            stop_bit
                        )
                    else:
                        qmask[
                            int(j0)
                        ] = 0

            root_coverage += int(
                branch_best
            )

        characterization_S = max(
            characterization_S,
            int(
                root_coverage
            ),
        )

        root_records.append({
            "root":
                int(p),

            "answers":
                root_answers,

            "qmask":
                qmask,

            "coverage":
                int(
                    root_coverage
                ),
        })

    return {
        "target_ids":
            target_ids,

        "roots":
            root_records,

        "characterization_S":
            int(
                characterization_S
            ),

        "raw_stop_coverage":
            int(
                raw_stop_coverage
            ),

        "predicate_count":
            P,
    }


def subset_is_realizable(
    local_indices,
    characterization,
):
    """
    Exact depth<=2 joint realizability test.

    For a fixed root p, all targets taking branch a must share
    at least one common feasible continuation:
        intersection_t Q_p(t) != empty.

    Branches are independent, hence both branch intersections
    must be nonempty.

    Empty branches impose no constraint.
    """

    target_ids = (
        characterization[
            "target_ids"
        ]
    )

    # Empty subset.
    if len(local_indices) == 0:
        return True

    # Rootless STOP policy.
    if all(
        int(
            target_ids[j]
        )
        <
        B
        for j in local_indices
    ):
        return True

    for rec in characterization[
        "roots"
    ]:

        inter0 = None
        inter1 = None

        ok = True

        for j in local_indices:

            m = int(
                rec[
                    "qmask"
                ][
                    j
                ]
            )

            if m == 0:
                ok = False
                break

            a = int(
                rec[
                    "answers"
                ][
                    j
                ]
            )

            if a == ABSENT:

                if inter0 is None:
                    inter0 = m
                else:
                    inter0 &= m

                if inter0 == 0:
                    ok = False
                    break

            else:

                if inter1 is None:
                    inter1 = m
                else:
                    inter1 &= m

                if inter1 == 0:
                    ok = False
                    break

        if ok:
            return True

    return False


# ======================================================================
# Pair / triple minimal-obstruction audit
# ======================================================================

def find_minimal_obstruction_upto3(
    characterization,
    shared_vec: np.ndarray,
    candidates: np.ndarray,
):
    """
    Search exact minimal obstruction order up to 3.

    Search-space reduction is exact:

    The targets covered by the frozen optimal policy are themselves
    jointly realizable by that policy. Therefore an unrealizable subset
    cannot lie entirely inside that covered set. Every obstruction must
    contain at least one residual target.
    """

    target_ids = (
        characterization[
            "target_ids"
        ]
    )

    n = len(
        target_ids
    )

    covered_local = (
        shared_vec[
            target_ids
        ]
    )

    residual_local = (
        ~covered_local
    )

    residual_indices = set(
        int(x)
        for x in np.flatnonzero(
            residual_local
        )
    )

    residual_n = len(
        residual_indices
    )

    assert (
        residual_n
        >
        0
    )

    # The full individually-repairable set MUST be unrealizable
    # whenever Gamma > 0.
    full_realizable = (
        subset_is_realizable(
            tuple(
                range(n)
            ),
            characterization,
        )
    )

    assert not full_realizable, (
        "Gamma > 0 but full individual set "
        "was classified jointly realizable."
    )

    # --------------------------------------------------------------
    # Order 2
    # --------------------------------------------------------------

    pair_tests = 0

    for i in range(n):
        for j in range(
            i + 1,
            n,
        ):

            if (
                i
                not in
                residual_indices
                and
                j
                not in
                residual_indices
            ):
                continue

            pair_tests += 1

            subset = (
                i,
                j,
            )

            if not subset_is_realizable(
                subset,
                characterization,
            ):

                return {
                    "class":
                        "PAIR",

                    "min_order":
                        2,

                    "min_order_exact":
                        True,

                    "pair_tests":
                        pair_tests,

                    "triple_tests":
                        0,

                    "witness":
                        witness_json(
                            subset,
                            target_ids,
                            candidates,
                            shared_vec,
                        ),
                }

    # --------------------------------------------------------------
    # Order 3
    # --------------------------------------------------------------

    triple_tests = 0

    for subset in itertools.combinations(
        range(n),
        3,
    ):

        if not any(
            x
            in
            residual_indices
            for x in subset
        ):
            continue

        triple_tests += 1

        if not subset_is_realizable(
            subset,
            characterization,
        ):

            # Since all order-2 candidates were already exhaustively
            # verified realizable, this is a true pairwise-compatible
            # but jointly-unrealizable triple.
            return {
                "class":
                    "TRIPLE",

                "min_order":
                    3,

                "min_order_exact":
                    True,

                "pair_tests":
                    pair_tests,

                "triple_tests":
                    triple_tests,

                "witness":
                    witness_json(
                        subset,
                        target_ids,
                        candidates,
                        shared_vec,
                    ),
            }

    # No obstruction of size <= 3.
    return {
        "class":
            "GE4",

        "min_order":
            4,

        "min_order_exact":
            False,

        "pair_tests":
            pair_tests,

        "triple_tests":
            triple_tests,

        "witness":
            None,
    }


# ======================================================================
# Main
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=568,
    )

    parser.add_argument(
        "--max-gap-queries",
        type=int,
        default=0,
        help=(
            "0 means no limit after Gamma>0 filtering."
        ),
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="run",
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assert (
        0
        <=
        args.start
        <=
        args.stop
        <=
        568
    )

    print("=" * 132)
    print(
        "DEPTH-2 SHARED-REPAIR OBSTRUCTION AUDIT v0"
    )
    print(
        "NO NEW PREDICATES / NO ALGORITHM MODIFICATION / FIT STRUCTURAL AUDIT ONLY"
    )
    print("=" * 132)

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    observed = {
        "A4_script":
            sha256_file(
                A4_SCRIPT
            ),

        "GAP_script":
            sha256_file(
                GAP_SCRIPT
            ),

        "TECS_script":
            sha256_file(
                TECS_SCRIPT
            ),

        "AST_cache":
            sha256_file(
                AST_CACHE
            ),

        "RPS_merged":
            sha256_file(
                RPS_MERGED
            ),
    }

    expected = {
        "A4_script":
            EXPECTED_A4_SHA,

        "GAP_script":
            EXPECTED_GAP_SHA,

        "TECS_script":
            EXPECTED_TECS_SHA,

        "AST_cache":
            EXPECTED_AST_SHA,

        "RPS_merged":
            EXPECTED_RPS_SHA,
    }

    for k in observed:
        print(
            f"{k:<14} = {observed[k]}"
        )
        assert (
            observed[k]
            ==
            expected[k]
        ), (
            k,
            observed[k],
            expected[k],
        )

    print(
        "HASH INTEGRITY = PASS"
    )

    # ------------------------------------------------------------------
    # Modules / data
    # ------------------------------------------------------------------

    a4 = load_module(
        A4_SCRIPT,
        "obs_a4",
    )

    gap = load_module(
        GAP_SCRIPT,
        "obs_gap",
    )

    tecs = load_module(
        TECS_SCRIPT,
        "obs_tecs",
    )

    a2 = np.load(
        A2_ART,
        allow_pickle=True,
    )

    az = np.load(
        AST_CACHE,
        allow_pickle=True,
    )

    rps = np.load(
        RPS_MERGED,
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

    assert (
        candidates_all.shape
        ==
        (
            568,
            E,
        )
    )

    assert (
        valid_all.shape
        ==
        (
            568,
            E,
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

    ast_valid = (
        az[
            "valid"
        ]
        .astype(
            bool
        )
    )

    allowed_ids = np.flatnonzero(
        names_all
        !=
        "max_ast_depth"
    )

    assert (
        len(
            allowed_ids
        )
        ==
        44
    )

    assert np.array_equal(
        names_all[
            allowed_ids
        ],
        a2[
            "feature_names"
        ].astype(
            str
        ),
    )

    assert np.array_equal(
        ast_valid[
            candidates_all
        ],
        valid_all,
    )

    base_O = (
        rps[
            "base_O"
        ]
        .astype(
            np.int16
        )
    )

    base_S = (
        rps[
            "base_S"
        ]
        .astype(
            np.int16
        )
    )

    base_G = (
        rps[
            "base_Gamma"
        ]
        .astype(
            np.int16
        )
    )

    assert int(
        base_O.sum()
    ) == 44711

    assert int(
        base_S.sum()
    ) == 43319

    assert int(
        base_G.sum()
    ) == 1392

    assert int(
        np.sum(
            base_G
            >
            0
        )
    ) == 333

    print(
        "FROZEN FIT ANCHOR = PASS"
    )

    # ------------------------------------------------------------------
    # Query audit
    # ------------------------------------------------------------------

    records = []

    processed_gap = 0

    total_start = (
        time.perf_counter()
    )

    for qi in range(
        args.start,
        args.stop,
    ):

        if int(
            base_G[
                qi
            ]
        ) <= 0:
            continue

        if (
            args.max_gap_queries
            >
            0
            and
            processed_gap
            >=
            args.max_gap_queries
        ):
            break

        t0 = (
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

        built = (
            tecs.build_tecs_library(
                atomic,
                valid,
                a4,
            )
        )

        facts = (
            built[
                "library"
            ]
        )

        # --------------------------------------------------------------
        # Frozen individual / shared parity
        # --------------------------------------------------------------

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

        policy, s0, s1 = (
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

        G = (
            O
            -
            S
        )

        assert np.all(
            (~shared)
            |
            ind
        )

        assert (
            O
            ==
            int(
                base_O[
                    qi
                ]
            )
        ), (
            qi,
            O,
            int(
                base_O[
                    qi
                ]
            ),
        )

        assert (
            S
            ==
            int(
                base_S[
                    qi
                ]
            )
        ), (
            qi,
            S,
            int(
                base_S[
                    qi
                ]
            ),
        )

        assert (
            G
            ==
            int(
                base_G[
                    qi
                ]
            )
        ), (
            qi,
            G,
            int(
                base_G[
                    qi
                ]
            ),
        )

        target_ids = (
            np.flatnonzero(
                ind
            )
        )

        residual = (
            ind
            &
            (~shared)
        )

        residual_n = int(
            residual.sum()
        )

        assert (
            residual_n
            ==
            G
        ), (
            qi,
            residual_n,
            G,
        )

        # --------------------------------------------------------------
        # New exact characterization
        # --------------------------------------------------------------

        tc0 = (
            time.perf_counter()
        )

        char = (
            build_root_characterization(
                facts,
                valid,
                ind,
                a4,
            )
        )

        char_seconds = (
            time.perf_counter()
            -
            tc0
        )

        # CRITICAL THEORY PARITY GATE.
        assert (
            int(
                char[
                    "characterization_S"
                ]
            )
            ==
            S
        ), (
            "CHARACTERIZATION PARITY FAILED",
            qi,
            char[
                "characterization_S"
            ],
            S,
        )

        assert (
            len(
                char[
                    "target_ids"
                ]
            )
            ==
            O
        )

        # Frozen optimal covered set must itself be realizable.
        shared_local = tuple(
            int(j)
            for j in np.flatnonzero(
                shared[
                    char[
                        "target_ids"
                    ]
                ]
            )
        )

        assert subset_is_realizable(
            shared_local,
            char,
        )

        # Whole individual set must NOT be realizable because G > 0.
        assert not subset_is_realizable(
            tuple(
                range(
                    O
                )
            ),
            char,
        )

        # --------------------------------------------------------------
        # Minimal obstruction <= 3
        # --------------------------------------------------------------

        ts0 = (
            time.perf_counter()
        )

        obs = (
            find_minimal_obstruction_upto3(
                char,
                shared,
                candidates,
            )
        )

        search_seconds = (
            time.perf_counter()
            -
            ts0
        )

        elapsed = (
            time.perf_counter()
            -
            t0
        )

        rec = {
            "query":
                int(qi),

            "valid_n":
                int(
                    valid.sum()
                ),

            "predicate_count":
                int(
                    facts.shape[1]
                ),

            "O":
                O,

            "S":
                S,

            "Gamma":
                G,

            "residual_n":
                residual_n,

            "eligible_roots":
                int(
                    len(
                        char[
                            "roots"
                        ]
                    )
                ),

            "characterization_S":
                int(
                    char[
                        "characterization_S"
                    ]
                ),

            "class":
                obs[
                    "class"
                ],

            "min_order":
                int(
                    obs[
                        "min_order"
                    ]
                ),

            "min_order_exact":
                bool(
                    obs[
                        "min_order_exact"
                    ]
                ),

            "pair_tests":
                int(
                    obs[
                        "pair_tests"
                    ]
                ),

            "triple_tests":
                int(
                    obs[
                        "triple_tests"
                    ]
                ),

            "witness":
                obs[
                    "witness"
                ],

            "characterization_seconds":
                float(
                    char_seconds
                ),

            "search_seconds":
                float(
                    search_seconds
                ),

            "runtime_seconds":
                float(
                    elapsed
                ),
        }

        records.append(
            rec
        )

        processed_gap += 1

        print(
            f"q={qi:03d} "
            f"V={int(valid.sum()):02d} "
            f"P={facts.shape[1]:04d} "
            f"O={O:02d} "
            f"S={S:02d} "
            f"G={G:02d} "
            f"R={residual_n:02d} "
            f"roots={len(char['roots']):04d} "
            f"charS={char['characterization_S']:02d} "
            f"class={obs['class']:<6} "
            f"pairTests={obs['pair_tests']:05d} "
            f"tripleTests={obs['triple_tests']:07d} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    total_elapsed = (
        time.perf_counter()
        -
        total_start
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    classes = {
        "PAIR": 0,
        "TRIPLE": 0,
        "GE4": 0,
    }

    for r in records:
        classes[
            r[
                "class"
            ]
        ] += 1

    print()
    print("=" * 132)
    print(
        "OBSTRUCTION AUDIT v0 SUMMARY"
    )
    print("=" * 132)

    print(
        "gap queries processed =",
        len(
            records
        ),
    )

    print(
        "PAIR   =",
        classes[
            "PAIR"
        ],
    )

    print(
        "TRIPLE =",
        classes[
            "TRIPLE"
        ],
    )

    print(
        "GE4    =",
        classes[
            "GE4"
        ],
    )

    print(
        "characterization parity failures = 0"
    )

    print(
        "total runtime seconds =",
        total_elapsed,
    )

    print()
    print(
        "PAIR   : certified minimal obstruction order = 2"
    )

    print(
        "TRIPLE : certified minimal obstruction order = 3 "
        "(all pairs realizable, some triple unrealizable)"
    )

    print(
        "GE4    : exhaustive proof that no obstruction of order <= 3 exists"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    stem = (
        f"depth2_obstruction_v0_"
        f"{args.tag}_"
        f"{args.start}_{args.stop}"
    )

    out_json = (
        OUT_DIR
        /
        f"{stem}.json"
    )

    out_npz = (
        OUT_DIR
        /
        f"{stem}.npz"
    )

    payload = {
        "name":
            "Depth-2 Shared-Repair Obstruction Audit v0",

        "scope":
            "FIT structural audit only",

        "start":
            int(
                args.start
            ),

        "stop":
            int(
                args.stop
            ),

        "max_gap_queries":
            int(
                args.max_gap_queries
            ),

        "records":
            records,

        "summary": {
            "processed_gap_queries":
                int(
                    len(
                        records
                    )
                ),

            "PAIR":
                int(
                    classes[
                        "PAIR"
                    ]
                ),

            "TRIPLE":
                int(
                    classes[
                        "TRIPLE"
                    ]
                ),

            "GE4":
                int(
                    classes[
                        "GE4"
                    ]
                ),

            "characterization_parity_failures":
                0,

            "runtime_seconds":
                float(
                    total_elapsed
                ),
        },

        "integrity":
            observed,
    }

    out_json.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        out_npz,

        query=np.asarray(
            [
                r["query"]
                for r in records
            ],
            dtype=np.int16,
        ),

        O=np.asarray(
            [
                r["O"]
                for r in records
            ],
            dtype=np.int16,
        ),

        S=np.asarray(
            [
                r["S"]
                for r in records
            ],
            dtype=np.int16,
        ),

        Gamma=np.asarray(
            [
                r["Gamma"]
                for r in records
            ],
            dtype=np.int16,
        ),

        residual_n=np.asarray(
            [
                r["residual_n"]
                for r in records
            ],
            dtype=np.int16,
        ),

        predicate_count=np.asarray(
            [
                r["predicate_count"]
                for r in records
            ],
            dtype=np.int16,
        ),

        eligible_roots=np.asarray(
            [
                r["eligible_roots"]
                for r in records
            ],
            dtype=np.int16,
        ),

        obstruction_class=np.asarray(
            [
                r["class"]
                for r in records
            ],
            dtype="<U8",
        ),

        min_order=np.asarray(
            [
                r["min_order"]
                for r in records
            ],
            dtype=np.int8,
        ),

        min_order_exact=np.asarray(
            [
                r["min_order_exact"]
                for r in records
            ],
            dtype=bool,
        ),

        pair_tests=np.asarray(
            [
                r["pair_tests"]
                for r in records
            ],
            dtype=np.int32,
        ),

        triple_tests=np.asarray(
            [
                r["triple_tests"]
                for r in records
            ],
            dtype=np.int32,
        ),

        runtime_seconds=np.asarray(
            [
                r["runtime_seconds"]
                for r in records
            ],
            dtype=np.float64,
        ),
    )

    print()
    print(
        "JSON =",
        out_json,
    )

    print(
        "NPZ  =",
        out_npz,
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
