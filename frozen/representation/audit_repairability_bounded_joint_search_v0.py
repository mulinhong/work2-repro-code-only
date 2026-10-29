#!/usr/bin/env python3

from pathlib import Path
import hashlib
import importlib.util
import json
import time
import numpy as np


ROOT = Path(
    "/root/autodl-tmp/work2-algorithm-innovation"
)

A2_ART = ROOT / "fhex_v1/results/bcr_a2_exact_program_fact_bcct.npz"
TECS_ART = ROOT / "fhex_v1/results/tree_embeddable_certificate_synthesis_v0.npz"
AST_CACHE = ROOT / "fhex_v1/cache/python_train_ast_signature_v0.npz"

A4_SCRIPT = ROOT / "fhex_v1/scripts/audit_bcr_a4_rcps.py"
TECS_SCRIPT = ROOT / "fhex_v1/scripts/audit_tree_embeddable_certificate_synthesis_v0.py"

OUT_NPZ = ROOT / "fhex_v1/results/repairability_bounded_joint_search_v0.npz"
OUT_JSON = ROOT / "fhex_v1/results/repairability_bounded_joint_search_v0.json"


A2_SHA = (
    "4ce63e6e5eabf2662a291ebf9064b923"
    "880df124d5afe617e14ee357c0248d85"
)

TECS_SHA = (
    "eea14a27eae6f0aee02a6f74caed4294"
    "c2227977a0fb165b26957ced143c7729"
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
            x = f.read(1 << 20)
            if not x:
                break
            h.update(x)

    return h.hexdigest()


def load_module(path, name):

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(path)

    m = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        m
    )

    return m


def stats(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "sum": float(x.sum()),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, .90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def policy_key(policy):

    if policy is None:

        return (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        )

    return (
        int(policy["coverage"]),
        int(policy["gain"]),
        -int(policy["worst_leaf"]),
        -int(policy["first_feature"]),
        -int(policy["second_zero"]),
        -int(policy["second_one"]),
    )


def better_policy(
    cand,
    best,
):

    if best is None:
        return True

    kc = (
        cand["coverage"],
        cand["gain"],
        -cand["worst_leaf"],
        -cand["first_feature"],
    )

    kb = (
        best["coverage"],
        best["gain"],
        -best["worst_leaf"],
        -best["first_feature"],
    )

    return kc > kb


def exact_baseline_branch_masks(
    facts,
    valid,
):

    """
    Number of unique second-stage branch masks evaluated by
    the frozen exhaustive choose_bcct implementation.
    """

    s0, s1 = a4.build_survival_masks(
        facts
    )

    valid_ids = np.flatnonzero(
        valid
    )

    eligible_roots = 0
    masks = set()

    for root in range(
        facts.shape[1]
    ):

        ans = facts[
            valid_ids,
            root
        ]

        if not (
            np.any(
                ans == ABSENT
            )
            and
            np.any(
                ans == PRESENT
            )
        ):
            continue

        eligible_roots += 1

        masks.add(
            np.packbits(
                s0[root]
            ).tobytes()
        )

        masks.add(
            np.packbits(
                s1[root]
            ).tobytes()
        )

    return (
        eligible_roots,
        len(masks),
    )


def build_atomic_signatures(
    atomic,
    valid,
):

    """
    Exact 44-bit signatures for AST-valid candidates.

    Invalid candidates receive no signature because they
    are universally UNKNOWN and are separately treated as
    irreducible blockers.
    """

    sig = np.zeros(
        E,
        dtype=object,
    )

    for i in range(E):

        if not valid[i]:
            sig[i] = None
            continue

        bits = 0

        for j in range(
            atomic.shape[1]
        ):

            if atomic[i, j] == PRESENT:
                bits |= (
                    1 << j
                )

        sig[i] = bits

    return sig


def branch_repair_upper_bound(
    mask,
    valid,
    signatures,
):

    """
    Admissible TECS child-coverage upper bound.

    Any candidate before target t that is:
      (1) AST-invalid, hence UNKNOWN for every predicate, or
      (2) AST-valid with the identical complete atomic
          signature as t

    cannot be separated from t by any signed conjunction
    synthesized from those atomic facts.

    Therefore the count of such surviving prefix candidates
    lower-bounds t's achievable rank under ANY TECS child.

    A binary child also creates at most two leaves and each
    leaf can certify at most B valid hypotheses, giving the
    independent 2B capacity upper bound.
    """

    valid_targets = np.flatnonzero(
        valid
        &
        mask
    )

    if len(valid_targets) == 0:
        return 0

    possible = 0

    for t in valid_targets:

        target_sig = signatures[t]

        irreducible_rank = 0

        for j in range(
            t + 1
        ):

            if not mask[j]:
                continue

            if not valid[j]:

                # Invalid AST => UNKNOWN under all atomic
                # predicates and all TECS signed conjunctions.
                irreducible_rank += 1

            elif signatures[j] == target_sig:

                # Identical atomic signature => identical
                # answer under any TECS predicate.
                irreducible_rank += 1

        if irreducible_rank <= B:
            possible += 1

    return min(
        possible,
        2 * B,
    )


def bounded_choose_bcct(
    facts,
    atomic,
    valid,
):

    """
    Exact depth-2 BCCT search with an admissible
    repairability-specific root upper bound.

    Root is pruned ONLY when:

        UB(root) < incumbent coverage.

    Equality is NOT pruned because gain / worst-leaf /
    root-id tie-breaking may still improve the frozen
    lexicographic objective.
    """

    survive0, survive1 = (
        a4.build_survival_masks(
            facts
        )
    )

    valid_ids = np.flatnonzero(
        valid
    )

    signatures = build_atomic_signatures(
        atomic,
        valid,
    )

    roots = []

    for root in range(
        facts.shape[1]
    ):

        ans = facts[
            valid_ids,
            root
        ]

        if not (
            np.any(
                ans == ABSENT
            )
            and
            np.any(
                ans == PRESENT
            )
        ):
            continue

        m0 = survive0[root]
        m1 = survive1[root]

        ub0 = branch_repair_upper_bound(
            m0,
            valid,
            signatures,
        )

        ub1 = branch_repair_upper_bound(
            m1,
            valid,
            signatures,
        )

        ub = (
            ub0
            +
            ub1
        )

        roots.append(
            (
                -ub,
                root,
                ub,
            )
        )

    roots.sort()

    cache = {}

    exact_child_solves = 0

    def get_branch(mask):

        nonlocal exact_child_solves

        key = np.packbits(
            mask
        ).tobytes()

        if key not in cache:

            cache[key] = a4.choose_second(
                facts,
                valid,
                survive0,
                survive1,
                mask,
            )

            exact_child_solves += 1

        return cache[key]

    best = None

    pruned_roots = 0
    evaluated_roots = 0

    for _, root, ub in roots:

        if (
            best is not None
            and
            ub
            <
            best[
                "coverage"
            ]
        ):

            pruned_roots += 1
            continue

        evaluated_roots += 1

        m0 = survive0[
            root
        ]

        m1 = survive1[
            root
        ]

        p0 = get_branch(
            m0
        )

        p1 = get_branch(
            m1
        )

        cand = {
            "first_feature":
                int(root),

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

        if better_policy(
            cand,
            best,
        ):
            best = cand

    return {
        "policy":
            best,

        "eligible_roots":
            len(
                roots
            ),

        "evaluated_roots":
            evaluated_roots,

        "pruned_roots":
            pruned_roots,

        "exact_child_solves":
            exact_child_solves,

        "unique_evaluated_masks":
            len(
                cache
            ),
    }


def main():

    global a4

    t0 = time.perf_counter()

    print("=" * 150)
    print("REPAIRABILITY-BOUNDED JOINT SEARCH v0")
    print("=" * 150)

    print("FIT only.")
    print("P1 is NOT opened.")
    print("P2 is NOT opened.")
    print("Frozen TECS-v0 predicate family is unchanged.")

    print(
        "This audit changes ONLY the exact tree solver."
    )

    print(
        "Pruning uses irreducible prefix blockers + "
        "binary child capacity."
    )

    print(
        "No learned ordering, no tuned threshold, "
        "no heuristic pruning."
    )

    assert (
        sha256_file(
            A2_ART
        )
        ==
        A2_SHA
    )

    assert (
        sha256_file(
            TECS_ART
        )
        ==
        TECS_SHA
    )

    a4 = load_module(
        A4_SCRIPT,
        "rbjs_a4",
    )

    tecsmod = load_module(
        TECS_SCRIPT,
        "rbjs_tecs",
    )

    a2 = np.load(
        A2_ART,
        allow_pickle=True,
    )

    tecs = np.load(
        TECS_ART,
        allow_pickle=True,
    )

    ast = np.load(
        AST_CACHE,
        allow_pickle=True,
    )

    candidates = a2[
        "candidates"
    ].astype(
        np.int32
    )

    valid_all = a2[
        "candidate_ast_valid"
    ].astype(
        bool
    )

    names = ast[
        "feature_names"
    ].astype(
        str
    )

    signature = ast[
        "signature"
    ].astype(
        np.float32
    )

    allowed_ids = np.flatnonzero(
        names
        !=
        "max_ast_depth"
    )

    assert len(
        allowed_ids
    ) == 44

    frozen_S = tecs[
        "tecs_S"
    ].astype(
        np.int16
    )

    n = len(
        candidates
    )

    assert n == 568

    bounded_S = np.zeros(
        n,
        dtype=np.int16,
    )

    eligible_roots = np.zeros(
        n,
        dtype=np.int32,
    )

    evaluated_roots = np.zeros(
        n,
        dtype=np.int32,
    )

    pruned_roots = np.zeros(
        n,
        dtype=np.int32,
    )

    baseline_child_masks = np.zeros(
        n,
        dtype=np.int32,
    )

    bounded_child_solves = np.zeros(
        n,
        dtype=np.int32,
    )

    exact_policy_mismatch = 0
    coverage_mismatch = 0

    loop_t0 = time.perf_counter()

    for qi in range(n):

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
                candidates[qi]
            ][
                :,
                allowed_ids
            ]
            >
            0.0
        ).astype(
            np.int8
        )

        valid = valid_all[
            qi
        ]

        atomic[
            valid
        ] = presence[
            valid
        ]

        built = tecsmod.build_tecs_library(
            atomic,
            valid,
            a4,
        )

        facts = built[
            "library"
        ]

        # Frozen exhaustive reference.
        (
            baseline_policy,
            _,
            _,
        ) = a4.choose_bcct(
            facts,
            valid,
        )

        (
            base_roots,
            base_masks,
        ) = exact_baseline_branch_masks(
            facts,
            valid,
        )

        # Repairability-bounded exact solver.
        solved = bounded_choose_bcct(
            facts,
            atomic,
            valid,
        )

        policy = solved[
            "policy"
        ]

        if policy_key(
            policy
        ) != policy_key(
            baseline_policy
        ):

            exact_policy_mismatch += 1

        if policy is None:

            cov = 0

        else:

            cov = int(
                policy[
                    "coverage"
                ]
            )

        bounded_S[
            qi
        ] = cov

        if (
            cov
            !=
            int(
                frozen_S[
                    qi
                ]
            )
        ):

            coverage_mismatch += 1

        eligible_roots[
            qi
        ] = int(
            base_roots
        )

        baseline_child_masks[
            qi
        ] = int(
            base_masks
        )

        evaluated_roots[
            qi
        ] = int(
            solved[
                "evaluated_roots"
            ]
        )

        pruned_roots[
            qi
        ] = int(
            solved[
                "pruned_roots"
            ]
        )

        bounded_child_solves[
            qi
        ] = int(
            solved[
                "exact_child_solves"
            ]
        )

        if (
            (qi + 1) % 25 == 0
            or
            qi == n - 1
        ):

            print(
                f"[{qi+1:>3}/{n}]",
                "S=",
                int(
                    bounded_S[
                        :qi+1
                    ].sum()
                ),
                "roots_pruned=",
                int(
                    pruned_roots[
                        :qi+1
                    ].sum()
                ),
                "/",
                int(
                    eligible_roots[
                        :qi+1
                    ].sum()
                ),
                "child_solves=",
                int(
                    bounded_child_solves[
                        :qi+1
                    ].sum()
                ),
                "/",
                int(
                    baseline_child_masks[
                        :qi+1
                    ].sum()
                ),
                "elapsed=",
                round(
                    time.perf_counter()
                    -
                    loop_t0,
                    1,
                ),
                flush=True,
            )

    print()
    print("=" * 150)
    print("EXACTNESS INVARIANTS")
    print("=" * 150)

    print(
        "exact policy mismatches =",
        exact_policy_mismatch,
    )

    print(
        "coverage mismatches vs frozen TECS =",
        coverage_mismatch,
    )

    print(
        "bounded shared total =",
        int(
            bounded_S.sum()
        ),
    )

    print(
        "frozen TECS shared total =",
        int(
            frozen_S.sum()
        ),
    )

    assert exact_policy_mismatch == 0
    assert coverage_mismatch == 0

    # ========================================================
    # Efficiency.
    # ========================================================

    total_roots = int(
        eligible_roots.sum()
    )

    total_pruned = int(
        pruned_roots.sum()
    )

    total_eval = int(
        evaluated_roots.sum()
    )

    total_base_masks = int(
        baseline_child_masks.sum()
    )

    total_child = int(
        bounded_child_solves.sum()
    )

    root_prune_fraction = (
        float(total_pruned)
        /
        float(total_roots)
        if total_roots > 0
        else 0.0
    )

    child_reduction_fraction = (
        1.0
        -
        (
            float(total_child)
            /
            float(total_base_masks)
        )
        if total_base_masks > 0
        else 0.0
    )

    per_query_prune = (
        pruned_roots.astype(
            np.float64
        )
        /
        np.maximum(
            eligible_roots,
            1
        )
    )

    per_query_child_reduction = (
        1.0
        -
        (
            bounded_child_solves.astype(
                np.float64
            )
            /
            np.maximum(
                baseline_child_masks,
                1
            )
        )
    )

    print()
    print("=" * 150)
    print("SEARCH REDUCTION")
    print("=" * 150)

    print(
        "eligible roots total =",
        total_roots,
    )

    print(
        "evaluated roots total =",
        total_eval,
    )

    print(
        "pruned roots total =",
        total_pruned,
    )

    print(
        "root prune fraction =",
        root_prune_fraction,
    )

    print(
        "per-query root prune fraction =",
        stats(
            per_query_prune
        ),
    )

    print()
    print(
        "baseline unique child masks =",
        total_base_masks,
    )

    print(
        "bounded exact child solves =",
        total_child,
    )

    print(
        "child-subproblem reduction fraction =",
        child_reduction_fraction,
    )

    print(
        "per-query child reduction =",
        stats(
            per_query_child_reduction
        ),
    )

    # ========================================================
    # Frozen solver gate.
    # ========================================================

    gate_exact = (
        exact_policy_mismatch == 0
        and
        coverage_mismatch == 0
    )

    gate_root = (
        root_prune_fraction
        >
        0.50
    )

    gate_child = (
        child_reduction_fraction
        >
        0.50
    )

    continue_solver = (
        gate_exact
        and
        gate_root
        and
        gate_child
    )

    print()
    print("=" * 150)
    print("SPECIALIZED-SOLVER DECISION")
    print("=" * 150)

    print(
        "Gate 1 exact TECS parity =",
        gate_exact,
    )

    print(
        "Gate 2 >50% root pruning =",
        gate_root,
    )

    print(
        "Gate 3 >50% child-subproblem reduction =",
        gate_child,
    )

    print(
        "CONTINUE_REPAIRABILITY_BOUNDED_SOLVER =",
        continue_solver,
    )

    if continue_solver:

        diagnosis = (
            "STRONG SOLVER SIGNAL. The retrieval-repair-specific "
            "irreducible-prefix bound preserves the exact TECS "
            "optimum while removing the majority of root and "
            "child search. Promote this bound into the v2 exact "
            "solver and next replace exhaustive TECS predicate "
            "materialization with on-demand minimal-certificate "
            "pricing."
        )

    elif gate_exact:

        diagnosis = (
            "BOUND IS SOUND BUT TOO WEAK UNDER THE FROZEN "
            "EFFICIENCY GATES. Preserve TECS-v0 as the strong "
            "v2 synthesis mechanism, but do not promote this "
            "particular pruning bound. Move to a tighter "
            "prefix-deficit / certificate-aware pricing bound."
        )

    else:

        diagnosis = (
            "SOLVER PARITY FAILED. Do not use this bound. "
            "Frozen TECS-v0 remains unaffected."
        )

    print(
        "DIAGNOSIS =",
        diagnosis,
    )

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "repairability_bounded_joint_search_v0"
        ),

        bounded_S=
            bounded_S,

        frozen_tecs_S=
            frozen_S,

        eligible_roots=
            eligible_roots,

        evaluated_roots=
            evaluated_roots,

        pruned_roots=
            pruned_roots,

        baseline_child_masks=
            baseline_child_masks,

        bounded_child_solves=
            bounded_child_solves,
    )

    result = {
        "name":
            "Repairability-Bounded Joint Search v0",

        "exactness": {
            "policy_mismatches":
                exact_policy_mismatch,

            "coverage_mismatches":
                coverage_mismatch,

            "shared_total":
                int(
                    bounded_S.sum()
                ),
        },

        "search": {
            "eligible_roots":
                total_roots,

            "evaluated_roots":
                total_eval,

            "pruned_roots":
                total_pruned,

            "root_prune_fraction":
                root_prune_fraction,

            "baseline_child_masks":
                total_base_masks,

            "bounded_child_solves":
                total_child,

            "child_reduction_fraction":
                child_reduction_fraction,
        },

        "gates": {
            "exact":
                bool(
                    gate_exact
                ),

            "root_prune_gt_half":
                bool(
                    gate_root
                ),

            "child_reduction_gt_half":
                bool(
                    gate_child
                ),

            "continue":
                bool(
                    continue_solver
                ),
        },

        "diagnosis":
            diagnosis,

        "runtime_seconds":
            float(
                time.perf_counter()
                -
                t0
            ),

        "protocol": [
            "FIT only.",
            "No P1 access.",
            "No P2 access.",
            "Frozen TECS-v0 predicate family unchanged.",
            "B=20 and depth=2 unchanged.",
            "The only modification is exact root pruning using an admissible retrieval-repair-specific upper bound.",
            "AST-invalid candidates are irreducible UNKNOWN blockers.",
            "AST-valid candidates with identical complete atomic signatures are irreducible under all TECS signed conjunctions.",
            "A binary child can certify at most 2B valid hypotheses.",
            "A root is pruned only when its admissible coverage upper bound is strictly below incumbent coverage.",
            "Roots tied on coverage upper bound are never pruned, preserving frozen gain/worst-leaf/root-id tie-breaking.",
            "No learned ordering.",
            "No threshold tuning.",
            "No heuristic pruning.",
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
    print("=" * 150)
    print("AUDIT COMPLETE")
    print("=" * 150)

    print(
        "artifact =",
        OUT_NPZ,
    )

    print(
        "artifact SHA256 =",
        sha256_file(
            OUT_NPZ
        ),
    )

    print(
        "metrics =",
        OUT_JSON,
    )

    print(
        "runtime_seconds =",
        result[
            "runtime_seconds"
        ],
    )


if __name__ == "__main__":
    main()
