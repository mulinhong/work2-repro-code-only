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
GAP_ART = ROOT / "fhex_v1/results/rcps_all_hypothesis_shared_gap_v0.npz"
IRCF_ART = ROOT / "fhex_v1/results/iterative_residual_certificate_folding_v1.npz"
AST_CACHE = ROOT / "fhex_v1/cache/python_train_ast_signature_v0.npz"

A2_SCRIPT = ROOT / "fhex_v1/scripts/audit_bcr_a2_exact_program_fact_bcct.py"
A4_SCRIPT = ROOT / "fhex_v1/scripts/audit_bcr_a4_rcps.py"
GAP_SCRIPT = ROOT / "fhex_v1/scripts/audit_rcps_all_hypothesis_shared_gap_v0.py"

OUT_NPZ = ROOT / "fhex_v1/results/tree_embeddable_certificate_synthesis_v0.npz"
OUT_JSON = ROOT / "fhex_v1/results/tree_embeddable_certificate_synthesis_v0.json"


A2_SHA = (
    "4ce63e6e5eabf2662a291ebf9064b923"
    "880df124d5afe617e14ee357c0248d85"
)

GAP_SHA = (
    "d509387302f80291ed92ba613cc1f115d"
    "5aae04410919b94b30baeebbd1825c9"
)

IRCF_SHA = (
    "41cc1c10c09f65b19f009a73d4197588"
    "fd63be50ff759eb9089900390a21784d"
)


E = 80
B = 20

UNKNOWN = -1
ABSENT = 0
PRESENT = 1

IRCF_SHARED = 41230
IRCF_CCG = 363

RCPS_SHARED = 43192

RCPS_GAIN_PER_ADDED = (
    0.01357612566847905
)


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

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def tri_not(x):
    out = np.full_like(
        x,
        UNKNOWN,
    )

    out[
        x == ABSENT
    ] = PRESENT

    out[
        x == PRESENT
    ] = ABSENT

    return out


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


def atomic_unique_count(atomic):
    return len({
        atomic[:, j].tobytes()
        for j in range(
            atomic.shape[1]
        )
    })


def build_tecs_library(
    atomic,
    valid,
    a4,
):
    """
    Atomic predicates +
    ALL minimal two-question B20 repair-certificate folds.

    A pair (p,q) is minimal for target t iff:
        rank(p) > B
        rank(q) > B
        rank(p,q) <= B

    Therefore no proper one-question subcertificate
    already repairs that target.
    """

    s0, s1 = a4.build_survival_masks(
        atomic
    )

    columns = [
        atomic[:, j].copy()
        for j in range(
            atomic.shape[1]
        )
    ]

    seen = {
        col.tobytes()
        for col in columns
    }

    unique_added = 0
    raw_certificate_count = 0

    targets_with_pair_certificate = 0

    pattern_count = {
        "00": 0,
        "01": 0,
        "10": 0,
        "11": 0,
    }

    valid_targets = np.flatnonzero(
        valid
    )

    for t in valid_targets:

        answers = atomic[
            t
        ]

        assert np.all(
            (
                answers == ABSENT
            )
            |
            (
                answers == PRESENT
            )
        )

        single_masks = []
        single_ranks = []

        for p in range(
            atomic.shape[1]
        ):
            a = int(
                answers[p]
            )

            m = (
                s1[p]
                if a == PRESENT
                else s0[p]
            )

            single_masks.append(
                m
            )

            single_ranks.append(
                int(
                    m[
                        :t + 1
                    ].sum()
                )
            )

        target_has_pair = False

        for p in range(
            atomic.shape[1]
        ):

            if single_ranks[p] <= B:
                continue

            ap = int(
                answers[p]
            )

            for q in range(
                p + 1,
                atomic.shape[1],
            ):

                if single_ranks[q] <= B:
                    continue

                aq = int(
                    answers[q]
                )

                survivor = (
                    single_masks[p]
                    &
                    single_masks[q]
                )

                rank = int(
                    survivor[
                        :t + 1
                    ].sum()
                )

                if rank > B:
                    continue

                raw_certificate_count += 1
                target_has_pair = True

                pattern = (
                    f"{ap}{aq}"
                )

                pattern_count[
                    pattern
                ] += 1

                lp = (
                    atomic[:, p]
                    if ap == PRESENT
                    else tri_not(
                        atomic[:, p]
                    )
                )

                lq = (
                    atomic[:, q]
                    if aq == PRESENT
                    else tri_not(
                        atomic[:, q]
                    )
                )

                fold = a4.tri_and(
                    lp,
                    lq,
                ).astype(
                    np.int8,
                    copy=False,
                )

                assert (
                    int(
                        fold[t]
                    )
                    ==
                    PRESENT
                )

                fold_survivor = (
                    (fold == PRESENT)
                    |
                    (fold == UNKNOWN)
                )

                assert np.array_equal(
                    fold_survivor,
                    survivor,
                )

                key = fold.tobytes()

                if key in seen:
                    continue

                seen.add(
                    key
                )

                columns.append(
                    fold.copy()
                )

                unique_added += 1

        if target_has_pair:
            targets_with_pair_certificate += 1

    library = np.stack(
        columns,
        axis=1,
    )

    return {
        "library":
            library,

        "raw_certificate_count":
            int(
                raw_certificate_count
            ),

        "unique_added":
            int(
                unique_added
            ),

        "targets_with_pair_certificate":
            int(
                targets_with_pair_certificate
            ),

        "pattern_count":
            pattern_count,
    }


def exact_shared_vector(
    facts,
    valid,
    a4,
    gap,
):
    (
        policy,
        s0,
        s1,
    ) = a4.choose_bcct(
        facts,
        valid,
    )

    return gap.eval_compound_policy_all(
        policy,
        facts,
        s0,
        s1,
        valid,
        a4,
    )


def main():

    t0 = time.perf_counter()

    print("=" * 150)
    print("TREE-EMBEDDABLE CERTIFICATE SYNTHESIS v0")
    print("=" * 150)

    print("FIT only.")
    print("P1 is NOT opened.")
    print("P2 is NOT opened.")
    print("Frozen RCPS-v1 is NOT modified.")

    print(
        "Candidate predicates are synthesized ONLY "
        "from minimal atomic B20 repair certificates."
    )

    print(
        "All alternative minimal certificates are retained "
        "after exact semantic deduplication."
    )

    print(
        "Root / child0 / child1 are optimized jointly."
    )

    print(
        "No top-k, no threshold, no route-first staging, "
        "no iterative round count."
    )

    # ========================================================
    # Integrity.
    # ========================================================

    assert (
        sha256_file(A2_ART)
        ==
        A2_SHA
    )

    assert (
        sha256_file(GAP_ART)
        ==
        GAP_SHA
    )

    assert (
        sha256_file(IRCF_ART)
        ==
        IRCF_SHA
    )

    a2 = np.load(
        A2_ART,
        allow_pickle=True,
    )

    gap_art = np.load(
        GAP_ART,
        allow_pickle=True,
    )

    ircf = np.load(
        IRCF_ART,
        allow_pickle=True,
    )

    ast = np.load(
        AST_CACHE,
        allow_pickle=True,
    )

    a2mod = load_module(
        A2_SCRIPT,
        "tecs_a2",
    )

    a4mod = load_module(
        A4_SCRIPT,
        "tecs_a4",
    )

    gapmod = load_module(
        GAP_SCRIPT,
        "tecs_gap",
    )

    candidates = a2[
        "candidates"
    ].astype(
        np.int32
    )

    candidate_valid = a2[
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

    n = len(
        candidates
    )

    assert n == 568

    # ========================================================
    # Frozen baselines.
    # ========================================================

    O_atomic = gap_art[
        "atomic_individual_count"
    ].astype(
        np.int16
    )

    S_atomic = gap_art[
        "atomic_exact_shared_count"
    ].astype(
        np.int16
    )

    O_rcps = gap_art[
        "compound_individual_count"
    ].astype(
        np.int16
    )

    S_rcps = gap_art[
        "compound_exact_shared_count"
    ].astype(
        np.int16
    )

    rcps_unique = gap_art[
        "unique_compound_predicates"
    ].astype(
        np.int16
    )

    S_ircf = ircf[
        "iterative_S"
    ].astype(
        np.int16
    )

    CCG_ircf = ircf[
        "compatibility_closure_gain"
    ].astype(
        np.int16
    )

    assert int(
        S_ircf.sum()
    ) == IRCF_SHARED

    assert int(
        CCG_ircf.sum()
    ) == IRCF_CCG

    assert int(
        S_rcps.sum()
    ) == RCPS_SHARED

    # ========================================================
    # Outputs.
    # ========================================================

    O_tecs = np.zeros(
        n,
        dtype=np.int16,
    )

    S_tecs = np.zeros(
        n,
        dtype=np.int16,
    )

    GBS_tecs = np.zeros(
        n,
        dtype=np.int16,
    )

    atomic_unique = np.zeros(
        n,
        dtype=np.int16,
    )

    unique_added = np.zeros(
        n,
        dtype=np.int32,
    )

    raw_certificates = np.zeros(
        n,
        dtype=np.int32,
    )

    targets_with_pair = np.zeros(
        n,
        dtype=np.int16,
    )

    final_library_size = np.zeros(
        n,
        dtype=np.int32,
    )

    patterns = {
        "00": 0,
        "01": 0,
        "10": 0,
        "11": 0,
    }

    atomic_parity_fail = 0
    monotonicity_fail = 0

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

        valid = candidate_valid[
            qi
        ]

        atomic[
            valid
        ] = presence[
            valid
        ]

        atomic_unique[
            qi
        ] = atomic_unique_count(
            atomic
        )

        # ----------------------------------------------------
        # Atomic parity.
        # ----------------------------------------------------

        ind_atomic = gapmod.individual_b20_vector(
            atomic,
            valid,
            a4mod,
        )

        atomic_policy = a2mod.choose_bcct(
            atomic,
            valid,
        )

        shared_atomic = gapmod.eval_atomic_policy_all(
            atomic_policy,
            atomic,
            valid,
            a2mod,
        )

        if (
            int(
                ind_atomic.sum()
            )
            !=
            int(
                O_atomic[qi]
            )
            or
            int(
                shared_atomic.sum()
            )
            !=
            int(
                S_atomic[qi]
            )
        ):
            atomic_parity_fail += 1
            continue

        # ----------------------------------------------------
        # Synthesize all minimal certificates.
        # ----------------------------------------------------

        built = build_tecs_library(
            atomic,
            valid,
            a4mod,
        )

        library = built[
            "library"
        ]

        unique_added[
            qi
        ] = built[
            "unique_added"
        ]

        raw_certificates[
            qi
        ] = built[
            "raw_certificate_count"
        ]

        targets_with_pair[
            qi
        ] = built[
            "targets_with_pair_certificate"
        ]

        final_library_size[
            qi
        ] = int(
            library.shape[1]
        )

        for k, v in built[
            "pattern_count"
        ].items():
            patterns[k] += int(v)

        # ----------------------------------------------------
        # Joint root/children optimization.
        # ----------------------------------------------------

        ind_tecs = gapmod.individual_b20_vector(
            library,
            valid,
            a4mod,
        )

        shared_tecs = exact_shared_vector(
            library,
            valid,
            a4mod,
            gapmod,
        )

        gbs_tecs = gapmod.eval_compound_gbs_all(
            library,
            valid,
            a2mod,
        )

        O_tecs[
            qi
        ] = int(
            ind_tecs.sum()
        )

        S_tecs[
            qi
        ] = int(
            shared_tecs.sum()
        )

        GBS_tecs[
            qi
        ] = int(
            gbs_tecs.sum()
        )

        if (
            O_tecs[qi]
            <
            O_atomic[qi]
            or
            S_tecs[qi]
            <
            S_atomic[qi]
        ):
            monotonicity_fail += 1

        if (
            (qi + 1) % 25 == 0
            or
            qi == n - 1
        ):
            print(
                f"[{qi+1:>3}/{n}]",
                "O=",
                int(
                    O_tecs[
                        :qi+1
                    ].sum()
                ),
                "S=",
                int(
                    S_tecs[
                        :qi+1
                    ].sum()
                ),
                "added=",
                int(
                    unique_added[
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

    # ========================================================
    # Invariants.
    # ========================================================

    print()
    print("=" * 150)
    print("INVARIANTS")
    print("=" * 150)

    print(
        "atomic O/S parity failures =",
        atomic_parity_fail,
    )

    print(
        "atomic -> TECS monotonicity failures =",
        monotonicity_fail,
    )

    assert atomic_parity_fail == 0
    assert monotonicity_fail == 0

    # ========================================================
    # Metrics.
    # ========================================================

    Gamma_atomic = (
        O_atomic
        -
        S_atomic
    ).astype(
        np.int16
    )

    Gamma_tecs = (
        O_tecs
        -
        S_tecs
    ).astype(
        np.int16
    )

    Gamma_rcps = (
        O_rcps
        -
        S_rcps
    ).astype(
        np.int16
    )

    dO = (
        O_tecs
        -
        O_atomic
    ).astype(
        np.int16
    )

    dS = (
        S_tecs
        -
        S_atomic
    ).astype(
        np.int16
    )

    CCG = (
        dS
        -
        dO
    ).astype(
        np.int16
    )

    assert np.array_equal(
        CCG,
        Gamma_atomic
        -
        Gamma_tecs,
    )

    atomic_shared_total = int(
        S_atomic.sum()
    )

    tecs_shared_total = int(
        S_tecs.sum()
    )

    rcps_shared_total = int(
        S_rcps.sum()
    )

    tecs_gain = (
        tecs_shared_total
        -
        atomic_shared_total
    )

    rcps_gain = (
        rcps_shared_total
        -
        atomic_shared_total
    )

    tecs_added_total = int(
        unique_added.sum()
    )

    rcps_added = (
        rcps_unique.astype(
            np.int64
        )
        -
        atomic_unique.astype(
            np.int64
        )
    )

    rcps_added[
        rcps_added < 0
    ] = 0

    rcps_added_total = int(
        rcps_added.sum()
    )

    tecs_eff = (
        float(tecs_gain)
        /
        float(tecs_added_total)
        if tecs_added_total > 0
        else 0.0
    )

    rcps_eff = (
        float(rcps_gain)
        /
        float(rcps_added_total)
    )

    assert abs(
        rcps_eff
        -
        RCPS_GAIN_PER_ADDED
    ) < 1e-12

    ccg_total = int(
        CCG.sum()
    )

    print()
    print("=" * 150)
    print("TECS RESULT")
    print("=" * 150)

    print(
        "Atomic O/S/Gamma =",
        stats(O_atomic),
        stats(S_atomic),
        stats(Gamma_atomic),
    )

    print(
        "TECS O/S/Gamma =",
        stats(O_tecs),
        stats(S_tecs),
        stats(Gamma_tecs),
    )

    print(
        "RCPS O/S/Gamma =",
        stats(O_rcps),
        stats(S_rcps),
        stats(Gamma_rcps),
    )

    print()
    print(
        "shared totals atomic / IRCF / TECS / RCPS =",
        atomic_shared_total,
        IRCF_SHARED,
        tecs_shared_total,
        rcps_shared_total,
    )

    print(
        "shared gains TECS / RCPS =",
        tecs_gain,
        rcps_gain,
    )

    print()
    print(
        "sum dO TECS =",
        int(
            dO.sum()
        ),
    )

    print(
        "sum dS TECS =",
        int(
            dS.sum()
        ),
    )

    print(
        "Compatibility Closure Gain sum(dS-dO) =",
        ccg_total,
    )

    print(
        "positive-CCG queries =",
        int(
            np.sum(
                CCG > 0
            )
        ),
        "/",
        n,
    )

    print(
        "pure compatibility queries dO=0,dS>0 =",
        int(
            np.sum(
                (dO == 0)
                &
                (dS > 0)
            )
        ),
        "/",
        n,
    )

    print(
        "gap reduced queries =",
        int(
            np.sum(
                Gamma_tecs
                <
                Gamma_atomic
            )
        ),
        "/",
        n,
    )

    print(
        "gap eliminated queries =",
        int(
            np.sum(
                (Gamma_atomic > 0)
                &
                (Gamma_tecs == 0)
            )
        ),
        "/",
        n,
    )

    print()
    print("=" * 150)
    print("CERTIFICATE SYNTHESIS STRUCTURE")
    print("=" * 150)

    print(
        "raw minimal pair certificates total =",
        int(
            raw_certificates.sum()
        ),
    )

    print(
        "semantic unique certificate predicates added =",
        tecs_added_total,
    )

    print(
        "mean added predicates/query =",
        float(
            unique_added.mean()
        ),
    )

    print(
        "median added predicates/query =",
        float(
            np.median(
                unique_added
            )
        ),
    )

    print(
        "mean targets with pair certificates/query =",
        float(
            targets_with_pair.mean()
        ),
    )

    print(
        "certificate answer patterns =",
        patterns,
    )

    print()
    print("=" * 150)
    print("PREDICATE EFFICIENCY")
    print("=" * 150)

    print(
        "TECS added predicates total =",
        tecs_added_total,
    )

    print(
        "RCPS added predicates total =",
        rcps_added_total,
    )

    print(
        "TECS library fraction vs RCPS =",
        (
            float(tecs_added_total)
            /
            float(rcps_added_total)
            if rcps_added_total > 0
            else 0.0
        ),
    )

    print(
        "TECS shared gain per added predicate =",
        tecs_eff,
    )

    print(
        "RCPS shared gain per added predicate =",
        rcps_eff,
    )

    print()
    print(
        "TECS GBS coverage =",
        stats(
            GBS_tecs
        ),
    )

    print(
        "sum exact-minus-GBS TECS =",
        int(
            (
                S_tecs
                -
                GBS_tecs
            ).sum()
        ),
    )

    # ========================================================
    # Pre-registered decision.
    # ========================================================

    strong_capacity = (
        tecs_shared_total
        >=
        RCPS_SHARED
    )

    smaller_than_rcps = (
        tecs_added_total
        <
        rcps_added_total
    )

    gate_shared = (
        tecs_shared_total
        >
        IRCF_SHARED
    )

    gate_ccg = (
        ccg_total
        >
        IRCF_CCG
    )

    gate_eff = (
        tecs_eff
        >
        RCPS_GAIN_PER_ADDED
    )

    strong_promote = (
        strong_capacity
        and
        smaller_than_rcps
    )

    medium_promote = (
        gate_shared
        and
        gate_ccg
        and
        gate_eff
    )

    print()
    print("=" * 150)
    print("V2 PRE-REGISTERED DECISION")
    print("=" * 150)

    print(
        "Strong Gate A: S_TECS >= RCPS (43192) =",
        strong_capacity,
    )

    print(
        "Strong Gate B: TECS uses fewer added predicates =",
        smaller_than_rcps,
    )

    print(
        "Medium Gate 1: S_TECS > IRCF (41230) =",
        gate_shared,
    )

    print(
        "Medium Gate 2: CCG_TECS > IRCF CCG (363) =",
        gate_ccg,
    )

    print(
        "Medium Gate 3: gain/predicate > RCPS =",
        gate_eff,
    )

    print(
        "STRONG_PROMOTE_TECS =",
        strong_promote,
    )

    print(
        "MEDIUM_PROMOTE_TECS =",
        medium_promote,
    )

    if strong_promote:

        diagnosis = (
            "STRONG POSITIVE. Certificate-complete synthesis "
            "matches/exceeds full RCPS shared capacity while "
            "using fewer synthesized predicates. Promote "
            "tree-embeddable repair-certificate synthesis as "
            "the primary v2 mechanism; next replace exhaustive "
            "certificate materialization with a specialized "
            "joint pricing / branch-and-bound solver."
        )

    elif medium_promote:

        diagnosis = (
            "POSITIVE V2 SIGNAL. Joint optimization over all "
            "minimal repair-certificate predicates beats IRCF "
            "and is more predicate-efficient than blind RCPS. "
            "Preserve TECS as the v2 synthesis object and move "
            "to a specialized joint solver."
        )

    else:

        diagnosis = (
            "CERTIFICATE-COMPLETE FOLD SYNTHESIS DOES NOT "
            "PASS THE PRE-REGISTERED V2 PROMOTION GATES. "
            "Do not tune certificate thresholds or add larger "
            "certificate grammar post hoc. Preserve the Shared "
            "Repair Gap and joint-routing result, and move to "
            "direct prefix-blocker optimization rather than "
            "certificate folding."
        )

    print(
        "DIAGNOSIS =",
        diagnosis,
    )

    # ========================================================
    # Save.
    # ========================================================

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "tree_embeddable_certificate_synthesis_v0"
        ),

        atomic_O=
            O_atomic,

        atomic_S=
            S_atomic,

        tecs_O=
            O_tecs,

        tecs_S=
            S_tecs,

        tecs_GBS=
            GBS_tecs,

        tecs_Gamma=
            Gamma_tecs,

        rcps_O=
            O_rcps,

        rcps_S=
            S_rcps,

        dO=
            dO,

        dS=
            dS,

        compatibility_closure_gain=
            CCG,

        raw_minimal_certificate_count=
            raw_certificates,

        unique_certificate_predicates_added=
            unique_added,

        targets_with_pair_certificate=
            targets_with_pair,

        final_library_size=
            final_library_size,

        atomic_unique_predicates=
            atomic_unique,

        rcps_unique_predicates=
            rcps_unique,
    )

    result = {
        "name":
            "Tree-Embeddable Certificate Synthesis v0",

        "shared": {
            "atomic":
                atomic_shared_total,

            "ircf":
                IRCF_SHARED,

            "tecs":
                tecs_shared_total,

            "rcps":
                rcps_shared_total,
        },

        "compatibility": {
            "sum_dO":
                int(
                    dO.sum()
                ),

            "sum_dS":
                int(
                    dS.sum()
                ),

            "sum_CCG":
                ccg_total,
        },

        "synthesis": {
            "raw_minimal_pair_certificates":
                int(
                    raw_certificates.sum()
                ),

            "unique_added_predicates":
                tecs_added_total,

            "patterns":
                patterns,
        },

        "efficiency": {
            "tecs_gain":
                tecs_gain,

            "rcps_gain":
                rcps_gain,

            "tecs_gain_per_added":
                tecs_eff,

            "rcps_gain_per_added":
                rcps_eff,
        },

        "gates": {
            "strong_capacity":
                bool(
                    strong_capacity
                ),

            "smaller_than_rcps":
                bool(
                    smaller_than_rcps
                ),

            "shared_gt_ircf":
                bool(
                    gate_shared
                ),

            "CCG_gt_ircf":
                bool(
                    gate_ccg
                ),

            "efficiency_gt_rcps":
                bool(
                    gate_eff
                ),

            "strong_promote":
                bool(
                    strong_promote
                ),

            "medium_promote":
                bool(
                    medium_promote
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
            "Frozen RCPS-v1 unchanged.",
            "B=20.",
            "Shared tree depth=2.",
            "Atomic family is frozen 44 AST predicates.",
            "Every AST-valid Local80 hypothesis may contribute minimal atomic repair certificates.",
            "A two-question certificate is included only when neither one-question subcertificate reaches B20 but their conjunction does.",
            "All alternative minimal certificates are retained.",
            "Each certificate is folded into a signed conjunction matching the target answer pattern.",
            "Predicates are exact-semantic deduplicated.",
            "Root and both children are jointly optimized under frozen BCCT.",
            "No route-first staging.",
            "No top-k.",
            "No score threshold.",
            "No tuned iteration count.",
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
