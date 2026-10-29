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

A2_ART = (
    ROOT
    / "fhex_v1/results/"
      "bcr_a2_exact_program_fact_bcct.npz"
)

TECS_ART = (
    ROOT
    / "fhex_v1/results/"
      "tree_embeddable_certificate_synthesis_v0.npz"
)

RBJS_ART = (
    ROOT
    / "fhex_v1/results/"
      "repairability_bounded_joint_search_v0.npz"
)

RUNTIME_JSON = (
    ROOT
    / "fhex_v1/results/"
      "tecs_rbjs_runtime_decomposition_v0.json"
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

TECS_SCRIPT = (
    ROOT
    / "fhex_v1/scripts/"
      "audit_tree_embeddable_certificate_synthesis_v0.py"
)

RBJS_SCRIPT = (
    ROOT
    / "fhex_v1/scripts/"
      "audit_repairability_bounded_joint_search_v0.py"
)

OUT_NPZ = (
    ROOT
    / "fhex_v1/results/"
      "answer_pattern_quotient_compiler_v0.npz"
)

OUT_JSON = (
    ROOT
    / "fhex_v1/results/"
      "answer_pattern_quotient_compiler_v0.json"
)


TECS_SHA = (
    "eea14a27eae6f0aee02a6f74caed4294"
    "c2227977a0fb165b26957ced143c7729"
)

RBJS_SHA = (
    "94c83912962fc57adab96bcc94fb4161d"
    "fc79dcf84544fc41b5cbc84293f02a2"
)


E = 80
F = 44
B = 20

UNKNOWN = -1
ABSENT = 0
PRESENT = 1

OLD_RBJS_PIPELINE_SECONDS = (
    165.20258580427617
)

BUILD_SPEEDUP_GATE = 3.0
PIPELINE_SPEEDUP_GATE = 1.4


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            x = f.read(
                1 << 20
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

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            path
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def stats(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "sum":
            float(
                x.sum()
            ),

        "mean":
            float(
                x.mean()
            ),

        "median":
            float(
                np.median(
                    x
                )
            ),

        "p90":
            float(
                np.quantile(
                    x,
                    0.90,
                )
            ),

        "min":
            float(
                x.min()
            ),

        "max":
            float(
                x.max()
            ),
    }


# ============================================================
# Frozen signed-pair universe:
#
#   (p,q,a,b)
#   p < q
#   a,b in {0,1}
#
# Size = 4*C(44,2) = 3784.
# ============================================================

PAIR_P = []
PAIR_Q = []
PAIR_A = []
PAIR_B = []

for p in range(F):

    for q in range(
        p + 1,
        F,
    ):

        for a in (
            ABSENT,
            PRESENT,
        ):

            for b in (
                ABSENT,
                PRESENT,
            ):

                PAIR_P.append(
                    p
                )

                PAIR_Q.append(
                    q
                )

                PAIR_A.append(
                    a
                )

                PAIR_B.append(
                    b
                )


PAIR_P = np.asarray(
    PAIR_P,
    dtype=np.int16,
)

PAIR_Q = np.asarray(
    PAIR_Q,
    dtype=np.int16,
)

PAIR_A = np.asarray(
    PAIR_A,
    dtype=np.int8,
)

PAIR_B = np.asarray(
    PAIR_B,
    dtype=np.int8,
)

K = len(
    PAIR_P
)

assert (
    K
    ==
    4
    *
    (
        F
        *
        (
            F - 1
        )
        //
        2
    )
)

assert (
    K
    ==
    3784
)


def apqc_build(
    atomic,
    valid,
):
    """
    Exact TECS compiler by answer-pattern quotienting.

    Instead of enumerating:
        target -> (p,q)

    enumerate only:
        (p,q,a,b)

    A signed-pair class is activated iff at least one valid
    target with answers (a,b) has:

        single(p) > B
        single(q) > B
        pair(p,q) <= B

    Ordering is reconstructed EXACTLY as the frozen TECS
    target-wise implementation:

        first witness target,
        then p,
        then q.

    Query-local semantic deduplication is also exact.
    """

    assert atomic.shape == (
        E,
        F,
    )

    # --------------------------------------------------------
    # Literal positive-compatible survivor masks.
    #
    # [F,2,E]
    # --------------------------------------------------------

    literal_survive = np.empty(
        (
            F,
            2,
            E,
        ),
        dtype=bool,
    )

    literal_survive[
        :,
        ABSENT,
        :,
    ] = (
        (
            atomic.T
            ==
            ABSENT
        )
        |
        (
            atomic.T
            ==
            UNKNOWN
        )
    )

    literal_survive[
        :,
        PRESENT,
        :,
    ] = (
        (
            atomic.T
            ==
            PRESENT
        )
        |
        (
            atomic.T
            ==
            UNKNOWN
        )
    )

    # [F,2,E]
    literal_prefix = np.cumsum(
        literal_survive,
        axis=2,
        dtype=np.int16,
    )

    # --------------------------------------------------------
    # All 3784 signed-pair survivor masks.
    #
    # [K,E]
    # --------------------------------------------------------

    pair_survive = (
        literal_survive[
            PAIR_P,
            PAIR_A,
            :,
        ]
        &
        literal_survive[
            PAIR_Q,
            PAIR_B,
            :,
        ]
    )

    pair_prefix = np.cumsum(
        pair_survive,
        axis=1,
        dtype=np.int16,
    )

    # --------------------------------------------------------
    # Valid-target answer-pattern match.
    #
    # For AST-valid candidates, every atomic answer is 0/1.
    #
    # [K,E]
    # --------------------------------------------------------

    ans_p = atomic[
        :,
        PAIR_P,
    ].T

    ans_q = atomic[
        :,
        PAIR_Q,
    ].T

    target_match = (
        valid[
            None,
            :
        ]
        &
        (
            ans_p
            ==
            PAIR_A[
                :,
                None
            ]
        )
        &
        (
            ans_q
            ==
            PAIR_B[
                :,
                None
            ]
        )
    )

    # --------------------------------------------------------
    # Single-question ranks for every signed pair's two
    # literals at every possible target.
    #
    # [K,E]
    # --------------------------------------------------------

    rank_p = literal_prefix[
        PAIR_P,
        PAIR_A,
        :,
    ]

    rank_q = literal_prefix[
        PAIR_Q,
        PAIR_B,
        :,
    ]

    # --------------------------------------------------------
    # Minimal-certificate witness incidence.
    #
    # This is EXACTLY the frozen TECS condition.
    #
    # [K,E]
    # --------------------------------------------------------

    certificate = (
        target_match
        &
        (
            rank_p
            >
            B
        )
        &
        (
            rank_q
            >
            B
        )
        &
        (
            pair_prefix
            <=
            B
        )
    )

    raw_certificate_count = int(
        certificate.sum()
    )

    active = np.any(
        certificate,
        axis=1,
    )

    active_ids = np.flatnonzero(
        active
    )

    # First witness target for frozen target-wise ordering.
    first_target = np.argmax(
        certificate[
            active_ids
        ],
        axis=1,
    ).astype(
        np.int16
    )

    # Old TECS loop ordering:
    #
    #   for target ascending:
    #       for p ascending:
    #           for q ascending:
    #
    order = np.lexsort(
        (
            PAIR_Q[
                active_ids
            ],
            PAIR_P[
                active_ids
            ],
            first_target,
        )
    )

    ordered_ids = active_ids[
        order
    ]

    # --------------------------------------------------------
    # Semantic quotient.
    #
    # For every AST-valid candidate, a folded signed
    # conjunction is PRESENT iff both signed literals match;
    # otherwise ABSENT.
    #
    # Every AST-invalid candidate is UNKNOWN for every such
    # fold.
    #
    # Therefore, with fixed valid mask, predicate semantics
    # are uniquely determined by the PRESENT set.
    # --------------------------------------------------------

    columns = [
        atomic[
            :,
            j
        ].copy()
        for j in range(F)
    ]

    seen = {
        col.tobytes()
        for col in columns
    }

    unique_added = 0

    for k in ordered_ids:

        present = target_match[
            k
        ]

        col = np.where(
            valid,
            present.astype(
                np.int8
            ),
            UNKNOWN,
        ).astype(
            np.int8,
            copy=False,
        )

        key = col.tobytes()

        if key in seen:
            continue

        seen.add(
            key
        )

        columns.append(
            col
        )

        unique_added += 1

    library = np.stack(
        columns,
        axis=1,
    )

    # Number of targets that have at least one minimal pair.
    target_has_pair = np.any(
        certificate,
        axis=0,
    )

    patterns = {
        "00":
            int(
                certificate[
                    (
                        PAIR_A == 0
                    )
                    &
                    (
                        PAIR_B == 0
                    )
                ].sum()
            ),

        "01":
            int(
                certificate[
                    (
                        PAIR_A == 0
                    )
                    &
                    (
                        PAIR_B == 1
                    )
                ].sum()
            ),

        "10":
            int(
                certificate[
                    (
                        PAIR_A == 1
                    )
                    &
                    (
                        PAIR_B == 0
                    )
                ].sum()
            ),

        "11":
            int(
                certificate[
                    (
                        PAIR_A == 1
                    )
                    &
                    (
                        PAIR_B == 1
                    )
                ].sum()
            ),
    }

    return {
        "library":
            library,

        "raw_certificate_count":
            raw_certificate_count,

        "unique_added":
            int(
                unique_added
            ),

        "targets_with_pair_certificate":
            int(
                target_has_pair.sum()
            ),

        "pattern_count":
            patterns,

        "active_answer_pattern_classes":
            int(
                active.sum()
            ),
    }


def main():

    print(
        "=" * 150
    )

    print(
        "ANSWER-PATTERN QUOTIENT CERTIFICATE COMPILER v0"
    )

    print(
        "=" * 150
    )

    print(
        "FIT only. No P1/P2."
    )

    print(
        "Frozen TECS predicate family and RBJS objective "
        "are unchanged."
    )

    print(
        "Only certificate construction is replaced."
    )

    print(
        "No heuristic, no threshold, no top-k."
    )

    print(
        "Exact library order and semantics must match "
        "frozen TECS."
    )

    assert (
        sha256_file(
            TECS_ART
        )
        ==
        TECS_SHA
    )

    assert (
        sha256_file(
            RBJS_ART
        )
        ==
        RBJS_SHA
    )

    a4 = load_module(
        A4_SCRIPT,
        "apqc_a4",
    )

    tecsmod = load_module(
        TECS_SCRIPT,
        "apqc_tecs",
    )

    rbjs = load_module(
        RBJS_SCRIPT,
        "apqc_rbjs",
    )

    rbjs.a4 = a4

    a2 = np.load(
        A2_ART,
        allow_pickle=True,
    )

    tecs_art = np.load(
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

    allowed = np.flatnonzero(
        names
        !=
        "max_ast_depth"
    )

    assert len(
        allowed
    ) == F

    frozen_S = tecs_art[
        "tecs_S"
    ].astype(
        np.int16
    )

    n = len(
        candidates
    )

    assert n == 568

    old_build_time = np.zeros(
        n,
        dtype=np.float64,
    )

    new_build_time = np.zeros(
        n,
        dtype=np.float64,
    )

    rbjs_time = np.zeros(
        n,
        dtype=np.float64,
    )

    old_lib_size = np.zeros(
        n,
        dtype=np.int16,
    )

    new_lib_size = np.zeros(
        n,
        dtype=np.int16,
    )

    active_classes = np.zeros(
        n,
        dtype=np.int16,
    )

    raw_certificate_old = np.zeros(
        n,
        dtype=np.int32,
    )

    raw_certificate_new = np.zeros(
        n,
        dtype=np.int32,
    )

    unique_added_old = np.zeros(
        n,
        dtype=np.int16,
    )

    unique_added_new = np.zeros(
        n,
        dtype=np.int16,
    )

    library_mismatch = 0
    raw_count_mismatch = 0
    unique_count_mismatch = 0
    target_count_mismatch = 0
    pattern_mismatch = 0
    coverage_mismatch = 0
    policy_mismatch = 0

    wall0 = time.perf_counter()

    for qi in range(
        n
    ):

        atomic = np.full(
            (
                E,
                F,
            ),
            UNKNOWN,
            dtype=np.int8,
        )

        presence = (
            signature[
                candidates[
                    qi
                ]
            ][
                :,
                allowed
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

        # ====================================================
        # Frozen target-wise compiler.
        # ====================================================

        t0 = time.perf_counter()

        old = tecsmod.build_tecs_library(
            atomic,
            valid,
            a4,
        )

        old_build_time[
            qi
        ] = (
            time.perf_counter()
            -
            t0
        )

        old_library = old[
            "library"
        ]

        old_lib_size[
            qi
        ] = int(
            old_library.shape[
                1
            ]
        )

        raw_certificate_old[
            qi
        ] = int(
            old[
                "raw_certificate_count"
            ]
        )

        unique_added_old[
            qi
        ] = int(
            old[
                "unique_added"
            ]
        )

        # ====================================================
        # APQC compiler.
        # ====================================================

        t0 = time.perf_counter()

        new = apqc_build(
            atomic,
            valid,
        )

        new_build_time[
            qi
        ] = (
            time.perf_counter()
            -
            t0
        )

        new_library = new[
            "library"
        ]

        new_lib_size[
            qi
        ] = int(
            new_library.shape[
                1
            ]
        )

        active_classes[
            qi
        ] = int(
            new[
                "active_answer_pattern_classes"
            ]
        )

        raw_certificate_new[
            qi
        ] = int(
            new[
                "raw_certificate_count"
            ]
        )

        unique_added_new[
            qi
        ] = int(
            new[
                "unique_added"
            ]
        )

        # ====================================================
        # Strongest possible compiler parity:
        # exact shape, order and ternary values.
        # ====================================================

        if not np.array_equal(
            old_library,
            new_library,
        ):

            library_mismatch += 1

        if (
            old[
                "raw_certificate_count"
            ]
            !=
            new[
                "raw_certificate_count"
            ]
        ):

            raw_count_mismatch += 1

        if (
            old[
                "unique_added"
            ]
            !=
            new[
                "unique_added"
            ]
        ):

            unique_count_mismatch += 1

        if (
            old[
                "targets_with_pair_certificate"
            ]
            !=
            new[
                "targets_with_pair_certificate"
            ]
        ):

            target_count_mismatch += 1

        if (
            old[
                "pattern_count"
            ]
            !=
            new[
                "pattern_count"
            ]
        ):

            pattern_mismatch += 1

        # ====================================================
        # Frozen exhaustive policy for policy-id parity.
        # ====================================================

        (
            reference_policy,
            _,
            _,
        ) = a4.choose_bcct(
            old_library,
            valid,
        )

        # ====================================================
        # New compiler + frozen RBJS solver.
        # ====================================================

        t0 = time.perf_counter()

        solved = rbjs.bounded_choose_bcct(
            new_library,
            atomic,
            valid,
        )

        rbjs_time[
            qi
        ] = (
            time.perf_counter()
            -
            t0
        )

        new_policy = solved[
            "policy"
        ]

        if rbjs.policy_key(
            new_policy
        ) != rbjs.policy_key(
            reference_policy
        ):

            policy_mismatch += 1

        if (
            int(
                new_policy[
                    "coverage"
                ]
            )
            !=
            int(
                frozen_S[
                    qi
                ]
            )
        ):

            coverage_mismatch += 1

        if (
            (qi + 1) % 50 == 0
            or
            qi == n - 1
        ):

            print(
                f"[{qi+1:>3}/{n}]",
                "old_build=",
                round(
                    float(
                        old_build_time[
                            :qi+1
                        ].sum()
                    ),
                    2,
                ),
                "APQC=",
                round(
                    float(
                        new_build_time[
                            :qi+1
                        ].sum()
                    ),
                    2,
                ),
                "RBJS=",
                round(
                    float(
                        rbjs_time[
                            :qi+1
                        ].sum()
                    ),
                    2,
                ),
                "lib_mismatch=",
                library_mismatch,
                flush=True,
            )

    # ========================================================
    # Exactness.
    # ========================================================

    print()
    print(
        "=" * 150
    )

    print(
        "EXACTNESS INVARIANTS"
    )

    print(
        "=" * 150
    )

    print(
        "exact TECS library mismatches =",
        library_mismatch,
    )

    print(
        "raw certificate count mismatches =",
        raw_count_mismatch,
    )

    print(
        "unique predicate count mismatches =",
        unique_count_mismatch,
    )

    print(
        "target-with-pair count mismatches =",
        target_count_mismatch,
    )

    print(
        "answer-pattern count mismatches =",
        pattern_mismatch,
    )

    print(
        "exact policy mismatches =",
        policy_mismatch,
    )

    print(
        "coverage mismatches =",
        coverage_mismatch,
    )

    assert library_mismatch == 0
    assert raw_count_mismatch == 0
    assert unique_count_mismatch == 0
    assert target_count_mismatch == 0
    assert pattern_mismatch == 0
    assert policy_mismatch == 0
    assert coverage_mismatch == 0

    # ========================================================
    # Runtime.
    # ========================================================

    old_build_total = float(
        old_build_time.sum()
    )

    new_build_total = float(
        new_build_time.sum()
    )

    rbjs_total = float(
        rbjs_time.sum()
    )

    old_pipeline = (
        old_build_total
        +
        rbjs_total
    )

    new_pipeline = (
        new_build_total
        +
        rbjs_total
    )

    build_speedup = (
        old_build_total
        /
        new_build_total
    )

    direct_pipeline_speedup = (
        old_pipeline
        /
        new_pipeline
    )

    frozen_pipeline_speedup = (
        OLD_RBJS_PIPELINE_SECONDS
        /
        new_pipeline
    )

    build_fraction_new = (
        new_build_total
        /
        new_pipeline
    )

    print()
    print(
        "=" * 150
    )

    print(
        "APQC RUNTIME RESULT"
    )

    print(
        "=" * 150
    )

    print(
        "old target-wise TECS build seconds =",
        old_build_total,
    )

    print(
        "APQC build seconds =",
        new_build_total,
    )

    print(
        "RBJS seconds on exact APQC library =",
        rbjs_total,
    )

    print()

    print(
        "certificate-build speedup =",
        build_speedup,
    )

    print(
        "same-run old-build+RBJS seconds =",
        old_pipeline,
    )

    print(
        "APQC+RBJS seconds =",
        new_pipeline,
    )

    print(
        "same-run pipeline speedup =",
        direct_pipeline_speedup,
    )

    print(
        "speedup vs frozen 165.20s RBJS pipeline =",
        frozen_pipeline_speedup,
    )

    print(
        "APQC synthesis fraction of new pipeline =",
        build_fraction_new,
    )

    print()
    print(
        "old build/query =",
        stats(
            old_build_time
        ),
    )

    print(
        "APQC build/query =",
        stats(
            new_build_time
        ),
    )

    print(
        "RBJS/query =",
        stats(
            rbjs_time
        ),
    )

    print()
    print(
        "TECS library size/query =",
        stats(
            new_lib_size
        ),
    )

    print(
        "active answer-pattern classes/query =",
        stats(
            active_classes
        ),
    )

    # ========================================================
    # Frozen decision.
    # ========================================================

    gate_exact = (
        library_mismatch == 0
        and
        policy_mismatch == 0
        and
        coverage_mismatch == 0
    )

    gate_build = (
        build_speedup
        >=
        BUILD_SPEEDUP_GATE
    )

    gate_pipeline = (
        frozen_pipeline_speedup
        >=
        PIPELINE_SPEEDUP_GATE
    )

    promote = (
        gate_exact
        and
        gate_build
        and
        gate_pipeline
    )

    print()
    print(
        "=" * 150
    )

    print(
        "APQC PRE-REGISTERED DECISION"
    )

    print(
        "=" * 150
    )

    print(
        "Gate 1 exact library/policy parity =",
        gate_exact,
    )

    print(
        "Gate 2 certificate build speedup >= 3x =",
        gate_build,
    )

    print(
        "Gate 3 end-to-end speedup >= 1.4x =",
        gate_pipeline,
    )

    print(
        "PROMOTE_APQC =",
        promote,
    )

    if promote:

        diagnosis = (
            "STRONG POSITIVE. Answer-pattern quotienting "
            "reproduces the exact TECS predicate library and "
            "RBJS optimum while eliminating target-wise "
            "certificate materialization as the main runtime "
            "bottleneck. Promote APQC as the exact certificate "
            "compiler paired with RBJS. Generic column "
            "generation is not currently necessary."
        )

    elif gate_exact:

        diagnosis = (
            "APQC IS EXACT BUT DOES NOT PASS THE FROZEN "
            "RUNTIME GATES. Keep TECS/RBJS as the algorithmic "
            "result; do not tune APQC. Only then consider a "
            "repair-specific on-demand pricing formulation."
        )

    else:

        diagnosis = (
            "APQC EXACT PARITY FAILED. Do not use this "
            "compiler. Frozen TECS/RBJS remain unaffected."
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
            "answer_pattern_quotient_compiler_v0"
        ),

        old_build_time=
            old_build_time,

        apqc_build_time=
            new_build_time,

        rbjs_time=
            rbjs_time,

        library_size=
            new_lib_size,

        active_answer_pattern_classes=
            active_classes,

        raw_certificate_count=
            raw_certificate_new,

        unique_added=
            unique_added_new,
    )

    result = {
        "name":
            "Answer-Pattern Quotient Certificate Compiler v0",

        "exactness": {
            "library_mismatches":
                library_mismatch,

            "raw_count_mismatches":
                raw_count_mismatch,

            "unique_count_mismatches":
                unique_count_mismatch,

            "target_count_mismatches":
                target_count_mismatch,

            "pattern_count_mismatches":
                pattern_mismatch,

            "policy_mismatches":
                policy_mismatch,

            "coverage_mismatches":
                coverage_mismatch,
        },

        "runtime": {
            "old_build_seconds":
                old_build_total,

            "apqc_build_seconds":
                new_build_total,

            "rbjs_seconds":
                rbjs_total,

            "apqc_rbjs_pipeline_seconds":
                new_pipeline,

            "build_speedup":
                build_speedup,

            "same_run_pipeline_speedup":
                direct_pipeline_speedup,

            "frozen_pipeline_speedup":
                frozen_pipeline_speedup,

            "new_build_fraction":
                build_fraction_new,
        },

        "gates": {
            "exact":
                bool(
                    gate_exact
                ),

            "build_speedup_ge_3":
                bool(
                    gate_build
                ),

            "pipeline_speedup_ge_1_4":
                bool(
                    gate_pipeline
                ),

            "promote":
                bool(
                    promote
                ),
        },

        "diagnosis":
            diagnosis,

        "wall_seconds":
            float(
                time.perf_counter()
                -
                wall0
            ),

        "protocol": [
            "FIT only.",
            "No P1 access.",
            "No P2 access.",
            "Frozen TECS predicate family unchanged.",
            "Frozen RBJS objective and solver unchanged.",
            "Every minimal target-specific two-question certificate is quotiented by feature-pair answer pattern.",
            "The signed-pair universe has exactly 4*C(44,2)=3784 classes.",
            "A class is active iff at least one valid target witnesses the frozen TECS minimal-certificate condition.",
            "First-witness target plus feature ids reconstruct the exact frozen TECS insertion order.",
            "Query-local semantic deduplication is exact.",
            "No heuristic filtering.",
            "No top-k.",
            "No threshold tuning.",
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


if __name__ == "__main__":
    main()
