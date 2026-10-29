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

EVALDIR = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/evaluation_v1"
)

RETRIEVAL = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "retrieval_v3_trainonly_exactkernel/"
      "jters_p3_qwen3_top500_trainonly_exactkernel_v1.npz"
)

RETRIEVAL_MEMO = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "retrieval_v3_trainonly_exactkernel/"
      "p3_trainonly_exactkernel_freeze_v1.json"
)

RETRIEVAL_SCRIPT = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "retrieval_v3_trainonly_exactkernel/"
      "build_p3_trainonly_exactkernel_v1.py"
)

P3_MANIFEST = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2/"
      "jters_p3_manifest_v1.npz"
)

AST_CACHE = (
    ROOT
    / "fhex_v1/cache/python_train_ast_signature_v0.npz"
)

A2_SCRIPT = (
    ROOT
    / "fhex_v1/scripts/"
      "audit_bcr_a2_exact_program_fact_bcct.py"
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

RBJS_SCRIPT = (
    ROOT
    / "fhex_v1/frozen_v2/"
      "audit_repairability_bounded_joint_search_v0.py"
)

APQC_SCRIPT = (
    ROOT
    / "fhex_v1/frozen_v2/"
      "audit_answer_pattern_quotient_compiler_v0.py"
)

FROZEN_PROTOCOL = (
    ROOT
    / "fhex_v1/frozen_v2/"
      "jters_v2_frozen_protocol.json"
)

FREEZE_MANIFEST = (
    ROOT
    / "fhex_v1/frozen_v2/"
      "jters_v2_freeze_manifest.json"
)

OUT_NPZ = (
    EVALDIR
    / "jters_p3_frozen_evaluation_v1.npz"
)

OUT_JSON = (
    EVALDIR
    / "jters_p3_frozen_evaluation_v1.json"
)


EXPECTED = {
    "retrieval":
        "d119a1180e560b182e50f152ce5b8a403c193c38292e3dab50678000b687f323",

    "retrieval_memo":
        "784be3a499be5fd4bc94327473515645588dcd9c03a28e8beebe33eb63e9c385",

    "retrieval_script":
        "21c6134a7dfd89614a02ceff5af032f531dc08c81757f1a7f6e9a70536c16b7b",

    "p3_manifest":
        "07650ed49593985ce506a728597e83fd84f76c6ad78d82d4f731d122133db7d0",

    "ast_cache":
        "4316bf9327754a47a0ad97e868087cd8c4ced236800b11346a3d073f733f0e36",

    "a2":
        "1d27808e3f86cd8956e6e342a342efd0a8f9d78e35c7232e25483e1df57b5851",

    "a4":
        "580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9",

    "gap":
        "df2d8f33869ed623122f480bcd7a1d4f79a48a083a791b8ac43ac05746712e15",

    "tecs":
        "12c31549062575e68b4cd0daa57659ad4572d372c16d11cf366a8c5ba7221d98",

    "rbjs":
        "d6ca28ab382d7965a21544228df9c52fee37f2926b846748143b2bcf1ceebc21",

    "apqc":
        "a5d14eba3a6634c26defa13f4ba42561347d7d5c19b3449a3df889bc0cc208df",

    "protocol":
        "ce22ddf88da8a1c15727346d2679248d102eb11a65a54b641c99095394b95601",

    "freeze_manifest":
        "7ec59b91d1803688144b914cf1727054e1a85099255594acf32c0470bc5d77be",
}


E = 80
B = 20
ATOMIC_F = 44
LOCAL_START = 100
LOCAL_STOP = 180
TRAIN_ROWS = 412178
P3_N = 6000

UNKNOWN = -1


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            b = f.read(1 << 20)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def require_sha(name, path):

    actual = sha256(path)
    expected = EXPECTED[name]

    ok = actual == expected

    print(
        f"{name:<22}",
        "PASS" if ok else "FAIL",
    )

    if not ok:

        print(" expected =", expected)
        print(" actual   =", actual)

        raise RuntimeError(
            f"Frozen provenance mismatch: {name}"
        )


def load_module(name, path):

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            f"Cannot load {path}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def policy_tuple(policy):

    if policy is None:
        return None

    return (
        int(
            policy["first_feature"]
        ),
        int(
            policy["second_zero"]
        ),
        int(
            policy["second_one"]
        ),
        int(
            policy["coverage"]
        ),
        int(
            policy["gain"]
        ),
        int(
            policy["worst_leaf"]
        ),
    )


def build_atomic(
    candidate_ids,
    signature,
    ast_valid,
    allowed_ids,
):

    valid = ast_valid[
        candidate_ids
    ]

    atomic = np.full(
        (
            E,
            ATOMIC_F,
        ),
        UNKNOWN,
        dtype=np.int8,
    )

    present = (
        signature[
            candidate_ids
        ][:, allowed_ids]
        >
        0.0
    ).astype(
        np.int8
    )

    atomic[
        valid
    ] = present[
        valid
    ]

    return (
        atomic,
        valid,
    )


def main():

    print("=" * 160)
    print("P3 FROZEN JTE-RCS / TECS-RBJS-APQC PROSPECTIVE EVALUATION v1")
    print("=" * 160)

    print(
        "THIS IS THE FIRST P3 ALGORITHM OUTCOME EVALUATION."
    )

    print(
        "NO POST-P3 METHOD TUNING IS PERMITTED."
    )

    print()

    # ========================================================
    # 1. Frozen provenance.
    # ========================================================

    require_sha(
        "retrieval",
        RETRIEVAL,
    )

    require_sha(
        "retrieval_memo",
        RETRIEVAL_MEMO,
    )

    require_sha(
        "retrieval_script",
        RETRIEVAL_SCRIPT,
    )

    require_sha(
        "p3_manifest",
        P3_MANIFEST,
    )

    require_sha(
        "ast_cache",
        AST_CACHE,
    )

    require_sha(
        "a2",
        A2_SCRIPT,
    )

    require_sha(
        "a4",
        A4_SCRIPT,
    )

    require_sha(
        "gap",
        GAP_SCRIPT,
    )

    require_sha(
        "tecs",
        TECS_SCRIPT,
    )

    require_sha(
        "rbjs",
        RBJS_SCRIPT,
    )

    require_sha(
        "apqc",
        APQC_SCRIPT,
    )

    require_sha(
        "protocol",
        FROZEN_PROTOCOL,
    )

    require_sha(
        "freeze_manifest",
        FREEZE_MANIFEST,
    )

    # ========================================================
    # 2. Load frozen algorithm implementations.
    # ========================================================

    a2 = load_module(
        "p3_frozen_a2",
        A2_SCRIPT,
    )

    a4 = load_module(
        "p3_frozen_a4",
        A4_SCRIPT,
    )

    gap = load_module(
        "p3_frozen_gap",
        GAP_SCRIPT,
    )

    tecs = load_module(
        "p3_frozen_tecs",
        TECS_SCRIPT,
    )

    rbjs = load_module(
        "p3_frozen_rbjs",
        RBJS_SCRIPT,
    )

    apqc = load_module(
        "p3_frozen_apqc",
        APQC_SCRIPT,
    )

    # Frozen RBJS source expects the A4 module as a global.
    rbjs.a4 = a4

    assert a2.E80 == E
    assert a2.B == B

    assert a4.E80 == E
    assert a4.B == B

    assert gap.E == E
    assert gap.B == B

    assert tecs.B == B

    assert rbjs.E == E
    assert rbjs.B == B

    assert apqc.E == E
    assert apqc.F == ATOMIC_F
    assert apqc.B == B

    # ========================================================
    # 3. Load frozen P3 retrieval.
    # ========================================================

    with np.load(
        RETRIEVAL,
        allow_pickle=False,
    ) as z:

        source_rows = np.asarray(
            z[
                "source_row_indices"
            ],
            dtype=np.int64,
        )

        shard_ids = np.asarray(
            z[
                "shard_ids"
            ],
            dtype=np.int16,
        )

        gt = np.asarray(
            z[
                "ground_truth_code_indices"
            ],
            dtype=np.int32,
        )

        retrieval = np.asarray(
            z[
                "qwen3_indices"
            ],
            dtype=np.int32,
        )

        catalog_rows = int(
            np.asarray(
                z[
                    "catalog_rows"
                ]
            ).item()
        )

    assert source_rows.shape == (
        P3_N,
    )

    assert shard_ids.shape == (
        P3_N,
    )

    assert gt.shape == (
        P3_N,
    )

    assert retrieval.shape == (
        P3_N,
        500,
    )

    assert catalog_rows == TRAIN_ROWS

    assert np.array_equal(
        source_rows,
        gt.astype(
            np.int64
        ),
    )

    # ========================================================
    # 4. Define frozen failure/frontier sets.
    #
    # Base failure:
    #     GT not in Top100.
    #
    # Primary all-hypothesis evaluation:
    #     every base failure's ranks 101..180.
    #
    # Secondary true-GT evaluation:
    #     only failures whose actual GT lies in 101..180.
    # ========================================================

    match = (
        retrieval
        ==
        gt[
            :,
            None
        ]
    )

    assert np.all(
        match.sum(
            axis=1
        )
        <=
        1
    )

    gt_found = match.any(
        axis=1
    )

    gt_top500_pos = np.full(
        P3_N,
        500,
        dtype=np.int16,
    )

    found_ids = np.flatnonzero(
        gt_found
    )

    gt_top500_pos[
        found_ids
    ] = np.argmax(
        match[
            found_ids
        ],
        axis=1,
    ).astype(
        np.int16
    )

    base_failure = (
        gt_top500_pos
        >=
        LOCAL_START
    )

    failure_p3_pos = np.flatnonzero(
        base_failure
    )

    local80_true_gt = (
        gt_found
        &
        (
            gt_top500_pos
            >=
            LOCAL_START
        )
        &
        (
            gt_top500_pos
            <
            LOCAL_STOP
        )
    )

    local80_p3_pos = np.flatnonzero(
        local80_true_gt
    )

    n_fail = len(
        failure_p3_pos
    )

    n_local = len(
        local80_p3_pos
    )

    if n_fail == 0:

        raise RuntimeError(
            "P3 contains zero base failures."
        )

    if n_local == 0:

        raise RuntimeError(
            "P3 contains zero Local80-recoverable true-GT failures."
        )

    print()
    print("=" * 160)
    print("P3 FROZEN DENOMINATORS")
    print("=" * 160)

    print(
        "P3 queries =",
        P3_N,
    )

    print(
        "base Top100 failures =",
        n_fail,
    )

    print(
        "Local80-recoverable true-GT failures =",
        n_local,
    )

    print(
        "failure shard counts =",
        {
            int(s):
                int(
                    np.sum(
                        shard_ids[
                            failure_p3_pos
                        ]
                        ==
                        s
                    )
                )
            for s in range(5)
        },
    )

    # ========================================================
    # 5. Exact AST presence facts.
    # ========================================================

    with np.load(
        AST_CACHE,
        allow_pickle=False,
    ) as az:

        names_all = np.asarray(
            az[
                "feature_names"
            ]
        ).astype(
            str
        )

        signature = np.asarray(
            az[
                "signature"
            ],
            dtype=np.float32,
        )

        ast_valid = np.asarray(
            az[
                "valid"
            ],
            dtype=bool,
        )

    assert signature.shape == (
        TRAIN_ROWS,
        45,
    )

    assert ast_valid.shape == (
        TRAIN_ROWS,
    )

    allowed_ids = np.flatnonzero(
        names_all
        !=
        "max_ast_depth"
    )

    assert len(
        allowed_ids
    ) == ATOMIC_F

    # ========================================================
    # 6. Per-failure outputs.
    # ========================================================

    O_atomic = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    S_atomic = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    O_rcps = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    S_rcps = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    O_tecs = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    S_tecs = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    rcps_predicates = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    tecs_predicates = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    tecs_added = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    tecs_raw_certificates = np.zeros(
        n_fail,
        dtype=np.int32,
    )

    apqc_exact_library = np.zeros(
        n_fail,
        dtype=bool,
    )

    rbjs_policy_exact = np.zeros(
        n_fail,
        dtype=bool,
    )

    rbjs_coverage_mismatch = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    rbjs_pruned_roots = np.zeros(
        n_fail,
        dtype=np.int32,
    )

    rbjs_evaluated_roots = np.zeros(
        n_fail,
        dtype=np.int32,
    )

    rbjs_exact_child_solves = np.zeros(
        n_fail,
        dtype=np.int32,
    )

    valid_hypotheses = np.zeros(
        n_fail,
        dtype=np.int16,
    )

    candidate_ids_all = np.empty(
        (
            n_fail,
            E,
        ),
        dtype=np.int32,
    )

    # True-GT arrays are indexed over failure rows.
    true_gt_in_local80 = np.zeros(
        n_fail,
        dtype=bool,
    )

    gt_frontier_pos = np.full(
        n_fail,
        -1,
        dtype=np.int16,
    )

    rcps_gt_rank = np.full(
        n_fail,
        -1,
        dtype=np.int16,
    )

    tecs_gt_rank = np.full(
        n_fail,
        -1,
        dtype=np.int16,
    )

    rcps_gt_retained = np.zeros(
        n_fail,
        dtype=bool,
    )

    tecs_gt_retained = np.zeros(
        n_fail,
        dtype=bool,
    )

    start_time = time.perf_counter()

    # ========================================================
    # 7. FIRST P3 ALGORITHM OUTCOME EXECUTION.
    # ========================================================

    print()
    print("=" * 160)
    print("BEGIN FIRST P3 ALGORITHM OUTCOME EVALUATION")
    print("=" * 160)

    for j, p3i0 in enumerate(
        failure_p3_pos
    ):

        p3i = int(
            p3i0
        )

        candidates = retrieval[
            p3i,
            LOCAL_START:
            LOCAL_STOP
        ].astype(
            np.int32,
            copy=False,
        )

        assert candidates.shape == (
            E,
        )

        candidate_ids_all[
            j
        ] = candidates

        (
            atomic,
            valid,
        ) = build_atomic(
            candidates,
            signature,
            ast_valid,
            allowed_ids,
        )

        valid_hypotheses[
            j
        ] = int(
            valid.sum()
        )

        # ----------------------------------------------------
        # Atomic family.
        # ----------------------------------------------------

        ind_atomic = (
            gap.individual_b20_vector(
                atomic,
                valid,
                a4,
            )
        )

        p_atomic = a2.choose_bcct(
            atomic,
            valid,
        )

        shared_atomic = (
            gap.eval_atomic_policy_all(
                p_atomic,
                atomic,
                valid,
                a2,
            )
        )

        O_atomic[
            j
        ] = int(
            ind_atomic.sum()
        )

        S_atomic[
            j
        ] = int(
            shared_atomic.sum()
        )

        # ----------------------------------------------------
        # Frozen RCPS family:
        #
        # atomic + all AND + all OR, then exact semantic dedup.
        # ----------------------------------------------------

        (
            rcps_full,
            _,
            _,
            _,
        ) = a4.build_synth_library(
            atomic
        )

        (
            rcps_library,
            _,
        ) = a4.deduplicate_predicates(
            rcps_full,
            valid,
        )

        rcps_predicates[
            j
        ] = int(
            rcps_library.shape[
                1
            ]
        )

        ind_rcps = (
            gap.individual_b20_vector(
                rcps_library,
                valid,
                a4,
            )
        )

        (
            p_rcps,
            rcps_s0,
            rcps_s1,
        ) = a4.choose_bcct(
            rcps_library,
            valid,
        )

        shared_rcps = (
            gap.eval_compound_policy_all(
                p_rcps,
                rcps_library,
                rcps_s0,
                rcps_s1,
                valid,
                a4,
            )
        )

        O_rcps[
            j
        ] = int(
            ind_rcps.sum()
        )

        S_rcps[
            j
        ] = int(
            shared_rcps.sum()
        )

        # ----------------------------------------------------
        # Frozen TECS.
        # ----------------------------------------------------

        tecs_build = (
            tecs.build_tecs_library(
                atomic,
                valid,
                a4,
            )
        )

        tecs_library = (
            tecs_build[
                "library"
            ]
        )

        tecs_predicates[
            j
        ] = int(
            tecs_library.shape[
                1
            ]
        )

        tecs_added[
            j
        ] = int(
            tecs_build[
                "unique_added"
            ]
        )

        tecs_raw_certificates[
            j
        ] = int(
            tecs_build[
                "raw_certificate_count"
            ]
        )

        ind_tecs = (
            gap.individual_b20_vector(
                tecs_library,
                valid,
                a4,
            )
        )

        (
            p_tecs,
            tecs_s0,
            tecs_s1,
        ) = a4.choose_bcct(
            tecs_library,
            valid,
        )

        shared_tecs = (
            gap.eval_compound_policy_all(
                p_tecs,
                tecs_library,
                tecs_s0,
                tecs_s1,
                valid,
                a4,
            )
        )

        O_tecs[
            j
        ] = int(
            ind_tecs.sum()
        )

        S_tecs[
            j
        ] = int(
            shared_tecs.sum()
        )

        # ----------------------------------------------------
        # Frozen APQC exact compiler.
        # ----------------------------------------------------

        apqc_build = (
            apqc.apqc_build(
                atomic,
                valid,
            )
        )

        apqc_library = (
            apqc_build[
                "library"
            ]
        )

        apqc_exact_library[
            j
        ] = bool(
            np.array_equal(
                tecs_library,
                apqc_library,
            )
        )

        # ----------------------------------------------------
        # Frozen RBJS exact solver.
        # ----------------------------------------------------

        solved = (
            rbjs.bounded_choose_bcct(
                apqc_library,
                atomic,
                valid,
            )
        )

        p_rbjs = (
            solved[
                "policy"
            ]
        )

        rbjs_policy_exact[
            j
        ] = (
            policy_tuple(
                p_rbjs
            )
            ==
            policy_tuple(
                p_tecs
            )
        )

        (
            rbjs_s0,
            rbjs_s1,
        ) = a4.build_survival_masks(
            apqc_library
        )

        shared_rbjs = (
            gap.eval_compound_policy_all(
                p_rbjs,
                apqc_library,
                rbjs_s0,
                rbjs_s1,
                valid,
                a4,
            )
        )

        rbjs_coverage_mismatch[
            j
        ] = int(
            np.sum(
                shared_rbjs
                !=
                shared_tecs
            )
        )

        rbjs_pruned_roots[
            j
        ] = int(
            solved[
                "pruned_roots"
            ]
        )

        rbjs_evaluated_roots[
            j
        ] = int(
            solved[
                "evaluated_roots"
            ]
        )

        rbjs_exact_child_solves[
            j
        ] = int(
            solved[
                "exact_child_solves"
            ]
        )

        # ----------------------------------------------------
        # Secondary true-GT evaluation.
        # ----------------------------------------------------

        absolute_gt_pos = int(
            gt_top500_pos[
                p3i
            ]
        )

        if (
            absolute_gt_pos
            >=
            LOCAL_START
            and
            absolute_gt_pos
            <
            LOCAL_STOP
        ):

            true_gt_in_local80[
                j
            ] = True

            pos = (
                absolute_gt_pos
                -
                LOCAL_START
            )

            gt_frontier_pos[
                j
            ] = pos

            assert int(
                candidates[
                    pos
                ]
            ) == int(
                gt[
                    p3i
                ]
            )

            gt_is_valid = bool(
                ast_valid[
                    int(
                        gt[
                            p3i
                        ]
                    )
                ]
            )

            (
                rr,
                rret,
                _,
            ) = a4.execute_policy(
                p_rcps,
                rcps_library,
                rcps_s0,
                rcps_s1,
                pos,
                gt_is_valid,
            )

            (
                tr,
                tret,
                _,
            ) = a4.execute_policy(
                p_tecs,
                tecs_library,
                tecs_s0,
                tecs_s1,
                pos,
                gt_is_valid,
            )

            rcps_gt_rank[
                j
            ] = int(
                rr
            )

            tecs_gt_rank[
                j
            ] = int(
                tr
            )

            rcps_gt_retained[
                j
            ] = bool(
                rret
            )

            tecs_gt_retained[
                j
            ] = bool(
                tret
            )

        if (
            (j + 1) % 25 == 0
            or
            j + 1 == n_fail
        ):

            print(
                "evaluated",
                j + 1,
                "/",
                n_fail,
                "base failures",
                "elapsed=",
                round(
                    time.perf_counter()
                    -
                    start_time,
                    1,
                ),
                "sec",
            )

    # ========================================================
    # 8. Frozen structural quantities.
    # ========================================================

    Gamma_atomic = (
        O_atomic
        -
        S_atomic
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

    Gamma_tecs = (
        O_tecs
        -
        S_tecs
    ).astype(
        np.int16
    )

    sum_S_atomic = int(
        S_atomic.sum()
    )

    sum_S_rcps = int(
        S_rcps.sum()
    )

    sum_S_tecs = int(
        S_tecs.sum()
    )

    sum_Gamma_atomic = int(
        Gamma_atomic.sum()
    )

    sum_Gamma_rcps = int(
        Gamma_rcps.sum()
    )

    sum_Gamma_tecs = int(
        Gamma_tecs.sum()
    )

    # ========================================================
    # 9. Frozen shard gate.
    # ========================================================

    failure_shards = shard_ids[
        failure_p3_pos
    ]

    shard_result = {}

    nonworse_shards = 0
    better_shards = 0

    for shard in range(5):

        m = (
            failure_shards
            ==
            shard
        )

        n_shard = int(
            m.sum()
        )

        s_r = int(
            S_rcps[
                m
            ].sum()
        )

        s_t = int(
            S_tecs[
                m
            ].sum()
        )

        if s_t >= s_r:
            nonworse_shards += 1

        if s_t > s_r:
            better_shards += 1

        shard_result[
            str(shard)
        ] = {
            "base_failures":
                n_shard,

            "S_RCPS":
                s_r,

            "S_TECS":
                s_t,

            "delta":
                s_t
                -
                s_r,

            "nonworse":
                bool(
                    s_t
                    >=
                    s_r
                ),

            "better":
                bool(
                    s_t
                    >
                    s_r
                ),
        }

    # ========================================================
    # 10. Frozen exactness requirements.
    # ========================================================

    apqc_mismatch_queries = int(
        (
            ~apqc_exact_library
        ).sum()
    )

    rbjs_policy_mismatch_queries = int(
        (
            ~rbjs_policy_exact
        ).sum()
    )

    rbjs_vector_mismatches = int(
        rbjs_coverage_mismatch.sum()
    )

    apqc_exact_gate = (
        apqc_mismatch_queries
        ==
        0
    )

    rbjs_exact_gate = (
        rbjs_policy_mismatch_queries
        ==
        0
        and
        rbjs_vector_mismatches
        ==
        0
    )

    exactness_pass = bool(
        apqc_exact_gate
        and
        rbjs_exact_gate
    )

    # ========================================================
    # 11. Secondary true-GT gate.
    # ========================================================

    local_mask = (
        true_gt_in_local80
    )

    assert int(
        local_mask.sum()
    ) == n_local

    raw_gt_rank = (
        gt_frontier_pos[
            local_mask
        ]
        +
        1
    )

    rcps_rank_local = (
        rcps_gt_rank[
            local_mask
        ]
    )

    tecs_rank_local = (
        tecs_gt_rank[
            local_mask
        ]
    )

    raw_R20 = int(
        (
            raw_gt_rank
            <=
            B
        ).sum()
    )

    rcps_R20 = int(
        (
            rcps_rank_local
            <=
            B
        ).sum()
    )

    tecs_R20 = int(
        (
            tecs_rank_local
            <=
            B
        ).sum()
    )

    rcps_success = (
        rcps_rank_local
        <=
        B
    )

    tecs_success = (
        tecs_rank_local
        <=
        B
    )

    both = int(
        (
            rcps_success
            &
            tecs_success
        ).sum()
    )

    tecs_only = int(
        (
            tecs_success
            &
            ~rcps_success
        ).sum()
    )

    rcps_only = int(
        (
            rcps_success
            &
            ~tecs_success
        ).sum()
    )

    neither = int(
        (
            ~rcps_success
            &
            ~tecs_success
        ).sum()
    )

    rcps_retained_n = int(
        rcps_gt_retained[
            local_mask
        ].sum()
    )

    tecs_retained_n = int(
        tecs_gt_retained[
            local_mask
        ].sum()
    )

    gt_retention_gate = (
        tecs_retained_n
        ==
        n_local
    )

    secondary_gt_gate = (
        tecs_R20
        >=
        rcps_R20
    )

    # ========================================================
    # 12. Frozen primary gates.
    # ========================================================

    gate_shared = (
        sum_S_tecs
        >
        sum_S_rcps
    )

    gate_gamma = (
        sum_Gamma_tecs
        <
        sum_Gamma_rcps
    )

    gate_shards = (
        nonworse_shards
        >=
        4
    )

    primary_structural_pass = bool(
        gate_shared
        and
        gate_gamma
        and
        gate_shards
        and
        gt_retention_gate
    )

    # Exactness is a validity requirement, not a tunable
    # performance gate.
    p3_valid_confirmation = bool(
        exactness_pass
        and
        primary_structural_pass
    )

    # Secondary GT result is reported separately exactly as
    # frozen in the protocol.
    secondary_gt_pass = bool(
        secondary_gt_gate
    )

    # ========================================================
    # 13. Persist the first prospective outcome.
    # ========================================================

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "jters_p3_frozen_evaluation_v1"
        ),

        p3_failure_positions=
            failure_p3_pos.astype(
                np.int32
            ),

        source_row_indices=
            source_rows[
                failure_p3_pos
            ].astype(
                np.int64
            ),

        shard_ids=
            failure_shards.astype(
                np.int16
            ),

        ground_truth_code_indices=
            gt[
                failure_p3_pos
            ].astype(
                np.int32
            ),

        candidate_ids=
            candidate_ids_all,

        valid_hypotheses=
            valid_hypotheses,

        O_atomic=
            O_atomic,

        S_atomic=
            S_atomic,

        Gamma_atomic=
            Gamma_atomic,

        O_rcps=
            O_rcps,

        S_rcps=
            S_rcps,

        Gamma_rcps=
            Gamma_rcps,

        O_tecs=
            O_tecs,

        S_tecs=
            S_tecs,

        Gamma_tecs=
            Gamma_tecs,

        rcps_predicates=
            rcps_predicates,

        tecs_predicates=
            tecs_predicates,

        tecs_added=
            tecs_added,

        tecs_raw_certificates=
            tecs_raw_certificates,

        apqc_exact_library=
            apqc_exact_library,

        rbjs_policy_exact=
            rbjs_policy_exact,

        rbjs_coverage_mismatch=
            rbjs_coverage_mismatch,

        rbjs_pruned_roots=
            rbjs_pruned_roots,

        rbjs_evaluated_roots=
            rbjs_evaluated_roots,

        rbjs_exact_child_solves=
            rbjs_exact_child_solves,

        true_gt_in_local80=
            true_gt_in_local80,

        gt_frontier_pos=
            gt_frontier_pos,

        rcps_gt_rank=
            rcps_gt_rank,

        tecs_gt_rank=
            tecs_gt_rank,

        rcps_gt_retained=
            rcps_gt_retained,

        tecs_gt_retained=
            tecs_gt_retained,

        retrieval_sha256=np.asarray(
            EXPECTED[
                "retrieval"
            ]
        ),

        frozen_protocol_sha256=np.asarray(
            EXPECTED[
                "protocol"
            ]
        ),

        valid_accessed=np.asarray(
            False
        ),

        test_accessed=np.asarray(
            False
        ),
    )

    artifact_sha = sha256(
        OUT_NPZ
    )

    result = {
        "format":
            "jters_p3_frozen_evaluation_v1",

        "status":
            "FIRST P3 ALGORITHM OUTCOME; METHOD REMAINS FROZEN",

        "denominators": {
            "p3_queries":
                P3_N,

            "base_top100_failures":
                n_fail,

            "local80_recoverable_true_gt":
                n_local,
        },

        "structural": {
            "Atomic": {
                "O":
                    int(
                        O_atomic.sum()
                    ),

                "S":
                    sum_S_atomic,

                "Gamma":
                    sum_Gamma_atomic,
            },

            "RCPS": {
                "O":
                    int(
                        O_rcps.sum()
                    ),

                "S":
                    sum_S_rcps,

                "Gamma":
                    sum_Gamma_rcps,
            },

            "TECS": {
                "O":
                    int(
                        O_tecs.sum()
                    ),

                "S":
                    sum_S_tecs,

                "Gamma":
                    sum_Gamma_tecs,
            },

            "delta_TECS_minus_RCPS": {
                "O":
                    int(
                        O_tecs.sum()
                        -
                        O_rcps.sum()
                    ),

                "S":
                    sum_S_tecs
                    -
                    sum_S_rcps,

                "Gamma":
                    sum_Gamma_tecs
                    -
                    sum_Gamma_rcps,
            },
        },

        "shards":
            shard_result,

        "shard_summary": {
            "nonworse":
                nonworse_shards,

            "better":
                better_shards,
        },

        "exactness": {
            "APQC_library_mismatch_queries":
                apqc_mismatch_queries,

            "RBJS_policy_mismatch_queries":
                rbjs_policy_mismatch_queries,

            "RBJS_coverage_vector_mismatches":
                rbjs_vector_mismatches,

            "APQC_exact":
                apqc_exact_gate,

            "RBJS_exact":
                rbjs_exact_gate,

            "exactness_pass":
                exactness_pass,
        },

        "true_gt": {
            "n":
                n_local,

            "RAW_R20":
                raw_R20,

            "RCPS_R20":
                rcps_R20,

            "TECS_R20":
                tecs_R20,

            "both":
                both,

            "TECS_only":
                tecs_only,

            "RCPS_only":
                rcps_only,

            "neither":
                neither,

            "RCPS_GT_retained":
                rcps_retained_n,

            "TECS_GT_retained":
                tecs_retained_n,
        },

        "frozen_gates": {
            "sum_S_TECS_gt_sum_S_RCPS":
                gate_shared,

            "sum_Gamma_TECS_lt_sum_Gamma_RCPS":
                gate_gamma,

            "at_least_4_of_5_shards_nonworse":
                gate_shards,

            "GT_retention_100_percent":
                gt_retention_gate,

            "primary_structural_pass":
                primary_structural_pass,

            "R20_TECS_ge_R20_RCPS":
                secondary_gt_gate,

            "secondary_gt_pass":
                secondary_gt_pass,

            "exactness_pass":
                exactness_pass,

            "P3_VALID_PRIMARY_CONFIRMATION":
                p3_valid_confirmation,
        },

        "provenance": {
            k:
                v
            for k, v in EXPECTED.items()
        },

        "artifact": {
            "path":
                str(
                    OUT_NPZ
                ),

            "sha256":
                artifact_sha,
        },

        "elapsed_seconds":
            float(
                time.perf_counter()
                -
                start_time
            ),

        "post_p3_tuning_permitted":
            False,

        "valid_accessed":
            False,

        "test_accessed":
            False,
    }

    OUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    json_sha = sha256(
        OUT_JSON
    )

    # ========================================================
    # 14. First prospective reveal.
    # ========================================================

    print()
    print("=" * 160)
    print("P3 FIRST PROSPECTIVE REVEAL")
    print("=" * 160)

    print()
    print(
        "Atomic :",
        "O =",
        int(
            O_atomic.sum()
        ),
        "S =",
        sum_S_atomic,
        "Gamma =",
        sum_Gamma_atomic,
    )

    print(
        "RCPS   :",
        "O =",
        int(
            O_rcps.sum()
        ),
        "S =",
        sum_S_rcps,
        "Gamma =",
        sum_Gamma_rcps,
    )

    print(
        "TECS   :",
        "O =",
        int(
            O_tecs.sum()
        ),
        "S =",
        sum_S_tecs,
        "Gamma =",
        sum_Gamma_tecs,
    )

    print()

    print(
        "delta S (TECS-RCPS) =",
        sum_S_tecs
        -
        sum_S_rcps,
    )

    print(
        "delta Gamma (TECS-RCPS) =",
        sum_Gamma_tecs
        -
        sum_Gamma_rcps,
    )

    print()

    print(
        "shards nonworse =",
        nonworse_shards,
        "/ 5",
    )

    print(
        "shards better =",
        better_shards,
        "/ 5",
    )

    for shard in range(5):

        x = shard_result[
            str(shard)
        ]

        print(
            "shard",
            shard,
            ": n=",
            x[
                "base_failures"
            ],
            "RCPS=",
            x[
                "S_RCPS"
            ],
            "TECS=",
            x[
                "S_TECS"
            ],
            "delta=",
            x[
                "delta"
            ],
        )

    print()
    print(
        "APQC exact library mismatches =",
        apqc_mismatch_queries,
    )

    print(
        "RBJS policy mismatches =",
        rbjs_policy_mismatch_queries,
    )

    print(
        "RBJS coverage-vector mismatches =",
        rbjs_vector_mismatches,
    )

    print()

    print(
        "true-GT Local80 n =",
        n_local,
    )

    print(
        "RAW R20 =",
        raw_R20,
    )

    print(
        "RCPS R20 =",
        rcps_R20,
    )

    print(
        "TECS R20 =",
        tecs_R20,
    )

    print(
        "TECS-only / RCPS-only / both / neither =",
        tecs_only,
        "/",
        rcps_only,
        "/",
        both,
        "/",
        neither,
    )

    print(
        "RCPS GT retained =",
        rcps_retained_n,
        "/",
        n_local,
    )

    print(
        "TECS GT retained =",
        tecs_retained_n,
        "/",
        n_local,
    )

    print()
    print("=" * 160)
    print("FROZEN P3 GATE DECISION")
    print("=" * 160)

    print(
        "GATE 1: sum(S_TECS) > sum(S_RCPS) =",
        gate_shared,
    )

    print(
        "GATE 2: sum(Gamma_TECS) < sum(Gamma_RCPS) =",
        gate_gamma,
    )

    print(
        "GATE 3: >=4/5 shards nonworse =",
        gate_shards,
    )

    print(
        "GATE 4: GT retention 100% =",
        gt_retention_gate,
    )

    print(
        "P3_PRIMARY_STRUCTURAL_PASS =",
        primary_structural_pass,
    )

    print()

    print(
        "P3_EXACTNESS_PASS =",
        exactness_pass,
    )

    print(
        "P3_VALID_PRIMARY_CONFIRMATION =",
        p3_valid_confirmation,
    )

    print()

    print(
        "SECONDARY: R20_TECS >= R20_RCPS =",
        secondary_gt_gate,
    )

    print(
        "P3_SECONDARY_GT_PASS =",
        secondary_gt_pass,
    )

    print()

    print(
        "artifact SHA256 =",
        artifact_sha,
    )

    print(
        "report SHA256 =",
        json_sha,
    )

    print(
        "POST_P3_METHOD_TUNING_PERMITTED = False"
    )

    if p3_valid_confirmation:

        print(
            "STATUS = P3 PRIMARY STRUCTURAL "
            "CONFIRMATION PASSED"
        )

    else:

        print(
            "STATUS = P3 PRIMARY STRUCTURAL "
            "CONFIRMATION DID NOT PASS; "
            "FROZEN RESULT MUST BE RETAINED"
        )


if __name__ == "__main__":
    main()
