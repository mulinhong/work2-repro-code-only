from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")

E = 80
UNKNOWN = -1


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

TECS_ART = (
    ROOT
    / "fhex_v1/results/"
      "tree_embeddable_certificate_synthesis_v0.npz"
)

OUT_NPZ = (
    ROOT
    / "fhex_v1/obstruction_audit_v0/"
      "tecs_rcps_union_v0.npz"
)

OUT_JSON = (
    ROOT
    / "fhex_v1/obstruction_audit_v0/"
      "tecs_rcps_union_v0.json"
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

    mod = importlib.util.module_from_spec(spec)

    assert spec.loader is not None

    spec.loader.exec_module(mod)

    return mod


def exact_column_union(*libraries):
    """
    Exact ternary-column union.

    Keep the first occurrence of every complete 80-candidate
    ternary semantic vector.

    No GT/outcome-dependent filtering.
    """

    seen = set()
    columns = []

    source_counts = []

    for lib in libraries:

        added = 0

        for j in range(
            lib.shape[1]
        ):
            col = lib[:, j]

            key = col.tobytes()

            if key in seen:
                continue

            seen.add(key)

            columns.append(
                col.copy()
            )

            added += 1

        source_counts.append(
            int(added)
        )

    out = np.stack(
        columns,
        axis=1,
    )

    return (
        out,
        source_counts,
    )


def main():

    print("=" * 132)
    print(
        "TECS ∪ RCPS EXACT SUPERSET ROBUSTNESS AUDIT v0"
    )
    print(
        "DIAGNOSTIC ONLY / NO NEW METHOD / NO P3 CLAIM"
    )
    print("=" * 132)

    # ============================================================
    # Integrity
    # ============================================================

    observed = {
        "A4":
            sha256_file(
                A4_SCRIPT
            ),

        "GAP":
            sha256_file(
                GAP_SCRIPT
            ),

        "TECS":
            sha256_file(
                TECS_SCRIPT
            ),

        "AST":
            sha256_file(
                AST_CACHE
            ),
    }

    assert (
        observed["A4"]
        ==
        EXPECTED_A4_SHA
    )

    assert (
        observed["GAP"]
        ==
        EXPECTED_GAP_SHA
    )

    assert (
        observed["TECS"]
        ==
        EXPECTED_TECS_SHA
    )

    assert (
        observed["AST"]
        ==
        EXPECTED_AST_SHA
    )

    print(
        "HASH INTEGRITY = PASS"
    )

    # ============================================================
    # Load
    # ============================================================

    a4 = load_module(
        A4_SCRIPT,
        "union_a4",
    )

    gap = load_module(
        GAP_SCRIPT,
        "union_gap",
    )

    tecs = load_module(
        TECS_SCRIPT,
        "union_tecs",
    )

    a2 = np.load(
        A2_ART,
        allow_pickle=True,
    )

    az = np.load(
        AST_CACHE,
        allow_pickle=True,
    )

    frozen = np.load(
        TECS_ART,
        allow_pickle=True,
    )

    candidates = (
        a2[
            "candidates"
        ]
        .astype(
            np.int32
        )
    )

    candidate_valid = (
        a2[
            "candidate_ast_valid"
        ]
        .astype(
            bool
        )
    )

    assert (
        candidates.shape
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
            candidates
        ],
        candidate_valid,
    )

    # Frozen reference.
    O_tecs = (
        frozen[
            "tecs_O"
        ]
        .astype(
            np.int16
        )
    )

    S_tecs = (
        frozen[
            "tecs_S"
        ]
        .astype(
            np.int16
        )
    )

    G_tecs = (
        frozen[
            "tecs_Gamma"
        ]
        .astype(
            np.int16
        )
    )

    O_rcps = (
        frozen[
            "rcps_O"
        ]
        .astype(
            np.int16
        )
    )

    S_rcps = (
        frozen[
            "rcps_S"
        ]
        .astype(
            np.int16
        )
    )

    assert int(
        S_tecs.sum()
    ) == 43319

    assert int(
        G_tecs.sum()
    ) == 1392

    assert int(
        np.sum(
            G_tecs
            >
            0
        )
    ) == 333

    print(
        "FROZEN TECS ANCHOR = PASS"
    )

    # ============================================================
    # Outputs
    # ============================================================

    n = 568

    O_union = np.zeros(
        n,
        dtype=np.int16,
    )

    S_union = np.zeros(
        n,
        dtype=np.int16,
    )

    G_union = np.zeros(
        n,
        dtype=np.int16,
    )

    tecs_size = np.zeros(
        n,
        dtype=np.int16,
    )

    rcps_size = np.zeros(
        n,
        dtype=np.int16,
    )

    union_size = np.zeros(
        n,
        dtype=np.int16,
    )

    rcps_new_over_tecs = np.zeros(
        n,
        dtype=np.int16,
    )

    t_all = time.perf_counter()

    # ============================================================
    # Exact union audit
    # ============================================================

    for qi in range(n):

        t0 = time.perf_counter()

        cands = (
            candidates[
                qi
            ]
        )

        valid = (
            candidate_valid[
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
                cands
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

        # --------------------------------------------------------
        # Frozen TECS family
        # --------------------------------------------------------

        built_tecs = (
            tecs.build_tecs_library(
                atomic,
                valid,
                a4,
            )
        )

        lib_tecs = (
            built_tecs[
                "library"
            ]
        )

        # Strong per-query frozen parity.
        ind_tecs = (
            gap.individual_b20_vector(
                lib_tecs,
                valid,
                a4,
            )
        )

        shared_tecs = (
            tecs.exact_shared_vector(
                lib_tecs,
                valid,
                a4,
                gap,
            )
        )

        assert int(
            ind_tecs.sum()
        ) == int(
            O_tecs[
                qi
            ]
        )

        assert int(
            shared_tecs.sum()
        ) == int(
            S_tecs[
                qi
            ]
        )

        # --------------------------------------------------------
        # Frozen RCPS family
        # --------------------------------------------------------

        full_rcps, _, _, _ = (
            a4.build_synth_library(
                atomic
            )
        )

        lib_rcps, _ = (
            a4.deduplicate_predicates(
                full_rcps,
                valid,
            )
        )

        # Frozen RCPS parity.
        ind_rcps = (
            gap.individual_b20_vector(
                lib_rcps,
                valid,
                a4,
            )
        )

        policy_rcps, s0r, s1r = (
            a4.choose_bcct(
                lib_rcps,
                valid,
            )
        )

        shared_rcps = (
            gap.eval_compound_policy_all(
                policy_rcps,
                lib_rcps,
                s0r,
                s1r,
                valid,
                a4,
            )
        )

        assert int(
            ind_rcps.sum()
        ) == int(
            O_rcps[
                qi
            ]
        )

        assert int(
            shared_rcps.sum()
        ) == int(
            S_rcps[
                qi
            ]
        )

        # --------------------------------------------------------
        # Exact semantic union
        # --------------------------------------------------------

        (
            lib_union,
            source_added,
        ) = (
            exact_column_union(
                lib_tecs,
                lib_rcps,
            )
        )

        tecs_size[
            qi
        ] = int(
            lib_tecs.shape[1]
        )

        rcps_size[
            qi
        ] = int(
            lib_rcps.shape[1]
        )

        union_size[
            qi
        ] = int(
            lib_union.shape[1]
        )

        # Since TECS goes first, this is the number of exact RCPS
        # semantic columns not already contained in TECS.
        rcps_new_over_tecs[
            qi
        ] = int(
            source_added[1]
        )

        # --------------------------------------------------------
        # Exact union O / S
        # --------------------------------------------------------

        ind_union = (
            gap.individual_b20_vector(
                lib_union,
                valid,
                a4,
            )
        )

        policy_union, s0u, s1u = (
            a4.choose_bcct(
                lib_union,
                valid,
            )
        )

        shared_union = (
            gap.eval_compound_policy_all(
                policy_union,
                lib_union,
                s0u,
                s1u,
                valid,
                a4,
            )
        )

        O_union[
            qi
        ] = int(
            ind_union.sum()
        )

        S_union[
            qi
        ] = int(
            shared_union.sum()
        )

        G_union[
            qi
        ] = (
            O_union[
                qi
            ]
            -
            S_union[
                qi
            ]
        )

        # Exact superset monotonicity.
        assert (
            O_union[
                qi
            ]
            >=
            max(
                int(
                    O_tecs[
                        qi
                    ]
                ),
                int(
                    O_rcps[
                        qi
                    ]
                ),
            )
        )

        assert (
            S_union[
                qi
            ]
            >=
            max(
                int(
                    S_tecs[
                        qi
                    ]
                ),
                int(
                    S_rcps[
                        qi
                    ]
                ),
            )
        )

        elapsed = (
            time.perf_counter()
            -
            t0
        )

        if (
            qi % 10
            ==
            0
            or
            S_union[qi]
            >
            S_tecs[qi]
        ):

            print(
                f"q={qi:03d} "
                f"Ptecs={lib_tecs.shape[1]:04d} "
                f"Prcps={lib_rcps.shape[1]:04d} "
                f"Punion={lib_union.shape[1]:04d} "
                f"newRCPS={source_added[1]:04d} "
                f"O={int(O_tecs[qi]):02d}->{int(O_union[qi]):02d} "
                f"S={int(S_tecs[qi]):02d}->{int(S_union[qi]):02d} "
                f"G={int(G_tecs[qi]):02d}->{int(G_union[qi]):02d} "
                f"time={elapsed:.2f}s",
                flush=True,
            )

    runtime = (
        time.perf_counter()
        -
        t_all
    )

    # ============================================================
    # Summary
    # ============================================================

    dO = (
        O_union
        -
        O_tecs
    ).astype(
        np.int16
    )

    dS = (
        S_union
        -
        S_tecs
    ).astype(
        np.int16
    )

    dG = (
        G_union
        -
        G_tecs
    ).astype(
        np.int16
    )

    gap_closure = (
        G_tecs
        -
        G_union
    ).astype(
        np.int16
    )

    gap_mask = (
        G_tecs
        >
        0
    )

    print()
    print("=" * 132)
    print(
        "TECS ∪ RCPS SUPERSET ROBUSTNESS SUMMARY"
    )
    print("=" * 132)

    print(
        "queries =",
        n,
    )

    print()
    print(
        "TECS totals O / S / Gamma =",
        int(
            O_tecs.sum()
        ),
        int(
            S_tecs.sum()
        ),
        int(
            G_tecs.sum()
        ),
    )

    print(
        "RCPS totals O / S =",
        int(
            O_rcps.sum()
        ),
        int(
            S_rcps.sum()
        ),
    )

    print(
        "UNION totals O / S / Gamma =",
        int(
            O_union.sum()
        ),
        int(
            S_union.sum()
        ),
        int(
            G_union.sum()
        ),
    )

    print()
    print(
        "UNION vs TECS total dO =",
        int(
            dO.sum()
        ),
    )

    print(
        "UNION vs TECS total dS =",
        int(
            dS.sum()
        ),
    )

    print(
        "UNION vs TECS total gap closure =",
        int(
            gap_closure.sum()
        ),
    )

    print()
    print(
        "queries dO > 0 =",
        int(
            np.sum(
                dO
                >
                0
            )
        ),
    )

    print(
        "queries dS > 0 =",
        int(
            np.sum(
                dS
                >
                0
            )
        ),
    )

    print(
        "queries Gamma reduced =",
        int(
            np.sum(
                G_union
                <
                G_tecs
            )
        ),
    )

    print(
        "queries Gamma increased =",
        int(
            np.sum(
                G_union
                >
                G_tecs
            )
        ),
    )

    print()
    print(
        "original TECS gap-positive queries =",
        int(
            gap_mask.sum()
        ),
    )

    print(
        "among them UNION gap eliminated =",
        int(
            np.sum(
                gap_mask
                &
                (
                    G_union
                    ==
                    0
                )
            )
        ),
    )

    print(
        "among them UNION gap still positive =",
        int(
            np.sum(
                gap_mask
                &
                (
                    G_union
                    >
                    0
                )
            )
        ),
    )

    print()
    print(
        "mean RCPS semantic columns newly added over TECS =",
        float(
            rcps_new_over_tecs.mean()
        ),
    )

    print(
        "median RCPS semantic columns newly added over TECS =",
        float(
            np.median(
                rcps_new_over_tecs
            )
        ),
    )

    print(
        "runtime seconds =",
        runtime,
    )

    # Largest exact shared gains.
    order = np.argsort(
        -dS
    )

    print()
    print(
        "TOP UNION SHARED GAINS"
    )

    for qi in order[:20]:

        if int(
            dS[
                qi
            ]
        ) <= 0:
            break

        print(
            f"q={qi:03d} "
            f"dO={int(dO[qi]):+d} "
            f"dS={int(dS[qi]):+d} "
            f"G={int(G_tecs[qi])}->{int(G_union[qi])} "
            f"newRCPS={int(rcps_new_over_tecs[qi])}"
        )

    # ============================================================
    # Save
    # ============================================================

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "TECS_RCPS_EXACT_UNION_V0"
        ),

        tecs_O=O_tecs,
        tecs_S=S_tecs,
        tecs_Gamma=G_tecs,

        rcps_O=O_rcps,
        rcps_S=S_rcps,

        union_O=O_union,
        union_S=S_union,
        union_Gamma=G_union,

        dO=dO,
        dS=dS,
        dGamma=dG,
        gap_closure=gap_closure,

        tecs_library_size=tecs_size,
        rcps_library_size=rcps_size,
        union_library_size=union_size,

        rcps_new_semantic_columns_over_tecs=
            rcps_new_over_tecs,
    )

    result = {
        "name":
            "TECS + RCPS Exact Superset Robustness Audit v0",

        "scope":
            "FIT structural diagnostic only",

        "queries":
            int(n),

        "tecs": {
            "O":
                int(
                    O_tecs.sum()
                ),

            "S":
                int(
                    S_tecs.sum()
                ),

            "Gamma":
                int(
                    G_tecs.sum()
                ),

            "gap_positive_queries":
                int(
                    np.sum(
                        G_tecs
                        >
                        0
                    )
                ),
        },

        "union": {
            "O":
                int(
                    O_union.sum()
                ),

            "S":
                int(
                    S_union.sum()
                ),

            "Gamma":
                int(
                    G_union.sum()
                ),

            "dO_vs_tecs":
                int(
                    dO.sum()
                ),

            "dS_vs_tecs":
                int(
                    dS.sum()
                ),

            "gap_closure_vs_tecs":
                int(
                    gap_closure.sum()
                ),

            "queries_dS_positive":
                int(
                    np.sum(
                        dS
                        >
                        0
                    )
                ),

            "tecs_gap_queries_eliminated":
                int(
                    np.sum(
                        gap_mask
                        &
                        (
                            G_union
                            ==
                            0
                        )
                    )
                ),

            "tecs_gap_queries_remaining":
                int(
                    np.sum(
                        gap_mask
                        &
                        (
                            G_union
                            >
                            0
                        )
                    )
                ),
        },

        "runtime_seconds":
            float(
                runtime
            ),

        "integrity":
            observed,
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
