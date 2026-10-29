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

OUT_JSON = (
    ROOT
    / "fhex_v1/results/"
      "tecs_rbjs_runtime_decomposition_v0.json"
)


EXPECTED_TECS = (
    "eea14a27eae6f0aee02a6f74caed4294"
    "c2227977a0fb165b26957ced143c7729"
)

EXPECTED_RBJS = (
    "94c83912962fc57adab96bcc94fb4161d"
    "fc79dcf84544fc41b5cbc84293f02a2"
)

E = 80
UNKNOWN = -1


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
                    0.9,
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


def main():

    print(
        "=" * 145
    )

    print(
        "TECS / RBJS RUNTIME DECOMPOSITION v0"
    )

    print(
        "=" * 145
    )

    print(
        "FIT only. No P1/P2."
    )

    print(
        "No method changes. Timing only."
    )

    assert (
        sha256_file(
            TECS_ART
        )
        ==
        EXPECTED_TECS
    )

    assert (
        sha256_file(
            RBJS_ART
        )
        ==
        EXPECTED_RBJS
    )

    a4 = load_module(
        A4_SCRIPT,
        "bench_a4",
    )

    tecsmod = load_module(
        TECS_SCRIPT,
        "bench_tecs",
    )

    rbjs = load_module(
        RBJS_SCRIPT,
        "bench_rbjs",
    )

    # RBJS helper uses module-global a4.
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
    ].astype(str)

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

    frozen_S = tecs_art[
        "tecs_S"
    ].astype(
        np.int16
    )

    n = len(
        candidates
    )

    t_build = np.zeros(
        n,
        dtype=np.float64,
    )

    t_masks = np.zeros(
        n,
        dtype=np.float64,
    )

    t_exhaustive = np.zeros(
        n,
        dtype=np.float64,
    )

    t_bounded = np.zeros(
        n,
        dtype=np.float64,
    )

    lib_size = np.zeros(
        n,
        dtype=np.int32,
    )

    exhaustive_cov = np.zeros(
        n,
        dtype=np.int16,
    )

    bounded_cov = np.zeros(
        n,
        dtype=np.int16,
    )

    mismatch = 0

    total_t0 = time.perf_counter()

    for qi in range(
        n
    ):

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

        # --------------------------------------------
        # Certificate construction.
        # --------------------------------------------

        x0 = time.perf_counter()

        built = tecsmod.build_tecs_library(
            atomic,
            valid,
            a4,
        )

        facts = built[
            "library"
        ]

        x1 = time.perf_counter()

        t_build[
            qi
        ] = (
            x1
            -
            x0
        )

        lib_size[
            qi
        ] = int(
            facts.shape[
                1
            ]
        )

        # --------------------------------------------
        # Survival-mask construction only.
        # --------------------------------------------

        x0 = time.perf_counter()

        _ = a4.build_survival_masks(
            facts
        )

        x1 = time.perf_counter()

        t_masks[
            qi
        ] = (
            x1
            -
            x0
        )

        # --------------------------------------------
        # Exhaustive BCCT.
        # --------------------------------------------

        x0 = time.perf_counter()

        (
            policy_ex,
            _,
            _,
        ) = a4.choose_bcct(
            facts,
            valid,
        )

        x1 = time.perf_counter()

        t_exhaustive[
            qi
        ] = (
            x1
            -
            x0
        )

        exhaustive_cov[
            qi
        ] = int(
            policy_ex[
                "coverage"
            ]
        )

        # --------------------------------------------
        # RBJS.
        # --------------------------------------------

        x0 = time.perf_counter()

        solved = rbjs.bounded_choose_bcct(
            facts,
            atomic,
            valid,
        )

        x1 = time.perf_counter()

        t_bounded[
            qi
        ] = (
            x1
            -
            x0
        )

        bounded_cov[
            qi
        ] = int(
            solved[
                "policy"
            ][
                "coverage"
            ]
        )

        if (
            exhaustive_cov[
                qi
            ]
            !=
            frozen_S[
                qi
            ]
            or
            bounded_cov[
                qi
            ]
            !=
            frozen_S[
                qi
            ]
        ):

            mismatch += 1

        if (
            (qi + 1) % 50 == 0
            or
            qi
            ==
            n - 1
        ):

            print(
                f"[{qi+1:>3}/{n}]",
                "build=",
                round(
                    float(
                        t_build[
                            :qi+1
                        ].sum()
                    ),
                    2,
                ),
                "exhaustive=",
                round(
                    float(
                        t_exhaustive[
                            :qi+1
                        ].sum()
                    ),
                    2,
                ),
                "RBJS=",
                round(
                    float(
                        t_bounded[
                            :qi+1
                        ].sum()
                    ),
                    2,
                ),
                flush=True,
            )

    assert mismatch == 0

    build_total = float(
        t_build.sum()
    )

    masks_total = float(
        t_masks.sum()
    )

    exhaustive_total = float(
        t_exhaustive.sum()
    )

    bounded_total = float(
        t_bounded.sum()
    )

    exhaustive_pipeline = (
        build_total
        +
        exhaustive_total
    )

    bounded_pipeline = (
        build_total
        +
        bounded_total
    )

    solver_speedup = (
        exhaustive_total
        /
        bounded_total
    )

    pipeline_speedup = (
        exhaustive_pipeline
        /
        bounded_pipeline
    )

    synthesis_fraction_bounded = (
        build_total
        /
        bounded_pipeline
    )

    print()
    print(
        "=" * 145
    )

    print(
        "RUNTIME RESULT"
    )

    print(
        "=" * 145
    )

    print(
        "coverage mismatches =",
        mismatch,
    )

    print()

    print(
        "TECS certificate construction seconds =",
        build_total,
    )

    print(
        "survival-mask-only seconds =",
        masks_total,
    )

    print(
        "exhaustive BCCT seconds =",
        exhaustive_total,
    )

    print(
        "RBJS seconds =",
        bounded_total,
    )

    print()

    print(
        "exhaustive pipeline "
        "(certificate + solver) seconds =",
        exhaustive_pipeline,
    )

    print(
        "RBJS pipeline "
        "(certificate + solver) seconds =",
        bounded_pipeline,
    )

    print()

    print(
        "solver-only speedup exhaustive/RBJS =",
        solver_speedup,
    )

    print(
        "end-to-end speedup exhaustive/RBJS =",
        pipeline_speedup,
    )

    print(
        "certificate synthesis fraction "
        "of RBJS pipeline =",
        synthesis_fraction_bounded,
    )

    print()

    print(
        "certificate construction/query =",
        stats(
            t_build
        ),
    )

    print(
        "exhaustive solver/query =",
        stats(
            t_exhaustive
        ),
    )

    print(
        "RBJS solver/query =",
        stats(
            t_bounded
        ),
    )

    print(
        "TECS library size/query =",
        stats(
            lib_size
        ),
    )

    # Frozen interpretation boundary:
    # no threshold is used to choose an algorithm here.
    if (
        synthesis_fraction_bounded
        >
        0.50
    ):

        diagnosis = (
            "CERTIFICATE MATERIALIZATION IS NOW THE "
            "DOMINANT BOTTLENECK. Keep RBJS frozen and "
            "develop exact implicit/on-demand minimal-"
            "certificate generation next."
        )

    else:

        diagnosis = (
            "TREE SEARCH REMAINS A MATERIAL FRACTION OF "
            "END-TO-END COST. RBJS already addresses the "
            "dominant search component; do not introduce "
            "column generation solely for complexity."
        )

    print()
    print(
        "DIAGNOSIS =",
        diagnosis,
    )

    result = {
        "name":
            "TECS/RBJS Runtime Decomposition v0",

        "coverage_mismatches":
            mismatch,

        "seconds": {
            "certificate":
                build_total,

            "mask_only":
                masks_total,

            "exhaustive_solver":
                exhaustive_total,

            "rbjs_solver":
                bounded_total,

            "exhaustive_pipeline":
                exhaustive_pipeline,

            "rbjs_pipeline":
                bounded_pipeline,
        },

        "speedup": {
            "solver_only":
                solver_speedup,

            "end_to_end":
                pipeline_speedup,
        },

        "rbjs_pipeline_certificate_fraction":
            synthesis_fraction_bounded,

        "diagnosis":
            diagnosis,
    }

    OUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "metrics =",
        OUT_JSON,
    )

    print(
        "wall runtime =",
        time.perf_counter()
        -
        total_t0,
    )


if __name__ == "__main__":
    main()
