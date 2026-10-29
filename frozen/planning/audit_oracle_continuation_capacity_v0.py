from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.util
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")
D = ROOT / "fhex_v1/obstruction_audit_v0"

UNION_SCRIPT = D / "audit_tecs_rcps_union_v0.py"
UNION_ART = D / "tecs_rcps_union_v0.npz"
V0_SCRIPT = D / "audit_depth2_obstruction_v0.py"

EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_V0_SHA = (
    "aa4e1d359ff9dfe801a1ddfd23cc91c35c2f05e36dd08113f5d7af776b3f46e8"
)

E = 80
B = 20
CAP = B + 1
C = CAP + 1

UNKNOWN = -1
ABSENT = 0
PRESENT = 1

NEG = -10000


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
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# Precomputed DP transitions.
# ----------------------------------------------------------------------

c0_grid, c1_grid = np.meshgrid(
    np.arange(C, dtype=np.int16),
    np.arange(C, dtype=np.int16),
    indexing="ij",
)

flat_c0 = c0_grid.reshape(-1)
flat_c1 = c1_grid.reshape(-1)

new0 = np.minimum(
    flat_c0 + 1,
    CAP,
)

new1 = np.minimum(
    flat_c1 + 1,
    CAP,
)

IDX_ADD0 = (
    new0 * C
    +
    flat_c1
).astype(np.int32)

IDX_ADD1 = (
    flat_c0 * C
    +
    new1
).astype(np.int32)

IDX_ADD_BOTH = (
    new0 * C
    +
    new1
).astype(np.int32)

GAIN_IF_ADD0 = (
    new0 <= B
).astype(np.int16)

GAIN_IF_ADD1 = (
    new1 <= B
).astype(np.int16)


def arbitrary_binary_branch_optimum(
    current_mask,
    valid,
    target_mask,
):
    """
    Exact maximum target coverage obtainable by an arbitrary
    binary second-stage predicate on valid candidates.

    Invalid candidates are fixed UNKNOWN and therefore survive
    BOTH answer leaves.

    This is an upper bound over every realizable AST predicate,
    but preserves:
        - retrieval order
        - B=20
        - UNKNOWN semantics
        - one binary second question
    """

    target_mask = (
        np.asarray(target_mask, dtype=bool)
        &
        np.asarray(current_mask, dtype=bool)
    )

    target_n = int(
        target_mask.sum()
    )

    if target_n == 0:
        return 0

    # If whole root branch already fits in B, STOP repairs all.
    if int(current_mask.sum()) <= B:
        return target_n

    dp = np.full(
        C * C,
        NEG,
        dtype=np.int16,
    )

    dp[0] = 0

    for pos in np.flatnonzero(
        current_mask
    ):

        pos = int(pos)

        if not bool(valid[pos]):
            # UNKNOWN under every admissible AST-derived semantics:
            # it survives both leaves.
            ndp = np.full_like(
                dp,
                NEG,
            )

            np.maximum.at(
                ndp,
                IDX_ADD_BOTH,
                dp,
            )

            dp = ndp
            continue

        is_target = int(
            target_mask[pos]
        )

        ndp = np.full_like(
            dp,
            NEG,
        )

        # Assign this valid candidate answer 0.
        val0 = (
            dp
            +
            is_target
            *
            GAIN_IF_ADD0
        )

        np.maximum.at(
            ndp,
            IDX_ADD0,
            val0,
        )

        # Assign this valid candidate answer 1.
        val1 = (
            dp
            +
            is_target
            *
            GAIN_IF_ADD1
        )

        np.maximum.at(
            ndp,
            IDX_ADD1,
            val1,
        )

        dp = ndp

    return int(
        dp.max()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-gap-queries",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    print("=" * 132)
    print("ORACLE CONTINUATION CAPACITY AUDIT v0")
    print("ARBITRARY BINARY SECOND-STAGE LABELING / FIXED ROOT FAMILY")
    print("UNKNOWN SEMANTICS AND RANK BUDGET ARE PRESERVED")
    print("=" * 132)

    assert (
        sha256_file(UNION_SCRIPT)
        ==
        EXPECTED_UNION_SHA
    )

    assert (
        sha256_file(V0_SCRIPT)
        ==
        EXPECTED_V0_SHA
    )

    print("FROZEN SCRIPT INTEGRITY = PASS")

    unionmod = load_module(
        UNION_SCRIPT,
        "oracle_union",
    )

    v0 = load_module(
        V0_SCRIPT,
        "oracle_v0",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "oracle_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "oracle_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "oracle_tecs",
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

    candidates_all = (
        a2["candidates"]
        .astype(np.int32)
    )

    valid_all = (
        a2["candidate_ast_valid"]
        .astype(bool)
    )

    names_all = (
        az["feature_names"]
        .astype(str)
    )

    signature = (
        az["signature"]
        .astype(np.float32)
    )

    allowed_ids = np.flatnonzero(
        names_all != "max_ast_depth"
    )

    assert len(allowed_ids) == 44

    O_saved = (
        ua["union_O"]
        .astype(np.int16)
    )

    S_saved = (
        ua["union_S"]
        .astype(np.int16)
    )

    G_saved = (
        ua["union_Gamma"]
        .astype(np.int16)
    )

    assert int(O_saved.sum()) == 44711
    assert int(S_saved.sum()) == 43330
    assert int(G_saved.sum()) == 1381
    assert int(np.sum(G_saved > 0)) == 329

    print("UNION ARTIFACT ANCHOR = PASS")

    processed = 0

    total_O = 0
    total_S = 0
    total_oracle = 0

    total_expr_headroom = 0
    total_intrinsic_residual = 0

    closed_by_oracle = 0
    oracle_improved = 0

    t_all = time.perf_counter()

    for qi in range(568):

        if int(G_saved[qi]) <= 0:
            continue

        if (
            args.max_gap_queries > 0
            and
            processed >= args.max_gap_queries
        ):
            break

        q_t0 = time.perf_counter()

        candidates = candidates_all[qi]
        valid = valid_all[qi]

        atomic = np.full(
            (E, 44),
            UNKNOWN,
            dtype=np.int8,
        )

        presence = (
            signature[candidates][:, allowed_ids]
            > 0.0
        ).astype(np.int8)

        atomic[valid] = presence[valid]

        built = tecs.build_tecs_library(
            atomic,
            valid,
            a4,
        )

        lib_tecs = built["library"]

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

        facts, _ = (
            unionmod.exact_column_union(
                lib_tecs,
                lib_rcps,
            )
        )

        ind = gap.individual_b20_vector(
            facts,
            valid,
            a4,
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

        assert O == int(O_saved[qi])
        assert S == int(S_saved[qi])
        assert O - S == int(G_saved[qi])

        char = (
            v0.build_root_characterization(
                facts,
                valid,
                ind,
                a4,
            )
        )

        assert (
            int(char["characterization_S"])
            ==
            S
        )

        survive0, survive1 = (
            a4.build_survival_masks(
                facts
            )
        )

        # Raw STOP baseline.
        target_ids = np.flatnonzero(
            ind
        )

        oracle_S = int(
            np.sum(
                target_ids
                <
                B
            )
        )

        best_root = -1

        # Cache oracle branch solution by actual root branch mask.
        # For valid targets, membership in current_mask exactly
        # determines which root-answer branch they take.
        cache = {}

        def get_branch(current_mask):

            key = np.packbits(
                current_mask
            ).tobytes()

            if key not in cache:

                target_mask = (
                    ind
                    &
                    current_mask
                )

                cache[key] = (
                    arbitrary_binary_branch_optimum(
                        current_mask,
                        valid,
                        target_mask,
                    )
                )

            return int(
                cache[key]
            )

        # Existing legal root family; arbitrary second continuation.
        for rec in char["roots"]:

            p = int(
                rec["root"]
            )

            b0 = get_branch(
                survive0[p]
            )

            b1 = get_branch(
                survive1[p]
            )

            oracle_root = (
                b0
                +
                b1
            )

            # Existing realizable continuation coverage under this
            # root must never exceed arbitrary-label oracle coverage.
            assert (
                oracle_root
                >=
                int(rec["coverage"])
            ), (
                qi,
                p,
                oracle_root,
                int(rec["coverage"]),
            )

            if oracle_root > oracle_S:

                oracle_S = int(
                    oracle_root
                )

                best_root = p

        assert (
            S
            <=
            oracle_S
            <=
            O
        ), (
            qi,
            S,
            oracle_S,
            O,
        )

        expr_headroom = (
            oracle_S
            -
            S
        )

        intrinsic_residual = (
            O
            -
            oracle_S
        )

        assert (
            expr_headroom
            +
            intrinsic_residual
            ==
            O - S
        )

        if expr_headroom > 0:
            oracle_improved += 1

        if intrinsic_residual == 0:
            closed_by_oracle += 1

        total_O += O
        total_S += S
        total_oracle += oracle_S

        total_expr_headroom += (
            expr_headroom
        )

        total_intrinsic_residual += (
            intrinsic_residual
        )

        elapsed = (
            time.perf_counter()
            -
            q_t0
        )

        print(
            f"q={qi:03d} "
            f"O={O:02d} "
            f"S={S:02d} "
            f"Oracle2={oracle_S:02d} "
            f"G={O-S:02d} "
            f"expr={expr_headroom:02d} "
            f"intrinsic={intrinsic_residual:02d} "
            f"bestRoot={best_root:04d} "
            f"branchStates={len(cache):04d} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

        processed += 1

    runtime = (
        time.perf_counter()
        -
        t_all
    )

    print()
    print("=" * 132)
    print("ORACLE CONTINUATION CAPACITY SUMMARY")
    print("=" * 132)

    print(
        "residual queries processed =",
        processed,
    )

    print(
        "sum O / union S / oracle2 S =",
        total_O,
        total_S,
        total_oracle,
    )

    print(
        "union residual gap =",
        total_O - total_S,
    )

    print(
        "continuation expressivity headroom =",
        total_expr_headroom,
    )

    print(
        "root/depth/rank residual =",
        total_intrinsic_residual,
    )

    print(
        "queries oracle2 improves union =",
        oracle_improved,
    )

    print(
        "queries oracle2 closes full gap =",
        closed_by_oracle,
    )

    print(
        "decomposition check =",
        total_expr_headroom
        +
        total_intrinsic_residual,
    )

    assert (
        total_expr_headroom
        +
        total_intrinsic_residual
        ==
        total_O
        -
        total_S
    )

    print(
        "runtime seconds =",
        runtime,
    )

    print()
    print("ORACLE CONTINUATION DECOMPOSITION = PASS")

    print(
        "SCRIPT SHA256 =",
        sha256_file(
            Path(__file__)
        ),
    )

    print("=" * 132)


if __name__ == "__main__":
    main()
