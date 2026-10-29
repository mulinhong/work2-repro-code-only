from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import importlib.util
import json
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")
D = ROOT / "fhex_v1/obstruction_audit_v0"

UNION_SCRIPT = D / "audit_tecs_rcps_union_v0.py"
UNION_ART = D / "tecs_rcps_union_v0.npz"

ALL_OPT_SCRIPT = D / "audit_all_optimal_backbones_one_leaf_v0.py"
ALL_OPT_JSON = D / "all_optimal_backbones_one_leaf_v0_full329.json"

EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_ALL_OPT_SHA = (
    "a6ca6cfe7f48ed53155aac96b50b4dc8f4d622f2d4cf8dcaef6f34dffb12ebb0"
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
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def bool_to_bits(x):
    out = 0
    for i in np.flatnonzero(x):
        out |= 1 << int(i)
    return out


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-hard-queries",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="smoke10",
    )

    args = parser.parse_args()

    print("=" * 148)
    print("JOINT K<=5 UNION-PREDICATE EXACT POLICY AUDIT v0")
    print("MAX DEPTH=3 / LEAF BUDGET=5 / FIXED H_ind / EXACT DP")
    print("=" * 148)

    assert sha256_file(UNION_SCRIPT) == EXPECTED_UNION_SHA
    assert sha256_file(ALL_OPT_SCRIPT) == EXPECTED_ALL_OPT_SHA
    assert ALL_OPT_JSON.exists()

    print("FROZEN SCRIPT INTEGRITY = PASS")

    unionmod = load_module(
        UNION_SCRIPT,
        "jk5_union",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "jk5_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "jk5_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "jk5_tecs",
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

    allopt = json.loads(
        ALL_OPT_JSON.read_text(
            encoding="utf-8"
        )
    )

    hard = {}

    for rec in allopt["records"]:

        residual = int(
            rec[
                "residual_after_all_optimal_backbones"
            ]
        )

        if residual > 0:
            hard[
                int(
                    rec["query"]
                )
            ] = rec

    assert len(hard) == 82

    residual_sum = sum(
        int(
            x[
                "residual_after_all_optimal_backbones"
            ]
        )
        for x in hard.values()
    )

    assert residual_sum == 113

    print("HARD-CORE ANCHOR = PASS")
    print("hard queries =", len(hard))
    print("hard residual =", residual_sum)

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

    O_saved = ua["union_O"].astype(np.int16)
    S_saved = ua["union_S"].astype(np.int16)

    processed = 0

    sum_O = 0
    sum_S2 = 0
    sum_allopt_plus1 = 0
    sum_joint5 = 0

    joint_beats_allopt = 0
    joint_closes_O = 0

    total_joint_headroom = 0
    total_remaining = 0

    records = []

    t_all = time.perf_counter()

    for qi in sorted(hard):

        if (
            args.max_hard_queries > 0
            and
            processed >= args.max_hard_queries
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

        O = int(ind.sum())

        assert O == int(O_saved[qi])

        policy, s0, s1 = a4.choose_bcct(
            facts,
            valid,
        )

        shared = gap.eval_compound_policy_all(
            policy,
            facts,
            s0,
            s1,
            valid,
            a4,
        )

        S2 = int(
            np.sum(
                shared
                &
                ind
            )
        )

        assert S2 == int(S_saved[qi])

        allopt_plus1 = int(
            hard[qi][
                "optimal_backbone_arbitrary_plus1"
            ]
        )

        # ==========================================================
        # Bitset representation
        # ==========================================================

        P = facts.shape[1]

        full_mask = (1 << E) - 1
        valid_bits = bool_to_bits(valid)
        target_bits = bool_to_bits(ind)

        survive0_np, survive1_np = (
            a4.build_survival_masks(
                facts
            )
        )

        survive0 = [
            bool_to_bits(
                survive0_np[p]
            )
            for p in range(P)
        ]

        survive1 = [
            bool_to_bits(
                survive1_np[p]
            )
            for p in range(P)
        ]

        valid0 = []
        valid1 = []

        for p in range(P):

            valid0.append(
                bool_to_bits(
                    valid
                    &
                    (
                        facts[:, p]
                        ==
                        ABSENT
                    )
                )
            )

            valid1.append(
                bool_to_bits(
                    valid
                    &
                    (
                        facts[:, p]
                        ==
                        PRESENT
                    )
                )
            )

        target_ids = [
            int(x)
            for x in np.flatnonzero(ind)
        ]

        prefix_bits = [
            (1 << (t + 1)) - 1
            for t in range(E)
        ]

        # ----------------------------------------------------------
        # Exact STOP coverage
        # ----------------------------------------------------------

        @lru_cache(maxsize=None)
        def stop_cov(mask):

            c = 0

            alive_targets = (
                mask
                &
                target_bits
            )

            for t in target_ids:

                if not (
                    alive_targets
                    &
                    (
                        1 << t
                    )
                ):
                    continue

                rank = (
                    mask
                    &
                    prefix_bits[t]
                ).bit_count()

                if rank <= B:
                    c += 1

            return int(c)

        # ----------------------------------------------------------
        # Exact Bellman DP.
        #
        # depth = remaining questions allowed.
        # leaves = maximum leaves available to this subtree.
        #
        # V is "at most leaves", because recursive subtrees may STOP.
        # ----------------------------------------------------------

        choice = {}

        @lru_cache(maxsize=None)
        def V(mask, depth, leaves):

            base = stop_cov(mask)

            best = base
            best_choice = (
                "STOP",
                -1,
                -1,
            )

            if (
                depth <= 0
                or
                leaves <= 1
                or
                mask.bit_count() <= B
            ):
                choice[
                    (
                        mask,
                        depth,
                        leaves,
                    )
                ] = best_choice

                return int(best)

            for p in range(P):

                # Exact frozen legality:
                # both binary answers among valid survivors.
                if not (
                    mask
                    &
                    valid0[p]
                ):
                    continue

                if not (
                    mask
                    &
                    valid1[p]
                ):
                    continue

                m0 = (
                    mask
                    &
                    survive0[p]
                )

                m1 = (
                    mask
                    &
                    survive1[p]
                )

                # No semantic progress.
                if (
                    m0 == mask
                    or
                    m1 == mask
                ):
                    # Legal predicates should normally progress on
                    # valid candidates, but UNKNOWN overlap can keep
                    # many candidates. Exact equality is useless.
                    continue

                # Split the total leaf budget between children.
                for l0 in range(
                    1,
                    leaves
                ):

                    l1 = (
                        leaves
                        -
                        l0
                    )

                    cand = (
                        V(
                            m0,
                            depth - 1,
                            l0,
                        )
                        +
                        V(
                            m1,
                            depth - 1,
                            l1,
                        )
                    )

                    if cand > best:

                        best = int(cand)

                        best_choice = (
                            "SPLIT",
                            int(p),
                            int(l0),
                        )

                        # Cannot exceed all individually repairable
                        # targets in this subtree.
                        subtree_target_n = (
                            mask
                            &
                            target_bits
                        ).bit_count()

                        if best == subtree_target_n:
                            break

                if (
                    best
                    ==
                    (
                        mask
                        &
                        target_bits
                    ).bit_count()
                ):
                    break

            choice[
                (
                    mask,
                    depth,
                    leaves,
                )
            ] = best_choice

            return int(best)

        # ----------------------------------------------------------
        # Hard cross-parity:
        # max depth 2 / <=4 leaves must recover frozen UNION S.
        # ----------------------------------------------------------

        exact_S2 = V(
            full_mask,
            2,
            4,
        )

        assert exact_S2 == S2, (
            qi,
            exact_S2,
            S2,
        )

        joint5 = V(
            full_mask,
            3,
            5,
        )

        assert (
            S2
            <=
            joint5
            <=
            O
        )

        joint_headroom = int(
            joint5
            -
            allopt_plus1
        )

        remaining = int(
            O
            -
            joint5
        )

        # If positive, this is the important same-family separation:
        #
        # joint existing-predicate K5 >
        # ALL depth2-optimal UNION backbones + ARBITRARY q3.
        if joint_headroom > 0:
            joint_beats_allopt += 1

        if joint5 == O:
            joint_closes_O += 1

        sum_O += O
        sum_S2 += S2
        sum_allopt_plus1 += allopt_plus1
        sum_joint5 += joint5

        total_joint_headroom += (
            joint_headroom
        )

        total_remaining += (
            remaining
        )

        elapsed = (
            time.perf_counter()
            -
            q_t0
        )

        states = V.cache_info()

        records.append(
            {
                "query":
                    int(qi),

                "O":
                    O,

                "S2":
                    S2,

                "all_optimal_backbone_arbitrary_plus1":
                    allopt_plus1,

                "joint_union_K5":
                    int(joint5),

                "joint_headroom_over_all_optimal_backbones":
                    int(joint_headroom),

                "remaining_to_O":
                    int(remaining),

                "dp_cache_hits":
                    int(states.hits),

                "dp_cache_misses":
                    int(states.misses),

                "runtime_seconds":
                    float(elapsed),
            }
        )

        processed += 1

        print(
            f"q={qi:03d} "
            f"O={O:02d} "
            f"S2={S2:02d} "
            f"OptBackArb={allopt_plus1:02d} "
            f"JointK5={joint5:02d} "
            f"jointHead={joint_headroom:+02d} "
            f"remain={remaining:02d} "
            f"states={states.misses} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    runtime = time.perf_counter() - t_all

    print()
    print("=" * 148)
    print("JOINT K<=5 UNION EXACT SUMMARY")
    print("=" * 148)

    print(
        "hard queries processed =",
        processed,
    )

    print(
        "sum O / S2 / AllOptBackboneArb+1 / JointUnionK5 =",
        sum_O,
        sum_S2,
        sum_allopt_plus1,
        sum_joint5,
    )

    print(
        "joint K5 headroom over ALL depth2-optimal backbones + arbitrary q3 =",
        total_joint_headroom,
    )

    print(
        "remaining gap after exact JointUnionK5 =",
        total_remaining,
    )

    print(
        "queries JointUnionK5 beats all optimal backbones =",
        joint_beats_allopt,
    )

    print(
        "queries JointUnionK5 reaches O =",
        joint_closes_O,
    )

    print(
        "runtime seconds =",
        runtime,
    )

    OUT = (
        D
        /
        f"joint_k5_union_exact_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "name":
                    "Joint K<=5 UNION-Predicate Exact Policy Audit v0",

                "processed":
                    int(processed),

                "sum_O":
                    int(sum_O),

                "sum_S2":
                    int(sum_S2),

                "sum_all_optimal_backbone_arbitrary_plus1":
                    int(sum_allopt_plus1),

                "sum_joint_union_K5":
                    int(sum_joint5),

                "joint_headroom":
                    int(total_joint_headroom),

                "remaining_gap":
                    int(total_remaining),

                "queries_joint_beats_allopt":
                    int(joint_beats_allopt),

                "queries_joint_reaches_O":
                    int(joint_closes_O),

                "records":
                    records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("JSON =", OUT)

    print(
        "JOINT K5 EXACT DP = PASS"
    )

    print(
        "SCRIPT SHA256 =",
        sha256_file(
            Path(__file__)
        ),
    )

    print("=" * 148)


if __name__ == "__main__":
    main()
