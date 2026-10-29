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
JOINT_SCRIPT = D / "audit_joint_k5_union_exact_v0.py"
RPC_SCRIPT = D / "audit_rpc_capacity_bound_value_v0.py"

RPC_FULL_JSON = D / "rpc_capacity_bound_value_v0_full82.json"

EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_JOINT_SHA = (
    "33d8a037da621e3213f94e8578a7bdb4d9e87483d5e9ef06f1eab885c0136d55"
)

EXPECTED_RPC_SHA = (
    "4da6e382b26b35506f8de2ccd6d107aed4a3e5918b8f26fa16d254ac73aa380a"
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

    print("=" * 158)
    print("RPC + DEPTH-QUOTA REACHABILITY BOUND AUDIT v0")
    print("RPC ONLY vs RPC + TARGET-SPECIFIC FINITE-DEPTH QUOTA BOUND")
    print("SAME EXACT JOINT K<=5 SOLVER / SAME SEARCH ORDER")
    print("=" * 158)

    assert sha256_file(UNION_SCRIPT) == EXPECTED_UNION_SHA
    assert sha256_file(JOINT_SCRIPT) == EXPECTED_JOINT_SHA
    assert sha256_file(RPC_SCRIPT) == EXPECTED_RPC_SHA
    assert RPC_FULL_JSON.exists()

    print("FROZEN SCRIPT INTEGRITY = PASS")

    unionmod = load_module(
        UNION_SCRIPT,
        "rqd_union",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "rqd_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "rqd_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "rqd_tecs",
    )

    a2 = np.load(
        unionmod.A2_ART,
        allow_pickle=True,
    )

    az = np.load(
        unionmod.AST_CACHE,
        allow_pickle=True,
    )

    frozen_rpc = json.loads(
        RPC_FULL_JSON.read_text(
            encoding="utf-8"
        )
    )

    frozen_records = {
        int(r["query"]): r
        for r in frozen_rpc["records"]
    }

    query_ids = sorted(frozen_records)

    assert len(query_ids) == 82

    candidates_all = a2["candidates"].astype(np.int32)
    valid_all = a2["candidate_ast_valid"].astype(bool)

    names_all = az["feature_names"].astype(str)
    signature = az["signature"].astype(np.float32)

    allowed_ids = np.flatnonzero(
        names_all != "max_ast_depth"
    )

    assert len(allowed_ids) == 44

    aggregate = {
        "rpc_states": 0,
        "dq_states": 0,

        "rpc_branches": 0,
        "dq_branches": 0,

        "rpc_pruned": 0,
        "dq_pruned": 0,

        "rpc_runtime": 0.0,
        "dq_runtime": 0.0,

        "dq_calls": 0,
        "dq_tightened_states": 0,

        "dq_targets_tested": 0,
        "dq_targets_proven_impossible": 0,
    }

    out_records = []

    processed = 0
    all_start = time.perf_counter()

    for qi in query_ids:

        if (
            args.max_hard_queries > 0
            and
            processed >= args.max_hard_queries
        ):
            break

        frozen = frozen_records[qi]

        expected_joint = int(
            frozen["joint_K5"]
        )

        candidates = candidates_all[qi]
        valid = valid_all[qi]

        # ==========================================================
        # Reconstruct exact UNION predicate family.
        # ==========================================================

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

        P = int(facts.shape[1])

        ind = gap.individual_b20_vector(
            facts,
            valid,
            a4,
        )

        # ==========================================================
        # Bitsets.
        # ==========================================================

        full_mask = (1 << E) - 1

        valid_bits = bool_to_bits(valid)
        target_bits = bool_to_bits(ind)

        target_ids = [
            int(x)
            for x in np.flatnonzero(ind)
        ]

        prefix_bits = [
            (1 << (t + 1)) - 1
            for t in range(E)
        ]

        survive0_np, survive1_np = (
            a4.build_survival_masks(
                facts
            )
        )

        survive0 = [
            bool_to_bits(survive0_np[p])
            for p in range(P)
        ]

        survive1 = [
            bool_to_bits(survive1_np[p])
            for p in range(P)
        ]

        valid0 = [
            bool_to_bits(
                valid
                &
                (
                    facts[:, p]
                    ==
                    ABSENT
                )
            )
            for p in range(P)
        ]

        valid1 = [
            bool_to_bits(
                valid
                &
                (
                    facts[:, p]
                    ==
                    PRESENT
                )
            )
            for p in range(P)
        ]

        # Valid targets must have definite answers.
        for t in target_ids:
            assert np.all(
                (
                    facts[t] == ABSENT
                )
                |
                (
                    facts[t] == PRESENT
                )
            )

        # ==========================================================
        # STOP.
        # ==========================================================

        @lru_cache(maxsize=None)
        def stop_cov(mask):

            alive = mask & target_bits
            reward = 0

            for t in target_ids:

                bit = 1 << t

                if not (alive & bit):
                    continue

                rank = (
                    mask
                    &
                    prefix_bits[t]
                ).bit_count()

                if rank <= B:
                    reward += 1

            return int(reward)

        # ==========================================================
        # Frozen RPC capacity bound.
        # ==========================================================

        @lru_cache(maxsize=None)
        def rpc_capacity(mask, leaves):

            leaves = int(leaves)

            slack = [
                B
                for _ in range(leaves)
            ]

            reward = 0

            for i in range(E):

                bit = 1 << i

                if not (mask & bit):
                    continue

                # UNKNOWN / invalid:
                if not (valid_bits & bit):

                    for j in range(leaves):

                        if slack[j] > 0:
                            slack[j] -= 1

                    continue

                # Valid target:
                if target_bits & bit:

                    chosen = -1

                    for j in range(leaves):

                        if slack[j] > 0:
                            chosen = j
                            break

                    if chosen >= 0:
                        reward += 1
                        slack[chosen] -= 1

                    continue

                # Valid non-target:
                if slack[0] > 0:
                    slack[0] -= 1

            return int(reward)

        # ==========================================================
        # Depth-Quota Reachability bound.
        #
        # For each target:
        #
        #   delta = current_rank - B
        #
        # Each future predicate p can eliminate at most sigma_p
        # currently surviving prefix blockers from t's true path.
        #
        # Over k remaining questions:
        #
        #   distinct eliminated blockers
        #       <= sum of k largest sigma_p
        #
        # because overlap is deliberately ignored.
        #
        # If even this optimistic sum < delta, t is impossible.
        #
        # Predicate legality/sharedness are also ignored:
        # therefore this remains an admissible relaxation.
        # ==========================================================

        dq_stats = {
            "calls": 0,
            "tightened": 0,
            "targets_tested": 0,
            "targets_impossible": 0,
        }

        @lru_cache(maxsize=None)
        def dq_reachable_bound(
            mask,
            depth,
            leaves,
        ):

            dq_stats["calls"] += 1

            k = min(
                int(depth),
                int(leaves) - 1,
            )

            if k <= 0:
                return stop_cov(mask)

            alive = mask & target_bits

            feasible = 0

            for t in target_ids:

                bit = 1 << t

                if not (alive & bit):
                    continue

                rank = (
                    mask
                    &
                    prefix_bits[t]
                ).bit_count()

                if rank <= B:
                    feasible += 1
                    continue

                dq_stats[
                    "targets_tested"
                ] += 1

                delta = int(
                    rank - B
                )

                blockers = (
                    mask
                    &
                    prefix_bits[t]
                )

                # Maintain the k largest individual sigma values.
                top = [
                    0
                    for _ in range(k)
                ]

                top_sum = 0

                possible = False

                for p in range(P):

                    ans = int(
                        facts[t, p]
                    )

                    if ans == ABSENT:
                        opposite = valid1[p]
                    else:
                        opposite = valid0[p]

                    sigma = (
                        blockers
                        &
                        opposite
                    ).bit_count()

                    if sigma <= top[0]:
                        continue

                    top_sum += (
                        sigma
                        -
                        top[0]
                    )

                    top[0] = sigma
                    top.sort()

                    # Once the optimistic upper bound already
                    # reaches the quota, this target may be
                    # feasible; no need to inspect more predicates.
                    if top_sum >= delta:
                        possible = True
                        break

                if possible:
                    feasible += 1
                else:
                    dq_stats[
                        "targets_impossible"
                    ] += 1

            return int(feasible)

        # ==========================================================
        # Exact solver.
        #
        # mode = RPC
        # mode = RPC_DQ
        #
        # Search order / recurrence are identical.
        # ==========================================================

        def run_solver(mode):

            assert mode in (
                "RPC",
                "RPC_DQ",
            )

            stats = {
                "states": 0,
                "branch_options": 0,
                "pruned": 0,
                "state_bound_stops": 0,
            }

            @lru_cache(maxsize=None)
            def V(mask, depth, leaves):

                stats["states"] += 1

                base = stop_cov(mask)

                rpc_ub = rpc_capacity(
                    mask,
                    leaves,
                )

                if mode == "RPC":

                    ub_state = rpc_ub

                else:

                    dq_ub = dq_reachable_bound(
                        mask,
                        depth,
                        leaves,
                    )

                    ub_state = min(
                        rpc_ub,
                        dq_ub,
                    )

                    if ub_state < rpc_ub:
                        dq_stats[
                            "tightened"
                        ] += 1

                assert base <= ub_state

                if ub_state == base:

                    stats[
                        "state_bound_stops"
                    ] += 1

                    return int(base)

                if (
                    depth <= 0
                    or
                    leaves <= 1
                    or
                    mask.bit_count() <= B
                ):

                    return int(base)

                best = int(base)

                for p in range(P):

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

                    if (
                        m0 == mask
                        or
                        m1 == mask
                    ):
                        continue

                    for l0 in range(
                        1,
                        leaves
                    ):

                        l1 = (
                            leaves
                            -
                            l0
                        )

                        stats[
                            "branch_options"
                        ] += 1

                        ub0_rpc = rpc_capacity(
                            m0,
                            l0,
                        )

                        ub1_rpc = rpc_capacity(
                            m1,
                            l1,
                        )

                        if mode == "RPC":

                            ub0 = ub0_rpc
                            ub1 = ub1_rpc

                        else:

                            ub0 = min(
                                ub0_rpc,
                                dq_reachable_bound(
                                    m0,
                                    depth - 1,
                                    l0,
                                ),
                            )

                            ub1 = min(
                                ub1_rpc,
                                dq_reachable_bound(
                                    m1,
                                    depth - 1,
                                    l1,
                                ),
                            )

                        if (
                            ub0
                            +
                            ub1
                            <=
                            best
                        ):

                            stats[
                                "pruned"
                            ] += 1

                            continue

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

                        if best == ub_state:
                            break

                    if best == ub_state:
                        break

                return int(best)

            t0 = time.perf_counter()

            value = V(
                full_mask,
                3,
                5,
            )

            runtime = (
                time.perf_counter()
                -
                t0
            )

            info = V.cache_info()

            stats[
                "cache_hits"
            ] = int(info.hits)

            stats[
                "cache_misses"
            ] = int(info.misses)

            stats[
                "runtime"
            ] = float(runtime)

            stats[
                "value"
            ] = int(value)

            return stats

        # ==========================================================
        # Run frozen RPC first.
        # ==========================================================

        rpc = run_solver(
            "RPC"
        )

        assert (
            rpc["value"]
            ==
            expected_joint
        )

        # Save DQ stats before second run for clean deltas.
        for k0 in dq_stats:
            dq_stats[k0] = 0

        rpc_dq = run_solver(
            "RPC_DQ"
        )

        assert (
            rpc_dq["value"]
            ==
            expected_joint
        )

        assert (
            rpc["value"]
            ==
            rpc_dq["value"]
        )

        state_reduction = (
            1.0
            -
            (
                rpc_dq["states"]
                /
                rpc["states"]
            )
        )

        speedup = (
            rpc["runtime"]
            /
            max(
                rpc_dq["runtime"],
                1e-12,
            )
        )

        print(
            f"q={qi:03d} "
            f"J5={expected_joint:02d} "
            f"states "
            f"{rpc['states']}->{rpc_dq['states']} "
            f"extraRed={100.0*state_reduction:.2f}% "
            f"pruned "
            f"{rpc['pruned']}->{rpc_dq['pruned']} "
            f"DQtight={dq_stats['tightened']} "
            f"DQimpossible={dq_stats['targets_impossible']} "
            f"time "
            f"{rpc['runtime']:.3f}s->"
            f"{rpc_dq['runtime']:.3f}s "
            f"speedup={speedup:.2f}x",
            flush=True,
        )

        aggregate[
            "rpc_states"
        ] += rpc["states"]

        aggregate[
            "dq_states"
        ] += rpc_dq["states"]

        aggregate[
            "rpc_branches"
        ] += rpc["branch_options"]

        aggregate[
            "dq_branches"
        ] += rpc_dq["branch_options"]

        aggregate[
            "rpc_pruned"
        ] += rpc["pruned"]

        aggregate[
            "dq_pruned"
        ] += rpc_dq["pruned"]

        aggregate[
            "rpc_runtime"
        ] += rpc["runtime"]

        aggregate[
            "dq_runtime"
        ] += rpc_dq["runtime"]

        aggregate[
            "dq_calls"
        ] += dq_stats["calls"]

        aggregate[
            "dq_tightened_states"
        ] += dq_stats["tightened"]

        aggregate[
            "dq_targets_tested"
        ] += dq_stats["targets_tested"]

        aggregate[
            "dq_targets_proven_impossible"
        ] += dq_stats["targets_impossible"]

        out_records.append(
            {
                "query":
                    int(qi),

                "joint_K5":
                    int(expected_joint),

                "rpc":
                    rpc,

                "rpc_dq":
                    rpc_dq,

                "additional_state_reduction":
                    float(state_reduction),

                "runtime_speedup":
                    float(speedup),

                "dq_stats":
                    dict(dq_stats),
            }
        )

        processed += 1

    # ==============================================================
    # Summary.
    # ==========================================================

    state_reduction = (
        1.0
        -
        (
            aggregate["dq_states"]
            /
            aggregate["rpc_states"]
        )
    )

    branch_reduction = (
        1.0
        -
        (
            aggregate["dq_branches"]
            /
            aggregate["rpc_branches"]
        )
    )

    speedup = (
        aggregate["rpc_runtime"]
        /
        max(
            aggregate["dq_runtime"],
            1e-12,
        )
    )

    print()
    print("=" * 158)
    print(
        "RPC + DEPTH-QUOTA SUMMARY"
    )
    print("=" * 158)

    print(
        "hard queries processed =",
        processed,
    )

    print(
        "RPC states =",
        aggregate["rpc_states"],
    )

    print(
        "RPC+DQ states =",
        aggregate["dq_states"],
    )

    print(
        "additional state reduction fraction =",
        state_reduction,
    )

    print(
        "RPC branch options =",
        aggregate["rpc_branches"],
    )

    print(
        "RPC+DQ branch options =",
        aggregate["dq_branches"],
    )

    print(
        "additional branch-option reduction fraction =",
        branch_reduction,
    )

    print(
        "RPC pruned =",
        aggregate["rpc_pruned"],
    )

    print(
        "RPC+DQ pruned =",
        aggregate["dq_pruned"],
    )

    print(
        "DQ bound calls =",
        aggregate["dq_calls"],
    )

    print(
        "states where DQ tightened RPC =",
        aggregate["dq_tightened_states"],
    )

    print(
        "targets tested by DQ =",
        aggregate["dq_targets_tested"],
    )

    print(
        "targets certified impossible by DQ =",
        aggregate[
            "dq_targets_proven_impossible"
        ],
    )

    print(
        "RPC runtime seconds =",
        aggregate["rpc_runtime"],
    )

    print(
        "RPC+DQ runtime seconds =",
        aggregate["dq_runtime"],
    )

    print(
        "incremental runtime speedup =",
        speedup,
    )

    print(
        "EXACT VALUE PARITY = PASS"
    )

    OUT = (
        D
        /
        f"rpc_depth_quota_bound_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "name":
                    "RPC + Depth-Quota Reachability Bound Audit v0",

                "processed":
                    int(processed),

                "aggregate":
                    aggregate,

                "additional_state_reduction_fraction":
                    float(state_reduction),

                "additional_branch_reduction_fraction":
                    float(branch_reduction),

                "runtime_speedup":
                    float(speedup),

                "records":
                    out_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "JSON =",
        OUT,
    )

    print(
        "SCRIPT SHA256 =",
        sha256_file(
            Path(__file__)
        ),
    )

    print("=" * 158)


if __name__ == "__main__":
    main()
