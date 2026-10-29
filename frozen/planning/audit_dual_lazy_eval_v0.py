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
THREEWAY_SCRIPT = D / "audit_dual_bound_threeway_v0.py"
DQ_FULL_JSON = D / "rpc_depth_quota_bound_v0_full82.json"

EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)

EXPECTED_THREEWAY_SHA = (
    "541fb6ad3cf8036c69030b7fb6249e5e1a973fdf6f5a14cba9685d429608b826"
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
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

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

    print("=" * 170)
    print("DUAL BOUND EXACT LAZY-EVALUATION AUDIT v0")
    print("EAGER RPC+DQ vs EXACT LAZY/CLIPPED RPC+DQ")
    print("SAME MATHEMATICAL BOUND / SAME RECURRENCE / SAME SEARCH ORDER")
    print("=" * 170)

    assert sha256_file(
        UNION_SCRIPT
    ) == EXPECTED_UNION_SHA

    assert sha256_file(
        THREEWAY_SCRIPT
    ) == EXPECTED_THREEWAY_SHA

    assert DQ_FULL_JSON.exists()

    print(
        "FROZEN SCRIPT INTEGRITY = PASS"
    )

    unionmod = load_module(
        UNION_SCRIPT,
        "lazy_union",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "lazy_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "lazy_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "lazy_tecs",
    )

    a2 = np.load(
        unionmod.A2_ART,
        allow_pickle=True,
    )

    az = np.load(
        unionmod.AST_CACHE,
        allow_pickle=True,
    )

    frozen = json.loads(
        DQ_FULL_JSON.read_text(
            encoding="utf-8"
        )
    )

    frozen_records = {
        int(r["query"]): r
        for r in frozen["records"]
    }

    query_ids = sorted(
        frozen_records
    )

    assert len(query_ids) == 82

    candidates_all = (
        a2["candidates"].astype(
            np.int32
        )
    )

    valid_all = (
        a2["candidate_ast_valid"].astype(
            bool
        )
    )

    names_all = (
        az["feature_names"].astype(str)
    )

    signature = (
        az["signature"].astype(
            np.float32
        )
    )

    allowed_ids = np.flatnonzero(
        names_all != "max_ast_depth"
    )

    assert len(allowed_ids) == 44

    aggregate = {
        "EAGER": {
            "states": 0,
            "branches": 0,
            "pruned": 0,
            "runtime": 0.0,
            "dq_calls": 0,
            "dq_target_tests": 0,
        },

        "LAZY": {
            "states": 0,
            "branches": 0,
            "pruned": 0,
            "runtime": 0.0,
            "dq_calls": 0,
            "dq_target_tests": 0,
            "rpc_only_branch_prunes": 0,
            "one_child_shortcuts": 0,
            "dq_cap_shortcuts": 0,
        },
    }

    out_records = []

    processed = 0

    for qi in query_ids:

        if (
            args.max_hard_queries > 0
            and
            processed >= args.max_hard_queries
        ):
            break

        expected_joint = int(
            frozen_records[qi][
                "joint_K5"
            ]
        )

        candidates = candidates_all[qi]
        valid = valid_all[qi]

        atomic = np.full(
            (E, 44),
            UNKNOWN,
            dtype=np.int8,
        )

        presence = (
            signature[candidates][
                :,
                allowed_ids
            ]
            > 0.0
        ).astype(np.int8)

        atomic[valid] = presence[valid]

        built = tecs.build_tecs_library(
            atomic,
            valid,
            a4,
        )

        lib_tecs = built[
            "library"
        ]

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

        P = int(
            facts.shape[1]
        )

        ind = gap.individual_b20_vector(
            facts,
            valid,
            a4,
        )

        full_mask = (
            (1 << E) - 1
        )

        valid_bits = bool_to_bits(
            valid
        )

        target_bits = bool_to_bits(
            ind
        )

        target_ids = [
            int(x)
            for x in np.flatnonzero(
                ind
            )
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

        def run(mode):

            assert mode in (
                "EAGER",
                "LAZY",
            )

            stats = {
                "states": 0,
                "branches": 0,
                "pruned": 0,
                "dq_calls": 0,
                "dq_target_tests": 0,
                "rpc_only_branch_prunes": 0,
                "one_child_shortcuts": 0,
                "dq_cap_shortcuts": 0,
            }

            @lru_cache(
                maxsize=None
            )
            def stop_cov(mask):

                reward = 0

                alive = (
                    mask
                    &
                    target_bits
                )

                for t in target_ids:

                    bit = 1 << t

                    if not (
                        alive & bit
                    ):
                        continue

                    rank = (
                        mask
                        &
                        prefix_bits[t]
                    ).bit_count()

                    if rank <= B:
                        reward += 1

                return int(reward)

            @lru_cache(
                maxsize=None
            )
            def rpc_capacity(
                mask,
                leaves,
            ):

                slack = [
                    B
                    for _ in range(
                        int(leaves)
                    )
                ]

                reward = 0

                for i in range(E):

                    bit = 1 << i

                    if not (
                        mask & bit
                    ):
                        continue

                    if not (
                        valid_bits & bit
                    ):

                        for j in range(
                            len(slack)
                        ):

                            if slack[j] > 0:
                                slack[j] -= 1

                        continue

                    if target_bits & bit:

                        chosen = -1

                        for j, s in enumerate(
                            slack
                        ):

                            if s > 0:
                                chosen = j
                                break

                        if chosen >= 0:

                            reward += 1
                            slack[chosen] -= 1

                        continue

                    if slack[0] > 0:
                        slack[0] -= 1

                return int(reward)

            # ------------------------------------------------------
            # Full DQ Q.
            # ------------------------------------------------------

            @lru_cache(
                maxsize=None
            )
            def dq_full(
                mask,
                depth,
                leaves,
            ):

                stats[
                    "dq_calls"
                ] += 1

                k = min(
                    int(depth),
                    int(leaves) - 1,
                )

                if k <= 0:
                    return stop_cov(mask)

                feasible = 0

                alive = (
                    mask
                    &
                    target_bits
                )

                for t in target_ids:

                    bit = 1 << t

                    if not (
                        alive & bit
                    ):
                        continue

                    rank = (
                        mask
                        &
                        prefix_bits[t]
                    ).bit_count()

                    if rank <= B:

                        feasible += 1
                        continue

                    stats[
                        "dq_target_tests"
                    ] += 1

                    delta = rank - B

                    blockers = (
                        mask
                        &
                        prefix_bits[t]
                    )

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

                        opposite = (
                            valid1[p]
                            if ans == ABSENT
                            else valid0[p]
                        )

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

                        if top_sum >= delta:

                            possible = True
                            break

                    if possible:
                        feasible += 1

                return int(feasible)

            # ------------------------------------------------------
            # Exact clipped dual:
            #
            # Returns min(C,Q), but stops DQ once feasible >= C.
            # ------------------------------------------------------

            @lru_cache(
                maxsize=None
            )
            def dual_clipped(
                mask,
                depth,
                leaves,
            ):

                C = rpc_capacity(
                    mask,
                    leaves,
                )

                base = stop_cov(mask)

                if base >= C:

                    stats[
                        "dq_cap_shortcuts"
                    ] += 1

                    return int(C)

                k = min(
                    int(depth),
                    int(leaves) - 1,
                )

                if k <= 0:
                    return int(base)

                stats[
                    "dq_calls"
                ] += 1

                feasible = int(base)

                alive = (
                    mask
                    &
                    target_bits
                )

                for t in target_ids:

                    bit = 1 << t

                    if not (
                        alive & bit
                    ):
                        continue

                    rank = (
                        mask
                        &
                        prefix_bits[t]
                    ).bit_count()

                    if rank <= B:
                        continue

                    stats[
                        "dq_target_tests"
                    ] += 1

                    delta = rank - B

                    blockers = (
                        mask
                        &
                        prefix_bits[t]
                    )

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

                        opposite = (
                            valid1[p]
                            if ans == ABSENT
                            else valid0[p]
                        )

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

                        if top_sum >= delta:

                            possible = True
                            break

                    if possible:

                        feasible += 1

                        if feasible >= C:

                            stats[
                                "dq_cap_shortcuts"
                            ] += 1

                            return int(C)

                return int(feasible)

            def eager_bound(
                mask,
                depth,
                leaves,
            ):

                C = rpc_capacity(
                    mask,
                    leaves,
                )

                Q = dq_full(
                    mask,
                    depth,
                    leaves,
                )

                return min(
                    C,
                    Q,
                )

            def lazy_bound(
                mask,
                depth,
                leaves,
            ):

                return dual_clipped(
                    mask,
                    depth,
                    leaves,
                )

            @lru_cache(
                maxsize=None
            )
            def V(
                mask,
                depth,
                leaves,
            ):

                stats[
                    "states"
                ] += 1

                base = stop_cov(mask)

                if mode == "EAGER":

                    ub_state = eager_bound(
                        mask,
                        depth,
                        leaves,
                    )

                else:

                    ub_state = lazy_bound(
                        mask,
                        depth,
                        leaves,
                    )

                assert base <= ub_state

                if ub_state == base:
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
                        mask & valid0[p]
                    ):
                        continue

                    if not (
                        mask & valid1[p]
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
                            leaves - l0
                        )

                        stats[
                            "branches"
                        ] += 1

                        if mode == "EAGER":

                            ub0 = eager_bound(
                                m0,
                                depth - 1,
                                l0,
                            )

                            ub1 = eager_bound(
                                m1,
                                depth - 1,
                                l1,
                            )

                            if (
                                ub0 + ub1
                                <= best
                            ):

                                stats[
                                    "pruned"
                                ] += 1

                                continue

                        else:

                            # --------------------------------------
                            # First, cheap RPC-only test.
                            # --------------------------------------

                            C0 = rpc_capacity(
                                m0,
                                l0,
                            )

                            C1 = rpc_capacity(
                                m1,
                                l1,
                            )

                            if (
                                C0 + C1
                                <= best
                            ):

                                stats[
                                    "pruned"
                                ] += 1

                                stats[
                                    "rpc_only_branch_prunes"
                                ] += 1

                                continue

                            # --------------------------------------
                            # Evaluate only child 0 DQ.
                            # --------------------------------------

                            ub0 = lazy_bound(
                                m0,
                                depth - 1,
                                l0,
                            )

                            if (
                                ub0 + C1
                                <= best
                            ):

                                stats[
                                    "pruned"
                                ] += 1

                                stats[
                                    "one_child_shortcuts"
                                ] += 1

                                continue

                            # --------------------------------------
                            # Only now evaluate child 1 DQ.
                            # --------------------------------------

                            ub1 = lazy_bound(
                                m1,
                                depth - 1,
                                l1,
                            )

                            if (
                                ub0 + ub1
                                <= best
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

            stats[
                "runtime"
            ] = float(runtime)

            stats[
                "value"
            ] = int(value)

            return stats

        eager = run(
            "EAGER"
        )

        lazy = run(
            "LAZY"
        )

        assert (
            eager["value"]
            ==
            expected_joint
        )

        assert (
            lazy["value"]
            ==
            expected_joint
        )

        # Exact same search result and search tree are expected.
        assert (
            eager["states"]
            ==
            lazy["states"]
        )

        assert (
            eager["branches"]
            ==
            lazy["branches"]
        )

        assert (
            eager["pruned"]
            ==
            lazy["pruned"]
        )

        expected_states = int(
            frozen_records[qi][
                "rpc_dq"
            ][
                "states"
            ]
        )

        assert (
            eager["states"]
            ==
            expected_states
        )

        speedup = (
            eager["runtime"]
            /
            max(
                lazy["runtime"],
                1e-12,
            )
        )

        print(
            f"q={qi:03d} "
            f"J5={expected_joint:02d} "
            f"states={eager['states']} "
            f"dqCalls "
            f"{eager['dq_calls']}->{lazy['dq_calls']} "
            f"targetTests "
            f"{eager['dq_target_tests']}->{lazy['dq_target_tests']} "
            f"RPCprune={lazy['rpc_only_branch_prunes']} "
            f"oneChild={lazy['one_child_shortcuts']} "
            f"capShort={lazy['dq_cap_shortcuts']} "
            f"time "
            f"{eager['runtime']:.3f}s->"
            f"{lazy['runtime']:.3f}s "
            f"speedup={speedup:.2f}x",
            flush=True,
        )

        for key, obj in (
            ("EAGER", eager),
            ("LAZY", lazy),
        ):

            for field in aggregate[key]:

                if field == "runtime":
                    aggregate[key][field] += float(
                        obj[field]
                    )

                else:
                    aggregate[key][field] += int(
                        obj.get(
                            field,
                            0,
                        )
                    )

        out_records.append(
            {
                "query": int(qi),
                "eager": eager,
                "lazy": lazy,
                "speedup": float(
                    speedup
                ),
            }
        )

        processed += 1

    E0 = aggregate["EAGER"]
    L0 = aggregate["LAZY"]

    speedup = (
        E0["runtime"]
        /
        max(
            L0["runtime"],
            1e-12,
        )
    )

    print()
    print("=" * 170)
    print("EXACT LAZY DUAL SUMMARY")
    print("=" * 170)

    print(
        "hard queries processed =",
        processed,
    )

    print(
        "states eager/lazy =",
        E0["states"],
        L0["states"],
    )

    print(
        "branches eager/lazy =",
        E0["branches"],
        L0["branches"],
    )

    print(
        "pruned eager/lazy =",
        E0["pruned"],
        L0["pruned"],
    )

    print(
        "DQ calls eager/lazy =",
        E0["dq_calls"],
        L0["dq_calls"],
    )

    print(
        "DQ target tests eager/lazy =",
        E0["dq_target_tests"],
        L0["dq_target_tests"],
    )

    print(
        "lazy RPC-only branch prunes =",
        L0[
            "rpc_only_branch_prunes"
        ],
    )

    print(
        "lazy one-child DQ shortcuts =",
        L0[
            "one_child_shortcuts"
        ],
    )

    print(
        "lazy clipped-cap shortcuts =",
        L0[
            "dq_cap_shortcuts"
        ],
    )

    print(
        "runtime eager/lazy =",
        E0["runtime"],
        L0["runtime"],
    )

    print(
        "lazy runtime speedup =",
        speedup,
    )

    print(
        "EXACT VALUE PARITY = PASS"
    )

    print(
        "EXACT SEARCH-TREE PARITY = PASS"
    )

    OUT = (
        D
        /
        f"dual_lazy_eval_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "processed":
                    processed,

                "aggregate":
                    aggregate,

                "speedup":
                    speedup,

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

    print("=" * 170)


if __name__ == "__main__":
    main()
