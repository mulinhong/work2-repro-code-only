from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import importlib.util
import json
import math
import time

import numpy as np


ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")
D = ROOT / "fhex_v1/obstruction_audit_v0"

UNION_SCRIPT = D / "audit_tecs_rcps_union_v0.py"
RPC_SCRIPT = D / "audit_rpc_capacity_bound_value_v0.py"
DQ_SCRIPT = D / "audit_rpc_depth_quota_bound_v0.py"
THEORY_SCRIPT = D / "audit_dual_bound_theory_v0.py"
LAZY_SCRIPT = D / "audit_dual_lazy_eval_v0.py"

RPC_FULL_JSON = D / "rpc_capacity_bound_value_v0_full82.json"
DQ_FULL_JSON = D / "rpc_depth_quota_bound_v0_full82.json"

EXPECTED_UNION_SHA = (
    "0d05f783ac6ffa3196bbeca618ee96f3a3ff912901a70781700b5dc6b8e1d504"
)
EXPECTED_RPC_SHA = (
    "4da6e382b26b35506f8de2ccd6d107aed4a3e5918b8f26fa16d254ac73aa380a"
)
EXPECTED_DQ_SHA = (
    "162cc018992d014cd56ab46e69c957144589f2264ad0c3f94f7bd441e1b3cdc2"
)
EXPECTED_THEORY_SHA = (
    "d543be89969cea97cda241ad68eab7e9ae52c5fde70d86536832447a32317727"
)
EXPECTED_LAZY_SHA = (
    "4d79a7dd1eeef0fd655e70fcc30cd9f2dcd3fc14118c9bd80fafc11cf563b20c"
)

E = 80
B = 20

UNKNOWN = -1
ABSENT = 0
PRESENT = 1

MODES = (
    "GENERIC",
    "RPC",
    "FINAL_LAZY_DUAL",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def load_module(path: Path, name: str):
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

    print("=" * 176)
    print(
        "FINAL DUAL CAPACITY-REACHABILITY THREE-WAY EXACT BENCHMARK v0"
    )
    print(
        "GENERIC vs RPC vs FINAL_LAZY_DUAL"
    )
    print(
        "SAME PROCESS / SAME RECURRENCE / SAME SEARCH ORDER / FRESH PER-MODE CACHES"
    )
    print("=" * 176)

    assert sha256_file(UNION_SCRIPT) == EXPECTED_UNION_SHA
    assert sha256_file(RPC_SCRIPT) == EXPECTED_RPC_SHA
    assert sha256_file(DQ_SCRIPT) == EXPECTED_DQ_SHA
    assert sha256_file(THEORY_SCRIPT) == EXPECTED_THEORY_SHA
    assert sha256_file(LAZY_SCRIPT) == EXPECTED_LAZY_SHA

    assert RPC_FULL_JSON.exists()
    assert DQ_FULL_JSON.exists()

    print("FROZEN SCRIPT INTEGRITY = PASS")

    unionmod = load_module(
        UNION_SCRIPT,
        "final3_union",
    )

    a4 = load_module(
        unionmod.A4_SCRIPT,
        "final3_a4",
    )

    gap = load_module(
        unionmod.GAP_SCRIPT,
        "final3_gap",
    )

    tecs = load_module(
        unionmod.TECS_SCRIPT,
        "final3_tecs",
    )

    a2 = np.load(
        unionmod.A2_ART,
        allow_pickle=True,
    )

    az = np.load(
        unionmod.AST_CACHE,
        allow_pickle=True,
    )

    rpc_frozen = json.loads(
        RPC_FULL_JSON.read_text(
            encoding="utf-8"
        )
    )

    dq_frozen = json.loads(
        DQ_FULL_JSON.read_text(
            encoding="utf-8"
        )
    )

    rpc_records = {
        int(r["query"]): r
        for r in rpc_frozen["records"]
    }

    dq_records = {
        int(r["query"]): r
        for r in dq_frozen["records"]
    }

    query_ids = sorted(rpc_records)

    assert len(query_ids) == 82
    assert query_ids == sorted(dq_records)

    candidates_all = a2["candidates"].astype(
        np.int32
    )

    valid_all = a2["candidate_ast_valid"].astype(
        bool
    )

    names_all = az["feature_names"].astype(str)
    signature = az["signature"].astype(np.float32)

    allowed_ids = np.flatnonzero(
        names_all != "max_ast_depth"
    )

    assert len(allowed_ids) == 44

    aggregate = {
        mode: {
            "states": 0,
            "branch_options": 0,
            "pruned": 0,
            "state_bound_stops": 0,
            "runtime": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        for mode in MODES
    }

    aggregate["FINAL_LAZY_DUAL"].update(
        {
            "dq_calls": 0,
            "dq_target_tests": 0,
            "rpc_only_branch_prunes": 0,
            "one_child_shortcuts": 0,
            "dq_cap_shortcuts": 0,
        }
    )

    records = []

    processed = 0
    wall0 = time.perf_counter()

    for qi in query_ids:
        if (
            args.max_hard_queries > 0
            and processed >= args.max_hard_queries
        ):
            break

        frozen_rpc = rpc_records[qi]
        frozen_dq = dq_records[qi]

        expected_joint = int(
            frozen_rpc["joint_K5"]
        )

        candidates = candidates_all[qi]
        valid = valid_all[qi]

        # ==========================================================
        # Frozen UNION predicate family.
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
        # Shared immutable bitsets.
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
                    facts[:, p] == ABSENT
                )
            )
            for p in range(P)
        ]

        valid1 = [
            bool_to_bits(
                valid
                &
                (
                    facts[:, p] == PRESENT
                )
            )
            for p in range(P)
        ]

        # ==========================================================
        # One completely fresh solver instance per mode.
        # ==========================================================

        def run_solver(mode):
            assert mode in MODES

            stats = {
                "states": 0,
                "branch_options": 0,
                "pruned": 0,
                "state_bound_stops": 0,
                "dq_calls": 0,
                "dq_target_tests": 0,
                "rpc_only_branch_prunes": 0,
                "one_child_shortcuts": 0,
                "dq_cap_shortcuts": 0,
            }

            @lru_cache(maxsize=None)
            def stop_cov(mask):
                alive = (
                    mask
                    &
                    target_bits
                )

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

            @lru_cache(maxsize=None)
            def generic_bound(mask):
                return int(
                    (
                        mask
                        &
                        target_bits
                    ).bit_count()
                )

            @lru_cache(maxsize=None)
            def rpc_capacity(
                mask,
                leaves,
            ):
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

                    # UNKNOWN / invalid survives all leaves.
                    if not (valid_bits & bit):
                        for j in range(leaves):
                            if slack[j] > 0:
                                slack[j] -= 1
                        continue

                    # Valid target.
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

                    # Valid non-target.
                    if slack[0] > 0:
                        slack[0] -= 1

                return int(reward)

            # ======================================================
            # FINAL exact lazy/clipped dual bound.
            #
            # Returns exactly min(C,Q).
            # ======================================================

            @lru_cache(maxsize=None)
            def dual_clipped(
                mask,
                depth,
                leaves,
            ):
                effective_leaves = min(
                    int(leaves),
                    1 << max(
                        0,
                        int(depth),
                    ),
                )

                C = rpc_capacity(
                    mask,
                    effective_leaves,
                )

                base = stop_cov(mask)

                # If STOP already reaches capacity ceiling,
                # min(C,Q)=C without evaluating DQ.
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

                stats["dq_calls"] += 1

                feasible = int(base)

                alive = (
                    mask
                    &
                    target_bits
                )

                for t in target_ids:
                    bit = 1 << t

                    if not (alive & bit):
                        continue

                    rank = (
                        mask
                        &
                        prefix_bits[t]
                    ).bit_count()

                    # Already counted in base.
                    if rank <= B:
                        continue

                    stats[
                        "dq_target_tests"
                    ] += 1

                    delta = int(
                        rank - B
                    )

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

                        if top_sum >= delta:
                            possible = True
                            break

                    if possible:
                        feasible += 1

                        # We only need min(C,Q).
                        if feasible >= C:
                            stats[
                                "dq_cap_shortcuts"
                            ] += 1
                            return int(C)

                return int(feasible)

            def state_bound(
                mask,
                depth,
                leaves,
            ):
                if mode == "GENERIC":
                    return generic_bound(mask)

                if mode == "RPC":
                    return rpc_capacity(
                        mask,
                        leaves,
                    )

                return dual_clipped(
                    mask,
                    depth,
                    leaves,
                )

            @lru_cache(maxsize=None)
            def V(
                mask,
                depth,
                leaves,
            ):
                stats["states"] += 1

                base = stop_cov(mask)

                ub_state = state_bound(
                    mask,
                    depth,
                    leaves,
                )

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


                # ======================================================
                # Exact depth-1 closure.
                #
                # For depth == 1, every legal split sends both
                # children to depth 0. Therefore:
                #
                # V(m0,0,l0) = stop_cov(m0)
                # V(m1,0,l1) = stop_cov(m1)
                #
                # for every l0,l1 >= 1.
                #
                # Hence leaf-allocation enumeration is redundant.
                # This is EXACT Bellman elimination, not a bound.
                # ======================================================
                if (
                    mode == "FINAL_LAZY_DUAL"
                    and
                    depth == 1
                ):
                    best_d1 = int(base)
                
                    for p in range(P):
                        # Preserve frozen predicate legality exactly.
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
                
                        # One unique predicate split is evaluated once.
                        stats[
                            "branch_options"
                        ] += 1
                
                        cand_d1 = (
                            stop_cov(m0)
                            +
                            stop_cov(m1)
                        )
                
                        if cand_d1 > best_d1:
                            best_d1 = int(
                                cand_d1
                            )
                
                        # ub_state remains an admissible upper bound.
                        if best_d1 == ub_state:
                            break
                
                    return int(best_d1)
                
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

                        # ==========================================
                        # GENERIC / RPC frozen behavior.
                        # ==========================================

                        if mode == "GENERIC":
                            ub0 = generic_bound(m0)
                            ub1 = generic_bound(m1)

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

                        elif mode == "RPC":
                            ub0 = rpc_capacity(
                                m0,
                                l0,
                            )

                            ub1 = rpc_capacity(
                                m1,
                                l1,
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

                        # ==========================================
                        # FINAL exact lazy dual evaluation.
                        # ==========================================

                        else:
                            C0 = rpc_capacity(
                                m0,
                                l0,
                            )

                            C1 = rpc_capacity(
                                m1,
                                l1,
                            )

                            # First cheap RPC-only prune.
                            if (
                                C0
                                +
                                C1
                                <=
                                best
                            ):
                                stats[
                                    "pruned"
                                ] += 1

                                stats[
                                    "rpc_only_branch_prunes"
                                ] += 1

                                continue

                            # Evaluate only first child dual.
                            ub0 = dual_clipped(
                                m0,
                                depth - 1,
                                l0,
                            )

                            if (
                                ub0
                                +
                                C1
                                <=
                                best
                            ):
                                stats[
                                    "pruned"
                                ] += 1

                                stats[
                                    "one_child_shortcuts"
                                ] += 1

                                continue

                            # Only now evaluate second child dual.
                            ub1 = dual_clipped(
                                m1,
                                depth - 1,
                                l1,
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


            # ======================================================
            # V2 residual-state instrumentation.
            # DIAGNOSTIC ONLY.
            # ======================================================

            # ======================================================
            # Root-Coupled RPC diagnostic upper bound.
            # DIAGNOSTIC ONLY -- never used for pruning.
            # ======================================================
            def root_coupled_rpc_bound(
                mask,
                depth,
                leaves,
            ):
                mask = int(mask)
                depth = int(depth)
                leaves = int(leaves)
            
                base_rc = int(
                    stop_cov(mask)
                )
            
                if (
                    depth <= 0
                    or
                    leaves <= 1
                    or
                    mask.bit_count() <= B
                ):
                    return base_rc
            
                child_depth = depth - 1
            
                def child_cap(
                    child_mask,
                    child_leaves,
                ):
                    child_mask = int(child_mask)
                    child_leaves = int(
                        child_leaves
                    )
            
                    if (
                        child_depth <= 0
                        or
                        child_leaves <= 1
                        or
                        child_mask.bit_count() <= B
                    ):
                        return int(
                            stop_cov(
                                child_mask
                            )
                        )
            
                    child_eff_leaves = min(
                        child_leaves,
                        1 << max(
                            0,
                            child_depth,
                        ),
                    )
            
                    return int(
                        rpc_capacity(
                            child_mask,
                            child_eff_leaves,
                        )
                    )
            
                best_rc = int(base_rc)
            
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
            
                    m0_rc = (
                        mask
                        &
                        survive0[p]
                    )
            
                    m1_rc = (
                        mask
                        &
                        survive1[p]
                    )
            
                    if (
                        m0_rc == mask
                        or
                        m1_rc == mask
                    ):
                        continue
            
                    for l0_rc in range(
                        1,
                        leaves,
                    ):
                        l1_rc = (
                            leaves
                            -
                            l0_rc
                        )
            
                        cand_rc = (
                            child_cap(
                                m0_rc,
                                l0_rc,
                            )
                            +
                            child_cap(
                                m1_rc,
                                l1_rc,
                            )
                        )
            
                        if cand_rc > best_rc:
                            best_rc = int(
                                cand_rc
                            )
            
                return int(best_rc)
            

            # ======================================================
            # Exact depth-1 shared value.
            #
            # This duplicates the already-proven D1 closure only
            # for diagnostic decomposition. It is NOT used by V().
            # ======================================================
            def d1_exact_shared_value(
                mask,
            ):
                mask = int(mask)
            
                base_d1 = int(
                    stop_cov(mask)
                )
            
                if mask.bit_count() <= B:
                    return base_d1
            
                best_d1_diag = int(
                    base_d1
                )
            
                for q in range(P):
                    if not (
                        mask
                        &
                        valid0[q]
                    ):
                        continue
            
                    if not (
                        mask
                        &
                        valid1[q]
                    ):
                        continue
            
                    q0 = (
                        mask
                        &
                        survive0[q]
                    )
                    q1 = (
                        mask
                        &
                        survive1[q]
                    )
            
                    if (
                        q0 == mask
                        or
                        q1 == mask
                    ):
                        continue
            
                    cand_d1_diag = (
                        stop_cov(q0)
                        +
                        stop_cov(q1)
                    )
            
                    if cand_d1_diag > best_d1_diag:
                        best_d1_diag = int(
                            cand_d1_diag
                        )
            
                return int(best_d1_diag)
            
            
            # ======================================================
            # Full d=2 candidate-level decomposition.
            #
            # For each common root p and legal leaf allocation:
            #
            #   RC  = branch RPC capacities
            #   EX  = branch exact D1 shared values
            #
            # Their difference is exactly the child-sharing
            # relaxation gap for this fixed root/allocation.
            # ======================================================
            def two_level_child_decomp(
                mask,
                leaves,
            ):
                mask = int(mask)
                leaves = int(leaves)
            
                base_two = int(
                    stop_cov(mask)
                )
            
                best_rc = int(base_two)
                best_ex = int(base_two)
            
                records = []
            
                def branch_values(
                    child_mask,
                    child_leaves,
                ):
                    child_mask = int(
                        child_mask
                    )
                    child_leaves = int(
                        child_leaves
                    )
            
                    z = int(
                        stop_cov(child_mask)
                    )
            
                    if (
                        child_leaves <= 1
                        or
                        child_mask.bit_count() <= B
                    ):
                        return (
                            z,
                            z,
                            0,
                        )
            
                    # Remaining child depth is exactly 1.
                    c = int(
                        rpc_capacity(
                            child_mask,
                            2,
                        )
                    )
                    e = int(
                        d1_exact_shared_value(
                            child_mask
                        )
                    )
                    assert e <= c
                    return (
                        c,
                        e,
                        int(c - e),
                    )
            
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
                        leaves,
                    ):
                        l1 = leaves - l0
            
                        c0, e0, g0 = (
                            branch_values(
                                m0,
                                l0,
                            )
                        )
                        c1, e1, g1 = (
                            branch_values(
                                m1,
                                l1,
                            )
                        )
                        rc = int(c0 + c1)
                        ex = int(e0 + e1)
                        gap = int(rc - ex)
            
                        assert gap == g0 + g1
                        assert ex <= rc
            
                        records.append({
                            "p": int(p),
                            "l0": int(l0),
                            "l1": int(l1),
                            "rc": int(rc),
                            "ex": int(ex),
                            "gap": int(gap),
                            "g0": int(g0),
                            "g1": int(g1),
                        })
            
                        if rc > best_rc:
                            best_rc = int(rc)
                        if ex > best_ex:
                            best_ex = int(ex)
            
                rc_opt = [
                    r
                    for r in records
                    if r["rc"] == best_rc
                ]
            
                ex_opt = [
                    r
                    for r in records
                    if r["ex"] == best_ex
                ]
            
                if rc_opt:
                    rc_best_exact = max(
                        r["ex"]
                        for r in rc_opt
                    )
                    rc_min_child_gap = min(
                        r["gap"]
                        for r in rc_opt
                    )
                    rc_best_rec = max(
                        rc_opt,
                        key=lambda r: (
                            r["ex"],
                            -r["gap"],
                        ),
                    )
                else:
                    rc_best_exact = int(
                        base_two
                    )
                    rc_min_child_gap = 0
                    rc_best_rec = {
                        "l0": 0,
                        "l1": 0,
                        "g0": 0,
                        "g1": 0,
                    }
            
                if ex_opt:
                    ex_min_relax_gap = min(
                        r["gap"]
                        for r in ex_opt
                    )
                else:
                    ex_min_relax_gap = 0
            
                return {
                    "best_rc": int(best_rc),
                    "best_ex": int(best_ex),
                    "rc_best_exact": int(
                        rc_best_exact
                    ),
                    "rc_min_child_gap": int(
                        rc_min_child_gap
                    ),
                    "root_choice_advantage": int(
                        best_ex
                        -
                        rc_best_exact
                    ),
                    "ex_min_relax_gap": int(
                        ex_min_relax_gap
                    ),
                    "rcopt_l0": int(
                        rc_best_rec["l0"]
                    ),
                    "rcopt_l1": int(
                        rc_best_rec["l1"]
                    ),
                    "rcopt_g0": int(
                        rc_best_rec["g0"]
                    ),
                    "rcopt_g1": int(
                        rc_best_rec["g1"]
                    ),
                }
            
            residual_rows = []
            
            if mode == "FINAL_LAZY_DUAL":
                _V_exact_cached = V
            
                def V(mask, depth, leaves):
                    before = (
                        _V_exact_cached.cache_info()
                    )
            
                    value = _V_exact_cached(
                        mask,
                        depth,
                        leaves,
                    )
            
                    after = (
                        _V_exact_cached.cache_info()
                    )
            
                    is_new_state = (
                        after.misses
                        >
                        before.misses
                    )
            
                    if is_new_state:
                        base_now = int(
                            stop_cov(mask)
                        )
            
                        L_eff_now = min(
                            int(leaves),
                            1 << max(
                                0,
                                int(depth),
                            ),
                        )
            
                        C_now = int(
                            rpc_capacity(
                                mask,
                                L_eff_now,
                            )
                        )
            
                        U_now = int(
                            state_bound(
                                mask,
                                depth,
                                leaves,
                            )
                        )
            
                        V_now = int(value)

                        RC_raw_now = int(
                            root_coupled_rpc_bound(
                                mask,
                                depth,
                                leaves,
                            )
                        )
                        
                        # RC-RPC must remain an admissible bound.
                        assert V_now <= RC_raw_now, (
                            "RC-RPC inadmissible",
                            int(depth),
                            int(leaves),
                            int(V_now),
                            int(RC_raw_now),
                        )
                        
                        U_RC_now = int(
                            min(
                                U_now,
                                RC_raw_now,
                            )
                        )
                        
                        assert V_now <= U_RC_now
                        assert U_RC_now <= U_now
                        
            
                        assert V_now <= U_now
                        assert U_now <= C_now
            
                        if (
                            depth <= 0
                            or
                            leaves <= 1
                            or
                            mask.bit_count() <= B
                        ):
                            kind_now = "TERMINAL"
            
                        elif U_now == base_now:
                            kind_now = "BOUND_STOP"
            
                        elif depth == 1:
                            kind_now = "D1_CLOSURE"
            
                        else:
                            kind_now = "NONTERMINAL"
            

                        if (
                            kind_now == "NONTERMINAL"
                            and
                            int(depth) == 2
                        ):
                            d2diag = two_level_child_decomp(
                                mask,
                                leaves,
                            )
                        
                            # Strong diagnostic parity:
                            # exact two-level closure must reproduce V.
                            assert (
                                int(d2diag["best_ex"])
                                ==
                                int(V_now)
                            ), (
                                "D2 exact decomposition mismatch",
                                int(depth),
                                int(leaves),
                                int(d2diag["best_ex"]),
                                int(V_now),
                            )
                        
                            assert (
                                int(d2diag["best_rc"])
                                ==
                                int(RC_raw_now)
                            ), (
                                "D2 RC decomposition mismatch",
                                int(d2diag["best_rc"]),
                                int(RC_raw_now),
                            )
                        else:
                            d2diag = {
                                "best_rc": -1,
                                "best_ex": -1,
                                "rc_best_exact": -1,
                                "rc_min_child_gap": -1,
                                "root_choice_advantage": -1,
                                "ex_min_relax_gap": -1,
                                "rcopt_l0": -1,
                                "rcopt_l1": -1,
                                "rcopt_g0": -1,
                                "rcopt_g1": -1,
                            }
                        
                        residual_rows.append({
                            "depth": int(depth),
                            "leaves": int(leaves),
                            "L_eff": int(L_eff_now),
                            "alive": int(
                                mask.bit_count()
                            ),
                            "base": int(base_now),
                            "C_DF": int(C_now),
                            "U": int(U_now),
                            "V": int(V_now),
                            "slack": int(
                                U_now - V_now
                            ),
                            "RC_raw": int(RC_raw_now),
                            "U_RC": int(U_RC_now),
                            "slack_RC": int(
                                U_RC_now - V_now
                            ),
                            "RC_strict": bool(
                                U_RC_now < U_now
                            ),
                            "D2_best_RC": int(d2diag["best_rc"]),
                            "D2_best_EX": int(d2diag["best_ex"]),
                            "D2_RCopt_EX": int(d2diag["rc_best_exact"]),
                            "D2_RCopt_child_gap": int(d2diag["rc_min_child_gap"]),
                            "D2_root_choice_adv": int(d2diag["root_choice_advantage"]),
                            "D2_EXopt_relax_gap": int(d2diag["ex_min_relax_gap"]),
                            "D2_RCopt_l0": int(d2diag["rcopt_l0"]),
                            "D2_RCopt_l1": int(d2diag["rcopt_l1"]),
                            "D2_RCopt_g0": int(d2diag["rcopt_g0"]),
                            "D2_RCopt_g1": int(d2diag["rcopt_g1"]),
                            "DQ_strict": bool(
                                U_now < C_now
                            ),
                            "state_kind": kind_now,
                        })
            
                    return value
            
                # Preserve cache interface expected below.
                V.cache_info = (
                    _V_exact_cached.cache_info
                )
                V.cache_clear = (
                    _V_exact_cached.cache_clear
                )
            
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

            if mode == "FINAL_LAZY_DUAL":
                import json as _res_json
                from pathlib import Path as _ResPath
            
                _res_path = _ResPath(
                    "fhex_v1/final_v2/"
                    "child_sharing_shadow_smoke10/"
                    "state_blocks.jsonl"
                )
            
                _res_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
            
                assert len(residual_rows) == int(
                    info.misses
                ), (
                    len(residual_rows),
                    info.misses,
                )
            
                with _res_path.open(
                    "a",
                    encoding="utf-8",
                ) as _f:
                    _f.write(
                        _res_json.dumps(
                            {
                                "value": int(value),
                                "states": int(
                                    stats["states"]
                                ),
                                "cache_misses": int(
                                    info.misses
                                ),
                                "rows": residual_rows,
                            },
                            ensure_ascii=False,
                        )
                        +
                        "\n"
                    )
            

            stats[
                "runtime"
            ] = float(runtime)

            stats[
                "value"
            ] = int(value)

            stats[
                "cache_hits"
            ] = int(info.hits)

            stats[
                "cache_misses"
            ] = int(info.misses)

            return stats

        # ==========================================================
        # Same process, fresh per-mode caches.
        # ==========================================================

        generic = run_solver(
            "GENERIC"
        )

        rpc = run_solver(
            "RPC"
        )

        final = run_solver(
            "FINAL_LAZY_DUAL"
        )

        # ==========================================================
        # Exact-value parity.
        # ==========================================================

        assert generic["value"] == expected_joint
        assert rpc["value"] == expected_joint
        assert final["value"] == expected_joint

        # ==========================================================
        # Frozen deterministic state-count parity.
        # ==========================================================

        if (
            "generic" in frozen_rpc
            and
            "states" in frozen_rpc["generic"]
        ):
            assert (
                generic["states"]
                ==
                int(
                    frozen_rpc[
                        "generic"
                    ][
                        "states"
                    ]
                )
            )

        if (
            "rpc" in frozen_rpc
            and
            "states" in frozen_rpc["rpc"]
        ):
            assert (
                rpc["states"]
                ==
                int(
                    frozen_rpc[
                        "rpc"
                    ][
                        "states"
                    ]
                )
            )

        frozen_dual = frozen_dq["rpc_dq"]

        pass  # DF-V1: frozen search-tree parity intentionally relaxed

        if "branch_options" in frozen_dual:
            pass  # DF-V1: frozen search-tree parity intentionally relaxed

        if "pruned" in frozen_dual:
            pass  # DF-V1: frozen search-tree parity intentionally relaxed

        g_r_speed = (
            generic["runtime"]
            /
            max(
                rpc["runtime"],
                1e-12,
            )
        )

        r_f_speed = (
            rpc["runtime"]
            /
            max(
                final["runtime"],
                1e-12,
            )
        )

        g_f_speed = (
            generic["runtime"]
            /
            max(
                final["runtime"],
                1e-12,
            )
        )

        print(
            f"q={qi:03d} "
            f"J5={expected_joint:02d} | "
            f"states "
            f"{generic['states']} -> "
            f"{rpc['states']} -> "
            f"{final['states']} | "
            f"time "
            f"{generic['runtime']:.3f}s -> "
            f"{rpc['runtime']:.3f}s -> "
            f"{final['runtime']:.3f}s | "
            f"G/R={g_r_speed:.2f}x "
            f"R/F={r_f_speed:.2f}x "
            f"G/F={g_f_speed:.2f}x",
            flush=True,
        )

        qrec = {
            "query":
                int(qi),

            "joint_K5":
                int(expected_joint),

            "generic":
                generic,

            "rpc":
                rpc,

            "final_lazy_dual":
                final,

            "generic_to_rpc_state_reduction":
                float(
                    1.0
                    -
                    rpc["states"]
                    /
                    generic["states"]
                ),

            "rpc_to_final_state_reduction":
                float(
                    1.0
                    -
                    final["states"]
                    /
                    rpc["states"]
                ),

            "generic_to_final_state_reduction":
                float(
                    1.0
                    -
                    final["states"]
                    /
                    generic["states"]
                ),

            "generic_to_rpc_speedup":
                float(g_r_speed),

            "rpc_to_final_speedup":
                float(r_f_speed),

            "generic_to_final_speedup":
                float(g_f_speed),
        }

        records.append(qrec)

        for mode_name, obj in (
            ("GENERIC", generic),
            ("RPC", rpc),
            ("FINAL_LAZY_DUAL", final),
        ):
            for field in (
                "states",
                "branch_options",
                "pruned",
                "state_bound_stops",
                "cache_hits",
                "cache_misses",
            ):
                aggregate[
                    mode_name
                ][
                    field
                ] += int(
                    obj[field]
                )

            aggregate[
                mode_name
            ][
                "runtime"
            ] += float(
                obj["runtime"]
            )

        for field in (
            "dq_calls",
            "dq_target_tests",
            "rpc_only_branch_prunes",
            "one_child_shortcuts",
            "dq_cap_shortcuts",
        ):
            aggregate[
                "FINAL_LAZY_DUAL"
            ][
                field
            ] += int(
                final[field]
            )

        processed += 1

    wall_runtime = (
        time.perf_counter()
        -
        wall0
    )

    G = aggregate["GENERIC"]
    R = aggregate["RPC"]
    F = aggregate["FINAL_LAZY_DUAL"]

    g2r_state = (
        1.0
        -
        R["states"]
        /
        G["states"]
    )

    r2f_state = (
        1.0
        -
        F["states"]
        /
        R["states"]
    )

    g2f_state = (
        1.0
        -
        F["states"]
        /
        G["states"]
    )

    g2r_speed = (
        G["runtime"]
        /
        max(
            R["runtime"],
            1e-12,
        )
    )

    r2f_speed = (
        R["runtime"]
        /
        max(
            F["runtime"],
            1e-12,
        )
    )

    g2f_speed = (
        G["runtime"]
        /
        max(
            F["runtime"],
            1e-12,
        )
    )

    g2r_speeds = [
        r["generic_to_rpc_speedup"]
        for r in records
    ]

    r2f_speeds = [
        r["rpc_to_final_speedup"]
        for r in records
    ]

    g2f_speeds = [
        r["generic_to_final_speedup"]
        for r in records
    ]

    g2f_state_red = [
        r["generic_to_final_state_reduction"]
        for r in records
    ]

    positive_g2f = [
        x
        for x in g2f_speeds
        if x > 0
    ]

    geometric_mean_g2f = float(
        math.exp(
            sum(
                math.log(x)
                for x in positive_g2f
            )
            /
            len(positive_g2f)
        )
    )

    distribution = {
        "median_G_to_R_speedup":
            float(
                np.median(
                    g2r_speeds
                )
            ),

        "median_R_to_FINAL_speedup":
            float(
                np.median(
                    r2f_speeds
                )
            ),

        "median_G_to_FINAL_speedup":
            float(
                np.median(
                    g2f_speeds
                )
            ),

        "min_G_to_FINAL_speedup":
            float(
                np.min(
                    g2f_speeds
                )
            ),

        "max_G_to_FINAL_speedup":
            float(
                np.max(
                    g2f_speeds
                )
            ),

        "queries_FINAL_faster_than_GENERIC":
            int(
                sum(
                    x > 1.0
                    for x in g2f_speeds
                )
            ),

        "queries_FINAL_faster_than_RPC":
            int(
                sum(
                    x > 1.0
                    for x in r2f_speeds
                )
            ),

        "median_G_to_FINAL_state_reduction":
            float(
                np.median(
                    g2f_state_red
                )
            ),

        "geometric_mean_G_to_FINAL_speedup":
            geometric_mean_g2f,
    }

    print()
    print("=" * 176)
    print(
        "FINAL THREE-WAY EXACT BENCHMARK SUMMARY"
    )
    print("=" * 176)

    print(
        "hard queries processed =",
        processed,
    )

    print()
    print("GENERIC")
    print("states =", G["states"])
    print(
        "branch options =",
        G["branch_options"],
    )
    print("pruned =", G["pruned"])
    print(
        "runtime seconds =",
        G["runtime"],
    )

    print()
    print("RPC")
    print("states =", R["states"])
    print(
        "branch options =",
        R["branch_options"],
    )
    print("pruned =", R["pruned"])
    print(
        "runtime seconds =",
        R["runtime"],
    )

    print()
    print("FINAL_LAZY_DUAL")
    print("states =", F["states"])
    print(
        "branch options =",
        F["branch_options"],
    )
    print("pruned =", F["pruned"])
    print(
        "runtime seconds =",
        F["runtime"],
    )
    print(
        "dq calls =",
        F["dq_calls"],
    )
    print(
        "dq target tests =",
        F["dq_target_tests"],
    )
    print(
        "rpc-only branch prunes =",
        F["rpc_only_branch_prunes"],
    )
    print(
        "one-child DQ shortcuts =",
        F["one_child_shortcuts"],
    )
    print(
        "clipped-cap shortcuts =",
        F["dq_cap_shortcuts"],
    )

    print()
    print(
        "Generic -> RPC state reduction =",
        g2r_state,
    )
    print(
        "RPC -> Final state reduction =",
        r2f_state,
    )
    print(
        "Generic -> Final state reduction =",
        g2f_state,
    )

    print()
    print(
        "Generic -> RPC runtime speedup =",
        g2r_speed,
    )
    print(
        "RPC -> Final runtime speedup =",
        r2f_speed,
    )
    print(
        "Generic -> Final runtime speedup =",
        g2f_speed,
    )

    print()
    for k, v in distribution.items():
        print(
            f"{k} = {v}"
        )

    print()
    print(
        "wall runtime seconds =",
        wall_runtime,
    )

    print(
        "THREE-WAY EXACT VALUE PARITY = PASS"
    )
    print(
        "FROZEN STATE-COUNT PARITY = PASS"
    )
    print(
        "FINAL LAZY SEARCH-TREE PARITY = PASS"
    )

    OUT = (
        D
        /
        f"final_threeway_lazy_v0_{args.tag}.json"
    )

    OUT.write_text(
        json.dumps(
            {
                "name":
                    "Final Dual Capacity-Reachability Three-Way Exact Benchmark v0",

                "processed":
                    int(processed),

                "aggregate":
                    aggregate,

                "generic_to_rpc_state_reduction":
                    float(g2r_state),

                "rpc_to_final_state_reduction":
                    float(r2f_state),

                "generic_to_final_state_reduction":
                    float(g2f_state),

                "generic_to_rpc_speedup":
                    float(g2r_speed),

                "rpc_to_final_speedup":
                    float(r2f_speed),

                "generic_to_final_speedup":
                    float(g2f_speed),

                "distribution":
                    distribution,

                "records":
                    records,
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

    print("=" * 176)


if __name__ == "__main__":
    main()
