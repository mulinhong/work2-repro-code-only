#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import time

import numpy as np


ROOT = Path('/root/autodl-tmp/work2-algorithm-innovation')
A2_ART = ROOT / 'fhex_v1/results/bcr_a2_exact_program_fact_bcct.npz'
A4_ART = ROOT / 'fhex_v1/results/bcr_a4_rcps.npz'
AST_CACHE = ROOT / 'fhex_v1/cache/python_train_ast_signature_v0.npz'
A2_SCRIPT = ROOT / 'fhex_v1/scripts/audit_bcr_a2_exact_program_fact_bcct.py'
A4_SCRIPT = ROOT / 'fhex_v1/scripts/audit_bcr_a4_rcps.py'

OUT_NPZ = ROOT / 'fhex_v1/results/rcps_all_hypothesis_shared_gap_v0.npz'
OUT_JSON = ROOT / 'fhex_v1/results/rcps_all_hypothesis_shared_gap_v0.json'

EXPECTED_A2_ART_SHA = '4ce63e6e5eabf2662a291ebf9064b923880df124d5afe617e14ee357c0248d85'
EXPECTED_A4_ART_SHA = 'fb61f7a1b30c69698b9ee7365cb92cd39c4931b9849be5d25c4e4a314943beb5'
EXPECTED_A2_SCRIPT_SHA = '1d27808e3f86cd8956e6e342a342efd0a8f9d78e35c7232e25483e1df57b5851'
EXPECTED_A4_SCRIPT_SHA = '580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9'

E = 80
B = 20
UNKNOWN = -1
ABSENT = 0
PRESENT = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bool_rows_to_ints(x: np.ndarray) -> list[int]:
    # x: [P,E], E=80. little-endian bits => bit j is candidate position j.
    packed = np.packbits(x.astype(np.uint8, copy=False), axis=1, bitorder='little')
    return [int.from_bytes(row.tobytes(), 'little', signed=False) for row in packed]


def prune_superset_masks(masks: list[int]) -> list[int]:
    # For rank minimization, if k subset m then m is dominated: for every
    # second question x, |k & x| <= |m & x|.
    uniq = sorted(set(masks), key=lambda z: (z.bit_count(), z))
    kept: list[int] = []
    for m in uniq:
        dominated = False
        for k in kept:
            if (m & k) == k:  # k subseteq m
                dominated = True
                break
        if not dominated:
            kept.append(m)
    return kept


def individual_b20_vector(facts: np.ndarray, valid_target: np.ndarray, a4mod) -> np.ndarray:
    """
    Exact B20 individual feasibility for every AST-valid target hypothesis.

    Candidate order remains the original E80 order. Invalid-AST candidates
    remain in the retrieval list and survive either answer via UNKNOWN.
    Invalid candidates are excluded only as *targets* because they cannot
    define a reliable exact answer path.

    A valid target's synthesized predicates are all known (0/1), so its
    answer path is unique. We test existence of a path of length <=2.
    """
    survive0, survive1 = a4mod.build_survival_masks(facts)
    bits0 = bool_rows_to_ints(survive0)
    bits1 = bool_rows_to_ints(survive1)

    out = np.zeros(E, dtype=bool)
    valid_ids = np.flatnonzero(valid_target)

    for t0 in valid_ids:
        t = int(t0)
        raw_rank = t + 1
        if raw_rank <= B:
            out[t] = True
            continue

        target = facts[t]
        if np.any(target == UNKNOWN):
            raise AssertionError('AST-valid target has UNKNOWN predicate value')

        prefix = (1 << (t + 1)) - 1
        masks = [
            ((bits1[p] if int(target[p]) == PRESENT else bits0[p]) & prefix)
            for p in range(facts.shape[1])
        ]

        # One question.
        if min(m.bit_count() for m in masks) <= B:
            out[t] = True
            continue

        # Only inclusion-minimal survivor sets can be useful in a pair.
        minimal = prune_superset_masks(masks)

        found = False
        for i in range(len(minimal)):
            mi = minimal[i]
            for j in range(i + 1, len(minimal)):
                if (mi & minimal[j]).bit_count() <= B:
                    found = True
                    break
            if found:
                break

        out[t] = found

    return out


def eval_atomic_policy_all(policy, facts, valid_target, a2mod) -> np.ndarray:
    success = np.zeros(E, dtype=bool)
    for t0 in np.flatnonzero(valid_target):
        t = int(t0)
        if policy is None:
            r = t + 1
        else:
            r, retained, _ = a2mod.execute_policy(policy, facts, t, facts[t])
            if not retained:
                r = E + 1
        success[t] = (r <= B)
    return success


def eval_compound_policy_all(policy, facts, s0, s1, valid_target, a4mod) -> np.ndarray:
    success = np.zeros(E, dtype=bool)
    for t0 in np.flatnonzero(valid_target):
        t = int(t0)
        if policy is None:
            r = t + 1
        else:
            r, retained, _ = a4mod.execute_policy(policy, facts, s0, s1, t, True)
            if not retained:
                r = E + 1
        success[t] = (r <= B)
    return success


def eval_atomic_gbs_all(facts, valid_target, a2mod) -> np.ndarray:
    p = a2mod.choose_gbs(facts)
    return eval_atomic_policy_all(p, facts, valid_target, a2mod)


def eval_compound_gbs_all(facts, valid_target, a2mod) -> np.ndarray:
    # GBS policy format is compatible with A2 execute_policy.
    p = a2mod.choose_gbs(facts)
    return eval_atomic_policy_all(p, facts, valid_target, a2mod)


def qstats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        'sum': float(x.sum()),
        'mean': float(x.mean()),
        'median': float(np.median(x)),
        'p90': float(np.quantile(x, 0.90)),
        'min': float(x.min()),
        'max': float(x.max()),
    }


def main():
    t0 = time.perf_counter()

    print('=' * 150)
    print('RCPS ALL-HYPOTHESIS SHARED REPAIR GAP AUDIT v0')
    print('=' * 150)
    print('Consumed FIT only. All 568 FIT failure frontiers are audited.')
    print('No P1 access. No P2 access. RCPS-v1 is NOT modified.')
    print('GT identity is NOT used to define O_G or S_G.')
    print('Target hypothesis universe = AST-valid Local80 candidates.')
    print('AST-invalid candidates remain as UNKNOWN blockers and survive either answer.')

    # ------------------------------------------------------------------
    # Integrity.
    # ------------------------------------------------------------------
    observed = {
        'A2_artifact': sha256_file(A2_ART),
        'A4_artifact': sha256_file(A4_ART),
        'A2_script': sha256_file(A2_SCRIPT),
        'A4_script': sha256_file(A4_SCRIPT),
        'AST_cache': sha256_file(AST_CACHE),
    }
    assert observed['A2_artifact'] == EXPECTED_A2_ART_SHA
    assert observed['A4_artifact'] == EXPECTED_A4_ART_SHA
    assert observed['A2_script'] == EXPECTED_A2_SCRIPT_SHA
    assert observed['A4_script'] == EXPECTED_A4_SCRIPT_SHA

    print()
    for k, v in observed.items():
        print(k, '=', v)

    a2mod = load_module(A2_SCRIPT, 'gap_a2_native')
    a4mod = load_module(A4_SCRIPT, 'gap_a4_native')
    a2 = np.load(A2_ART, allow_pickle=True)
    a4 = np.load(A4_ART, allow_pickle=True)
    az = np.load(AST_CACHE, allow_pickle=True)

    candidates = a2['candidates'].astype(np.int32)
    candidate_valid = a2['candidate_ast_valid'].astype(bool)
    raw_rank = a2['raw_rank'].astype(np.int16)
    n = len(candidates)
    assert n == 568
    assert candidates.shape == (568, E)
    assert candidate_valid.shape == (568, E)

    # A2/A4 identity on shared data.
    assert np.array_equal(a2['gt'], a4['gt'])
    assert np.array_equal(a2['candidates'], a4['candidates'])
    assert np.array_equal(a2['raw_rank'], a4['raw_rank'])

    names_all = az['feature_names'].astype(str)
    signature = az['signature'].astype(np.float32)
    ast_valid = az['valid'].astype(bool)
    allowed_ids = np.flatnonzero(names_all != 'max_ast_depth')
    assert len(allowed_ids) == 44
    assert np.array_equal(names_all[allowed_ids], a2['feature_names'].astype(str))
    assert np.array_equal(ast_valid[candidates], candidate_valid)

    # Outputs per query.
    valid_n = candidate_valid.sum(axis=1).astype(np.int16)
    O_atomic = np.zeros(n, dtype=np.int16)
    O_compound = np.zeros(n, dtype=np.int16)
    S_atomic = np.zeros(n, dtype=np.int16)
    S_compound = np.zeros(n, dtype=np.int16)
    G_atomic = np.zeros(n, dtype=np.int16)
    G_compound = np.zeros(n, dtype=np.int16)
    unique_compound = np.zeros(n, dtype=np.int16)

    # Diagnostics and parity.
    atomic_policy_mismatch = 0
    compound_policy_mismatch_on_recoverable = 0
    atomic_policy_coverage_mismatch = 0
    compound_policy_coverage_mismatch = 0
    atomic_individual_gt_B20_mismatch = 0
    compound_individual_gt_B20_mismatch = 0

    gt_all = a2['gt'].astype(np.int32)
    atomic_oracle_saved = a2['exact_oracle_rank'].astype(np.int16)

    start = time.perf_counter()

    for qi in range(n):
        cands = candidates[qi]
        valid = candidate_valid[qi]

        atomic = np.full((E, 44), UNKNOWN, dtype=np.int8)
        pres = (signature[cands][:, allowed_ids] > 0.0).astype(np.int8)
        atomic[valid] = pres[valid]

        # --------------------------------------------------------------
        # Atomic individual and shared.
        # --------------------------------------------------------------
        ind_a = individual_b20_vector(atomic, valid, a4mod)
        O_atomic[qi] = int(ind_a.sum())

        p_a = a2mod.choose_bcct(atomic, valid)
        s_a_vec = eval_atomic_policy_all(p_a, atomic, valid, a2mod)
        S_atomic[qi] = int(s_a_vec.sum())

        if p_a is None:
            cov_a = int(np.sum(valid & (np.arange(E) < B)))
            f1a = f20a = f21a = -1
        else:
            cov_a = int(p_a['coverage'])
            f1a = int(p_a['first_feature'])
            f20a = int(p_a['second_zero'])
            f21a = int(p_a['second_one'])
        if cov_a != int(S_atomic[qi]):
            atomic_policy_coverage_mismatch += 1

        # Frozen A2 generates/stores BCCT policy only for
        # real-GT Local80-recoverable queries. For the other
        # FIT failures the frozen policy fields remain -1,
        # although this all-hypothesis audit legitimately
        # computes a shared policy for their Local80 frontier.
        #
        # Therefore policy-identity parity is meaningful only
        # on the same frozen A2 recoverable subset.
        if raw_rank[qi] <= E:
            if (
                f1a != int(a2['bcct_first_question'][qi])
                or f20a != int(a2['bcct_second_if_absent'][qi])
                or f21a != int(a2['bcct_second_if_present'][qi])
            ):
                atomic_policy_mismatch += 1

        g_a_vec = eval_atomic_gbs_all(atomic, valid, a2mod)
        G_atomic[qi] = int(g_a_vec.sum())

        # --------------------------------------------------------------
        # Compound family.
        # --------------------------------------------------------------
        full, _, _, _ = a4mod.build_synth_library(atomic)
        compound, keep = a4mod.deduplicate_predicates(full, valid)
        unique_compound[qi] = int(compound.shape[1])

        ind_c = individual_b20_vector(compound, valid, a4mod)
        O_compound[qi] = int(ind_c.sum())

        p_c, s0, s1 = a4mod.choose_bcct(compound, valid)
        s_c_vec = eval_compound_policy_all(p_c, compound, s0, s1, valid, a4mod)
        S_compound[qi] = int(s_c_vec.sum())

        if p_c is None:
            cov_c = int(np.sum(valid & (np.arange(E) < B)))
            g1c = g20c = g21c = -1
        else:
            cov_c = int(p_c['coverage'])
            g1c = int(keep[int(p_c['first_feature'])])
            g20c = (
                int(keep[int(p_c['second_zero'])])
                if int(p_c['second_zero']) >= 0 else -1
            )
            g21c = (
                int(keep[int(p_c['second_one'])])
                if int(p_c['second_one']) >= 0 else -1
            )
        if cov_c != int(S_compound[qi]):
            compound_policy_coverage_mismatch += 1

        # A4 froze policies only on Local80-recoverable real-GT queries.
        if raw_rank[qi] <= E:
            if (
                g1c != int(a4['first_predicate_global'][qi])
                or g20c != int(a4['second_if_absent_global'][qi])
                or g21c != int(a4['second_if_present_global'][qi])
            ):
                compound_policy_mismatch_on_recoverable += 1

        g_c_vec = eval_compound_gbs_all(compound, valid, a2mod)
        G_compound[qi] = int(g_c_vec.sum())

        # --------------------------------------------------------------
        # Real-GT parity is used ONLY as implementation validation.
        # It does not enter O/S definitions.
        # --------------------------------------------------------------
        if raw_rank[qi] <= E:
            pos = np.flatnonzero(cands == gt_all[qi])
            assert len(pos) == 1
            t = int(pos[0])
            if bool(ind_a[t]) != bool(atomic_oracle_saved[qi] <= B):
                atomic_individual_gt_B20_mismatch += 1
            # Frozen A4 RCPS oracle diagnostic in v1 showed 121/121, but
            # we validate the compound individual target directly here.
            if not bool(ind_c[t]):
                compound_individual_gt_B20_mismatch += 1

        if (qi + 1) % 25 == 0 or qi == n - 1:
            print(
                f'[{qi+1:>3}/{n}]',
                f'Oa={int(O_atomic[:qi+1].sum())}',
                f'Sa={int(S_atomic[:qi+1].sum())}',
                f'Oc={int(O_compound[:qi+1].sum())}',
                f'Sc={int(S_compound[:qi+1].sum())}',
                f'elapsed={time.perf_counter()-start:.1f}s',
                flush=True,
            )

    # ------------------------------------------------------------------
    # Hard invariants.
    # ------------------------------------------------------------------
    print()
    print('=' * 150)
    print('IMPLEMENTATION / OPTIMALITY INVARIANTS')
    print('=' * 150)
    print('atomic frozen-policy mismatches on 121 recoverable =', atomic_policy_mismatch)
    print('compound frozen-policy mismatches on 121 recoverable =', compound_policy_mismatch_on_recoverable)
    print('atomic policy coverage mismatches =', atomic_policy_coverage_mismatch)
    print('compound policy coverage mismatches =', compound_policy_coverage_mismatch)
    print('atomic real-GT individual B20 mismatches =', atomic_individual_gt_B20_mismatch)
    print('compound real-GT individual B20 failures =', compound_individual_gt_B20_mismatch)

    assert atomic_policy_mismatch == 0
    assert compound_policy_mismatch_on_recoverable == 0
    assert atomic_policy_coverage_mismatch == 0
    assert compound_policy_coverage_mismatch == 0
    assert atomic_individual_gt_B20_mismatch == 0
    assert compound_individual_gt_B20_mismatch == 0

    # Family containment invariants.
    if np.any(O_compound < O_atomic):
        raise AssertionError('Compound individual feasibility decreased despite atomic subset.')
    if np.any(S_compound < S_atomic):
        raise AssertionError('Compound exact shared optimum decreased despite atomic subset.')

    gamma_a = (O_atomic - S_atomic).astype(np.int16)
    gamma_c = (O_compound - S_compound).astype(np.int16)
    assert np.all(gamma_a >= 0)
    assert np.all(gamma_c >= 0)

    dO = (O_compound - O_atomic).astype(np.int16)
    dS = (S_compound - S_atomic).astype(np.int16)
    gap_reduction = (gamma_a - gamma_c).astype(np.int16)
    assert np.array_equal(gap_reduction, dS - dO)

    dG = (G_compound - G_atomic).astype(np.int16)
    interaction = (dS - dG).astype(np.int16)

    # Strict query-level witnesses.
    atomic_incompatibility_witness = gamma_a > 0
    compound_incompatibility_witness = gamma_c > 0
    gap_reduced = gamma_c < gamma_a
    gap_eliminated = (gamma_a > 0) & (gamma_c == 0)
    pure_compat_resolution = (dO == 0) & (dS > 0)
    individual_only_gain = (dO > 0) & (dS == 0)
    both_gain = (dO > 0) & (dS > 0)

    print()
    print('=' * 150)
    print('ALL-HYPOTHESIS SHARED REPAIR GAP')
    print('=' * 150)
    print('queries =', n)
    print('valid target hypotheses/query:', qstats(valid_n))
    print()
    print('ATOMIC')
    print('  individual O_G:', qstats(O_atomic))
    print('  exact shared S_G:', qstats(S_atomic))
    print('  gap Gamma=O-S:', qstats(gamma_a))
    print('  queries Gamma>0 =', int(atomic_incompatibility_witness.sum()), '/', n)
    print()
    print('COMPOUND')
    print('  individual O_G:', qstats(O_compound))
    print('  exact shared S_G:', qstats(S_compound))
    print('  gap Gamma=O-S:', qstats(gamma_c))
    print('  queries Gamma>0 =', int(compound_incompatibility_witness.sum()), '/', n)
    print()
    print('FAMILY EFFECT')
    print('  total individual gain sum(dO) =', int(dO.sum()))
    print('  total exact-shared gain sum(dS) =', int(dS.sum()))
    print('  total gap reduction sum(Gamma_a-Gamma_c) =', int(gap_reduction.sum()))
    print('  queries gap reduced =', int(gap_reduced.sum()), '/', n)
    print('  queries gap eliminated =', int(gap_eliminated.sum()), '/', n)
    print('  PURE compatibility-resolution queries (dO=0,dS>0) =', int(pure_compat_resolution.sum()), '/', n)
    print('  individual-only gain queries (dO>0,dS=0) =', int(individual_only_gain.sum()), '/', n)
    print('  both-gain queries (dO>0,dS>0) =', int(both_gain.sum()), '/', n)

    print()
    print('=' * 150)
    print('EXACT BUDGET-AWARE POLICY vs GENERIC GBS')
    print('=' * 150)
    print('Atomic GBS coverage:', qstats(G_atomic))
    print('Compound GBS coverage:', qstats(G_compound))
    print('Atomic exact coverage:', qstats(S_atomic))
    print('Compound exact coverage:', qstats(S_compound))
    print('sum compound effect under GBS =', int(dG.sum()))
    print('sum compound effect under exact shared objective =', int(dS.sum()))
    print('interaction sum [(Sc-Sa)-(Gc-Ga)] =', int(interaction.sum()))
    print('queries exact compound gain > GBS compound gain =', int(np.sum(dS > dG)), '/', n)

    # Pre-registered-style diagnostic gate for whether v2 compatibility work is justified.
    # This is on consumed FIT and is NOT a confirmatory statistical gate.
    gate = {
        'atomic_gap_exists_on_at_least_20pct_queries': bool(atomic_incompatibility_witness.mean() >= 0.20),
        'compound_reduces_total_gap': bool(gap_reduction.sum() > 0),
        'at_least_10pct_queries_are_pure_compatibility_resolution': bool(pure_compat_resolution.mean() >= 0.10),
        'exact_shared_compound_gain_exceeds_GBS_gain_in_total': bool(dS.sum() > dG.sum()),
    }
    gate['continue_algorithm_v2'] = bool(all(gate.values()))

    print()
    print('=' * 150)
    print('V2 CONTINUATION DIAGNOSTIC')
    print('=' * 150)
    for k, v in gate.items():
        print(k, '=', v)

    if gate['continue_algorithm_v2']:
        diagnosis = (
            'STRONG ALL-HYPOTHESIS COMPATIBILITY SIGNAL. Individual repairability and '
            'shared-policy realizability are measurably separated on many FIT frontiers; '
            'compound predicates reduce that separation, and the exact budget-aware shared '
            'objective exploits the expanded family better than generic balanced splitting. '
            'Proceed to conflict-guided predicate synthesis as a NEW v2 algorithm; do not '
            'modify frozen RCPS-v1 or P2.'
        )
    else:
        diagnosis = (
            'ALL-HYPOTHESIS COMPATIBILITY SIGNAL IS NOT STRONG ENOUGH UNDER THE FROZEN '
            'DIAGNOSTIC GATES. Keep RCPS-v1 as the empirical method, but do not elevate '
            'shared-certificate compatibility into the new algorithm core without a new '
            'mechanism hypothesis.'
        )

    print()
    print('DIAGNOSIS =', diagnosis)

    np.savez_compressed(
        OUT_NPZ,
        format=np.asarray('rcps_all_hypothesis_shared_gap_v0'),
        valid_hypothesis_count=valid_n,
        atomic_individual_count=O_atomic,
        compound_individual_count=O_compound,
        atomic_exact_shared_count=S_atomic,
        compound_exact_shared_count=S_compound,
        atomic_gbs_shared_count=G_atomic,
        compound_gbs_shared_count=G_compound,
        atomic_shared_gap=gamma_a,
        compound_shared_gap=gamma_c,
        individual_gain=dO,
        exact_shared_gain=dS,
        gbs_shared_gain=dG,
        shared_gap_reduction=gap_reduction,
        exact_vs_gbs_interaction=interaction,
        atomic_incompatibility_witness=atomic_incompatibility_witness,
        compound_incompatibility_witness=compound_incompatibility_witness,
        pure_compatibility_resolution=pure_compat_resolution,
        individual_only_gain=individual_only_gain,
        both_gain=both_gain,
        unique_compound_predicates=unique_compound,
    )

    result = {
        'name': 'RCPS All-Hypothesis Shared Repair Gap Audit v0',
        'scope': 'All 568 consumed FIT failure frontiers; no P1/P2; GT-independent O/S definitions.',
        'definition': {
            'target_universe': 'AST-valid Local80 candidate hypotheses; invalid candidates remain UNKNOWN blockers.',
            'O_G': 'number of target hypotheses individually B20-feasible with <=2 questions from family G',
            'S_G': 'maximum number of target hypotheses simultaneously B20-certified by one depth-2 shared policy under frozen BCCT objective',
            'Gamma_G': 'O_G - S_G',
        },
        'integrity': {
            **observed,
            'atomic_policy_mismatches_on_recoverable': int(atomic_policy_mismatch),
            'compound_policy_mismatches_on_recoverable': int(compound_policy_mismatch_on_recoverable),
            'atomic_policy_coverage_mismatches': int(atomic_policy_coverage_mismatch),
            'compound_policy_coverage_mismatches': int(compound_policy_coverage_mismatch),
            'atomic_real_GT_B20_mismatches': int(atomic_individual_gt_B20_mismatch),
            'compound_real_GT_B20_failures': int(compound_individual_gt_B20_mismatch),
        },
        'summary': {
            'queries': int(n),
            'valid_hypotheses': qstats(valid_n),
            'atomic_O': qstats(O_atomic),
            'atomic_S': qstats(S_atomic),
            'atomic_Gamma': qstats(gamma_a),
            'compound_O': qstats(O_compound),
            'compound_S': qstats(S_compound),
            'compound_Gamma': qstats(gamma_c),
            'total_individual_gain': int(dO.sum()),
            'total_exact_shared_gain': int(dS.sum()),
            'total_gap_reduction': int(gap_reduction.sum()),
            'queries_atomic_gap_positive': int(atomic_incompatibility_witness.sum()),
            'queries_compound_gap_positive': int(compound_incompatibility_witness.sum()),
            'queries_gap_reduced': int(gap_reduced.sum()),
            'queries_gap_eliminated': int(gap_eliminated.sum()),
            'pure_compatibility_resolution_queries': int(pure_compat_resolution.sum()),
            'individual_only_gain_queries': int(individual_only_gain.sum()),
            'both_gain_queries': int(both_gain.sum()),
            'atomic_GBS': qstats(G_atomic),
            'compound_GBS': qstats(G_compound),
            'total_GBS_compound_gain': int(dG.sum()),
            'exact_vs_GBS_interaction_sum': int(interaction.sum()),
            'queries_exact_gain_exceeds_GBS_gain': int(np.sum(dS > dG)),
        },
        'diagnostic_gate': gate,
        'diagnosis': diagnosis,
        'protocol': [
            'Consumed FIT only.',
            'No P1 access.',
            'No P2 access.',
            'No RCPS-v1 modification.',
            'All 568 FIT failure frontiers are included; selection does not require the real GT to be in Local80.',
            'Real GT is used only for implementation parity checks, not for O_G, S_G, or Gamma_G.',
            'Atomic family is the frozen 44 exact AST predicates.',
            'Compound family is frozen RCPS atomic + pairwise AND + pairwise OR with exact partition deduplication.',
            'B=20 and maximum depth=2 remain frozen.',
            'AST-invalid candidates stay in masks as UNKNOWN blockers; they are excluded only as answer-defining target hypotheses.',
        ],
        'runtime_seconds': float(time.perf_counter() - t0),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    out_sha = sha256_file(OUT_NPZ)

    print()
    print('=' * 150)
    print('AUDIT COMPLETE')
    print('=' * 150)
    print('artifact =', OUT_NPZ)
    print('artifact SHA256 =', out_sha)
    print('metrics =', OUT_JSON)
    print('runtime_seconds =', result['runtime_seconds'])


if __name__ == '__main__':
    main()
