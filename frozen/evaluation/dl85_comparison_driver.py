#!/usr/bin/env python3
"""SOTA S6: pinned DL8.5 vs pinned A4, 100 TRAIN queries, 3 paired repetitions."""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path('/root/autodl-tmp/work2-algorithm-innovation')
A1 = ROOT / 'a1_wpg256_formal_INSTALL_R2_20261027'
S5 = ROOT / 'sota_s5_dl85_official_R0'
S5R1 = ROOT / 'sota_s5r1_fulltecs_20261028_121702_163448'
PINS = {
    'A4': ('fhex_v1/scripts/audit_bcr_a4_rcps.py', '580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9'),
    'A1': ('a1_wpg256_formal_INSTALL_R2_20261027/main.py', 'c1826e79fac2775d4138348d591bf1847e078fb0b13f2ed3af9bc470ed146340'),
    'S5R1': ('sota_s5r1_fulltecs_20261028_121702_163448/run.py', '45e13381fc050ddf3945dfa0dca9df6150b092adb71d11ec1e17dbd87e652a05'),
}
DL85_COMMIT = 'a7f0fee72b4c17bbcb727050ba102db7dda10119'
CASES, REPS, LIMIT, HARD = 100, 3, 45, 85

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def require(ok, msg):
    if not ok:
        raise RuntimeError('FAIL_STOP_' + str(msg))

def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def readcase(p):
    with np.load(p, allow_pickle=False) as z:
        valid = np.asarray(z['valid'], dtype=bool)
        lib = np.ascontiguousarray(z['tecs'], dtype=np.int8)
        expect = int(z['expected'])
        p3 = int(z['p3_row'])
    require(valid.shape == (80,) and lib.ndim == 2 and lib.shape[0] == 80, 'DATA_SHAPE')
    require(np.isin(lib[valid], [0, 1]).all() and np.all(lib[~valid] == -1), 'UNKNOWN_DOMAIN')
    return valid, lib, expect, p3

def leaf_loss(tids, positive, invalid):
    ids = np.asarray(list(tids), dtype=np.int64)
    require(len(set(ids.tolist())) == len(ids), 'DUPLICATE_LEAF_ID')
    require(np.all((ids >= 0) & (ids < len(positive))), 'LEAF_ID_RANGE')
    mask = np.zeros(80, dtype=bool)
    mask[invalid] = True
    chosen = positive[ids]
    mask[chosen] = True
    return int(len(ids) - np.count_nonzero(np.cumsum(mask)[chosen] <= 20))

def worker(out, mode, rep, number):
    valid, lib, expected, p3 = readcase(out / 'inputs' / f'case_{number:03d}.npz')
    row = {'rep': rep, 'case': number, 'p3_row': p3, 'mode': mode,
           'expected': expected, 'features': int(lib.shape[1]),
           'input_sha256': sha(out / 'inputs' / f'case_{number:03d}.npz')}
    if mode == 'A4':
        source = ROOT / PINS['A4'][0]
        spec = importlib.util.spec_from_file_location('s6_pinned_a4', source)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        require(mod.E80 == 80 and mod.B == 20, 'A4_SETTINGS')
        t0 = time.perf_counter()
        tree = mod.choose_bcct(lib, valid)[0]
        row['solver_s'] = time.perf_counter() - t0
        row['coverage'] = int(tree['coverage']) if tree is not None else int(valid[:20].sum())
        row['timeout'] = False
        require(row['coverage'] == expected, f'A4_HISTORICAL_MISMATCH_{number}')
    elif mode == 'DL85':
        from pydl85 import DL85Predictor
        pos = np.flatnonzero(valid)
        invalid = np.flatnonzero(~valid)
        x = np.ascontiguousarray(lib[valid], dtype=np.int32)
        calls = [0]
        def loss(tids):
            calls[0] += 1
            return leaf_loss(tids, pos, invalid)
        model = DL85Predictor(max_depth=2, min_sup=1, error_function=loss,
                              leaf_value_function=lambda ids: 0,
                              time_limit=LIMIT, quiet=True)
        t0 = time.perf_counter()
        model.fit(x)
        row['solver_s'] = time.perf_counter() - t0
        row['timeout'] = bool(model.timeout_)
        row['tree'] = model.tree_
        row['leaf_calls'] = calls[0]
        row['lattice_size'] = int(model.lattice_size_)
        row['error'] = None if model.tree_ is None else float(model.error_)
        row['coverage'] = None
        if model.tree_ is not None:
            count = [0]
            def replay(node, ids, depth):
                if 'feat' not in node:
                    require(depth <= 2 and len(ids) > 0, 'REPLAY_LEAF')
                    error = leaf_loss(ids, pos, invalid)
                    require(abs(error - float(node['error'])) < 1e-7, 'REPLAY_LEAF_SCORE')
                    count[0] += 1
                    return error
                k = int(node['feat'])
                require(0 <= k < x.shape[1] and depth < 2, 'REPLAY_FEATURE')
                left = ids[x[ids, k] == 1]
                right = ids[x[ids, k] == 0]
                require(len(left) > 0 and len(right) > 0, 'REPLAY_EMPTY_BRANCH')
                return replay(node['left'], left, depth + 1) + replay(node['right'], right, depth + 1)
            err = replay(model.tree_, np.arange(len(pos)), 0)
            require(count[0] <= 4 and abs(err - row['error']) < 1e-7, 'REPLAY_TREE_ERROR')
            row['coverage'] = int(len(pos) - err)
            require(row['coverage'] <= expected, 'DL85_BETTER_THAN_A4_OPTIMUM')
            if not row['timeout']:
                require(row['coverage'] == expected, 'DL85_PROVEN_OPTIMUM_MISMATCH')
        else:
            require(row['timeout'], 'NO_TREE_NO_TIMEOUT')
    else:
        require(False, 'UNKNOWN_MODE')
    p = out / 'records' / f'r{rep:02d}_c{number:03d}_{mode}.json'
    save(p, row)
    return row

def independently_replay(r, inp):
    if r['mode'] != 'DL85' or r.get('tree') is None:
        return
    valid, lib, expected, _ = readcase(inp)
    pos = np.flatnonzero(valid)
    invalid = np.flatnonzero(~valid)
    x = lib[valid]
    count = [0]
    def rec(node, ids, depth):
        if 'feat' not in node:
            require(depth <= 2 and len(ids) > 0, 'HOST_REPLAY_LEAF')
            chosen = pos[ids]
            survivor = np.zeros(80, dtype=bool)
            survivor[invalid] = True
            survivor[chosen] = True
            ranking = np.cumsum(survivor)
            err = int(len(ids) - sum(ranking[p] <= 20 for p in chosen))
            require(err == float(node['error']), 'HOST_LEAF_ERROR')
            count[0] += 1
            return err
        feat = int(node['feat'])
        require(0 <= feat < x.shape[1] and depth < 2, 'HOST_TREE_DEPTH_FEATURE')
        a = ids[x[ids, feat] == 1]
        b = ids[x[ids, feat] == 0]
        require(len(a) > 0 and len(b) > 0, 'HOST_EMPTY_BRANCH')
        return rec(node['left'], a, depth + 1) + rec(node['right'], b, depth + 1)
    err = rec(r['tree'], np.arange(len(pos), dtype=int), 0)
    require(count[0] <= 4, 'HOST_EXCESS_LEAVES')
    require(len(pos) - err == r['coverage'] and abs(err - r['error']) < 1e-7, 'HOST_REPLAY_MISMATCH')
    require(r['coverage'] <= expected, 'HOST_BEATS_EXACT')

def preflight(out):
    require(out.parent == ROOT and out.name.startswith('sota_s6_dl85_train100_'), 'OUTPUT_SCOPE')
    require((S5 / 'status.txt').read_text().strip() == 'STATUS=PASS_S5_MICRO', 'S5_STATUS')
    require((S5R1 / 'status.txt').read_text().strip() == 'STATUS=PASS_S5R1_DIAGNOSTIC', 'S5R1_STATUS')
    require((A1 / 'status.txt').read_text().strip() == 'STATUS=PASS_100_RETROSPECTIVE_TRAIN_PAIRED', 'A1_STATUS')
    for label, (relative, digest) in PINS.items():
        require(sha(ROOT / relative) == digest, f'SOURCE_PIN_{label}')
    head = subprocess.check_output(['git', '-C', str(S5 / 'upstream'), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == DL85_COMMIT, 'DL85_COMMIT')
    require(not subprocess.check_output(['git', '-C', str(S5 / 'upstream'), 'status', '--porcelain'], text=True).strip(), 'DL85_WORKTREE_DIRTY')
    (out / 'inputs').mkdir()
    (out / 'records').mkdir()
    (out / 'worker_logs').mkdir()
    case_manifest = []
    for i in range(1, CASES + 1):
        source = A1 / 'cases' / f'case_{i:03d}.npz'
        meta = A1 / 'cases' / f'case_{i:03d}.json'
        require(source.is_file() and meta.is_file(), f'MISSING_CASE_{i}')
        with np.load(source, allow_pickle=False) as z:
            valid = np.asarray(z['valid'], bool)
            lib = np.ascontiguousarray(z['tecs'], dtype=np.int8)
        hist = json.loads(meta.read_text(encoding='utf-8'))
        expected = int(hist['modes']['TECS']['score'])
        budget = int(hist['budget'])
        p3 = int(hist['p3_row'])
        require(valid.shape == (80,) and lib.shape == (80, budget), f'CASE_SHAPE_{i}')
        require(np.isin(lib[valid], [0, 1]).all() and np.all(lib[~valid] == -1), f'CASE_DOMAIN_{i}')
        dst = out / 'inputs' / f'case_{i:03d}.npz'
        np.savez_compressed(dst, valid=valid, tecs=lib, expected=np.array(expected, np.int32), p3_row=np.array(p3, np.int32))
        case_manifest.append({'case': i, 'p3_row': p3, 'feature_count': budget,
                              'historical_expected': expected, 'original_input_sha256': sha(source),
                              'diagnostic_input_sha256': sha(dst), 'original_meta_sha256': sha(meta)})
        if i % 20 == 0:
            print(f'PREFLIGHT={i}/{CASES} PCT={i}%', flush=True)
    require(len({x['p3_row'] for x in case_manifest}) == CASES, 'DUPLICATE_P3_ROW')
    save(out / 'INPUT_MANIFEST.json', case_manifest)
    protocol = {
        'status': 'PRE_REGISTERED_TRAIN_ONLY', 'cases': CASES, 'repetitions': REPS,
        'depth': 2, 'max_leaves': 4, 'Top_B': 20, 'E': 80,
        'official_dl85_version': '0.1.8', 'official_commit': DL85_COMMIT,
        'dl85_internal_timeout_seconds': LIMIT, 'external_worker_timeout_seconds': HARD,
        'order': 'rep1 ascending, rep2 descending, rep3 rotated at 50; within each case first solver toggles with (case+rep)%2',
        'same_preselected_tecs_feature_library_per_method': True,
        'not_feature_selection_comparison': True,
        'no_test_R12_access': True, 'all_complete_or_censored_cases_preserved': True,
        'source_pins': PINS, 'input_manifest_sha256': sha(out / 'INPUT_MANIFEST.json'),
        'no_gpu_or_work1_process_modification': True,
        'statistical_unit': 'query, not repeated run',
        'report_solver_only_and_external_worker_wall_separately': True,
    }
    save(out / 'PROTOCOL.json', protocol)
    return case_manifest

def main(out):
    print('PHASE=1/4 PREFLIGHT_AND_FREEZE_100_TRAIN_INPUTS', flush=True)
    cases = preflight(out)
    print('PHASE=2/4 THREE_REPETITIONS_PAIRED_SOLVERS', flush=True)
    result = []
    errors = []
    for rep in range(1, REPS + 1):
        order = list(range(1, CASES + 1))
        if rep == 2:
            order.reverse()
        elif rep == 3:
            order = order[49:] + order[:49]
        for idx, i in enumerate(order):
            modes = ('A4', 'DL85') if (i + rep) % 2 == 0 else ('DL85', 'A4')
            for mode in modes:
                cmd = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', str(out), mode, str(rep), str(i)]
                log = out / 'worker_logs' / f'r{rep:02d}_c{i:03d}_{mode}.txt'
                t0 = time.perf_counter()
                try:
                    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=HARD)
                    elapsed = time.perf_counter() - t0
                    log.write_text((cp.stdout or '') + '\nSTDERR:\n' + (cp.stderr or ''), encoding='utf-8')
                    require(cp.returncode == 0, f'WORKER_FAILED_{rep}_{i}_{mode}_RC{cp.returncode}')
                    row = json.loads((out / 'records' / f'r{rep:02d}_c{i:03d}_{mode}.json').read_text())
                    require(row['mode'] == mode and row['rep'] == rep and row['case'] == i, 'ROW_IDENTITY')
                    require(row['input_sha256'] == sha(out / 'inputs' / f'case_{i:03d}.npz'), 'ROW_INPUT_DRIFT')
                    row['worker_wall_s'] = elapsed
                    independently_replay(row, out / 'inputs' / f'case_{i:03d}.npz')
                    if mode == 'DL85':
                        require(row['timeout'] or row['coverage'] == row['expected'], 'DL85_UNPROVEN_RESULT')
                    else:
                        require(row['coverage'] == row['expected'], 'A4_NOT_EXACT')
                except subprocess.TimeoutExpired as exc:
                    elapsed = time.perf_counter() - t0
                    if mode != 'DL85':
                        raise RuntimeError(f'FAIL_STOP_A4_EXTERNAL_TIMEOUT_REP{rep}_CASE{i}') from exc
                    log.write_text('DL85_EXTERNAL_TIMEOUT\n' + str(exc), encoding='utf-8')
                    row = {'rep': rep, 'case': i, 'mode': mode, 'p3_row': cases[i-1]['p3_row'],
                           'features': cases[i-1]['feature_count'], 'expected': cases[i-1]['historical_expected'],
                           'coverage': None, 'timeout': True, 'external_timeout': True,
                           'solver_s': None, 'worker_wall_s': elapsed}
                row['worker_wall_s'] = elapsed
                save(out / 'records' / f'r{rep:02d}_c{i:03d}_{mode}.json', row)
                result.append(row)
            done = ((rep - 1) * CASES + idx + 1)
            paired = result[-2:]
            by = {r['mode']:r for r in paired}
            print(f'PROGRESS={done}/{CASES*REPS} PCT={100*done/(CASES*REPS):.1f}% '
                  f'REP={rep} CASE={i} FEATURES={by["A4"]["features"]} '
                  f'A4={by["A4"]["coverage"]} DL85={by["DL85"]["coverage"]} '
                  f'TIMEOUT={by["DL85"]["timeout"]} '
                  f'A4_SOLVER_S={by["A4"].get("solver_s",-1):.4f} '
                  f'DL85_SOLVER_S={by["DL85"].get("solver_s")}', flush=True)
    print('PHASE=3/4 SUMMARY_AND_POSTFLIGHT', flush=True)
    for label, (relative, digest) in PINS.items():
        require(sha(ROOT / relative) == digest, f'POSTFLIGHT_{label}')
    for case in cases:
        dest = out / 'inputs' / f'case_{case["case"]:03d}.npz'
        require(sha(dest) == case['diagnostic_input_sha256'], 'INPUT_POSTFLIGHT_DRIFT')
    sums = []
    for rep in range(1, REPS + 1):
        a = [r for r in result if r['rep'] == rep and r['mode'] == 'A4']
        b = [r for r in result if r['rep'] == rep and r['mode'] == 'DL85']
        complete = [r for r in b if not r['timeout']]
        sums.append({'rep': rep, 'a4_solver_total_s': sum(r['solver_s'] for r in a),
                     'dl85_solver_observed_s': sum(r['solver_s'] or 0.0 for r in b),
                     'dl85_timeouts': len(b)-len(complete),
                     'dl85_proven_coverage_parity': len(complete),
                     'all_complete': len(complete) == CASES})
    with (out / 'PAIRED_RESULTS.csv').open('w', newline='') as f:
        cols = ['rep', 'case', 'p3_row', 'mode', 'features', 'expected', 'coverage', 'timeout', 'solver_s', 'worker_wall_s']
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result)
    timed = sum(r['timeout'] for r in result if r['mode'] == 'DL85')
    summary = {'status': 'PASS_100_TRAIN_WITH_CENSORS' if timed else 'PASS_100_TRAIN_PAIRED_EXACT',
               'queries': CASES, 'reps': REPS, 'paired_runs': CASES*REPS,
               'solver_jobs': len(result), 'dl85_timeouts': timed,
               'all_300_a4_historical_parity': True,
               'dl85_complete_and_replayed': CASES*REPS-timed,
               'per_rep': sums, 'wall_times_are_observational_not_isolated': True,
               'no_independent_test': True, 'no_claim_full_search_budget_equal_to_feature_selection': True}
    # Query is the statistical unit: repetitions of a query are not new queries.
    complete = timed == 0
    if complete:
        import numpy as np
        aggregated = []
        for number in range(1, CASES + 1):
            ar = [r for r in result if r['case'] == number and r['mode'] == 'A4']
            dr = [r for r in result if r['case'] == number and r['mode'] == 'DL85']
            require(len(ar) == REPS and len(dr) == REPS, 'MISSING_REPETITIONS')
            aggregated.append((sum(r['solver_s'] for r in ar), sum(r['solver_s'] for r in dr)))
        ag = np.asarray(aggregated, dtype=np.float64)
        ratio = float(ag[:, 1].sum() / ag[:, 0].sum())
        rng = np.random.default_rng(20261028)
        indices = rng.integers(0, CASES, size=(10000, CASES))
        bootstrap = ag[indices].sum(axis=1)
        ratios = bootstrap[:, 1] / bootstrap[:, 0]
        summary['solver_only_pooled_dl85_over_a4'] = ratio
        summary['query_cluster_bootstrap_95ci'] = [float(x) for x in np.quantile(ratios, [0.025, 0.975])]
        summary['query_cluster_bootstrap_seed'] = 20261028
        summary['individual_queries_a4_faster'] = int((ag[:, 0] < ag[:, 1]).sum())
    else:
        summary['solver_only_pooled_dl85_over_a4'] = None
        summary['query_cluster_bootstrap_95ci'] = None
        summary['speedup_claim_suppressed_due_to_censoring'] = True
    save(out / 'SUMMARY.json', summary)
    with (out / 'PAPER_TABLE.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rep', 'a4_solver_total_s', 'dl85_solver_observed_s', 'dl85_timeouts', 'dl85_proven_coverage_parity', 'all_complete'])
        writer.writeheader()
        writer.writerows(sums)
    lines = ['# DL8.5 vs A4: frozen TRAIN100 evidence (solver-only)', '',
             f'- Dataset: {CASES} historical TRAIN cases, {REPS} repetitions, {len(result)} total isolated solver jobs.',
             f'- DL8.5: official pydl8.5 0.1.8, pinned commit {DL85_COMMIT}.',
             f'- Constraints: E=80, Top-B=20, depth=2, at most 4 leaves, identical full TECS-selected predicate library per query.',
             f'- Correctness: {CASES*REPS}/{CASES*REPS} A4 source/historical parity; {CASES*REPS-timed}/{CASES*REPS} DL8.5 runs complete and replayed; {timed} censored timeouts.',
             '- Scope: TRAIN retrospective only; this is NOT independent test evidence or TECS feature-selection algorithm comparison.',
             '- Timing: solver-only, separate fresh processes, order alternated; Work1/GPU keepalive unchanged.',
             '- Limitations: a single machine, fixed 45-second DL8.5 limit and observational CPU contention; disclose all negative results.', '']
    if complete:
        ci = summary['query_cluster_bootstrap_95ci']
        lines += [f'- Pooled DL8.5/A4 solver-only time ratio: {summary["solver_only_pooled_dl85_over_a4"]:.6f}x.',
                  f'- Query-resampled 95% percentile CI: [{ci[0]:.6f}, {ci[1]:.6f}] (100 query clusters, not 300 independent samples).']
    else:
        lines += ['- No exact pooled speed ratio or ordinary bootstrap CI reported because some DL8.5 results were censored.']
    lines += ['', '## Per-repetition totals', '']
    for entry in sums:
        lines.append(f'- Round {entry["rep"]}: A4={entry["a4_solver_total_s"]:.6f}s; observed DL8.5={entry["dl85_solver_observed_s"]:.6f}s; timeouts={entry["dl85_timeouts"]}.')
    (out / 'PAPER_EVIDENCE_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('PHASE=4/4 DONE', flush=True)
    print('STATUS=' + summary['status'], flush=True)

if __name__ == '__main__':
    try:
        if len(sys.argv) == 6 and sys.argv[1] == '--worker':
            worker(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
        elif len(sys.argv) == 2:
            main(Path(sys.argv[1]))
        else:
            raise RuntimeError('FAIL_STOP_ARGUMENTS')
    except BaseException as exc:
        if len(sys.argv) == 2:
            (Path(sys.argv[1])/'FAIL_STOP.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(20)
