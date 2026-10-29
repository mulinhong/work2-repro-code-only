#!/usr/bin/env python3
"""B2: new independent repeat timing, frozen R3B worker, TRAIN82 only."""
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/root/autodl-tmp/work2-algorithm-innovation')
B1 = ROOT / 'rpc_b1_source_lock_BZpBLm1p'
OUT = Path(sys.argv[1])
MODES = ('generic', 'rpc', 'dq_only', 'final_lazy_dual')
WORKER_SHA = 'f3e08c098c9d868c7c42405307ac60e0ad721554b76a33df721aabb70a421a45'
HISTORICAL = 3.1624692067388707


def log(message):
    print(message, flush=True)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError('FAIL_STOP_' + message)


def capture_environment(label):
    block = {'label': label, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'hostname': platform.node(), 'python': sys.version,
             'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
             'loadavg': os.getloadavg(), 'cpu_count': os.cpu_count()}
    try:
        block['meminfo'] = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                            for line in Path('/proc/meminfo').read_text().splitlines()
                            if line.startswith(('MemAvailable:', 'MemTotal:', 'SwapFree:'))}
    except OSError as exc:
        block['memory_error'] = repr(exc)
    for tag, cmd in [('processes', ['ps', '-eo', 'pid,ppid,%cpu,%mem,stat,args', '--sort=-%cpu']),
                     ('gpu', ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,power.draw', '--format=csv,noheader'])]:
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=12, check=False)
            block[tag] = '\n'.join(result.stdout.splitlines()[:28]) if tag == 'processes' else result.stdout.strip()
            block[tag + '_exit'] = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            block[tag + '_unavailable'] = repr(exc)
    return block


def main():
    require(OUT.parent == ROOT and OUT.name.startswith('rpc_b2_repeats_') and not OUT.is_symlink(), 'OUTPUT_ISOLATION')
    require(not (OUT / 'SUMMARY.json').exists(), 'ALREADY_COMPLETED')
    log('PHASE=1/4 B1_PROTOCOL_AND_SOURCE_SHA_GATE')
    require((B1 / 'status.txt').read_text().strip() == 'STATUS=PASS_B1', 'B1_STATUS')
    require((B1 / 'exit_code.txt').read_text().strip() == '0', 'B1_EXIT')
    previous = json.loads((B1 / 'SUMMARY.json').read_text())
    require(previous['status'] == 'PASS_B1_SOURCE_AND_PROTOCOL_PREFLIGHT', 'B1_SUMMARY')
    fixed = json.loads((B1 / 'B2_FROZEN_PROTOCOL.json').read_text())
    require(fixed['new_repetitions'] == 3 and fixed['worker_sha256'] == WORKER_SHA, 'B1_PROTOCOL')
    plans = fixed['query_orders']
    require([v['repeat'] for v in plans] == [1, 2, 3], 'REPETITION_IDS')
    ids = plans[0]['order']
    require(len(ids) == 82 and len(set(ids)) == 82 and ids == sorted(ids), 'COHORT82')
    require(plans[1]['order'] == ids[::-1] and plans[2]['order'] == ids[41:] + ids[:41], 'QUERY_ORDER')
    require(fixed['mode_rotation'] == 'query_id_mod_4_unchanged', 'MODE_ROTATION')
    source_hashes = {}
    for label, item in fixed['sources'].items():
        relative = Path(item['path'])
        require(not relative.is_absolute() and all('test' not in p.lower() and 'r12' not in p.lower() and not p.lower().startswith('work1') for p in relative.parts), 'FORBIDDEN_PATH_' + label)
        p = ROOT / relative
        require(p.is_file() and not p.is_symlink() and digest(p) == item['sha256'], 'SOURCE_DRIFT_' + label)
        source_hashes[label] = p
    worker = B1 / 'r3b_derived_worker.py'
    require(worker.is_file() and digest(worker) == WORKER_SHA, 'WORKER_SHA')
    copy = OUT / 'frozen_worker.py'
    copy.write_bytes(worker.read_bytes())
    require(digest(copy) == WORKER_SHA, 'WORKER_COPY_SHA')
    expected = json.loads((B1 / 'HISTORICAL_82_REFERENCE.json').read_text())
    expectations = {int(v['query']): v for v in expected}
    require(sorted(expectations) == ids, 'HISTORICAL82')
    rpc_ref = json.loads(source_hashes['RPC82'].read_text())
    dq_ref = json.loads(source_hashes['DQ82'].read_text())
    rpc_records = {int(r['query']): r for r in rpc_ref['records']}
    dq_records = {int(r['query']): r for r in dq_ref['records']}
    require(sorted(rpc_records) == ids and sorted(dq_records) == ids, 'FROZEN_82_ALIGNMENT')
    write(OUT / 'B2_PROTOCOL_PINNED.json', fixed)
    write(OUT / 'SOURCE_SHA256.json', {k: {'path': str(v), 'sha256': fixed['sources'][k]['sha256']} for k, v in source_hashes.items()})
    write(OUT / 'ENVIRONMENT_PREFLIGHT.json', capture_environment('preflight'))
    log('PINS=15/15 PASS QUERY_IDS=82 REPETITIONS=3 WORKER_SHA=PASS')

    ps = subprocess.run(['ps', '-eo', 'pid,args'], text=True, capture_output=True, check=False)
    require(ps.returncode == 0, 'PROCESS_PREFLIGHT')
    conflicts = []
    for line in ps.stdout.splitlines()[1:]:
        pieces = line.strip().split(None, 1)
        if len(pieces) != 2 or not pieces[0].isdigit():
            continue
        pid, command = int(pieces[0]), pieces[1]
        if pid != os.getpid() and ('r3b_derived_worker.py' in command or '/frozen_worker.py' in command):
            comm = Path('/proc') / str(pid) / 'comm'
            try:
                process_name = comm.read_text().strip().lower()
            except OSError:
                continue
            if process_name.startswith('python'):
                conflicts.append((pid, command[:300]))
    require(not conflicts, 'CONCURRENT_RPC_WORKER')
    log('PROCESS_CONFLICT_GATE=PASS (GPU keepalive not treated as conflict)')
    log('PHASE=2/4 THREE_PAIRED_REPETITIONS')
    all_repetitions = []
    for rep in plans:
        n = rep['repeat']
        repdir = OUT / f'repeat_{n:02d}'
        workdir = repdir / 'worker_results'
        logs = repdir / 'worker_logs'
        workdir.mkdir(parents=True)
        logs.mkdir()
        write(repdir / 'environment_before.json', capture_environment(f'repeat_{n}_before'))
        totals = {mode: 0.0 for mode in MODES}
        rows = []
        log(f'REPEAT={n}/3 START QUERY_ORDER={"ASC" if n == 1 else "DESC" if n == 2 else "SHIFT41"}')
        with (repdir / 'paired_rows.jsonl').open('w', encoding='utf8') as journal:
            for i, q in enumerate(rep['order'], 1):
                result_path = workdir / f'r3b_case_{q:03d}.json'
                require(not result_path.exists(), 'OUTPUT_COLLISION')
                env = dict(os.environ)
                env['WORK2_R3B_OUT'] = str(workdir)
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                cmd = [sys.executable, '-B', '-u', str(copy), '--query-id', str(q), '--max-hard-queries', '1', '--tag', f'B2_REPEAT_{n}']
                before = {'loadavg': os.getloadavg(), 'monotonic': time.monotonic()}
                observed_mode_order = []
                with (logs / f'query_{q:03d}.txt').open('w', encoding='utf8') as logfile:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, bufsize=1)
                    for line in child.stdout:
                        logfile.write(line)
                        if line.startswith(f'PROGRESS QUERY={q} MODE=') and line.rstrip().endswith(' START'):
                            observed_mode_order.append(line.split(' MODE=', 1)[1].split(' ', 1)[0])
                        if line.startswith('PROGRESS QUERY=') or line.startswith(f'q={q:03d} '):
                            log(f'REPEAT={n} CASE={i}/82 {line.strip()}')
                    child.stdout.close()
                    returncode = child.wait()
                require(returncode == 0 and result_path.is_file(), f'WORKER_EXIT_REP{n}_Q{q}_RC{returncode}')
                nominal = ['GENERIC', 'RPC', 'DQ_ONLY', 'FINAL_LAZY_DUAL']
                rotation = q % 4
                nominal = nominal[rotation:] + nominal[:rotation]
                require(observed_mode_order == nominal, 'MODE_ORDER_OR_INCOMPLETE_WORKER_' + str(q))
                data = json.loads(result_path.read_text())
                require(data['processed'] == 1 and len(data['records']) == 1, 'WRONG_WORKER_CASE_COUNT')
                record = data['records'][0]
                ref = expectations[q]
                require(int(record['query']) == q and int(record['joint_K5']) == int(ref['expected_optimum']), 'OBJECTIVE_MISMATCH')
                for mode in MODES:
                    part = record[mode]
                    require(int(part['value']) == int(ref['expected_optimum']) and 0 < float(part['runtime']) < 1e7, 'VALUE_OR_RUNTIME_' + mode)
                    require(int(part['states']) >= 0, 'INVALID_STATE_' + mode)
                    totals[mode] += float(part['runtime'])
                for mode, name in [('generic', 'generic'), ('rpc', 'rpc')]:
                    require(int(record[mode]['states']) == int(ref[name + '_states']), 'FROZEN_STATES_' + mode)
                final_ref = dq_records[q]['rpc_dq']
                for metric in ('states', 'branch_options', 'pruned'):
                    if metric in final_ref:
                        require(int(record['final_lazy_dual'][metric]) == int(final_ref[metric]), 'FINAL_COUNTER_' + metric)
                if 'dq_only' in dq_records[q]:
                    dq_expected = dq_records[q]['dq_only']
                    for metric in ('states', 'branch_options', 'pruned'):
                        if metric in dq_expected:
                            require(int(record['dq_only'][metric]) == int(dq_expected[metric]), 'DQ_COUNTER_' + metric)
                observation = {'repeat': n, 'position': i, 'query': q,
                               'optimum': ref['expected_optimum'], 'mode_order': observed_mode_order, 'query_wall_s': time.monotonic() - before['monotonic'],
                               'load_before': before['loadavg'], 'load_after': os.getloadavg(),
                               'modes': {mode: {key: record[mode][key] for key in ('value', 'runtime', 'states', 'branch_options', 'pruned', 'cache_hits', 'cache_misses')} for mode in MODES}}
                journal.write(json.dumps(observation, ensure_ascii=False) + '\n')
                journal.flush()
                rows.append(observation)
                log(f'REPEAT={n}/3 DONE={i}/82 PCT={i*100/82:.1f}% Q={q} G={record["generic"]["runtime"]:.3f}s R={record["rpc"]["runtime"]:.3f}s CUM_G_R={totals["generic"]/totals["rpc"]:.4f}x PARITY=PASS')
        with (repdir / 'PAIRED_ROWS.csv').open('w', newline='', encoding='utf8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['repeat', 'position', 'query', 'optimum', 'mode_order', 'query_wall_s'] +
                            [f'{m}_runtime_s' for m in MODES] + [f'{m}_states' for m in MODES])
            for item in rows:
                writer.writerow([item['repeat'], item['position'], item['query'], item['optimum'],
                                 '|'.join(item['mode_order']), item['query_wall_s']] +
                                [item['modes'][m]['runtime'] for m in MODES] +
                                [item['modes'][m]['states'] for m in MODES])
        ratio = totals['generic'] / totals['rpc']
        result = {'repeat': n, 'query_count': len(rows), 'totals_solver_seconds': totals,
                  'generic_rpc_ratio': ratio, 'query_order': rep['order'],
                  'all_values_equal': True, 'deterministic_state_gates': True}
        write(repdir / 'REPETITION_SUMMARY.json', result)
        write(repdir / 'environment_after.json', capture_environment(f'repeat_{n}_after'))
        all_repetitions.append(result)
        write(OUT / 'REPETITIONS_COMPLETED_SO_FAR.json', all_repetitions)
        for label, path in source_hashes.items():
            require(digest(path) == fixed['sources'][label]['sha256'], 'POST_REPEAT_SHA_' + label)
        require(digest(copy) == WORKER_SHA, 'WORKER_DRIFT')
        log(f'REPEAT={n}/3 PASS GENERIC={totals["generic"]:.4f}s RPC={totals["rpc"]:.4f}s SPEEDUP={ratio:.5f}x')

    log('PHASE=3/4 RUN_TO_RUN_STABILITY')
    ratios = [r['generic_rpc_ratio'] for r in all_repetitions]
    summary = {'status': 'PASS_B2_THREE_REPETITIONS', 'cohort': 'R3B_TRAIN82',
               'new_repetitions': 3, 'successful_query_executions': 246,
               'successful_solver_mode_executions': 984, 'historical_ratio_separate': HISTORICAL,
               'new_ratios': ratios, 'new_mean_ratio': statistics.mean(ratios),
               'new_sample_sd_ratio': statistics.stdev(ratios),
               'new_min_ratio': min(ratios), 'new_max_ratio': max(ratios),
               'new_cv_ratio': statistics.stdev(ratios) / statistics.mean(ratios),
               'repetitions': all_repetitions,
               'notes': 'Solver-only. Three repeated runs, not external generalization or 3 independent datasets. Keep historical separate.'}
    write(OUT / 'SUMMARY.json', summary)
    write(OUT / 'ENVIRONMENT_POSTFLIGHT.json', capture_environment('postflight'))
    log('PHASE=4/4 FINAL_SHA_AND_STATUS')
    for label, path in source_hashes.items():
        require(digest(path) == fixed['sources'][label]['sha256'], 'FINAL_SOURCE_SHA_' + label)
    require(digest(copy) == WORKER_SHA, 'FINAL_WORKER_SHA')
    log('STATUS=PASS_B2_THREE_REPETITIONS RATIOS=' + ','.join(f'{x:.5f}' for x in ratios))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (OUT / 'FAIL_STOP.txt').write_text(traceback.format_exc(), encoding='utf8')
        log('STATUS=FAIL_STOP ERROR=' + repr(exc))
        sys.exit(20)
