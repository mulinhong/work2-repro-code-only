from pathlib import Path
from collections import Counter, defaultdict
import fcntl, hashlib, json, math, os, random, re
import resource, signal, subprocess, sys, time
import tarfile, traceback

ROOT = Path('/root/autodl-tmp/work2-algorithm-innovation')
O = ROOT/'tecs_paper_r4e2_three_round_replication_v1/round_01'
R3 = ROOT/'tecs_paper_r3b_fourway_train82_v1'
R4 = ROOT/'tecs_paper_r4d_o3_streed_train82_v1'
R0 = ROOT/'tecs_paper_r4c0_train82_input_contract_v1'
E0 = ROOT/'tecs_paper_r4e0_historical_paired_audit_v1'

PYTHON = '/root/miniconda3/envs/work2-algo/bin/python'
BINARY = R4/'rankloss_cli'

PINS = {
 'r3b_summary':
 '96fc53d92b73c5fb056d392f26afc1581530c63292b5fa9d3a8f865b24b12079',
 'streed_binary':
 '15487bc335d99b16f69ac79adff765526282a8765995c55ac815b4eb906ad1b2',
 'streed_protocol':
 '9abe431c5a688de436f62869b212d657c7ad8912fb0fda9735df9c7f240e84c5',
 'r4d_archive':
 'b452434c2c5c1316784750747f41e7f0bcf6a33a333ef60c2b0676394c88120f',
 'r4e0_archive':
 '0b4e2377743c8bd312072454097673357c50bb5119428f29d5f89fd72c9f4dd7'
}

MODES = ('RPC','STREED')
NATIVE_CAP = 60.0
RPC_WALL_CAP = 90.0
STREED_WALL_CAP = 75.0
GIB = 12

def say(msg):
    print(msg,flush=True)

def need(condition,message):
    if not condition:
        raise RuntimeError('FAIL_STOP_'+message)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()

def stable(obj):
    return json.dumps(
        obj,sort_keys=True,indent=2,
        ensure_ascii=False
    )+'\n'

def write_immutable(p,text):
    if p.exists():
        need(p.read_text()==text,
             'IMMUTABLE_DRIFT_'+p.name)
    else:
        p.write_text(text)

def pin(p,digest,name):
    need(p.is_file() and not p.is_symlink(),
         'MISSING_'+name)
    actual=sha(p)
    need(actual==digest,
         'PIN_'+name+'_'+actual)
    say('PIN_'+name+'=PASS')

def load(p):
    return json.loads(p.read_text())

def child_limits():
    limit=GIB*1024**3
    resource.setrlimit(
        resource.RLIMIT_AS,(limit,limit)
    )
    os.sched_setaffinity(0,{CPU})

def run_child(mode,q,case,logs):
    log=logs/f'q{q:03d}_{mode.lower()}.stdout.txt'

    if mode=='RPC':
        cmd=[
            PYTHON,'-u',str(WORKER),
            '--query-id',str(q),
            '--tag','r4e1'
        ]
        cap=RPC_WALL_CAP
    else:
        cmd=[str(BINARY),str(case)]
        cap=STREED_WALL_CAP

    env=os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES':'',
        'OPENBLAS_NUM_THREADS':'1',
        'OMP_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1',
        'NUMEXPR_NUM_THREADS':'1',
        'PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONUNBUFFERED':'1',
        'WORK2_R3B_OUT':str(SPOOL)
    })

    start=time.monotonic()
    timed_out=False

    with log.open('w') as f:
        proc=subprocess.Popen(
            cmd,cwd=str(O),
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=child_limits,
            start_new_session=True
        )

        last=0
        while True:
            try:
                proc.wait(timeout=3)
                break

            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-start

                if elapsed-last>=15:
                    say(
                        f'HEARTBEAT QUERY={q} MODE={mode} '
                        f'ELAPSED_S={elapsed:.1f} '
                        f'WALL_CAP_S={cap}'
                    )
                    last=elapsed

                if elapsed>=cap:
                    timed_out=True
                    try:
                        os.killpg(
                            proc.pid,signal.SIGKILL
                        )
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    break

    elapsed=time.monotonic()-start

    return {
        'returncode':proc.returncode,
        'wall_s':round(elapsed,6),
        'watchdog':timed_out,
        'stdout':log.read_text(errors='replace'),
        'log':str(log.relative_to(O))
    }

def native_result(text):
    matches=re.findall(
        r'RESULT STATUS=(OPTIMAL|TIME_LIMIT) '
        r'RECOVERED=(\d+)\s+'
        r'TARGETS=(\d+)\s+'
        r'NODES=(\d+)\s+'
        r'DEPTH=(\d+)\s+'
        r'SECONDS=([\d.eE+\-]+)',
        text
    )
    need(len(matches)<=1,'MULTIPLE_NATIVE_RESULTS')
    return matches[0] if matches else None

def family_hash(case):
    # Same valid masks and feature matrix:
    # target flags deliberately omitted.
    h=hashlib.sha256()

    with case.open() as f:
        header=f.readline().split()
        need(len(header)==5,'FAMILY_HEADER')
        p=int(header[1])
        h.update((' '.join(header)+'\n').encode())

        for line in f:
            row=line.split()
            need(len(row)==p+2,'FAMILY_ROW')
            h.update(
                (' '.join(
                    (row[0],*row[2:])
                )+'\n').encode()
            )

    return h.hexdigest()

def ci_bootstrap(rows,seed,n=5000):
    rng=random.Random(seed)
    values=[]

    for _ in range(n):
        sample=[
            rows[rng.randrange(len(rows))]
            for j in range(len(rows))
        ]
        a=sum(x['rpc_s'] for x in sample)
        b=sum(x['streed_s'] for x in sample)
        values.append(b/a)

    values.sort()

    def quantile(p):
        x=(len(values)-1)*p
        i=int(x)
        j=min(i+1,len(values)-1)
        return values[i]+(
            values[j]-values[i]
        )*(x-i)

    return [quantile(.025),quantile(.975)]

def ci_cluster(groups,seed,n=5000):
    rng=random.Random(seed)
    clusters=list(groups.values())
    values=[]

    for _ in range(n):
        sample=[
            clusters[rng.randrange(len(clusters))]
            for j in range(len(clusters))
        ]
        a=sum(
            x['rpc_s']
            for group in sample for x in group
        )
        b=sum(
            x['streed_s']
            for group in sample for x in group
        )
        values.append(b/a)

    values.sort()

    def quantile(p):
        x=(len(values)-1)*p
        i=int(x)
        j=min(i+1,len(values)-1)
        return values[i]+(
            values[j]-values[i]
        )*(x-i)

    return [quantile(.025),quantile(.975)]

def main():
    O.mkdir(exist_ok=True)

    lock=(O/'RUN.lock').open('a+')
    try:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX|fcntl.LOCK_NB
        )
    except BlockingIOError:
        say('FINAL_STATUS=ALREADY_RUNNING_R4E1')
        return

    say('========== WORK2 R4E1 PAIRED BENCHMARK ==========')
    say(
        'NO_TEST_R12=YES NO_WORK1=YES '
        'NO_GPU=YES NO_FROZEN_WRITES=YES'
    )

    say('PHASE=1/5 VERIFY_FROZEN_SOURCES')

    for key,path in [
        ('r3b_summary',R3/'SUMMARY.json'),
        ('streed_binary',BINARY),
        ('streed_protocol',R4/'PROTOCOL.json'),
        ('r4d_archive',
         R4/'PAPER_R4D_STREED_O3_TRAIN82.evidence.tar.gz'),
        ('r4e0_archive',
         E0/'PAPER_R4E0_HISTORICAL_AUDIT.evidence.tar.gz')
    ]:
        pin(path,PINS[key],key.upper())

    r3proto=load(R3/'PROTOCOL.json')

    need(
        sha(R3/'PROTOCOL.json')
        ==load(R3/'SUMMARY.json')['protocol_sha256'],
        'R3B_PROTOCOL'
    )

    origin=R3/'r3b_derived_worker.py'

    pin(
        origin,
        r3proto['derived_worker_sha256'],
        'R3B_DERIVED_WORKER'
    )

    need(
        r3proto['kind']
        =='development_only_not_independent_dev_or_test',
        'TRAIN_SCOPE'
    )

    for rel,expected in (
        r3proto['data_and_source_sha256'].items()
    ):
        path=Path(rel)
        if not path.is_absolute():
            path=ROOT/rel

        need(
            'r12' not in str(path).lower()
            and 'test' not in str(path).lower(),
            'FORBIDDEN_PATH'
        )
        need(
            path.is_file()
            and sha(path)==expected,
            'SOURCE_DRIFT_'+rel
        )

    say('R3B_FROZEN_SOURCE_PINS=PASS')

    # Exact source block from the frozen R3B controller.
    old='''        order = list(MODES)
        rotation = int(qi) % len(order)
        order = order[rotation:] + order[:rotation]
        results = {}
        for which in order:
            print(f"PROGRESS QUERY={qi} MODE={which} START", flush=True)
            results[which] = run_solver(which)
            print(f"PROGRESS QUERY={qi} MODE={which} DONE", flush=True)
        generic = results["GENERIC"]
        rpc = results["RPC"]
        dq = results["DQ_ONLY"]
        final = results["FINAL_LAZY_DUAL"]'''

    new='        # Frozen run_solver core; only orchestration changes.\n        import signal as _sig\n        import time as _time\n        import json as _json\n        import os as _os\n        from pathlib import Path as _Path\n\n        class _TimeCap(BaseException):\n            pass\n\n        def _alarm(a,b):\n            raise _TimeCap()\n\n        _prior=_sig.getsignal(_sig.SIGALRM)\n        _start=_time.perf_counter()\n        _status="OPTIMAL"\n        _value=None\n\n        print(f"PROGRESS QUERY={qi} MODE=RPC START",flush=True)\n\n        _sig.signal(_sig.SIGALRM,_alarm)\n        try:\n            _sig.setitimer(_sig.ITIMER_REAL,60.0)\n            try:\n                _value=run_solver("RPC")\n            except _TimeCap:\n                _status="TIME_LIMIT"\n        finally:\n            _sig.setitimer(_sig.ITIMER_REAL,0.0)\n            _sig.signal(_sig.SIGALRM,_prior)\n\n        _elapsed=_time.perf_counter()-_start\n\n        if _status=="OPTIMAL":\n            assert isinstance(_value,dict)\n            assert int(_value["value"])==int(expected_joint)\n            assert _value["runtime"]>=0\n            assert _value["runtime"]<=_elapsed+0.1\n\n            if "rpc" in frozen_rpc and "states" in frozen_rpc["rpc"]:\n                assert int(_value["states"])==int(frozen_rpc["rpc"]["states"])\n\n            _answer=dict(_value)\n            _answer["status"]="OPTIMAL"\n        else:\n            _answer={\n                "status":"TIME_LIMIT",\n                "runtime":60.0,\n                "value":None,\n                "states":None\n            }\n\n        _answer["wrapper_solver_wall_s"]=float(_elapsed)\n\n        _payload={\n            "processed":1,\n            "rpc_only":True,\n            "limit_method":"SIGALRM_60S",\n            "records":[{\n                "query":int(qi),\n                "joint_K5":int(expected_joint),\n                "rpc":_answer\n            }]\n        }\n\n        _folder=_Path(_os.environ["WORK2_R3B_OUT"])\n        _folder.mkdir(parents=True,exist_ok=True)\n\n        _target=_folder/f"r3b_case_{int(qi):03d}.json"\n        _temp=_target.with_suffix(".json.tmp")\n\n        _temp.write_text(\n            _json.dumps(_payload,sort_keys=True)+chr(10)\n        )\n        _os.replace(_temp,_target)\n\n        print(\n            f"PROGRESS QUERY={qi} MODE=RPC DONE STATUS={_status} "\n            f"SOLVER_WALL_S={_elapsed:.4f}",flush=True\n        )\n        return'

    source=origin.read_text()
    need(
        source.count(old)==1,
        'RPC_ONLY_UNIQUE_PATCH_ANCHOR'
    )

    modified=source.replace(old,new)
    need(hashlib.sha256(modified.encode()).hexdigest() == '76303d287974e109167a8d789180fd7835f63886983ecc854714e916e046efbd', 'GENERATED_WORKER_SHA')
    compile(modified,'r4e1_rpc_only.py','exec')

    global WORKER,SPOOL,CPU

    WORKER=O/'r4e1_rpc_only.py'
    SPOOL=O/'rpc_raw'
    SPOOL.mkdir(exist_ok=True)

    write_immutable(WORKER,modified)
    need(sha(WORKER)=='76303d287974e109167a8d789180fd7835f63886983ecc854714e916e046efbd','FROZEN_R7_RPC_WORKER_SHA')

    need(
        modified.count('run_solver("RPC")')==1,
        'RPC_ONLY_SOLVER_CALL_COUNT'
    )

    cpus=sorted(os.sched_getaffinity(0))
    need(bool(cpus),'CPU_AFFINITY')
    CPU=cpus[0]

    manifest=load(
        R0/'R4C0_CASE_MANIFEST.json'
    )
    entries={
        int(x['query']):x
        for x in manifest['cases']
    }
    ids=sorted(entries)

    need(
        len(ids)==82
        and sorted(r3proto['case_ids'])==ids,
        'TRAIN82_QUERY_SET'
    )

    need(
        load(R4/'SUMMARY.json')['counts']
        =={
            'PARITY_PASS':80,
            'TIME_LIMIT_CENSORED':2
        },
        'R4D_RESULT_COUNTS'
    )

    families={}

    for i,q in enumerate(ids,1):
        case=R0/'cases'/f'case_{q:03d}.txt'
        entry=entries[q]

        need(
            case.is_file()
            and sha(case)==entry['case_sha256'],
            'CASE_SHA_'+str(q)
        )

        original=load(
            R3/f'r3b_case_{q:03d}.json'
        )['records'][0]

        need(
            original['rpc']['value']
            ==entry['expected_recovered'],
            'HISTORICAL_RPC_OBJECTIVE_'+str(q)
        )

        families[q]=family_hash(case)

        if i%20==0 or i==82:
            say(
                f'PROGRESS PHASE=INPUT_VERIFY '
                f'DONE={i}/82 '
                f'PCT={100*i/82:.1f}%'
            )

    protocol={
        'study':'R4E1 same-session paired Train82',
        'r3b_protocol_sha256':sha(R3/'PROTOCOL.json'),
        'original_worker_sha256':sha(origin),
        'rpc_only_worker_sha256':sha(WORKER),
        'streed_binary_sha256':sha(BINARY),
        'input_manifest_sha256':
            sha(R0/'R4C0_CASE_MANIFEST.json'),
        'queries':ids,
        'execution':'one CPU core, serial, alternating order',
        'cpu_core':CPU,
        'native_solver_budget_s':NATIVE_CAP,
        'rpc_wall_watchdog_s':RPC_WALL_CAP,
        'streed_wall_watchdog_s':STREED_WALL_CAP,
        'memory_limit_gib':GIB,
        'no_test_r12':True,
        'no_work1':True,
        'no_gpu':True,
        'limitation':(
            'RPC and STreeD native solver timers '
            'exclude somewhat different preprocessing.'
        )
    }
    protocol.update({'study':'R4E2 three-round frozen Train82 replication','replication_round':1,'order_policy':'REVERSE','source_r7_sha256':'08627d5ac394d8475635350912c6ff60f64286acef6ac7248c499e6736bc2b91','rpc_timeout_method':'SIGALRM_ITIMER_REAL_60S','timing_scope_caveat':'RPC and STreeD native timing scopes differ'})

    pp=O/'PROTOCOL.json'
    write_immutable(pp,stable(protocol))
    ph=sha(pp)

    say('PROTOCOL_SHA256='+ph)
    say(f'CPU_CORE={CPU}')
    say('PHASE=2/5 RUN_82_PAIRED_CASES')

    records=O/'records'
    logs=O/'case_logs'
    records.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)

    for i,q in enumerate(ids,1):
        entry=entries[q]
        case=R0/'cases'/f'case_{q:03d}.txt'
        path=records/f'q{q:03d}.json'

        if path.exists():
            row=load(path)
            need(
                row['query']==q
                and row['protocol_sha256']==ph
                and row['case_sha256']==entry['case_sha256']
                and row['rpc_worker_sha']==sha(WORKER)
                and row['streed_binary_sha']==sha(BINARY),
                'RESUME_'+str(q)
            )
            say(
                f'PROGRESS PHASE=CASES DONE={i}/82 '
                f'PCT={100*i/82:.1f}% '
                f'QUERY={q} REUSED=VERIFIED'
            )
            continue

        order=(('STREED','RPC') if i%2 else ('RPC','STREED'))

        say(
            f'CASE_START INDEX={i}/82 '
            f'QUERY={q} P={entry["P"]} '
            f'EXPECTED={entry["expected_recovered"]} '
            f'ORDER={order}'
        )

        pair={}

        for mode in order:
            result=run_child(mode,q,case,logs)

            if mode=='RPC':
                rawpath=SPOOL/f'r3b_case_{q:03d}.json'

                if result['watchdog']:
                    status='WALL_TIMEOUT_CENSORED'
                    solver_s=None
                    observed=None
                else:
                    need(
                        result['returncode']==0 and rawpath.is_file(),
                        'RPC_WORKER_Q'+str(q)+'_TAIL_'+result['stdout'][-300:]
                    )

                    receipt=load(rawpath)

                    need(
                        receipt.get('processed')==1
                        and receipt.get('rpc_only') is True
                        and receipt.get('limit_method')=='SIGALRM_60S'
                        and len(receipt.get('records',[]))==1,
                        'RPC_RECEIPT_Q'+str(q)
                    )

                    data=receipt['records'][0]

                    need(
                        int(data['query'])==q
                        and int(data['joint_K5'])==entry['expected_recovered'],
                        'RPC_QUERY_ID_Q'+str(q)
                    )

                    rpc=data['rpc']
                    solver_s=float(rpc['runtime'])
                    wrapper_s=float(rpc['wrapper_solver_wall_s'])

                    need(
                        math.isfinite(solver_s) and solver_s>=0
                        and math.isfinite(wrapper_s) and wrapper_s>=0,
                        'RPC_RUNTIME_Q'+str(q)
                    )

                    if rpc.get('status')=='OPTIMAL':
                        observed=int(rpc['value'])

                        need(
                            observed==entry['expected_recovered']
                            and solver_s<=NATIVE_CAP+0.01
                            and solver_s<=wrapper_s+0.1,
                            'RPC_EXACTNESS_Q'+str(q)
                        )

                        status='PARITY_PASS'

                    elif rpc.get('status')=='TIME_LIMIT':
                        need(
                            rpc['value'] is None
                            and solver_s==NATIVE_CAP,
                            'RPC_TIMEOUT_Q'+str(q)
                        )

                        observed=None
                        status='TIME_LIMIT_CENSORED'
                    else:
                        raise RuntimeError(
                            'RPC_UNKNOWN_STATUS_Q'+str(q)
                        )

            else:
                parsed=native_result(
                    result['stdout']
                )

                if result['watchdog']:
                    status='WALL_TIMEOUT_CENSORED'
                    observed=None
                    solver_s=None
                else:
                    need(
                        parsed is not None,
                        'STREED_MISSING_RESULT_'+str(q)
                    )

                    state,v,targets,nodes,depth,seconds=parsed
                    observed=int(v)
                    solver_s=float(seconds)

                    need(
                        int(targets)==entry['target_rows']
                        and observed<=entry['expected_recovered'],
                        'STREED_VALUE_'+str(q)
                    )

                    if (
                        result['returncode']==0
                        and state=='OPTIMAL'
                    ):
                        need(
                            observed==entry['expected_recovered'],
                            'STREED_OPTIMAL_'+str(q)
                        )
                        status='PARITY_PASS'
                    elif (
                        result['returncode']==3
                        and state=='TIME_LIMIT'
                    ):
                        status='TIME_LIMIT_CENSORED'
                    else:
                        raise RuntimeError(
                            'STREED_FAILURE_'+str(q)
                        )

            pair[mode]={
                'status':status,
                'observed':observed,
                'solver_only_s':solver_s,
                'capped_solver_s':(
                    min(solver_s,NATIVE_CAP)
                    if solver_s is not None
                    else NATIVE_CAP
                ),
                'wall_s':result['wall_s'],
                'watchdog':result['watchdog'],
                'returncode':result['returncode'],
                'log':result['log']
            }

            say(
                f'PROGRESS PHASE=SOLVER '
                f'QUERY={q} MODE={mode} '
                f'STATUS={status} '
                f'SOLVER_S={solver_s} '
                f'WALL_S={result["wall_s"]}'
            )

        # Fail early if even the easiest initial query
        # cannot certify under the declared protocol.
        if q==6:
            need(
                pair['RPC']['status']=='PARITY_PASS'
                and pair['STREED']['status']=='PARITY_PASS',
                'CANARY_QUERY6_BUDGET_OR_SEMANTICS'
            )
            say('CANARY_QUERY6_BOTH_OPTIMAL=PASS')

        row={
            'query':q,
            'index':i,
            'family_sha256':families[q],
            'case_sha256':entry['case_sha256'],
            'expected':entry['expected_recovered'],
            'protocol_sha256':ph,
            'rpc_worker_sha':sha(WORKER),
            'streed_binary_sha':sha(BINARY),
            **pair
        }

        tmp=path.with_suffix('.json.tmp')
        tmp.write_text(stable(row))
        tmp.replace(path)

        say(
            f'PROGRESS PHASE=CASES DONE={i}/82 '
            f'PCT={100*i/82:.1f}% '
            f'QUERY={q} '
            f'RPC={pair["RPC"]["status"]} '
            f'STREED={pair["STREED"]["status"]}'
        )

    say('PHASE=3/5 PAIRED_STATISTICS')

    rows=[
        load(records/f'q{q:03d}.json')
        for q in ids
    ]

    certified=[
        r for r in rows
        if r['RPC']['status']=='PARITY_PASS'
        and r['STREED']['status']=='PARITY_PASS'
    ]

    comparisons=[
        {
            'query':r['query'],
            'family':r['family_sha256'],
            'rpc_s':r['RPC']['solver_only_s'],
            'streed_s':r['STREED']['solver_only_s']
        }
        for r in certified
    ]

    rpc_sum=sum(x['rpc_s'] for x in comparisons)
    st_sum=sum(x['streed_s'] for x in comparisons)

    groups=defaultdict(list)
    for x in comparisons:
        groups[x['family']].append(x)

    ci_query=(
        ci_bootstrap(comparisons,20261024)
        if comparisons else None
    )
    ci_family=(
        ci_cluster(groups,20261025)
        if comparisons else None
    )

    wins_rpc=sum(
        x['rpc_s']<x['streed_s']
        for x in comparisons
    )
    wins_st=sum(
        x['rpc_s']>x['streed_s']
        for x in comparisons
    )

    counts={
        mode:dict(Counter(
            r[mode]['status'] for r in rows
        ))
        for mode in MODES
    }

    capped={
        mode:sum(
            r[mode]['capped_solver_s']
            for r in rows
        )
        for mode in MODES
    }

    summary={
        'scope':'same-core same-session Train82',
        'protocol_sha256':ph,
        'n':82,
        'joint_certified':len(certified),
        'unique_feature_families':len(groups),
        'counts':counts,
        'wins':{
            'RPC':wins_rpc,
            'STREED':wins_st,
            'ties':len(comparisons)-wins_rpc-wins_st
        },
        'joint_rpc_solver_s':rpc_sum,
        'joint_streed_solver_s':st_sum,
        'joint_streed_over_rpc_ratio':(
            st_sum/rpc_sum if rpc_sum else None
        ),
        'query_bootstrap_95ci':ci_query,
        'family_cluster_bootstrap_95ci':ci_family,
        'all82_capped_solver_s':capped,
        'all82_capped_streed_over_rpc_ratio':
            capped['STREED']/capped['RPC'],
        'censored':[
            {
                'query':r['query'],
                'rpc_status':r['RPC']['status'],
                'streed_status':r['STREED']['status']
            }
            for r in rows
            if (
                r['RPC']['status']!='PARITY_PASS'
                or r['STREED']['status']!='PARITY_PASS'
            )
        ],
        'limitations':[
            'Solver-only timer scopes differ slightly',
            'Only one contemporaneous paired pass',
            'Train queries may be correlated',
            'RPC 60s solver cutoff is retrospective when a full run finishes',
            'RPC external watchdog is 90s',
            'Not an independent test set',
            'STreeD-derived RankLoss is not stock STreeD'
        ]
    }
    summary['replication_round']=1
    summary['order_policy']='REVERSE'
    summary['limitations']=['RPC uses SIGALRM 60-second cutoff; signal delivery can be deferred in native calls' if 'retrospective' in s else s for s in summary['limitations']]

    write_immutable(
        O/'SUMMARY.json',stable(summary)
    )

    say(
        f'SUMMARY JOINT_CERTIFIED={len(certified)}/82 '
        f'RPC_WINS={wins_rpc} STREED_WINS={wins_st}'
    )
    say(
        'SUMMARY JOINT_STREED_OVER_RPC_RATIO='
        +str(summary['joint_streed_over_rpc_ratio'])
    )
    say('SUMMARY QUERY_BOOTSTRAP_95CI='+str(ci_query))
    say('SUMMARY FAMILY_BOOTSTRAP_95CI='+str(ci_family))
    say(
        'SUMMARY ALL82_CAPPED_RATIO='
        +str(summary['all82_capped_streed_over_rpc_ratio'])
    )

    say('PHASE=4/5 PACKAGE_EVIDENCE')

    archive=O/'PAPER_R4E1_SAMECORE_PAIRED.evidence.tar.gz'

    with tarfile.open(archive,'w:gz') as tf:
        paths=[
            O/'PROTOCOL.json',
            O/'SUMMARY.json',
            WORKER,
            *sorted(records.glob('q*.json')),
            *sorted(logs.glob('q*.stdout.txt'))
        ]
        for p in paths:
            tf.add(
                p,
                arcname=str(p.relative_to(O)),
                recursive=False
            )

    say('EVIDENCE_PATH='+str(archive))
    say('EVIDENCE_SHA256='+sha(archive))

    say('PHASE=5/5 FINAL')

    if (
        len(certified)>=75
        and summary['joint_streed_over_rpc_ratio']>=1.2
        and ci_family is not None
        and ci_family[0]>1.0
    ):
        say(
            'FINAL_STATUS=PASS_R4E1_EXPLORATORY_ADVANTAGE_NEEDS_REPLICATION'
        )
    else:
        say(
            'FINAL_STATUS=PASS_R4E1_PAIRED_DATA_NO_STRONG_ADVANTAGE_GATE'
        )

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        say('ERROR='+repr(exc))
        traceback.print_exc()
        say('FINAL_STATUS=FAIL_STOP_R4E1')
        sys.exit(1)
