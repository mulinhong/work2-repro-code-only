import json, sys, time, hashlib, importlib.util
from pathlib import Path
import numpy as np
from pydl85 import DL85Predictor

ROOT=Path('/root/autodl-tmp/work2-algorithm-innovation')
A1=ROOT/'a1_wpg256_formal_INSTALL_R2_20261027'
OUT=Path(sys.argv[1])

assert OUT.parent==ROOT
assert OUT.name.startswith('sota_s5_dl85_')
assert (A1/'status.txt').read_text().strip()=='STATUS=PASS_100_RETROSPECTIVE_TRAIN_PAIRED'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert sha(A1/'main.py')=='c1826e79fac2775d4138348d591bf1847e078fb0b13f2ed3af9bc470ed146340'

source=ROOT/'fhex_v1/scripts/audit_bcr_a4_rcps.py'
assert sha(source)=='580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9'

spec=importlib.util.spec_from_file_location('s5_a4',source)
a4=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=a4
spec.loader.exec_module(a4)

assert a4.E80==80 and a4.B==20

records=[]

for i in range(1,6):
    print(f'PROGRESS=CASE {i}/5 START',flush=True)

    with np.load(A1/'cases'/f'case_{i:03d}.npz',
                 allow_pickle=False) as z:
        valid=z['valid'].astype(bool)
        full=z['full'].astype(np.int8)

    historical=json.loads(
        (A1/'cases'/f'case_{i:03d}.json').read_text()
    )

    indices=np.linspace(
        0,full.shape[1]-1,8,dtype=int
    )

    X=np.ascontiguousarray(
        full[valid][:,indices],dtype=np.int32
    )

    assert X.shape[1]==8
    assert np.isin(X,(0,1)).all()

    valid_pos=np.flatnonzero(valid)
    invalid_pos=np.flatnonzero(~valid)
    calls=[0]

    def leaf_error(tids):
        ids=np.asarray(list(tids),dtype=int)
        assert len(set(ids.tolist()))==len(ids)
        assert (ids>=0).all()
        assert (ids<len(valid_pos)).all()

        survivors=np.zeros(80,dtype=bool)
        survivors[invalid_pos]=True

        chosen=valid_pos[ids]
        survivors[chosen]=True

        ranks=np.cumsum(
            survivors,dtype=np.int16
        )

        covered=int(
            (ranks[chosen]<=20).sum()
        )

        calls[0]+=1
        return int(len(ids)-covered)

    def leaf_value(tids):
        return 0

    frozen_tree=a4.choose_bcct(
        np.ascontiguousarray(full[:,indices]),
        valid
    )[0]

    expected=(
        int(frozen_tree['coverage'])
        if frozen_tree is not None
        else int(valid[:20].sum())
    )

    start=time.perf_counter()

    model=DL85Predictor(
        max_depth=2,
        min_sup=1,
        error_function=leaf_error,
        leaf_value_function=leaf_value,
        time_limit=90,
        quiet=True
    )

    model.fit(X)
    elapsed=time.perf_counter()-start

    err=getattr(model,'error_',None)

    if err is None or not np.isfinite(float(err)):
        raise RuntimeError(
            f'NO_FINITE_DL85_OPTIMUM CASE={i}'
        )

    dl85_coverage=len(valid_pos)-float(err)
    parity=abs(dl85_coverage-expected)<1e-7

    row={
        'case':i,
        'p3_row':historical['p3_row'],
        'valid_count':int(valid.sum()),
        'feature_indices':indices.tolist(),
        'a4_optimum':expected,
        'dl85_coverage':dl85_coverage,
        'dl85_error':float(err),
        'wall_s':elapsed,
        'leaf_error_calls':calls[0],
        'parity':parity
    }

    records.append(row)

    (OUT/f'case_{i:03d}.json').write_text(
        json.dumps(row,indent=2)+'\n'
    )

    print(
        f'PROGRESS=CASE {i}/5 '
        f'A4={expected} '
        f'DL85={dl85_coverage} '
        f'PARITY={parity} '
        f'WALL_S={elapsed:.3f} '
        f'LEAF_CALLS={calls[0]}',
        flush=True
    )

    if not parity:
        raise RuntimeError(
            f'FAIL_STOP_COVERAGE_MISMATCH_{i}'
        )

summary={
    'status':'PASS_DL85_OFFICIAL_CORE_MICRO_EQUIVALENCE',
    'cases':5,
    'records':records,
    'scope':'Historical TRAIN diagnostic only',
    'not_formal_sota_timing':True,
    'not_independent_test':True,
    'original_code_modified':False,
    'custom_additive_leaf_error':True
}

(OUT/'SUMMARY.json').write_text(
    json.dumps(summary,indent=2)+'\n'
)

print(
    'STATUS=PASS_DL85_OFFICIAL_CORE_MICRO_EQUIVALENCE',
    flush=True
)
