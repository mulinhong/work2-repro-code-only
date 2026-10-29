import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT=Path("/root/autodl-tmp/work2-algorithm-innovation")
D1=ROOT/"a1_blr64_dev_F1_iYn2Qv3u"
A4=ROOT/"fhex_v1/scripts/audit_bcr_a4_rcps.py"
OUT=Path(sys.argv[1])

B=20
K=256
SALT=b"V14-CONTROL-GREEDY-V1"

A4_SHA="580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9"

CASE_SHA={
1:(
"9e92009a64ae3fb3d4233758f1a227e8345b1a1a1d6f83ae31f9e2d0b0529560",
"c4561e2a7d298a9e218912a48a9a3d2e3ce6fe342ec151befcb4cc60a33d1539"),
2:(
"6d264ec6954f608d699599cd4c6a605242939cf3ae3386aea14f137f46681f9d",
"2381cf5b687b728d973ce1c0d7af72fa00bba9094eccc6828458433db3c58a75"),
3:(
"1c3c9702cf289d610ac7ca002a0c277f224d7d474ad9c29474ffc7373a025841",
"ee62f1bfc4cc08b449aeaa52c855f20f2b279637222a1fdd1e4a4a26f7ef7643"),
4:(
"d20eeb4ce7e2d90bce0fd29b8f842b9b996e523343987686def12feaf2df0b7e",
"ab479c527ced27e14a2af66aa3c492a6d9325d99cb15a299f0bed3343393ab11"),
5:(
"ad24e47f6d52ba7b3d32688bd4ab0c6afabe88f79ee726b18a0a0f8c87d116c8",
"09aabf3c23b62294dfae099928c0cd5fcfc05eb9f97ee2e0501a28d2b761e3fb")
}

def log(x):
    print(x,flush=True)

def sha_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda:f.read(1048576),b""):
            h.update(x)
    return h.hexdigest()

def sha_array(x):
    return hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()

def save(path,obj):
    path.write_text(
        json.dumps(obj,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8"
    )

def load_a4():
    assert sha_file(A4)==A4_SHA
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location("frozen_a4_d2",A4)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.B==20 and module.E80==80
    return module

def greedy_fill(full,valid,selected,budget):
    p=full.shape[1]
    selected=list(selected)
    chosen=np.zeros(p,dtype=bool)
    chosen[selected]=True

    z0=np.cumsum(full!=1,axis=0,dtype=np.int16)
    z1=np.cumsum(full!=0,axis=0,dtype=np.int16)

    repair=np.where(full==0,z0,z1)<=B
    repair &= valid[:,None]
    repair[:B]=False

    covered=repair[:,selected].any(axis=1)

    priorities=[
        hashlib.sha256(SALT+full[:,j].tobytes()).digest()
        for j in range(p)
    ]

    while len(selected)<budget:
        available=np.flatnonzero(~chosen)
        gain=np.logical_and(
            repair[:,available],
            ~covered[:,None]
        ).sum(axis=0)

        maximum=int(gain.max())

        if maximum==0:
            fill=sorted(
                available.tolist(),
                key=lambda j:(priorities[j],j)
            )
            selected.extend(fill[:budget-len(selected)])
            break

        candidates=available[gain==maximum]
        j=min(
            (int(v) for v in candidates),
            key=lambda v:(priorities[v],v)
        )

        selected.append(j)
        chosen[j]=True
        covered |= repair[:,j]

    assert len(selected)==budget
    assert len(set(selected))==budget
    return selected

def witness_hits(full,valid,r,c0,c1,s0,s1):
    hits=np.zeros(80,dtype=bool)

    for t in np.flatnonzero(valid):
        a=int(full[t,r])
        assert a in (0,1)

        mask=(s0[r] if a==0 else s1[r]).copy()
        child=c0 if a==0 else c1

        if child>=0:
            b=int(full[t,child])
            assert b in (0,1)
            mask &= s0[child] if b==0 else s1[child]

        hits[t]=np.count_nonzero(mask[:t+1])<=B

    return hits

def build_witnesses(full,valid,a4):
    s0,s1=a4.build_survival_masks(full)
    roots=[]

    for r in range(full.shape[1]):
        answers=full[valid,r]

        if not (
            np.any(answers==0) and
            np.any(answers==1)
        ):
            continue

        left=a4.evaluate_stop(
            s0[r],np.flatnonzero(valid&s0[r])
        )
        right=a4.evaluate_stop(
            s1[r],np.flatnonzero(valid&s1[r])
        )

        roots.append((
            int(left["coverage"]+right["coverage"]),
            int(left["gain"]+right["gain"]),
            -max(
                int(left["worst_leaf"]),
                int(right["worst_leaf"])
            ),
            -r,r
        ))

    roots.sort(reverse=True)
    witnesses=[]

    for _,_,_,_,r in roots[:K]:
        left=a4.choose_second(
            full,valid,s0,s1,s0[r]
        )
        right=a4.choose_second(
            full,valid,s0,s1,s1[r]
        )

        c0=int(left["feature"])
        c1=int(right["feature"])

        hits=witness_hits(
            full,valid,r,c0,c1,s0,s1
        )

        coverage=int(left["coverage"]+right["coverage"])
        assert int(hits.sum())==coverage

        features=tuple(dict.fromkeys(
            x for x in (r,c0,c1) if x>=0
        ))

        witnesses.append({
            "root":r,
            "zero":c0,
            "one":c1,
            "features":features,
            "hits":hits,
            "coverage":coverage,
            "gain":int(left["gain"]+right["gain"]),
            "worst_leaf":max(
                int(left["worst_leaf"]),
                int(right["worst_leaf"])
            )
        })

    assert len(witnesses)>0

    witnesses.sort(key=lambda w:(
        -w["coverage"],
        -w["gain"],
        w["worst_leaf"],
        w["root"]
    ))

    return witnesses

def select_portfolio(full,valid,base,budget,witnesses):
    selected=list(range(base))
    chosen=set(selected)
    remaining=set(range(len(witnesses)))
    counts=np.zeros(80,dtype=np.int16)

    first=witnesses[0]
    seed=[
        x for x in first["features"]
        if x not in chosen
    ]

    assert len(selected)+len(seed)<=budget

    selected.extend(seed)
    chosen.update(seed)
    counts[first["hits"]]+=1

    portfolio=[0]
    remaining.remove(0)

    while remaining:
        candidates=[]

        for i in sorted(remaining):
            w=witnesses[i]
            new=[
                x for x in w["features"]
                if x not in chosen
            ]
            cost=len(new)

            if len(selected)+cost>budget:
                continue

            gain=float(np.sum(
                w["hits"]/(1.0+counts)
            ))

            score=gain/max(1,cost)

            candidates.append((
                score,
                w["coverage"],
                -cost,
                -w["root"],
                -i,i
            ))

        if not candidates:
            break

        i=max(candidates)[-1]
        w=witnesses[i]

        new=[
            x for x in w["features"]
            if x not in chosen
        ]

        selected.extend(new)
        chosen.update(new)
        counts[w["hits"]]+=1

        portfolio.append(i)
        remaining.remove(i)

    before_fill=len(selected)
    selected=greedy_fill(
        full,valid,selected,budget
    )

    return selected,{
        "witness_candidates":len(witnesses),
        "witnesses_selected":len(portfolio),
        "tree_guided_new_predicates":before_fill-base,
        "greedy_fill_predicates":budget-before_fill,
        "seed_joint_coverage":first["coverage"],
        "seed_root":first["root"],
        "witness_indices":portfolio
    }

def replay(library,valid,tree):
    if tree is None:
        return int(valid[:B].sum())

    root=int(tree["first_feature"])
    children=[
        int(tree["second_zero"]),
        int(tree["second_one"])
    ]

    score=0

    for t in np.flatnonzero(valid):
        a=int(library[t,root])
        assert a in (0,1)

        survival=(
            (library[:,root]==a)|
            (library[:,root]==-1)
        )

        child=children[a]

        if child>=0:
            b=int(library[t,child])
            assert b in (0,1)

            survival &= (
                (library[:,child]==b)|
                (library[:,child]==-1)
            )

        score+=int(
            np.count_nonzero(survival[:t+1])<=B
        )

    return score

def main():
    assert OUT.parent==ROOT
    assert OUT.name.startswith("a1_wpg256_dev_")
    assert not OUT.is_symlink()

    log("PHASE=1/4 SOURCE_AND_PROTOCOL")

    a4=load_a4()

    protocol={
        "phase":"A1-D3_DEVELOPMENT_ONLY",
        "baseline":"WPG256",
        "pilot_rows":[80,16,52,73,34],
        "E":80,
        "B":20,
        "depth":2,
        "max_leaves":4,
        "root_shortlist":K,
        "source":"A1-D1-F1 frozen TRAIN matrices",
        "root_order":"Single-split coverage and fixed ties",
        "witness":"Best child or STOP in both branches",
        "portfolio":"Diminishing-return recovered-target coverage per new predicate",
        "fill_salt":SALT.decode(),
        "original_BLR64_preserved":True,
        "adaptive_method_disclosure":
            "WPG256 designed after inspecting BLR16 and BLR64 TRAIN pilots",
        "not_confirmatory":True,
        "test_r12_access":False,
        "full_exact_solver_during_selection":False,
        "a4_sha256":A4_SHA,
        "input_sha256":CASE_SHA
    }
    save(OUT/"protocol.json",protocol)

    case_dir=OUT/"cases"
    case_dir.mkdir()

    rows=[]

    log("PHASE=2/4 FIVE_CASE_COMPARISON")

    for i in range(1,6):
        fn=D1/"cases"/f"case_{i:02}.npz"
        jn=D1/"cases"/f"case_{i:02}.json"

        assert sha_file(fn)==CASE_SHA[i][0]
        assert sha_file(jn)==CASE_SHA[i][1]

        previous=json.loads(jn.read_text())

        with np.load(fn,allow_pickle=False) as z:
            full=z["full"].copy()
            valid=z["valid"].copy()
            atomic=z["atomic"].copy()
            oldlibs={
                k:z[k].copy()
                for k in ("tecs","greedy","blr64")
            }

        assert full.shape[0]==80
        assert valid.shape==(80,)

        base=np.unique(atomic,axis=1).shape[1]
        budget=int(previous["budget"])

        assert base<=budget<=full.shape[1]
        assert np.all(full[~valid]==-1)

        lookup={
            full[:,j].tobytes():j
            for j in range(full.shape[1])
        }
        assert len(lookup)==full.shape[1]

        original_greedy=greedy_fill(
            full,valid,list(range(base)),budget
        )

        assert original_greedy==previous[
            "results"]["GREEDY"]["selected_full_indices"]

        core=[
            x for x in previous["core_features"]
            if x>=0
        ]

        fair_blr_indices=greedy_fill(
            full,valid,
            list(range(base))+[
                x for x in core if x not in range(base)
            ],
            budget
        )

        fair_blr=full[:,fair_blr_indices]

        t=time.perf_counter()
        witnesses=build_witnesses(
            full,valid,a4
        )
        witness_s=time.perf_counter()-t

        t=time.perf_counter()
        chosen,meta=select_portfolio(
            full,valid,base,budget,witnesses
        )
        portfolio_s=time.perf_counter()-t

        wpg=full[:,chosen]

        libraries={
            "TECS":oldlibs["tecs"],
            "GREEDY":oldlibs["greedy"],
            "BLR64":oldlibs["blr64"],
            "FAIR_BLR64":fair_blr,
            "WPG256":wpg,
            "FULL":full
        }

        results={}

        for name,library in libraries.items():
            expected=(
                full.shape[1]
                if name=="FULL" else budget
            )

            assert library.shape==(80,expected)
            assert np.all(library[~valid]==-1)

            ids=[
                lookup[library[:,j].tobytes()]
                for j in range(library.shape[1])
            ]
            assert len(ids)==len(set(ids))

            start=time.perf_counter()
            tree=a4.choose_bcct(library,valid)[0]
            solver_s=time.perf_counter()-start

            score=(
                int(tree["coverage"])
                if tree is not None
                else int(valid[:B].sum())
            )

            assert score==replay(
                library,valid,tree
            )

            if name in previous["results"]:
                assert score==previous[
                    "results"][name]["coverage"]

            policy=[]
            for k in (
                "first_feature",
                "second_zero",
                "second_one"
            ):
                j=int(tree[k]) if tree is not None else -1
                policy.append(ids[j] if j>=0 else -1)

            results[name]={
                "coverage":score,
                "solver_s":solver_s,
                "library_sha256":sha_array(library),
                "selected_full_indices":ids,
                "policy_full_indices":policy,
                "replay_pass":True
            }

        assert results["FULL"]["coverage"]>=max(
            r["coverage"] for r in results.values()
        )

        assert (
            results["WPG256"]["coverage"]>=
            meta["seed_joint_coverage"]
        )

        record={
            "case":i,
            "p3_row":int(previous["p3_row"]),
            "budget":budget,
            "full_predicates":int(full.shape[1]),
            "witness_evaluation_s":witness_s,
            "portfolio_selection_s":portfolio_s,
            "portfolio":meta,
            "results":results
        }
        rows.append(record)

        save(case_dir/f"case_{i:02}.json",record)

        np.savez_compressed(
            case_dir/f"case_{i:02}.npz",
            valid=valid,
            full=full,
            tecs=oldlibs["tecs"],
            greedy=oldlibs["greedy"],
            blr64=oldlibs["blr64"],
            fair_blr64=fair_blr,
            wpg256=wpg
        )

        log(
            f"PROGRESS={i}/5 PCT={i*20}% "
            f"ROW={previous['p3_row']} "
            f"TECS={results['TECS']['coverage']} "
            f"GREEDY={results['GREEDY']['coverage']} "
            f"BLR64={results['BLR64']['coverage']} "
            f"WPG256={results['WPG256']['coverage']} "
            f"FULL={results['FULL']['coverage']} "
            f"TREE_PRED={meta['tree_guided_new_predicates']} "
            f"REPLAY=PASS"
        )

    log("PHASE=3/4 POSTFLIGHT_SHA")

    assert sha_file(A4)==A4_SHA

    for i in range(1,6):
        assert sha_file(
            D1/"cases"/f"case_{i:02}.npz"
        )==CASE_SHA[i][0]
        assert sha_file(
            D1/"cases"/f"case_{i:02}.json"
        )==CASE_SHA[i][1]

    totals={
        name:sum(
            row["results"][name]["coverage"]
            for row in rows
        )
        for name in libraries
    }

    summary={
        "status":"PASS_DEV_ONLY_NOT_FORMAL",
        "cases":5,
        "policy_replays":30,
        "full_superset_checks":5,
        "coverage_totals":totals,
        "tree_guided_predicates":[
            x["portfolio"]["tree_guided_new_predicates"]
            for x in rows
        ],
        "witness_eval_total_s":sum(
            x["witness_evaluation_s"] for x in rows
        ),
        "portfolio_selection_total_s":sum(
            x["portfolio_selection_s"] for x in rows
        ),
        "formal_v14_executed":False
    }

    save(OUT/"summary.json",summary)

    log("PHASE=4/4 FINAL")
    log("COVERAGE_TOTALS="+str(totals))
    log("STATUS=PASS_DEV_ONLY_NOT_FORMAL")

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        (OUT/"FAIL_STOP.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8"
        )
        log("STATUS=FAIL_STOP ERROR="+repr(exc))
        sys.exit(20)
