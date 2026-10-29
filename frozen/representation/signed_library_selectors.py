import csv, hashlib, importlib.util, json, os, sys, tarfile, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(os.environ['V14_ROOT']).resolve()
OUT=Path(os.environ['V14_OUT']).resolve()
assert OUT.parent==ROOT and OUT.name.startswith('paper_v14_disjoint_controls_')
E,B,F=80,20,44
SEEDS=(13,29,42,73,101)
MODES=('TECS','FULL','GREEDY','SEED_13','SEED_29','SEED_42','SEED_73','SEED_101')
PER_SHARD=20
SKIP_PER_SHARD=4
T0=time.monotonic()
PINS={
 'A4':('fhex_v1/scripts/audit_bcr_a4_rcps.py','580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9'),
 'TECS':('fhex_v1/frozen_v2/audit_tree_embeddable_certificate_synthesis_v0.py','12c31549062575e68b4cd0daa57659ad4572d372c16d11cf366a8c5ba7221d98'),
 'RETRIEVAL':('fhex_v1/prospective/p3_jters_v2/retrieval_v3_trainonly_exactkernel/jters_p3_qwen3_top500_trainonly_exactkernel_v1.npz','d119a1180e560b182e50f152ce5b8a403c193c38292e3dab50678000b687f323'),
 'AST':('fhex_v1/cache/python_train_ast_signature_v0.npz','4316bf9327754a47a0ad97e868087cd8c4ced236800b11346a3d073f733f0e36'),
 'P3_PROTOCOL':('fhex_v1/frozen_v2/jters_v2_frozen_protocol.json','ce22ddf88da8a1c15727346d2679248d102eb11a65a54b641c99095394b95601')
}
def say(s):print(s,flush=True)
def phase(i,s):say(f'PHASE={i}/6 {s} ELAPSED_S={time.monotonic()-T0:.1f}')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for part in iter(lambda:f.read(1048576),b''): h.update(part)
 return h.hexdigest()
def write_json(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def imp(p,n):
 s=importlib.util.spec_from_file_location(n,p)
 assert s and s.loader
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def unique(lib):
 out=[];seen=set()
 for j in range(lib.shape[1]):
  col=np.asarray(lib[:,j],dtype=np.int8)
  key=col.tobytes()
  if key not in seen:seen.add(key);out.append(col.copy())
 return np.stack(out,axis=1),seen

def full_signed(atomic,a4,tecs):
 pos=[atomic[:,j].copy() for j in range(F)]
 neg=[tecs.tri_not(v) for v in pos]
 out=list(pos);seen={v.tobytes() for v in out}
 for p in range(F):
  for q in range(p+1,F):
   for u in (pos[p],neg[p]):
    for v in (pos[q],neg[q]):
     c=np.asarray(a4.tri_and(u,v),dtype=np.int8)
     if c.tobytes() not in seen:seen.add(c.tobytes());out.append(c.copy())
 return np.stack(out,axis=1)

def hash_pick(full,base,num,seed):
 assert base<=num<=full.shape[1]
 salt=f'V12-SIGNED-BUDGET-{seed}'.encode()
 indices=sorted(range(base,full.shape[1]),key=lambda j:(hashlib.sha256(salt+full[:,j].tobytes()).digest(),j))
 keep=list(range(base))+indices[:num-base]
 return full[:,keep].copy()

def greedy_pick(full,valid,base,num):
 """Noncertificate target-aware rank-prefix set cover; fixed deterministic tie order."""
 assert base<=num<=full.shape[1]
 # For every possible target and feature, find whether one compatible response repairs it.
 z0=np.cumsum(full!=1,axis=0,dtype=np.int16)
 z1=np.cumsum(full!=0,axis=0,dtype=np.int16)
 repair=((np.where(full==0,z0,z1)<=B) & valid[:,None])
 repair[:B,:]=False
 selected=list(range(base));chosen=np.zeros(full.shape[1],dtype=bool);chosen[:base]=True
 covered=np.any(repair[:,:base],axis=1)
 salt=b'V14-CONTROL-GREEDY-V1'
 priority=[hashlib.sha256(salt+full[:,j].tobytes()).digest() for j in range(full.shape[1])]
 for _ in range(num-base):
  avail=np.flatnonzero(~chosen)
  if not len(avail):break
  gain=np.logical_and(repair[:,avail],(~covered)[:,None]).sum(axis=0)
  best=int(gain.max())
  if best==0:break
  good=avail[gain==best]
  j=min((int(v) for v in good),key=lambda k:(priority[k],k))
  selected.append(j);chosen[j]=True;covered|=repair[:,j]
 if len(selected)<num:
  fill=sorted(np.flatnonzero(~chosen).tolist(),key=lambda j:(priority[j],j))
  selected+=fill[:num-len(selected)]
 assert len(selected)==num and len(set(selected))==num
 return full[:,selected].copy()

def oracle(facts,valid):
 r0=np.cumsum(facts!=1,axis=0,dtype=np.int16)
 r1=np.cumsum(facts!=0,axis=0,dtype=np.int16)
 count=0
 for t in np.flatnonzero(valid):
  a=facts[t]
  rank=np.where(a==0,r0[t],r1[t])
  count+=int(t<B or bool(np.any(rank<=B)))
 return count

def replay(facts,valid,tree):
 if tree is None:return int(valid[:B].sum())
 p=int(tree['first_feature']); assert 0<=p<facts.shape[1]
 children=(int(tree['second_zero']),int(tree['second_one']))
 total=0
 for t in np.flatnonzero(valid):
  a=int(facts[t,p]);assert a in (0,1)
  subset=(facts[:,p]==a)|(facts[:,p]==-1)
  child=children[a]
  if child>=0:
   assert child<facts.shape[1]
   b=int(facts[t,child]);assert b in (0,1)
   subset &= (facts[:,child]==b)|(facts[:,child]==-1)
  total+=int(np.count_nonzero(subset[:t+1])<=B)
 return total

def make_archive(status):
 (OUT/'V14_STATUS.txt').write_text(status+'\n')
 with (OUT/'V14_OUTPUT_SHA256.txt').open('w') as f:
  for p in sorted(OUT.iterdir()):
   if p.is_file() and p.name not in ('V14_OUTPUT_SHA256.txt','EVIDENCE_V14.tar.gz','RUN_FULL.txt'):
    f.write(sha(p)+'  '+p.name+'\n')
 pkg=OUT/'EVIDENCE_V14.tar.gz'
 with tarfile.open(pkg,'w:gz') as tar:
  for p in sorted(OUT.iterdir()):
   if p.is_file() and p.name not in ('EVIDENCE_V14.tar.gz','RUN_FULL.txt'):
    tar.add(p,arcname=p.name,recursive=False)
 say('FINAL_STATUS='+status)
 say('PACKAGE='+str(pkg))
 say('PACKAGE_SHA256='+sha(pkg))

def main():
 phase(1,'VERIFY_PINNED_INPUTS_READ_ONLY')
 pinned={};pins=[]
 for name,(rel,expected) in PINS.items():
  p=ROOT/rel
  assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT)
  actual=sha(p)
  assert actual==expected,('FAIL_STOP_SOURCE_PIN',name,actual)
  pinned[name]=p;pins.append({'name':name,'path':rel,'sha256':actual,'status':'PASS'})
  say(f'PIN={name} PASS')
 write_json(OUT/'V14_SOURCE_PINS.json',pins)
 a4=imp(pinned['A4'],'v14_a4')
 tecs=imp(pinned['TECS'],'v14_tecs')
 assert a4.E80==E and a4.B==B and tecs.B==B
 phase(2,'PRELOCK_DISJOINT_TRAIN_COHORT_AND_PROTOCOL')
 with np.load(pinned['RETRIEVAL'],allow_pickle=False) as z:
  rank=np.asarray(z['qwen3_indices'],dtype=np.int64)
  gt=np.asarray(z['ground_truth_code_indices'],dtype=np.int64)
  shards=np.asarray(z['shard_ids'],dtype=np.int64)
  source_row=np.asarray(z['source_row_indices'],dtype=np.int64)
 assert rank.shape==(6000,500) and gt.shape==(6000,) and np.array_equal(gt,source_row)
 misses=np.flatnonzero(~(rank[:,:100]==gt[:,None]).any(axis=1))
 old=[];selected=[]
 for s in range(5):
  candidates=misses[shards[misses]==s]
  assert len(candidates)>=SKIP_PER_SHARD+PER_SHARD
  old.extend(int(v) for v in candidates[:SKIP_PER_SHARD])
  selected.extend(int(v) for v in candidates[SKIP_PER_SHARD:SKIP_PER_SHARD+PER_SHARD])
 assert len(old)==20 and len(selected)==100 and len(set(selected))==100
 assert not set(old).intersection(selected)
 cohort=[{'p3_row':p,'shard':int(shards[p]),'source_row_index':int(source_row[p])} for p in selected]
 co_hash=hashlib.sha256(json.dumps(cohort,sort_keys=True).encode()).hexdigest()
 proto={'mode':'prelocked_disjoint_TRAIN_only','selection':'For each of 5 fixed shards, Top100-failure rows offsets [4,24), excluding V12/V13 pilot offsets [0,4)', 'E':E,'B':B,'F':F,'cases':100,'depth_max':2,'implicit_max_leaves':4,'modes':MODES,'hash_seeds':SEEDS,'method':'frozen_A4_choose_bcct_exhaustive','noncertificate_strong_control':'greedy one-question rank-prefix repair-set coverage over full signed space, deterministic tie SHA256','times':'one internal solver timer per case-mode, excludes library construction, separate build timers','cohort_hash':co_hash,'no_test_r12':True,'no_work1':True,'no_retrieval_training':True,'not_original_p3_487':True,'not_TECS_RPC_joint':True}
 write_json(OUT/'V14_PRELOCKED_PROTOCOL.json',proto)
 write_json(OUT/'V14_COHORT.json',cohort)
 say(f'COHORT_LOCKED=100 PILOT_OVERLAP=0 COHORT_SHA256={co_hash}')
 # Run a tiny self-test before any real solver calls.
 toy=np.random.default_rng(20261026).integers(0,2,size=(E,4),dtype=np.int8)
 self_policy=a4.choose_bcct(toy,np.ones(E,dtype=bool))[0]
 assert replay(toy,np.ones(E,dtype=bool),self_policy)==int(self_policy['coverage'])
 say('INDEPENDENT_TREE_REPLAY_SYNTHETIC_SELFTEST=PASS')
 phase(3,'LOAD_P3_AST44_SOURCE')
 with np.load(pinned['AST'],allow_pickle=False) as z:
  names=np.asarray(z['feature_names']).astype(str)
  ids=np.flatnonzero(names!='max_ast_depth')
  valid_all=np.asarray(z['valid'],dtype=bool)
  signature=np.asarray(z['signature'],dtype=np.float32)
 assert len(ids)==F and signature.shape[1]==45
 phase(4,'BUILD_LIBRARIES_AND_RUN_EXACT_DEPTH2_CONTROL')
 fields=['case','p3_row','shard','mode','predicates','score','oracle','solver_s','build_s','replay_pass','root','second_zero','second_one']
 records=[]
 with (OUT/'V14_100X8_CASE_ROWS.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for i,p in enumerate(selected,1):
   ids80=rank[p,100:180]
   assert ids80.shape==(E,) and (ids80>=0).all() and (ids80<len(valid_all)).all()
   valid=valid_all[ids80]
   atomic=np.full((E,F),-1,dtype=np.int8)
   atomic[valid]=(signature[ids80][:,ids]>0).astype(np.int8)[valid]
   st=time.perf_counter()
   tecs_obj=tecs.build_tecs_library(atomic,valid,a4)
   tecs_lib,tset=unique(np.asarray(tecs_obj['library'],dtype=np.int8))
   tecs_build=time.perf_counter()-st
   st=time.perf_counter()
   full_lib,fset=unique(full_signed(atomic,a4,tecs))
   full_build=time.perf_counter()-st
   base=unique(atomic)[0].shape[1]
   n=tecs_lib.shape[1]
   assert tset.issubset(fset) and base<=n<=full_lib.shape[1]
   for k in range(base):
    assert np.array_equal(tecs_lib[:,k],full_lib[:,k])
   libs={'TECS':tecs_lib,'FULL':full_lib};buildtimes={'TECS':tecs_build,'FULL':full_build}
   st=time.perf_counter()
   libs['GREEDY']=greedy_pick(full_lib,valid,base,n)
   buildtimes['GREEDY']=full_build+(time.perf_counter()-st)
   for seed in SEEDS:
    st=time.perf_counter()
    mode='SEED_'+str(seed)
    libs[mode]=hash_pick(full_lib,base,n,seed)
    buildtimes[mode]=full_build+(time.perf_counter()-st)
   for mode in MODES:
    a=libs[mode]
    assert a.shape[1]==(full_lib.shape[1] if mode=='FULL' else n)
    assert np.all((a==-1)|(a==0)|(a==1))
    assert np.all(a[~valid]==-1)
   oracle_tecs=oracle(tecs_lib,valid)
   assert oracle_tecs==oracle(full_lib,valid)
   order=MODES[(i-1)%len(MODES):]+MODES[:(i-1)%len(MODES)]
   scores={}
   for mode in order:
    facts=libs[mode]
    st=time.perf_counter();tree=a4.choose_bcct(facts,valid)[0];sec=time.perf_counter()-st
    val=int(tree['coverage']) if tree is not None else int(valid[:B].sum())
    assert val==replay(facts,valid,tree),('REPLAY_MISMATCH',i,mode)
    assert 0<=val<=int(valid.sum())
    row={'case':i,'p3_row':p,'shard':int(shards[p]),'mode':mode,'predicates':facts.shape[1],'score':val,'oracle':oracle_tecs,'solver_s':round(sec,7),'build_s':round(buildtimes[mode],7),'replay_pass':1,'root':int(tree['first_feature']) if tree else -1,'second_zero':int(tree['second_zero']) if tree else -1,'second_one':int(tree['second_one']) if tree else -1}
    scores[mode]=val;records.append(row);writer.writerow(row);f.flush()
   assert scores['FULL']>=max(scores.values()),('IMPOSSIBLE_SUPERSET_VIOLATION',i,scores)
   say(f'EXACT_CASE={i}/100 P3_ROW={p} VALID={int(valid.sum())} TECS={scores["TECS"]} FULL={scores["FULL"]} GREEDY={scores["GREEDY"]} RAND={[scores["SEED_"+str(s)] for s in SEEDS]} TECS_COLS={n} FULL_COLS={full_lib.shape[1]} ELAPSED_S={time.monotonic()-T0:.1f}')
 phase(5,'INDEPENDENT_AGGREGATES_AND_PAIRED_BOOTSTRAP')
 assert len(records)==800
 by={(int(r['case']),r['mode']):r for r in records}
 assert len(by)==800
 totals={}
 for mode in MODES:
  rs=[by[(i,mode)] for i in range(1,101)]
  totals[mode]={'shared_score':sum(int(x['score']) for x in rs),'solver_s':round(sum(float(x['solver_s']) for x in rs),6),'build_s':round(sum(float(x['build_s']) for x in rs),6),'predicates_total':sum(int(x['predicates']) for x in rs),'wins_vs_tecs':sum(int(x['score'])>int(by[(i,'TECS')]['score']) for i,x in enumerate(rs,1))}
 paired={}
 for mode in MODES:
  if mode=='TECS':continue
  vals=np.array([int(by[(i,'TECS')]['score'])-int(by[(i,mode)]['score']) for i in range(1,101)],dtype=np.int32)
  paired[mode]={'tecs_minus_control':int(vals.sum()),'tecs_wins':int((vals>0).sum()),'ties':int((vals==0).sum()),'tecs_losses':int((vals<0).sum())}
 # Query, not seed, is the independent resampling unit; keep five seeds grouped per query.
 d=np.array([[int(by[(i,'TECS')]['score'])-int(by[(i,'SEED_'+str(s))]['score']) for s in SEEDS] for i in range(1,101)],dtype=float)
 ds=d.mean(axis=1)
 rng=np.random.default_rng(20261026)
 draws=rng.integers(0,100,size=(10000,100))
 ci=np.percentile(ds[draws].sum(axis=1),[2.5,97.5]).tolist()
 output={'scope':'NEW 100 nonoverlapping TRAIN-only queries; 5 shards x 20; not independent test','cohort_sha256':co_hash,'modes':MODES,'n_queries':100,'solve_count':800,'correct_replay_count':800,'full_signed_superset_verified':True,'summary':totals,'paired':paired,'mean_seed_deficit_sum':float(ds.sum()),'posthoc_query_bootstrap_95_mean_seed_deficit':[float(v) for v in ci],'one_timing_per_case_mode':True,'solver_time_excludes_library_generation':True,'construction_uses_raw_library_generation_no_APQC':True,'not_original_P3_487':True,'no_test_r12':True,'no_work1':True}
 write_json(OUT/'V14_SUMMARY.json',output)
 for mode in MODES:
  row=totals[mode]
  say(f'TOTAL={mode} SHARED={row["shared_score"]} SOLVER_S={row["solver_s"]} BUILD_S={row["build_s"]} COLS={row["predicates_total"]}')
 for mode,v in paired.items():say(f'TECS_MINUS_{mode}={v}')
 say('STATUS=DISJOINT_TRAIN_REPLICATION_NOT_HELD_OUT_TEST')
 phase(6,'PACKAGE')
 make_archive('PASS_V14_100_TRAIN_CASES_800_EXACT_SOLVES')

if __name__=='__main__':
 try:main()
 except Exception:
  err=traceback.format_exc()
  (OUT/'V14_FAILURE_TRACE.txt').write_text(err,encoding='utf-8')
  say(err)
  make_archive('FAIL_STOP_V14_INCOMPLETE')
  sys.exit(2)
