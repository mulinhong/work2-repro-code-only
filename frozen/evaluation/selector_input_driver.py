import csv, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path('/root/autodl-tmp/work2-algorithm-innovation')
OUT=Path(sys.argv[1])
SOURCE={
 'V14':('paper_v14_disjoint_controls_Mt5Q0mY6/run.py','aaa0d54faea9b21957e1149c6eee18efa51207d38281b7fcc1218b6618355b8a'),
 'V14_COHORT':('paper_v14_disjoint_controls_Mt5Q0mY6/V14_COHORT.json','2677790ddf2c1739ba323020b9d0eb19e7150489c5c779dd87e73e43ec7fb945'),
 'V14_CASES':('paper_v14_disjoint_controls_Mt5Q0mY6/V14_100X8_CASE_ROWS.csv','d484dc6d78ed60359be22671cd7f28a01162097b39b56afc3429c83a6f15e8b9'),
 'WPG':('a1_wpg256_dev_vydlsZPM/run.py','fc46588b049b4fa715ae76297ffd6222faac27b4f67c287534af6ade35bd9331'),
 'A4':('fhex_v1/scripts/audit_bcr_a4_rcps.py','580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9'),
 'TECS':('fhex_v1/frozen_v2/audit_tree_embeddable_certificate_synthesis_v0.py','12c31549062575e68b4cd0daa57659ad4572d372c16d11cf366a8c5ba7221d98'),
 'RETRIEVAL':('fhex_v1/prospective/p3_jters_v2/retrieval_v3_trainonly_exactkernel/jters_p3_qwen3_top500_trainonly_exactkernel_v1.npz','d119a1180e560b182e50f152ce5b8a403c193c38292e3dab50678000b687f323'),
 'AST':('fhex_v1/cache/python_train_ast_signature_v0.npz','4316bf9327754a47a0ad97e868087cd8c4ced236800b11346a3d073f733f0e36'),
 'P3_PROTOCOL':('fhex_v1/frozen_v2/jters_v2_frozen_protocol.json','ce22ddf88da8a1c15727346d2679248d102eb11a65a54b641c99095394b95601')
}
E,B,F=80,20,44
MODES=('TECS','GREEDY','ROOT256','WPG256','FULL')
START=time.monotonic()

def say(s):print(s,flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(1<<20),b''):h.update(part)
 return h.hexdigest()
def arysha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def imported(p,name):
 spec=importlib.util.spec_from_file_location(name,p)
 if spec is None or spec.loader is None:raise RuntimeError('BAD_MODULE_SPEC:'+name)
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def run():
 assert OUT.parent==ROOT and OUT.name.startswith('a1_wpg256_formal_') and not OUT.is_symlink()
 say('PHASE=1/5 SOURCE_PINS_AND_HISTORICAL_COHORT')
 pin={}
 for key,(r,expected) in SOURCE.items():
  p=ROOT/r
  assert p.is_file() and not p.is_symlink(),('MISSING_PINNED_SOURCE',key,r)
  actual=sha(p);assert actual==expected,('SOURCE_SHA_DRIFT',key,actual,expected)
  pin[key]=p
  say('PIN='+key+' PASS')
 # V14 module's top-level environment parameters are read only at import.
 os.environ['V14_ROOT']=str(ROOT)
 os.environ['V14_OUT']=str((ROOT/'paper_v14_disjoint_controls_Mt5Q0mY6').resolve())
 v14=imported(pin['V14'],'v14_formal_frozen')
 wpg=imported(pin['WPG'],'wpg256_formal_frozen')
 a4=imported(pin['A4'],'a4_formal_frozen')
 tecs=imported(pin['TECS'],'tecs_formal_frozen')
 assert a4.B==B and a4.E80==E and tecs.B==B and wpg.K==256
 assert wpg.SALT==b'V14-CONTROL-GREEDY-V1'
 cohort=json.loads(pin['V14_COHORT'].read_text())
 assert len(cohort)==100
 with pin['V14_CASES'].open(newline='',encoding='utf8') as f:
  old=list(csv.DictReader(f))
 prior={(int(x['case']),x['mode']):int(x['score']) for x in old}
 assert len(prior)==800
 with np.load(pin['RETRIEVAL'],allow_pickle=False) as z:
  rank=np.asarray(z['qwen3_indices'],dtype=np.int64)
  gt=np.asarray(z['ground_truth_code_indices'],dtype=np.int64)
  shards=np.asarray(z['shard_ids'],dtype=np.int64)
  srcrow=np.asarray(z['source_row_indices'],dtype=np.int64)
 assert rank.shape==(6000,500) and gt.shape==(6000,) and np.array_equal(gt,srcrow)
 misses=np.flatnonzero(~(rank[:,:100]==gt[:,None]).any(axis=1))
 prior_pilot=[];cohort_ids=[]
 for s in range(5):
  candidates=misses[shards[misses]==s]
  assert len(candidates)>=24
  prior_pilot.extend(int(v) for v in candidates[:4])
  cohort_ids.extend(int(v) for v in candidates[4:24])
 assert cohort_ids==[int(x['p3_row']) for x in cohort]
 assert not set(cohort_ids)&set(prior_pilot)
 for i,c in enumerate(cohort):
  p=int(c['p3_row']);assert int(c['shard'])==int(shards[p]);assert int(c['source_row_index'])==int(srcrow[p])
 protocol={'phase':'A1_F1_VERSIONED_PAIRED_TRAIN_COMPARISON','status':'FROZEN_BEFORE_FIRST_CASE',
  'interpretation':'Retrospective original V14 TRAIN cohort; not independent TEST, no held-out novelty assertion',
  'cohort_source_sha256':SOURCE['V14_COHORT'][1], 'cases':100, 'cohort_rows':cohort_ids,
  'depth':2,'max_leaves':4,'E':E,'B':B,'AST':F,'UNKNOWN':-1,'BUDGET':'TECS deduplicated tri-vector count per query',
  'modes':MODES,'solver_order':'Rotate 5 modes one position per query, predetermined',
  'WPG_K':256,'selection':'Unmodified pinned A1D3 WPG-256 with same GREEDY fill salt',
  'ROOT256':'First highest joint-coverage witness from WPG256 pool + same-salt GREEDY fill',
  'costs':'Full-Signed construction + method selection + exact solve; separate additive local components',
  'bootstrap':'10,000 query-resampled paired sums, seed 20261027; exploratory TRAIN',
  'no_test_r12':True,'no_work1':True,'adaptive_baseline_development_disclosed':True,
  'source_pins':{k:{'path':r,'sha256':v} for k,(r,v) in SOURCE.items()}}
 save(OUT/'FROZEN_PROTOCOL.json',protocol)
 with np.load(pin['AST'],allow_pickle=False) as z:
  names=np.asarray(z['feature_names']).astype(str)
  featureids=np.flatnonzero(names!='max_ast_depth')
  valid_all=np.asarray(z['valid'],dtype=bool)
  signature=np.asarray(z['signature'],dtype=np.float32)
 assert len(featureids)==F and signature.shape[1]==45
 say('PHASE=2/5 RECONSTRUCT_AND_SOLVE_100_TRAIN_CASES')
 dest=OUT/'cases';dest.mkdir()
 short=[]
 for idx,p in enumerate(cohort_ids,1):
  ids80=rank[p,100:180]
  assert ids80.shape==(E,) and np.all((ids80>=0)&(ids80<len(valid_all)))
  valid=valid_all[ids80]
  atomic=np.full((E,F),-1,dtype=np.int8)
  atomic[valid]=(signature[ids80][:,featureids]>0).astype(np.int8)[valid]
  t=time.perf_counter();tl,tset=v14.unique(np.asarray(tecs.build_tecs_library(atomic,valid,a4)['library'],dtype=np.int8));tc=time.perf_counter()-t
  t=time.perf_counter();full,fset=v14.unique(v14.full_signed(atomic,a4,tecs));fc=time.perf_counter()-t
  base=v14.unique(atomic)[0].shape[1];budget=tl.shape[1]
  assert tset.issubset(fset) and base<=budget<=full.shape[1] and all(np.array_equal(tl[:,j],full[:,j]) for j in range(base))
  assert np.all(full[~valid]==-1) and np.all(tl[~valid]==-1)
  t=time.perf_counter();greedy=v14.greedy_pick(full,valid,base,budget);gc=time.perf_counter()-t
  # Independent agreement of frozen V14 GREEDY and the new method's fill.
  assert np.array_equal(greedy,full[:,wpg.greedy_fill(full,valid,list(range(base)),budget)])
  t=time.perf_counter();witnesses=wpg.build_witnesses(full,valid,a4);wc=time.perf_counter()-t
  assert witnesses and len(witnesses)<=256
  t=time.perf_counter()
  seed=list(range(base));seed.extend(x for x in witnesses[0]['features'] if x not in seed)
  rootids=wpg.greedy_fill(full,valid,seed,budget)
  root=full[:,rootids]
  rc=time.perf_counter()-t
  t=time.perf_counter();wpgids,meta=wpg.select_portfolio(full,valid,base,budget,witnesses);pc=time.perf_counter()-t
  wpg_lib=full[:,wpgids]
  libs={'TECS':tl,'GREEDY':greedy,'ROOT256':root,'WPG256':wpg_lib,'FULL':full}
  selection={'TECS':tc,'GREEDY':fc+gc,'ROOT256':fc+wc+rc,'WPG256':fc+wc+pc,'FULL':fc}
  components={'tecs_generation_s':tc,'full_generation_s':fc,'greedy_selection_s':gc,
    'witness_construction_s':wc,'root_fill_selection_s':rc,'portfolio_selection_s':pc}
  lookup={full[:,j].tobytes():j for j in range(full.shape[1])}
  assert len(lookup)==full.shape[1]
  details={};values={}
  order=MODES[(idx-1)%len(MODES):]+MODES[:(idx-1)%len(MODES)]
  for mode in order:
   lib=libs[mode]
   assert lib.shape==(E,full.shape[1] if mode=='FULL' else budget) and np.all(lib[~valid]==-1)
   chosen=[lookup[lib[:,j].tobytes()] for j in range(lib.shape[1])]
   assert len(set(chosen))==len(chosen)
   t=time.perf_counter();tree=a4.choose_bcct(lib,valid)[0];solver_s=time.perf_counter()-t
   cov=int(tree['coverage']) if tree is not None else int(valid[:B].sum())
   assert cov==v14.replay(lib,valid,tree)==wpg.replay(lib,valid,tree),('REPLAY_FAIL',idx,mode)
   if mode in ('TECS','GREEDY','FULL'):
    assert cov==prior[(idx,mode)],('V14_HISTORICAL_COVERAGE_MISMATCH',idx,mode,cov,prior[(idx,mode)])
   nodes=[int(tree[k]) if tree is not None else -1 for k in ('first_feature','second_zero','second_one')]
   policy_full=[chosen[x] if x>=0 else -1 for x in nodes]
   details[mode]={'score':cov,'predicate_count':int(lib.shape[1]),'selected_full_indices':chosen,
     'policy_local_indices':nodes,'policy_full_indices':policy_full,'library_sha256':arysha(lib),
     'generation_selection_s':selection[mode], 'solver_s':solver_s,
     'local_total_s':selection[mode]+solver_s,'replay_pass':True}
   values[mode]=cov
  assert values['FULL']>=max(values.values())
  assert values['WPG256']>=int(witnesses[0]['coverage'])
  assert details['TECS']['predicate_count']==details['GREEDY']['predicate_count']==details['ROOT256']['predicate_count']==details['WPG256']['predicate_count']==budget
  case={'case':idx,'p3_row':p,'shard':int(shards[p]),'valid_count':int(valid.sum()),'budget':budget,'full_count':int(full.shape[1]),
    'witness_pool_size':len(witnesses),'witness_selected':meta['witnesses_selected'],
    'witness_indices':meta['witness_indices'],'seed_witness_coverage':int(witnesses[0]['coverage']),
    'tree_guided_new_predicates':meta['tree_guided_new_predicates'],'greedy_fill_predicates':meta['greedy_fill_predicates'],
    'components':components,'modes':details,'input_sha256':{'atomic':arysha(atomic),'valid':arysha(valid),'full':arysha(full)}}
  save(dest/f'case_{idx:03d}.json',case)
  np.savez_compressed(dest/f'case_{idx:03d}.npz',valid=valid,full=full,atomic=atomic,tecs=tl,greedy=greedy,root256=root,wpg256=wpg_lib)
  short.append(case)
  say(f'PROGRESS={idx}/100 PCT={idx}% ROW={p} TECS={values["TECS"]} GREEDY={values["GREEDY"]} ROOT256={values["ROOT256"]} WPG256={values["WPG256"]} FULL={values["FULL"]} BUDGET={budget} WITNESSES={len(witnesses)} WPG_SELECT={meta["witnesses_selected"]} ELAPSED_S={time.monotonic()-START:.1f}')
 say('PHASE=3/5 PAIRED_BOOTSTRAP_AND_AGGREGATES')
 assert len(short)==100
 rows=[]
 for mode in MODES:
  rows.append({'mode':mode,'score':sum(c['modes'][mode]['score'] for c in short),
   'predicate_count_sum':sum(c['modes'][mode]['predicate_count'] for c in short),
   'generation_selection_s':sum(c['modes'][mode]['generation_selection_s'] for c in short),
   'solver_s':sum(c['modes'][mode]['solver_s'] for c in short),
   'local_total_s':sum(c['modes'][mode]['local_total_s'] for c in short)})
 comparisons={}; rng=np.random.default_rng(20261027)
 draws=rng.integers(0,100,size=(10000,100))
 for x,y in [('TECS','WPG256'),('TECS','ROOT256'),('WPG256','ROOT256'),('TECS','GREEDY')]:
  dif=np.asarray([c['modes'][x]['score']-c['modes'][y]['score'] for c in short])
  ci=np.percentile(dif[draws].sum(axis=1),[2.5,97.5]).tolist()
  comparisons[x+'_minus_'+y]={'sum_difference':int(dif.sum()),'wins':int((dif>0).sum()),'ties':int((dif==0).sum()),'losses':int((dif<0).sum()),'query_bootstrap_sum_ci95':ci}
 summary={'status':'PASS_100_RETROSPECTIVE_TRAIN_PAIRED','cases':len(short),'exact_solve_count':500,'historical_v14_parity_checks':300,'independent_policy_replays':500,
  'cohort_is_original_V14_TRAIN':True,'not_independent_external_validation':True,
  'solver_timing_one_observation_per_query_mode':True,'timing_note':'Local costs include method generation/selection and exact solve, exclude data decoding',
  'cost_shares_note':'Shared Full-Signed/witness pool work charged in full to each standalone method',
  'modes':rows,'paired':comparisons,
  'seed_matches_full_count':sum(c['seed_witness_coverage']==c['modes']['FULL']['score'] for c in short),
  'budget_constrained_witness_count':sum(c['witness_selected']<c['witness_pool_size'] for c in short)}
 save(OUT/'SUMMARY.json',summary)
 with (OUT/'PAIRED_ROWS.csv').open('w',newline='',encoding='utf8') as f:
  columns=['case','p3_row','budget','full_count','seed_coverage','witness_pool','witness_selected']+[k+'_coverage' for k in MODES]
  wr=csv.DictWriter(f,fieldnames=columns);wr.writeheader()
  for c in short:
   wr.writerow(dict(case=c['case'],p3_row=c['p3_row'],budget=c['budget'],full_count=c['full_count'],seed_coverage=c['seed_witness_coverage'],witness_pool=c['witness_pool_size'],witness_selected=c['witness_selected'],**{k+'_coverage':c['modes'][k]['score'] for k in MODES}))
 say('PHASE=4/5 PIN_POSTFLIGHT')
 for key,(r,h) in SOURCE.items():assert sha(pin[key])==h,('PIN_DRIFT_POSTFLIGHT',key)
 say('PHASE=5/5 FINAL')
 say('COVERAGE_TOTALS='+str({r['mode']:r['score'] for r in rows}))
 say('STATUS=PASS_100_RETROSPECTIVE_TRAIN_PAIRED')

if __name__=='__main__':
 try:run()
 except Exception as exc:
  (OUT/'FAIL_STOP.txt').write_text(traceback.format_exc(),encoding='utf8')
  say('STATUS=FAIL_STOP ERROR='+repr(exc))
  sys.exit(20)
