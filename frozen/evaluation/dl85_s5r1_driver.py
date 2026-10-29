import sys,json,time,hashlib,importlib.util,traceback,subprocess
from pathlib import Path
import numpy as np
from pydl85 import DL85Predictor
R=Path('/root/autodl-tmp/work2-algorithm-innovation')
A=R/'a1_wpg256_formal_INSTALL_R2_20261027'
P=R/'sota_s5_dl85_official_R0'
O=Path(sys.argv[1]); T=45
PINS={'dl85':'a7f0fee72b4c17bbcb727050ba102db7dda10119','a4':'580b1fa28560fccce123a57d3eea88f35dc5c6dc3ab310844cc531e3e5279dd9','a1':'c1826e79fac2775d4138348d591bf1847e078fb0b13f2ed3af9bc470ed146340'}
def fail(x):raise RuntimeError('FAIL_STOP_'+x)
def check(p,s):
 if not p:fail(s)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 check(O.parent==R and O.name.startswith('sota_s5r1_fulltecs_'),'OUTPUT_SCOPE')
 check((P/'status.txt').read_text().strip()=='STATUS=PASS_S5_MICRO','S5_STATUS')
 cp=subprocess.run(['git','-C',str(P/'upstream'),'rev-parse','HEAD'],capture_output=True,text=True,check=True)
 check(cp.stdout.strip()==PINS['dl85'],'DL85_COMMIT')
 src=R/'fhex_v1/scripts/audit_bcr_a4_rcps.py'
 check(sha(src)==PINS['a4'] and sha(A/'main.py')==PINS['a1'],'SOURCE_SHA')
 sp=importlib.util.spec_from_file_location('s5r1_a4',src)
 a4=importlib.util.module_from_spec(sp);sys.modules[sp.name]=a4;sp.loader.exec_module(a4)
 check(a4.E80==80 and a4.B==20,'A4_CONTRACT')
 save('PROTOCOL.json',{'purpose':'DL8.5 versus A4 on full identical TECS predicate library','cases':[1,2,3,4,5],
   'depth':2,'max_leaves':4,'dl85_timeout_s':T,'min_sup':1,'order':'odd DL85-first, even A4-first',
   'no_test':True,'no_selection_tuning':True,'status':'DEVELOPMENT_ONLY','source_pins':PINS})
 print('PHASE=1/3 SOURCE_PASS START_5_CASES',flush=True)
 rows=[]
 for i in range(1,6):
  path=A/'cases'/f'case_{i:03d}.npz'; meta=A/'cases'/f'case_{i:03d}.json'
  with np.load(path,allow_pickle=False) as z:
   valid=np.asarray(z['valid'],bool);lib=np.asarray(z['tecs'],np.int8)
  hist=json.loads(meta.read_text());expected=int(hist['modes']['TECS']['score'])
  check(valid.shape==(80,) and lib.shape[0]==80 and lib.shape[1]==int(hist['budget']),'SHAPE_BUDGET')
  check(np.isin(lib[valid],(0,1)).all() and np.all(lib[~valid]==-1),'THREE_VALUE_DOMAIN')
  pos=np.flatnonzero(valid);invalid=np.flatnonzero(~valid)
  X=np.ascontiguousarray(lib[valid],dtype=np.int32);calls=[0]
  def loss(ids):
   ids=np.asarray(list(ids),np.int64)
   check(len(set(ids.tolist()))==len(ids) and np.all((0<=ids)&(ids<len(pos))),'LEAF_IDS')
   survive=np.zeros(80,bool);survive[invalid]=True
   chosen=pos[ids];survive[chosen]=True
   calls[0]+=1
   return int(len(ids)-np.count_nonzero(np.cumsum(survive)[chosen]<=20))
  def solve_dl():
   model=DL85Predictor(max_depth=2,min_sup=1,error_function=loss,leaf_value_function=lambda ids:0,time_limit=T,quiet=True)
   start=time.perf_counter();model.fit(X);return model,time.perf_counter()-start
  def solve_a4():
   start=time.perf_counter();tree=a4.choose_bcct(lib,valid)[0];elapsed=time.perf_counter()-start
   value=int(tree['coverage']) if tree is not None else int(valid[:20].sum())
   check(value==expected,'A4_HISTORICAL_PARITY');return value,elapsed
  print(f'PROGRESS=START {i}/5 FEATURE_COUNT={X.shape[1]} EXPECTED={expected}',flush=True)
  if i%2: model,td=solve_dl();v,ta=solve_a4()
  else:v,ta=solve_a4();model,td=solve_dl()
  timed=bool(model.timeout_);tree=model.tree_;cov=None;count=[0]
  if tree is not None:
   def walk(node,ids,depth):
    if 'feat' not in node:
     check(len(ids)>0 and depth<=2,'LEAF_VALIDITY')
     e=loss(ids);check(abs(e-float(node['error']))<1e-7,'LEAF_ERROR')
     count[0]+=1;return e
    k=int(node['feat']);check(0<=k<X.shape[1] and depth<2,'TREE_FEATURE')
    left=ids[X[ids,k]==1];right=ids[X[ids,k]==0]
    check(len(left)>0 and len(right)>0 and len(left)+len(right)==len(ids),'TREE_PARTITION')
    return walk(node['left'],left,depth+1)+walk(node['right'],right,depth+1)
   error=walk(tree,np.arange(len(pos)),0)
   check(count[0]<=4 and abs(error-float(model.error_))<1e-7,'TREE_REPLAY')
   cov=int(len(pos)-error);check(cov<=expected,'A4_EXACT_VIOLATED')
   if not timed:check(cov==expected,'DL85_NON_TIMEOUT_MISMATCH')
  else:check(timed,'EMPTY_TREE_WITHOUT_TIMEOUT')
  row={'case':i,'p3_row':int(hist['p3_row']),'input_sha256':sha(path),'feature_count':X.shape[1],
   'a4_coverage':v,'a4_solver_wall_s':ta,'dl85_coverage_replayed':cov,'dl85_wall_s':td,
   'dl85_timeout':timed,'dl85_tree':tree,'dl85_error':None if tree is None else float(model.error_),
   'dl85_lattice':int(model.lattice_size_),'dl85_leaf_calls':calls[0],'dl85_leaves':count[0]}
  save(f'case_{i:03d}.json',row);rows.append(row)
  print(f'PROGRESS=DONE {i}/5 A4={v} DL85={cov} TIMEOUT={timed} A4_S={ta:.4f} DL85_S={td:.4f}',flush=True)
 print('PHASE=2/3 POSTFLIGHT',flush=True)
 check(sha(src)==PINS['a4'] and sha(A/'main.py')==PINS['a1'],'POSTFLIGHT_SHA')
 censored=sum(r['dl85_timeout'] for r in rows)
 save('SUMMARY.json',{'status':'PASS_WITH_TIMEOUTS' if censored else 'PASS_FULL_LIBRARY_PARITY','cases':5,
  'timed_out':censored,'a4_total_s':sum(r['a4_solver_wall_s'] for r in rows),
  'dl85_total_s':sum(r['dl85_wall_s'] for r in rows),'not_formal_100case_comparison':True,'rows':rows})
 print('PHASE=3/3 FINISHED',flush=True)
 print('STATUS=PASS_S5R1_WITH_TIMEOUTS' if censored else 'STATUS=PASS_S5R1_FULL_LIBRARY_PARITY',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  (O/'FAIL_STOP.txt').write_text(traceback.format_exc())
  print('STATUS=FAIL_STOP '+repr(e),flush=True);sys.exit(20)
