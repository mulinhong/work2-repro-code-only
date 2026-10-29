import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    records=[]
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,relative=line.split('  ',1)
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError('Invalid manifest path')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('SHA256 mismatch: '+relative)
        records.append({'path':relative,'sha256':actual})
    report={'scope':'Frozen source byte identity only; no experiment is run.','files':len(records),'records':records}
    if a.output:Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Frozen source hashes matched:',len(records))

if __name__=='__main__':main()
