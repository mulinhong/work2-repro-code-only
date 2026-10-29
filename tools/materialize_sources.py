"""Restore original source paths without changing any existing different file."""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    destination=a.destination.resolve();mapping=json.loads((ROOT/'manifests/sources.json').read_text())
    pending=[]
    for row in mapping:
        relative=Path(row['original_path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid original relative path')
        if row['original_path'].startswith('generated/'):continue
        target=(destination/relative).resolve()
        if not target.is_relative_to(destination):raise ValueError('Source path outside destination')
        data=(ROOT/row['repository_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Source SHA256 mismatch')
        if target.exists() and target.read_bytes()!=data:raise ValueError('Refusing to replace different existing source: '+str(target))
        pending.append((target,data))
    for target,data in pending:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(data)
    print('Original source paths restored:',len(pending))

if __name__=='__main__':main()
