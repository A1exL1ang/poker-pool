"""Verify original and new frozen files before any pool20 match."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 m=json.loads((ROOT/'evaluation/frozen_bots20.json').read_text());bad=[];count=0
 for n,files in m['bots'].items():
  for name,h in files.items():
   p=ROOT/f'bot_{n}'/name.replace('\\','/');count+=1
   if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(f'bot_{n}/{name}')
 print(json.dumps({'pool20_frozen_files':count,'mismatches':bad}),flush=True)
 if bad:raise SystemExit(1)
if __name__=='__main__':main()
