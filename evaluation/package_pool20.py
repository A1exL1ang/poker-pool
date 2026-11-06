"""Remote-only verification and compression of reproducible evaluation outputs."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evaluation'
def main():
 expected=json.loads((E/'pool20_harness_manifest.json').read_text())['files']
 bad=[p for p,h in expected.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
 if bad:raise RuntimeError({'harness_changed':bad})
 artifacts={}
 groups={'pool20_raw_results.tar.gz':sorted((E/'pool20_results').glob('production_*.jsonl*')),'similarity_raw_results.tar.gz':[E/'similarity_corpus.jsonl',E/'similarity_replay.jsonl']}
 for name,files in groups.items():
  if not files:raise RuntimeError('missing outputs')
  with tarfile.open(E/name,'w:gz',compresslevel=6) as archive:
   for p in files:archive.add(p,arcname=p.relative_to(E).as_posix(),recursive=False)
  artifacts[name]={'bytes':(E/name).stat().st_size,'sha256':hashlib.sha256((E/name).read_bytes()).hexdigest(),'files':{p.relative_to(E).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
 summary=json.loads((E/'pool20_summary.json').read_text())
 if summary['tables']!=82080:raise RuntimeError('incomplete summary')
 report={'harness_files_verified':len(expected),'harness_mismatches':bad,'tables':summary['tables'],'artifacts':artifacts}
 (E/'pool20_release_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({'complete':True,'tables':summary['tables'],'harness_files':len(expected)}))
if __name__=='__main__':main()
