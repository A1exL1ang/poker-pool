"""Create a small runtime ZIP and an explicit private validation archive."""
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'dist';out.mkdir(exist_ok=True)
    runtime=[root/'agent.py']+sorted((root/'kestrel').glob('*.py'))+[root/'kestrel'/'preflop.json']
    if (root/'README.md').exists():runtime.append(root/'README.md')
    manifest={}
    with zipfile.ZipFile(out/'kestrel8.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for f in sorted(runtime):
            data=f.read_bytes();name=f.relative_to(root).as_posix()
            info=zipfile.ZipInfo(name,date_time=(2026,11,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
            manifest[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    data=(out/'kestrel8.zip').read_bytes()
    summary={'runtime_files':manifest,'zip_bytes':len(data),'zip_sha256':hashlib.sha256(data).hexdigest(),
             'uncompressed_bytes':sum(x['bytes'] for x in manifest.values()),'network_required':False,
             'dependencies':'Python standard library, Python 3.10+','official_adapter_required':True}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    validation=runtime+sorted((root/'tools').glob('*.py'))+sorted((root/'tools').glob('*.sh'))+sorted((root/'tests').glob('*.py'))
    with tarfile.open(out/'validation.tar.gz','w:gz') as tar:
        for f in validation:tar.add(f,arcname=f.relative_to(root).as_posix())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
