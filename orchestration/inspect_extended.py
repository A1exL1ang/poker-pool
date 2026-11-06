"""Compact root-only progress; never forward other teams' details to a builder."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent/'extended'
s=json.loads((root/'status.json').read_text())
out={k:s.get(k) for k in ('status','last_poll_at','supervisor_pid','build_deadline','service_tier','helpers_per_team')}
out['workers']=[]
for w in s['workers']:
    row={k:w.get(k) for k in ('number','model','reasoning','status','pid','started_at','exit_code','export_path','export_error')}
    progress=Path(w['workspace'])/'.control/STATUS.txt'
    row['progress']=progress.read_text(errors='replace')[-1600:] if progress.exists() else None
    log=root/'logs'/f"bot_{w['number']}.jsonl"
    events=[]
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-131072));tail=f.read().decode('utf8',errors='replace')
        for line in tail.splitlines():
            try:events.append(json.loads(line))
            except ValueError:pass
        row['log_bytes']=log.stat().st_size
    row['failures']=[e for e in events if e.get('type') in ('error','turn.failed')][-2:]
    messages=[e.get('item',{}).get('text','') for e in events if e.get('item',{}).get('type')=='agent_message']
    row['last_message']=messages[-1][-900:] if messages else None
    row['recent_item_types']=list(dict.fromkeys(e.get('item',{}).get('type') for e in events if e.get('item')))
    out['workers'].append(row)
print(json.dumps(out,indent=2))
