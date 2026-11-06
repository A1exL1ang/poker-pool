"""Validation-only model-capacity comparison; preserves the first checkpoint."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    archive=Path('checkpoints/equity_v1')
    archive.mkdir(parents=True,exist_ok=True)
    for name in ('assets/equity_model.json','measurements/equity_training.json','measurements/equity_heldout.json'):
        dest=archive/Path(name).name
        if not dest.exists():
            shutil.copy2(name,dest)
    rows=[]
    for trees,depth in ((192,6),(192,8),(256,8)):
        model=f'checkpoints/equity_d{depth}_t{trees}.json'
        report=f'measurements/equity_d{depth}_t{trees}.json'
        subprocess.run([sys.executable,'training/train.py','--train','training/data/train.bin',
                        '--valid','training/data/validation.bin','--trees',str(trees),'--depth',str(depth),
                        '--threads','8','--output',model,'--report',report],check=True)
        r=json.loads(Path(report).read_text())
        rmse=sum(v['metrics']['rmse'] for v in r['streets'].values())/3
        rows.append({'trees':trees,'depth':depth,'model':model,'report':report,'validation_mean_rmse':rmse})
    old=json.loads((archive/'equity_training.json').read_text())
    old_rmse=sum(v['metrics']['rmse'] for v in old['streets'].values())/3
    chosen=rows[0]
    for row in rows[1:]:
        if row['validation_mean_rmse'] < .95*chosen['validation_mean_rmse']:
            chosen=row
    if chosen['validation_mean_rmse'] >= .97*old_rmse:
        chosen={'trees':192,'depth':6,'model':str(archive/'equity_model.json'),
                'report':str(archive/'equity_training.json'),'validation_mean_rmse':old_rmse}
    shutil.copy2(chosen['model'],'assets/equity_model.json')
    shutil.copy2(chosen['report'],'measurements/equity_training.json')
    Path('measurements/model_selection.json').write_text(json.dumps({
        'rule':'new features must improve mean validation RMSE at least3%; each deeper/larger candidate must improve current choice at least5%',
        'original_validation_rmse':old_rmse,'candidates':rows,'selected':chosen,
        'heldout_used_for_selection':False},indent=2)+'\n')
    print(json.dumps(chosen),flush=True)


if __name__=='__main__':
    main()
