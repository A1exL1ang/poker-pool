"""Fresh-state calibration evaluation of frozen forest models (offline only)."""
import json
from pathlib import Path
import sys
import numpy as np

data=np.loadtxt(sys.argv[1],delimiter=',',dtype=np.float64)
models=sys.argv[2:] or ['equity_model_8.json','equity_model.json']
reports={}
for filename in models:
    trees=json.loads(Path(filename).read_text())['trees']
    predictions=np.zeros((len(data),4))
    for t in trees:
        for row,f in enumerate(data[:,:-4]):
            i=0
            while t['left'][i]!=-1:
                i=t['left'][i] if f[t['feature'][i]]<=t['threshold'][i] else t['right'][i]
            predictions[row]+=t['value'][i]
    predictions/=len(trees);error=predictions-data[:,-4:]
    reports[filename]={'states':len(data),'trees':len(trees),
       'rmse':np.sqrt(np.mean(error**2,axis=0)).tolist(),
       'mae':np.mean(abs(error),axis=0).tolist(),
       'abs_error_p95':np.quantile(abs(error),.95,axis=0).tolist(),
       'by_street':{str(n):np.sqrt(np.mean(error[data[:,0]==n]**2,axis=0)).tolist() for n in (3,4,5)}}
Path('results/equity_heldout.json').write_text(json.dumps({'seed':99010,'samples_per_label_per_range':512,'reports':reports,'note':'Fresh states unused for model selection; Monte Carlo labels include sampling noise. Training distributions are heuristic ranges, not real opponent ranges.'},indent=2))
print(json.dumps(reports,indent=2))
