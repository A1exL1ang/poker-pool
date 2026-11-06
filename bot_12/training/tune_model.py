"""Select complexity on a training holdback; never reads final held-out data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from train_model import extract_models,model_source

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',nargs='+',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output',default='training/tuning_report.json');a=p.parse_args()
    data=np.concatenate([np.loadtxt(x,delimiter=',',dtype=np.float32) for x in a.train])
    val=np.loadtxt(a.validation,delimiter=',',dtype=np.float32)
    X,Y=data[:,:-2],data[:,-2:];VX,VY=val[:,:-2],val[:,-2:]
    reports=[]
    for leaves,depth in [(31,7),(63,9)]:
        models=[HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=leaves,max_depth=depth,
                 learning_rate=.09,min_samples_leaf=35,l2_regularization=.1,early_stopping=False,
                 random_state=1212+k,warm_start=True) for k in range(2)]
        for iterations in [100,140,200]:
            start=time.time();pred=[]
            for k,model in enumerate(models):
                model.set_params(max_iter=iterations);model.fit(X,Y[:,k]);pred.append(model.predict(VX))
            pred=np.array(pred).T;base,trees=extract_models(models);src=model_source(base,trees,X.shape[1]);ns={};exec(src,ns)
            f=ns['predict'];rows=[list(map(int,fs)) for fs in VX[:5000]]
            # Single-call timing includes interpreter dispatch, excludes feature
            # extraction. Repeat once for compilation/cache warmup.
            [f(fs) for fs in rows[:100]]
            times=[];pythonpred=[]
            for fs in rows:
                t=time.perf_counter_ns();pythonpred.append(f(fs));times.append((time.perf_counter_ns()-t)/1000)
            export_error=float(np.max(abs(np.array(pythonpred)-pred[:len(rows)])))
            assert export_error<1e-6,export_error
            report={'iterations':iterations,'leaves':leaves,'depth':depth,
                    'validation_mae':abs(pred-VY).mean(axis=0).tolist(),
                    'validation_rmse':np.sqrt(((pred-VY)**2).mean(axis=0)).tolist(),
                    'prediction_us_p50':float(np.quantile(times,.5)),'prediction_us_p99':float(np.quantile(times,.99)),
                    'prediction_us_max':max(times),'source_bytes':len(src),'export_max_error':export_error,'seconds':time.time()-start}
            reports.append(report);print(json.dumps(report),flush=True)
    eligible=[r for r in reports if r['prediction_us_p99']<400]
    best=min(eligible,key=lambda r:sum(r['validation_mae']))
    report={'train_rows':len(X),'validation_rows':len(VX),'train_files':[Path(x).name for x in a.train],
            'validation_file':Path(a.validation).name,'experiments':reports,'selected':best,
            'selection':'minimum sum of HU validation MAEs with generated prediction p99<400us; no final heldout read'}
    Path(a.output).write_text(json.dumps(report,indent=2));print('SELECTED',json.dumps(best))

if __name__=='__main__':main()
