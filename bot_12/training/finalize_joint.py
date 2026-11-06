"""Fit the validation-selected joint forest, evaluate holdout once, export.

The selected settings are fixed:140 stages,127 leaves,depth10,80k row sample,
learning rate0.1. This is not a search and uses no held-out feedback.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from train_joint import export_tree
from train_model import model_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',nargs='+',required=True);p.add_argument('--heldout',required=True)
    p.add_argument('--output-dir',default='assets');a=p.parse_args()
    started=time.time();data=np.concatenate([np.loadtxt(x,delimiter=',',dtype=np.float32) for x in a.train])
    X,Y=data[:,:-2],data[:,-2:];base=Y.mean(axis=0).astype(float);pred=np.tile(base,(len(X),1));trees=[];models=[]
    rng=np.random.default_rng(1212088)
    for i in range(1,141):
        ix=rng.choice(len(X),size=min(80000,len(X)),replace=False)
        model=DecisionTreeRegressor(max_leaf_nodes=127,max_depth=10,min_samples_leaf=35,random_state=1212+i)
        model.fit(X[ix],Y[ix]-pred[ix]);pred+=.1*model.predict(X)
        models.append(model);trees.append(export_tree(model,.1))
        if i%20==0:print('trained',i,'of140','seconds',round(time.time()-started,2),flush=True)
    # Only after all model choices/fitting are complete is the holdout opened.
    test=np.loadtxt(a.heldout,delimiter=',',dtype=np.float32);TX,TY=test[:,:-2],test[:,-2:]
    tp=np.tile(base,(len(TX),1))
    for model in models:tp+=.1*model.predict(TX)
    modeldata={'version':1,'algorithm':'joint_residual_boosting','features':X.shape[1],
               'base':base.tolist(),'trees':trees}
    src=model_source(base.tolist(),trees,X.shape[1]);ns={};exec(src,ns);f=ns['predict']
    exportpred=np.array([f(list(map(int,row))) for row in TX[:2000]])
    exporterr=float(np.max(abs(exportpred-tp[:len(exportpred)])));assert exporterr<1e-6,exporterr
    out=Path(a.output_dir);out.mkdir(exist_ok=True,parents=True)
    source=out/'equity_predict.py';raw=out/'equity_model.json'
    source.with_suffix('.py.tmp').write_text(src)
    raw.with_suffix('.json.tmp').write_text(json.dumps(modeldata,separators=(',',':')))
    raw.with_suffix('.json.tmp').replace(raw);source.with_suffix('.py.tmp').replace(source)
    report={'algorithm':'joint_residual_boosting_with_multioutput_regression_trees',
            'train_rows':len(X),'heldout_rows':len(TX),'features':X.shape[1],
            'trees':140,'outputs_per_tree':2,'leaves':127,'depth':10,'subsample_rows':80000,
            'learning_rate':.1,'min_samples_leaf':35,'rng_seed':1212088,
            'seconds':time.time()-started,'model_bytes':raw.stat().st_size,'prediction_source_bytes':source.stat().st_size,
            'targets':['equity_random','equity_top20pct'],'export_max_error':exporterr,
            'mae':abs(tp-TY).mean(axis=0).tolist(),'rmse':np.sqrt(((tp-TY)**2).mean(axis=0)).tolist(),
            'p95_abs_error':np.quantile(abs(tp-TY),.95,axis=0).tolist(),
            'by_street':{str(s):{'rows':int(sum(TX[:,0]==s)),
                'mae':abs(tp[TX[:,0]==s]-TY[TX[:,0]==s]).mean(axis=0).tolist(),
                'rmse':np.sqrt(((tp[TX[:,0]==s]-TY[TX[:,0]==s])**2).mean(axis=0)).tolist()} for s in [3,4,5]},
            'train_files':[Path(x).name for x in a.train],'heldout_file':Path(a.heldout).name,
            'selection':'Training0..4fit/Training5validation only; final model fitall6 and heldout evaluated once for this model. Prior full forest evaluated same independently seeded heldout; no heldout feedback used for joint selection.',
            'scope':'Learned HU forest only. Production uses exact river enumeration, so forest river errors do not describe production river accuracy.'}
    Path('training/model_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
