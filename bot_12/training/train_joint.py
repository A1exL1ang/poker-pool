"""Joint residual boosting with two targets per leaf; training-only selection.

This experiment never reads the final heldout set and never changes assets.
Uses bounded four-thread numerical libs, single-thread tree fitting.
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
from train_model import model_source


def export_tree(tree,learning_rate):
    t=tree.tree_;v=np.array(t.value).reshape(t.node_count,2)*learning_rate
    return [t.feature.tolist(),t.threshold.tolist(),t.children_left.tolist(),t.children_right.tolist(),
            np.round(v,10).tolist()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',nargs='+',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output-dir',default='.control/joint_equity');p.add_argument('--subsample',type=int,default=80000)
    p.add_argument('--iterations',type=int,default=200);p.add_argument('--leaves',type=int,default=63)
    p.add_argument('--depth',type=int,default=9);p.add_argument('--rate',type=float,default=.1)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    data=np.concatenate([np.loadtxt(x,delimiter=',',dtype=np.float32) for x in a.train])
    val=np.loadtxt(a.validation,delimiter=',',dtype=np.float32)
    X,Y=data[:,:-2],data[:,-2:];VX,VY=val[:,:-2],val[:,-2:]
    base=Y.mean(axis=0).astype(float);pred=np.tile(base,(len(Y),1));vp=np.tile(base,(len(VY),1));trees=[];reports=[]
    rng=np.random.default_rng(1212088);started=time.time()
    for i in range(1,a.iterations+1):
        ix=rng.choice(len(X),size=min(a.subsample,len(X)),replace=False)
        residual=Y[ix]-pred[ix]
        model=DecisionTreeRegressor(max_leaf_nodes=a.leaves,max_depth=a.depth,min_samples_leaf=35,random_state=1212+i)
        model.fit(X[ix],residual)
        pred+=a.rate*model.predict(X);vp+=a.rate*model.predict(VX)
        trees.append(export_tree(model,a.rate))
        if i in {60,100,140,200,a.iterations}:
            src=model_source(base.tolist(),trees,X.shape[1]);ns={};exec(src,ns);f=ns['predict']
            rows=[list(map(int,fs)) for fs in VX[:5000]];[f(fs) for fs in rows[:100]]
            ts=[];yp=[]
            for fs in rows:
                t=time.perf_counter_ns();yp.append(f(fs));ts.append((time.perf_counter_ns()-t)/1000)
            exporterr=float(np.max(abs(np.array(yp)-vp[:len(yp)])));assert exporterr<1e-6,exporterr
            report={'iterations':i,'leaves':a.leaves,'depth':a.depth,'subsample':a.subsample,'rate':a.rate,
                    'validation_mae':abs(vp-VY).mean(axis=0).tolist(),
                    'validation_rmse':np.sqrt(((vp-VY)**2).mean(axis=0)).tolist(),
                    'prediction_us_p50':float(np.quantile(ts,.5)),'prediction_us_p99':float(np.quantile(ts,.99)),
                    'prediction_us_max':max(ts),'source_bytes':len(src),'export_max_error':exporterr,'seconds':time.time()-started}
            reports.append(report);print(json.dumps(report),flush=True)
            (out/f'joint_{i}.py').write_text(src)
            (out/f'joint_{i}.json').write_text(json.dumps({'version':1,'features':X.shape[1],'base':base.tolist(),'trees':trees},separators=(',',':')))
    summary={'training_rows':len(X),'validation_rows':len(VX),'validation_file':Path(a.validation).name,
             'train_files':[Path(x).name for x in a.train],'experiments':reports,
             'scope':'Training holdback only. No final heldout read. No gameplay asset overwritten.'}
    (out/'report.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
