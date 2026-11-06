"""Fit an offline multi-output ExtraTrees equity surrogate and export stdlib JSON.

Run from workspace: python training/equity_train.py --train artifacts/equity_train.bin
  --valid artifacts/equity_valid.bin --test artifacts/equity_test.bin
Three independent generator seeds separate fit, model selection, final reporting.
"""
import argparse
import gzip
import json
import struct
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def read(path):
    with open(path, 'rb') as f:
        n, nf = struct.unpack('<ii', f.read(8))
        dt = np.dtype([('x', '<f4', nf), ('y', '<f4', 4), ('cards', '<i4', 7)])
        a = np.frombuffer(f.read(), dtype=dt)
        assert len(a) == n
    return a['x'].copy(), a['y'].copy(), a['cards'].copy()


def metrics(model, x, y):
    p = model.predict(x)
    return {'mae': float(mean_absolute_error(y, p)), 'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'mae_by_range': np.mean(np.abs(p-y),axis=0).tolist(),
            'p95_absolute_error': float(np.quantile(np.abs(p-y), .95)),
            'by_street': {str(int(k)): {'n': int((x[:,-1]==k).sum()), 'mae': float(np.mean(np.abs((p-y)[x[:,-1]==k])))} for k in (3,4,5)}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--valid',required=True);p.add_argument('--test',required=True);p.add_argument('--out',default='artifacts/equity_model.json.gz');p.add_argument('--trees',type=int,default=24);args=p.parse_args()
    x,y,_=read(args.train);vx,vy,_=read(args.valid);tx,ty,_=read(args.test)
    records=[];best=None;bestscore=1
    for leaf,depth in [(5,20),(12,18),(24,16)]:
        model=ExtraTreesRegressor(n_estimators=args.trees,max_depth=depth,min_samples_leaf=leaf,max_features=.9,n_jobs=8,random_state=151)
        model.fit(x,y);m=metrics(model,vx,vy);records.append({'min_samples_leaf':leaf,'max_depth':depth,'validation':m});print(records[-1],flush=True)
        if m['mae']<bestscore: bestscore=m['mae'];best=model
    trees=[]
    for est in best.estimators_:
        t=est.tree_;fs=t.feature.astype(int).tolist();vs=[]
        for i,f in enumerate(fs):
            if f<0:vs.append([round(float(v),7) for v in t.value[i,:,0]]);fs[i]=-len(vs)
        trees.append([fs,t.threshold.round(6).tolist(),t.children_left.tolist(),t.children_right.tolist(),vs])
    with gzip.open(args.out,'wt',compresslevel=9) as f:json.dump({'version':1,'fractions':[.07,.2,.5,1.],'trees':trees},f,separators=(',',':'))
    report={'train_rows':len(x),'validation_rows':len(vx),'test_rows':len(tx),'candidates':records,'selected':best.get_params(),'held_out_test':metrics(best,tx,ty),'artifact_bytes':Path(args.out).stat().st_size,'label_note':'Monte Carlo sampled labels; measured error includes label noise. Top preflop ranges do not condition on postflop actions.'}
    Path('artifacts/equity_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
