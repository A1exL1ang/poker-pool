"""Fit compact offline histogram boosted trees. Scientific deps: training only.

Usage: python training/train_model.py --train train0.csv train1.csv ...
       --test heldout.csv --preflop preflop.csv --order order.txt
The held-out file must use a different generator seed and is never fitted.
"""
import argparse
import json
import time
import os
import math
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def extract_models(models):
    base=[];trees=[]
    for k,model in enumerate(models):
        base.append(float(model._baseline_prediction[0,0]))
        for stage in model._predictors:
            nodes=stage[0].nodes
            f=nodes['feature_idx'].tolist();th=nodes['num_threshold'].tolist()
            l=nodes['left'].tolist();r=nodes['right'].tolist();v=[]
            for i,n in enumerate(nodes):
                if n['is_leaf']: l[i]=-1;r[i]=-1
                z=[0.,0.];z[k]=round(float(n['value']),10);v.append(z)
            trees.append([f,th,l,r,v])
    return base,trees


def model_source(base,trees,nfeatures):
    lines=['# Generated offline by training/train_model.py; no third-party runtime dependency.',
           'def predict(fs):', '    '+','.join('f'+str(i) for i in range(nfeatures))+' = fs',
           '    y0 = '+repr(base[0]),'    y1 = '+repr(base[1])]
    for f,th,l,r,v in trees:
        joint=any(row[0] for row in v) and any(row[1] for row in v)
        target=0 if any(row[0] for row in v) else 1
        def expr(node):
            if l[node]<0:return repr(tuple(v[node])) if joint else repr(v[node][target])
            # Every feature is integer-valued; integer thresholds preserve the
            # split exactly while avoiding int/float comparisons at gameplay.
            return '('+expr(l[node])+' if f'+str(f[node])+' <= '+str(math.floor(th[node]))+' else '+expr(r[node])+')'
        if joint:
            lines.append('    v0,v1 = '+expr(0))
            lines.append('    y0 += v0')
            lines.append('    y1 += v1')
        else:
            lines.append('    y'+str(target)+' += '+expr(0))
    lines.append('    return y0,y1')
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',nargs='+',required=True)
    p.add_argument('--test',required=True)
    p.add_argument('--preflop',required=True)
    p.add_argument('--order',required=True)
    p.add_argument('--iterations',type=int,default=140)
    p.add_argument('--leaves',type=int,default=31)
    p.add_argument('--depth',type=int,default=7)
    a=p.parse_args()
    started=time.time()
    data=np.concatenate([np.loadtxt(x,delimiter=',',dtype=np.float32) for x in a.train])
    test=np.loadtxt(a.test,delimiter=',',dtype=np.float32)
    X,Y=data[:,:-2],data[:,-2:]
    TX,TY=test[:,:-2],test[:,-2:]
    models=[];pred=[];trees=[];base=[]
    for k in range(2):
        model=HistGradientBoostingRegressor(max_iter=a.iterations,max_leaf_nodes=a.leaves,max_depth=a.depth,
                  learning_rate=.09,min_samples_leaf=35,l2_regularization=.1,
                  early_stopping=False,random_state=1212+k)
        model.fit(X,Y[:,k]);models.append(model);pred.append(model.predict(TX))
    base,trees=extract_models(models)
    assets=Path('assets');assets.mkdir(exist_ok=True)
    modelpath=assets/'equity_model.json'
    modelpath.write_text(json.dumps({'version':1,'features':int(X.shape[1]),'base':base,'trees':trees},separators=(',',':')))
    # A generated local-variable expression forest avoids Python node-array
    # traversal overhead. It is ordinary offline source, no runtime compilation.
    (assets/'equity_predict.py').write_text(model_source(base,trees,X.shape[1]))
    pf=np.loadtxt(a.preflop,delimiter=',')
    order=np.loadtxt(a.order)
    (assets/'preflop.json').write_text(json.dumps({'ranges':[.05,.10,.20,.35,.50,1.],
                   'range_definition':'top hand classes ranked by independently estimated HU equity versus uniform cards, weighted by combo count',
                   'equities':[list(map(float,row[1:])) for row in sorted(pf,key=lambda z:z[0])],
                   'order':[int(row[2]) for row in sorted(order,key=lambda z:z[0])]},separators=(',',':')))
    pred=np.array(pred).T
    report={'train_rows':len(X),'heldout_rows':len(TX),'features':int(X.shape[1]),'trees':len(trees),'iterations_per_target':a.iterations,
            'seconds':time.time()-started,'model_bytes':modelpath.stat().st_size,
            'prediction_source_bytes':(assets/'equity_predict.py').stat().st_size,
            'targets':['equity_random','equity_top20pct'],'leaves':a.leaves,'depth':a.depth,
            'mae':[float(mean_absolute_error(TY[:,k],pred[:,k])) for k in range(2)],
            'rmse':[float(mean_squared_error(TY[:,k],pred[:,k])**.5) for k in range(2)],
            'p95_abs_error':np.quantile(abs(pred-TY),.95,axis=0).tolist(),
            'by_street':{str(s):{'rows':int(sum(TX[:,0]==s)),
                'mae':abs(pred[TX[:,0]==s]-TY[TX[:,0]==s]).mean(axis=0).tolist(),
                'rmse':np.sqrt(((pred[TX[:,0]==s]-TY[TX[:,0]==s])**2).mean(axis=0)).tolist()}
                for s in [3,4,5]},
            'train_files':[Path(x).name for x in a.train],'heldout_file':Path(a.test).name}
    Path('training/model_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
