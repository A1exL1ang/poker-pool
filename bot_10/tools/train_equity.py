"""Offline sklearn teacher export. Runtime uses only flat Python trees."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

p = argparse.ArgumentParser()
p.add_argument('--train', default='train.csv')
p.add_argument('--test', default='validation.csv')
p.add_argument('--output', default='equity_model.json')
p.add_argument('--trees', type=int, default=24)
p.add_argument('--depth', type=int, default=16)
a = p.parse_args()
start = time.time()
data = np.loadtxt(a.train, delimiter=',', dtype=np.float32)
test = np.loadtxt(a.test, delimiter=',', dtype=np.float32)
model = ExtraTreesRegressor(n_estimators=a.trees, max_depth=a.depth, min_samples_leaf=12, max_features=.9, n_jobs=8, random_state=1010)
model.fit(data[:, :-4], data[:, -4:])
pred = model.predict(test[:, :-4])
report = {'train_states':len(data), 'validation_states':len(test), 'trees':a.trees, 'depth':a.depth,
          'rmse_by_range':np.sqrt(np.mean((pred-test[:, -4:])**2, axis=0)).tolist(),
          'mae_by_range':np.mean(abs(pred-test[:, -4:]), axis=0).tolist(), 'seconds':time.time()-start,
          'note':'Labels are sampled equities; error includes Monte Carlo noise. Validation is not contest performance.'}
trees=[]
for estimator in model.estimators_:
    t=estimator.tree_
    trees.append({'left':t.children_left.tolist(), 'right':t.children_right.tolist(),
                  'feature':t.feature.tolist(), 'threshold':np.round(t.threshold,5).tolist(),
                  'value':np.round(t.value[:,:,0],5).tolist()})
Path(a.output).write_text(json.dumps({'ranges':[1,.6,.3,.12], 'trees':trees}, separators=(',',':')))
Path(a.output+'.report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
