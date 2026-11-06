"""Independent-seed exact river diagnostics and optional exhaustive evaluator QA.

python training/validate_equity.py --rivers 1000 --output training/exact_river_report.json
python training/validate_equity.py --exhaustive-five
"""
import argparse
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from equity import evaluate,analyze,hand_key,_PREFLOP

def main():
    p=argparse.ArgumentParser();p.add_argument('--rivers',type=int,default=300)
    p.add_argument('--seed',type=int,default=1212991)
    p.add_argument('--output',default='training/exact_river_report.json')
    p.add_argument('--exhaustive-five',action='store_true');a=p.parse_args()
    started=time.time()
    if a.exhaustive_five:
        counts=Counter(evaluate(cs)>>20 for cs in combinations(range(52),5))
        expected={0:1302540,1:1098240,2:123552,3:54912,4:10200,5:5108,6:3744,7:624,8:40}
        assert counts==expected,(counts,expected)
        print('All 2,598,960 five-card combinations match known combinatorial category counts.',time.time()-started)
        return
    r=random.Random(a.seed);rows=[];bycat=defaultdict(list)
    for _ in range(a.rivers):
        cs=r.sample(range(52),7);h,b=cs[:2],cs[2:];score=evaluate(cs)
        remain=sorted(set(range(52))-set(cs));w=t=wt=tt=nt=0
        for opp in combinations(remain,2):
            os=evaluate(opp+tuple(b));win=float(score>os)+.5*float(score==os)
            w+=win;t+=1
            if _PREFLOP['order'][hand_key(opp)]<266:wt+=win;nt+=1
        pred=analyze(h,b);target=[w/t,wt/nt];pr=[pred['equity_random'],pred['equity_tight']]
        err=[abs(x-y) for x,y in zip(pr,target)]
        row={'hole':h,'board':b,'category':score>>20,'prediction':pr,'target':target,'abs_error':err}
        rows.append(row);bycat[score>>20].append(row)
    def metrics(rs):
        return {'n':len(rs),'mae':[sum(x['abs_error'][k] for x in rs)/len(rs) for k in range(2)],
                'rmse':[(sum(x['abs_error'][k]**2 for x in rs)/len(rs))**.5 for k in range(2)],
                'p95_error':[sorted(x['abs_error'][k] for x in rs)[int(.95*(len(rs)-1))] for k in range(2)]}
    out={'method':'exact enumeration all990 legal opponent combos per river, uniform and top20percent preflop classes',
         'seed':a.seed,'seconds':time.time()-started,'overall':metrics(rows),
         'by_category':{k:metrics(v) for k,v in bycat.items()},'worst_random':sorted(rows,key=lambda x:x['abs_error'][0],reverse=True)[:10],
         'worst_tight':sorted(rows,key=lambda x:x['abs_error'][1],reverse=True)[:10]}
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out['overall'],indent=2))

if __name__=='__main__':main()
