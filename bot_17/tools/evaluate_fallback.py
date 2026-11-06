"""Compare fallback models against own high-sample, disjoint-board teacher."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import equity

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'data'/'equity_validation.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'data'/'equity_model_validation.json')
    args=p.parse_args();rows=[json.loads(x) for x in args.input.read_text(encoding='utf-8').splitlines()]
    featerr=0;ts=[];parity=0;rawparity=0;groups={};hse=mse=hmae=mmae=0;n=len(rows)
    for r in rows:
        f=r['features'];h=r['hole'];b=r['board'];opponents=int(f[1])
        ff=equity.model_features(h,b,opponents,f[2],f[3])
        featerr=max(featerr,max(abs(x-y) for x,y in zip(f,ff)))
        st=time.perf_counter_ns();m=equity.model_estimate(h,b,opponents,f[2],f[3]);ts.append((time.perf_counter_ns()-st)/1000)
        v=equity._heuristic(h,b,opponents,f[2],f[3]);y=r['teacher']
        mse+=(m-y)**2;hse+=(v-y)**2;mmae+=abs(m-y);hmae+=abs(v-y)
        if 'prediction' in r:
            parity=max(parity,abs(m-r['prediction']))
            if equity._COMPILED_PREDICTOR is not None:rawparity=max(rawparity,abs(equity._COMPILED_PREDICTOR(ff)-r['prediction']))
        key=f'board{len(b)}_opponents{opponents}'
        group=groups.setdefault(key,{'n':0,'model_squared_error':0.,'heuristic_squared_error':0.})
        group['n']+=1;group['model_squared_error']+=(m-y)**2;group['heuristic_squared_error']+=(v-y)**2
    result=dict(n=n,heuristic_rmse=math.sqrt(hse/n),model_rmse=math.sqrt(mse/n),heuristic_mae=hmae/n,model_mae=mmae/n,
        max_feature_error=featerr,max_prediction_error_after_clipping_and_exact_overrides=parity,max_raw_prediction_parity_error=rawparity,
        p50_us=sorted(ts)[n//2],p99_us=sorted(ts)[int(n*.99)],max_us=max(ts),over_1ms=sum(t>1000 for t in ts),
        groups=groups,source=str(args.input.name),evaluation_kind='validation used for development, not final held-out tournament evaluation')
    print(json.dumps(result,indent=2));args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert featerr<1e-8,'C++ and Python features disagree'

if __name__=='__main__':main()
