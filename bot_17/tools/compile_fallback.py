"""Convert our own JSON trees to static Python expressions (offline only)."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def compile_model(model_path,output):
    model=json.loads(model_path.read_text(encoding='utf-8'))
    def expr(tree,index):
        feature,threshold,left,right,value=tree[index]
        if feature<0:return repr(value*model['rate'])
        return f'({expr(tree,left)} if f[{feature}] < {threshold!r} else {expr(tree,right)})'
    lines=['"""Generated offline from our own trained model. No runtime code generation."""',
        f'MODEL_SIGNATURE = {(model["features"],len(model["trees"]),model["base"])!r}',
        'def predict(f):',f'    total = {model["base"]!r}']
    lines.extend('    total += '+expr(tree,0) for tree in model['trees'])
    lines.append('    return total')
    lines.extend(['','def predict_fast(f):',f'    total = {model["base"]!r}'])
    lines.extend('    total += '+expr(tree,0) for tree in model['trees'][:80])
    lines.append('    return total')
    temporary=output.with_suffix('.tmp.py');temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    temporary.replace(output)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=ROOT/'data'/'equity_model.json')
    p.add_argument('--output',type=Path,default=ROOT/'data'/'equity_predictor.py');args=p.parse_args()
    compile_model(args.model,args.output);print(args.output)

if __name__=='__main__':main()
