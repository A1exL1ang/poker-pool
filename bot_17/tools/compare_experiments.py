"""Describe paired training differences; does not silently select a winner."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('--reference',default='base')
    args=p.parse_args()
    summary=json.loads((args.directory/'summary.json').read_text())
    envs=[e[0] for e in summary['environments']]
    reference={e:json.loads((args.directory/(args.reference+'_'+e+'.json')).read_text()) for e in envs}
    output={}
    for profile in summary['profiles']:
        means=[]; variances=[]; details={}
        for e in envs:
            candidate=json.loads((args.directory/(profile+'_'+e+'.json')).read_text())
            pairs=list(zip(candidate['results'],reference[e]['results']))
            if any(x['seed']!=y['seed'] for x,y in pairs):
                raise SystemExit('Paired seeds differ')
            diffs=[x['hero_score']-y['hero_score'] for x,y in pairs]
            mean=statistics.fmean(diffs)
            varmean=statistics.variance(diffs)/len(diffs) if len(diffs)>1 else 0.
            means.append(mean);variances.append(varmean)
            details[e]={'mean':candidate['hero_score']['mean'],'paired_difference':mean,'paired_se':math.sqrt(varmean)}
        mean=statistics.fmean(means); se=math.sqrt(sum(variances))/len(envs)
        output[profile]={'mean':summary['mean_score_across_environments'][profile],
                         'difference_from_reference':mean,'paired_standard_error':se,
                         'descriptive_95_interval':[mean-1.96*se,mean+1.96*se],
                         'positive_environments':sum(x>0 for x in means),'environments':details}
    report={'reference':args.reference,'phase':summary['phase'],'comparisons':output,
            'warning':'Intervals are descriptive, unadjusted for selection/multiple comparisons. Confirm selected candidates on new training seeds before reserved evaluation.'}
    (args.directory/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='environments'} for k,v in output.items()},indent=2))


if __name__=='__main__':main()
