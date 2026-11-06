"""Audited strategy-only replacement for the v1 *source* similarity report.

The v1 recursive file scan accidentally included generated ICM, simulator and
schema code. This revision uses an explicit reviewed policy-file allowlist and
excludes mechanical functions even when embedded in policy files. It does not
change or invalidate the separately collected behavioral probe results.

Execute the full analysis remotely; no bot imports or inference are needed.
"""
import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
import re
from pathlib import Path

from similarity20 import ROOT, Normalize, function_records

VERSION = 'similarity-source-v2-policy-allowlist'

# Reviewed entrypoints holding authored betting/range/adaptation decisions.
# Asset loading, model execution, simulators and all generated files are absent.
POLICY_FILES = {
    1: ('pokerbot/agent.py',),
    2: ('pokerbot/agent.py',),
    3: ('bot3/agent.py',),
    4: ('bot/policy.py',),
    5: ('pokerbot/agent.py',),
    6: ('bot6/agent.py',),
    7: ('bot7/agent.py',),
    8: ('kestrel/policy.py',),
    9: ('bot.py',),
    10: ('agent.py',),
    11: ('pokerbot/agent.py',),
    12: ('agent.py',),
    13: ('bot13/agent.py',),
    14: ('pokerbot/strategy.py',),
    15: ('bot/agent.py',),
    16: ('bot/policy.py',),
    17: ('agent.py',),
    18: ('agent.py',),
    19: ('agent.py',),
    20: ('pokerbot/policy.py',),
}

# Common legal-output clamps, schema adapters, PRNG/reset/position bookkeeping,
# model setup and exact ICM/side-pot accounting are weak evidence of independently
# chosen poker strategy. Exclude them consistently rather than interpreting them
# as a strong shared implementation.
MECHANICAL_FUNCTIONS = {
    'reset', 'reset_hand', 'main', 'normalize', 'from_dict', 'to_dict', 'as_dict',
    'observation', 'legalize', '_legalize', 'safe_action', 'passive', 'fallback',
    '_safe', '_legal', '_raise', '_call', '_pass', 'calling', 'raising',
    'clamp', '_clamp', '_clip', '_random', '_seed', '_mix', '_finish',
    '_position', '_in_position', '_order', '_pre_order', '_live', '_behind',
    '_eligible_pot', '_eligible', '_pot_layers', '_pot_groups', '_icm_hero',
    'load_assets', 'selected_config', '_starting_tables', '_percentiles',
}

# These functions exist only as missing-import/asset fallback scaffolding;
# selected frozen submissions use their actual packaged equity implementations.
BOT_EXCLUSIONS = {
    9: {'preflop_equity', 'preflop_percentile', 'equity_vs_range'},
    19: {'preflop_equity', 'preflop_percentile', 'predict_from_features', 'multiway_equity'},
}


def function_allowed(number, name):
    return (not name.startswith('__') and name not in MECHANICAL_FUNCTIONS
            and name not in BOT_EXCLUSIONS.get(number, set()))


def location(record):
    return {key:record[key] for key in ('file','function','line','nodes')}


def policy_records(path, base, number):
    """Retain v1 normalization, also pruning excluded nested helper definitions."""
    records=function_records(path,base)
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    by_line={n.lineno:n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

    class Prune(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if not function_allowed(number,node.name):return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
        visit_AsyncFunctionDef=visit_FunctionDef

    for record in records:
        if not function_allowed(number,record['function']):
            yield record
            continue
        cloned=ast.parse(ast.unparse(by_line[record['line']])).body[0]
        pruned=Prune().visit(cloned)
        text=ast.dump(Normalize().visit(pruned),annotate_fields=True,include_attributes=False)
        tokens=re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s\w]",text)
        record=dict(record,fingerprint=hashlib.sha256(text.encode()).hexdigest(),
                    shingles={tuple(tokens[i:i+7]) for i in range(max(0,len(tokens)-6))})
        yield record


def analyze_source(root=ROOT):
    root=Path(root)
    functions={};audit={}
    for number, files in POLICY_FILES.items():
        base=root/f'bot_{number}'
        kept=[];excluded=[];hashes={}
        for relative in files:
            path=base/relative
            if not path.is_file():raise FileNotFoundError(f'Allowlisted policy source missing: {path}')
            hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
            for record in policy_records(path,base,number):
                if function_allowed(number,record['function']):kept.append(record)
                else:excluded.append(location(record))
        unique={}
        duplicates=[]
        for record in kept:
            if record['fingerprint'] in unique:duplicates.append(location(record))
            else:unique[record['fingerprint']]=record
        functions[number]=list(unique.values())
        audit[number]=dict(files=list(files),source_sha256=hashes,
                           included_functions=[location(x) for x in functions[number]],
                           excluded_mechanical_functions=excluded,
                           deduplicated_functions=duplicates)
        if not functions[number]:raise ValueError(f'No strategy functions selected for bot{number}')
    raw_sets={n:set().union(*(f['shingles'] for f in records)) for n,records in functions.items()}
    prevalence=Counter(s for values in raw_sets.values() for s in values)
    sets={n:{s for s in values if prevalence[s]<10} for n,values in raw_sets.items()}
    function_sets={n:[{s for s in f['shingles'] if prevalence[s]<10} for f in records]
                   for n,records in functions.items()}
    pairs=[]
    for a,b in itertools.combinations(range(1,21),2):
        sa,sb=sets[a],sets[b]
        by_hash={f['fingerprint']:f for f in functions[b]}
        exact=[]
        for fa in functions[a]:
            if fa['fingerprint'] in by_hash:
                exact.append(dict(a=location(fa),b=location(by_hash[fa['fingerprint']]),
                                  fingerprint=fa['fingerprint']))
        pairs.append(dict(a=a,b=b,token_shingle_jaccard=len(sa & sb)/max(1,len(sa | sb)),
                          shared_shingles=len(sa & sb),union_shingles=len(sa | sb),
                          exact_normalized_functions=exact))
    pairs.sort(key=lambda x:x['token_shingle_jaccard'],reverse=True)
    for pair in pairs[:20]:
        matches=[]
        for fa,sa in zip(functions[pair['a']],function_sets[pair['a']]):
            if not sa:continue
            for fb,sb in zip(functions[pair['b']],function_sets[pair['b']]):
                score=len(sa & sb)/max(1,len(sa | sb))
                if score>=.15:
                    matches.append(dict(score=score,shared_shingles=len(sa & sb),
                                        a=location(fa),b=location(fb)))
        pair['strongest_function_matches']=sorted(matches,key=lambda x:x['score'],reverse=True)[:5]
    return dict(version=VERSION,scope='Authored betting, range selection and opponent-adaptation policy functions only.',
        replaces='similarity20-v1 source pairs/function matches; do not use the superseded source ranking.',
        unaffected='The shared legal-state behavioral probe results and their statistics are unchanged. Asset hashes/shape metadata remain a separate descriptive inventory, not strategy-code evidence.',
        corrections=[
            'Explicit policy-file allowlist excludes bot18 policy/icm_fast.py, including generated _dp_7_5.',
            'All simulator files excluded by construction, including bot14 pokerbot/game.py and bot19 sim.py.',
            'Model/schema modules excluded, including model.py from_dict helpers.',
            'Inline exact ICM, side-pot accounting, input/output normalization, legal clamps, PRNG and position/reset boilerplate excluded consistently.',
            'Missing-import fallback equity scaffolding excluded for bots9 and19; packaged duplicate copies and generated asset tables never scanned.'
        ],
        method='Same v1 function-local alpha-normalized AST seven-token shingles, retaining constants/attribute names; excluded nested helper definitions replaced by Pass before normalization; fragments present in at least10 bots removed; identical functions within a bot deduplicated.',
        function_counts={n:len(fs) for n,fs in functions.items()},audit=audit,
        mechanical_function_exclusions=sorted(MECHANICAL_FUNCTIONS),
        per_bot_function_exclusions={n:sorted(v) for n,v in BOT_EXCLUSIONS.items()},pairs=pairs,
        limitations=[
            'This deliberately narrower metric measures selected policy structure, not whole-program similarity or all poker logic.',
            'Calls to separately implemented equity, risk and model modules can hide differences; behavioral comparisons complement this metric.',
            'Alpha-normalization retains constants and attribute/key names, so renaming or different program decompositions can lower similarity.',
            'Functions below35 AST nodes and large literal-dominated functions are excluded by the shared extractor.',
            'Shared standard poker decisions and common coding idioms can arise independently; resemblance does not establish copying.',
            'Pair scores and strongest function matches are descriptive, with no universal threshold defining equivalent policies.'
        ])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    result=analyze_source()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'version':VERSION,'output':str(out),'complete':True}))


if __name__=='__main__':main()
