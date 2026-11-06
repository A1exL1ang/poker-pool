"""Post-freeze source/behavior comparisons; execute substantial work remotely.

This measures similarity, not copying, provenance, poker strength or optimality.
Probe policies get identical legal public snapshots and fresh table-local state;
their supplied current-hand history remains visible. Long-run adaptation across
hands is deliberately outside this controlled diagnostic.
"""
import argparse
import ast
from collections import Counter, defaultdict
import dataclasses
import hashlib
import gzip
import itertools
import json
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'similarity20-v1'
EXCLUDE_DIRS = {'tests', 'tools', 'training', 'benchmarks', 'measurements', 'reports',
                'checkpoints', 'dist', 'artifacts', 'lab', 'sim', 'simulator', 'native',
                '__pycache__', '.git', '.control', 'plans'}
MECHANICAL_FILES = {'engine.py', 'simulator.py', 'cards.py', 'evaluator.py', 'icm.py',
                    'interface.py', 'protocol.py', 'native.py', 'run_bot.py', 'main.py',
                    'play.py', 'run_local.py', 'build_package.py', 'package_bot.py'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class Normalize(ast.NodeTransformer):
    """Alpha-normalize local names, retaining attribute/key names and literals."""
    def __init__(self):
        self.names = {}

    def name(self, value):
        if value not in self.names:
            self.names[value] = 'v' + str(len(self.names))
        return self.names[value]

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.name(node.id), ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg = self.name(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.name = 'function'
        node.returns = None
        node.decorator_list = []
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def function_records(path, root):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    records = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith('__') or node.name in {'reset', 'main', 'safe_action', 'passive', 'legalize'}:
            continue
        nodes = list(ast.walk(node))
        if len(nodes) < 35:
            continue
        # Large literal forests/generated inference tables do not count as authored logic.
        if len(nodes) > 2000 and sum(isinstance(x, ast.Constant) for x in nodes) / len(nodes) > .35:
            continue
        original_name, line = node.name, node.lineno
        normalized = Normalize().visit(ast.parse(ast.unparse(node)).body[0])
        text = ast.dump(normalized, annotate_fields=True, include_attributes=False)
        tokens = re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s\w]", text)
        shingles = {tuple(tokens[i:i+7]) for i in range(max(0, len(tokens)-6))}
        records.append(dict(file=path.relative_to(root).as_posix(), function=original_name,
                            line=line, nodes=len(nodes), fingerprint=hashlib.sha256(text.encode()).hexdigest(),
                            shingles=shingles))
    return records


def source_analysis():
    functions, skipped, asset_info = {}, {}, {}
    for number in range(1, 21):
        root = ROOT / f'bot_{number}'
        found, excluded = [], []
        for path in sorted(root.rglob('*.py')):
            rel = path.relative_to(root)
            if any(part in EXCLUDE_DIRS or part.startswith('.venv') for part in rel.parts) or path.name in MECHANICAL_FILES:
                continue
            if any(word in path.stem.lower() for word in ('generated', 'compiled', 'equity_fast', 'table_data')):
                excluded.append(rel.as_posix())
                continue
            found.extend(function_records(path, root))
        # Packaged copies and repeated normalized functions count only once per bot.
        unique = {}
        for item in found:
            unique.setdefault(item['fingerprint'], item)
        functions[number] = list(unique.values())
        skipped[number] = excluded
        assets = []
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            runtime_artifact = number == 15 and rel.as_posix() in {
                'artifacts/equity_model.json.gz','artifacts/equity_preflop.json','artifacts/policy.json'}
            if not runtime_artifact and any(x in EXCLUDE_DIRS or x.startswith('.venv') for x in rel.parts):
                continue
            if path.suffix in ('.json', '.bin', '.gz', '.z', '.npz', '.npy', '.pt', '.onnx'):
                raw = path.read_bytes()
                structure = dict(extension=path.suffix, header_hex=raw[:8].hex())
                try:
                    decoded = gzip.decompress(raw) if path.suffix == '.gz' else raw
                    value = json.loads(decoded)
                    structure = json_structure(value)
                except (ValueError, OSError, UnicodeError):
                    pass
                assets.append(dict(file=rel.as_posix(), bytes=path.stat().st_size,
                                   sha256=hashlib.sha256(raw).hexdigest(), structure=structure))
        asset_info[number] = assets
    sets = {n: set().union(*(f['shingles'] for f in fs)) if fs else set() for n, fs in functions.items()}
    prevalence = Counter(x for items in sets.values() for x in items)
    # Remove fragments in at least half the bots: common syntax/boilerplate is weak evidence.
    sets = {n: {x for x in values if prevalence[x] < 10} for n, values in sets.items()}
    pairs = []
    for a, b in itertools.combinations(range(1, 21), 2):
        sa, sb = sets[a], sets[b]
        by_hash = {x['fingerprint']: x for x in functions[b]}
        exact = []
        for fa in functions[a]:
            fb = by_hash.get(fa['fingerprint'])
            if fb:
                exact.append(dict(a={k:fa[k] for k in ('file','function','line','nodes')},
                                  b={k:fb[k] for k in ('file','function','line','nodes')},
                                  fingerprint=fa['fingerprint']))
        pairs.append(dict(a=a,b=b,token_shingle_jaccard=len(sa & sb)/max(1,len(sa | sb)),
                          exact_normalized_functions=exact,
                          identical_asset_hashes=sorted({x['sha256'] for x in asset_info[a]} & {x['sha256'] for x in asset_info[b]})))
    pairs.sort(key=lambda x:x['token_shingle_jaccard'], reverse=True)
    for pair in pairs[:20]:
        candidates=[]
        for fa in functions[pair['a']]:
            sa={x for x in fa['shingles'] if prevalence[x]<10}
            if not sa:continue
            for fb in functions[pair['b']]:
                sb={x for x in fb['shingles'] if prevalence[x]<10}
                score=len(sa & sb)/max(1,len(sa | sb))
                if score>=.15:
                    candidates.append(dict(score=score,
                        a={k:fa[k] for k in ('file','function','line','nodes')},
                        b={k:fb[k] for k in ('file','function','line','nodes')}))
        pair['strongest_function_matches']=sorted(candidates,key=lambda x:x['score'],reverse=True)[:5]
    return dict(version=VERSION, method='Function-local alpha-renamed AST seven-token shingles; constants and attributes preserved; fragments found in >=10 bots excluded.',
                excluded='Tests, simulators, evaluators, ICM/rules helpers, interfaces, packaging, training, checkpoints, generated tables and duplicate normalized functions.',
                caveat='Structural resemblance can arise independently from the same rules; it is not evidence of copying. This strategy-focused metric excludes mechanical evaluator similarity.',
                function_counts={n:len(v) for n,v in functions.items()}, skipped_generated=skipped,
                assets=asset_info, pairs=pairs)


def json_structure(value, depth=0):
    """Describe asset shapes without emitting learned weights/generated tables."""
    if isinstance(value, dict):
        keys=sorted(value)
        return dict(type='object',keys=keys[:32],key_count=len(keys),
                    children={k:json_structure(value[k],depth+1) for k in keys[:16]} if depth<2 else {})
    if isinstance(value, list):
        return dict(type='array',length=len(value),first=json_structure(value[0],depth+1) if value and depth<2 else None)
    return dict(type=type(value).__name__)


def stratum(o):
    depth = (o.stacks[o.seat] + o.street_bets[o.seat]) / o.bb
    return dict(street=o.street, seat=o.seat, relative_button=(o.seat-o.button)%o.n_seats,
                depth='short' if depth<=10 else 'medium' if depth<=40 else 'deep',
                pressure='facing_bet' if o.to_call>0 else 'free_option',
                choice='forced_check' if not o.to_call and not o.can_raise else 'facing_bet' if o.to_call else 'optional_check_or_bet',
                raise_available=bool(o.can_raise), active_players=sum(o.active))


def build_corpus(path, max_states=768, tables_per_profile=4):
    if max_states<1 or tables_per_profile<1:raise ValueError('positive corpus sizes required')
    from common.engine import Action, Config, run_tournament
    from schedule20 import PAYOUTS, keyed_int
    buckets=defaultdict(list)
    offered=Counter()

    class Scripted:
        def __init__(self, table_id, payout):
            self.table_id, self.payout = table_id, payout
        def reset(self, seed, seat, n):
            self.rng=random.Random(keyed_int('probe-script',self.table_id,seat))
            self.serial=0
        def act(self,o):
            self.serial+=1
            sid=f'{self.table_id}:{o.hand_number}:{o.seat}:{self.serial}'
            strata=dict(stratum(o),payout=self.payout)
            key=(strata['payout'],strata['street'],strata['relative_button'],strata['depth'],strata['pressure'])
            row=dict(state_id=sid,cluster=f'{self.table_id}:{o.hand_number}',stratum=strata,
                     observation=dataclasses.asdict(o),priority=keyed_int('probe-reservoir',sid))
            offered[str(key)]+=1
            buckets[key].append(row)
            buckets[key].sort(key=lambda x:x['priority'])
            del buckets[key][4:]
            u=self.rng.random()
            if o.can_raise and u<.18:
                target=min(o.max_raise_to,max(o.min_raise_to,o.street_bets[o.seat]+o.to_call+max(o.bb,int(o.pot*.6))))
                return Action('raise',target)
            if o.to_call:
                return Action('fold') if u<.32 else Action('call')
            return Action('check')

    tables=0
    for payout,pay in PAYOUTS.items():
        for stack in (300,1200,6000):
            for repetition in range(tables_per_profile):
                tid=f'{payout}-{stack}-{repetition}'
                bots=[Scripted(tid,payout) for _ in range(8)]
                result=run_tournament(bots,Config(starting_stack=stack,max_hands=60,payouts=pay,strict_timing=False),keyed_int('probe-deck',tid))
                if sum(result.errors) or sum(result.illegal):
                    raise RuntimeError('scripted corpus generator produced a fault')
                tables+=1
    selected=[]
    ordered=sorted(buckets,key=lambda key:keyed_int('probe-stratum-order',str(key)))
    # Round-robin the strata instead of letting common preflop states dominate.
    for layer in range(4):
        for key in ordered:
            if len(buckets[key])>layer:
                selected.append(buckets[key][layer])
    # Deterministic priority truncation within each round gives all strata an equal chance.
    selected=selected[:max_states]
    selected.sort(key=lambda x:x['state_id'])
    for row in selected:row.pop('priority')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in selected))
    meta=dict(version=VERSION,states=len(selected),scripted_tables=tables,available_strata=len(buckets),
              source='Original scripted legal agents, independent of all20 policies; no strength claims.',
              snapshot_fields='Only common Observation fields, public history and acting-player hole cards.',
              selection='Four smallest independent hash priorities per payout/street/button-position/depth/pressure stratum, round-robin truncated.',
              corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              counts={field:dict(Counter(str(row['stratum'][field]) for row in selected)) for field in ('street','relative_button','depth','pressure','payout','active_players')})
    path.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2))
    return meta


def restore_observation(data):
    from common.engine import Event, Observation
    d=dict(data)
    d['history']=tuple(Event(**e) for e in d['history'])
    for key in ('hole','board','stacks','street_bets','contributions','folded','all_in','active','starting_stacks','payouts'):
        d[key]=tuple(d[key])
    return Observation(**d)


def is_legal(action,o):
    if action.kind=='fold':return True
    if action.kind=='check':return o.to_call==0
    if action.kind=='call':return o.to_call>0
    return (action.kind=='raise' and o.can_raise and type(action.amount) is int
            and o.street_bets[o.seat]+o.to_call<action.amount<=o.max_raise_to
            and action.amount>=o.min_raise_to)


def replay(corpus, output, policy_seeds=3, repeats=2, numbers=range(1,21)):
    from run_pool20 import make_adapter, SeedSeparated
    from common.engine import Action
    rows=read_jsonl(corpus)
    numbers=list(numbers)
    if policy_seeds<1 or repeats<1 or not numbers or len(set(numbers))!=len(numbers) or any(n not in range(1,21) for n in numbers):
        raise ValueError('positive repetitions and unique bot IDs1..20 required')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('Refusing to overwrite a completed/partial replay; choose a fresh output')
    with out.open('w',buffering=1) as stream:
        for number in numbers:
            adapter=make_adapter(number)
            for row in rows:
                for policy_seed in range(policy_seeds):
                    # Same seed across repeats isolates timing/nondeterministic variation.
                    # Different bot IDs receive independent streams via SeedSeparated.
                    for repeat in range(repeats):
                        o=restore_observation(row['observation'])
                        wrapped=SeedSeparated(adapter,number,f'probe:{row["state_id"]}:{policy_seed}')
                        wrapped.reset(0,o.seat,o.n_seats)
                        error=None;action=None
                        start=time.perf_counter_ns()
                        try:action=wrapped.act(o)
                        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                        elapsed=time.perf_counter_ns()-start
                        valid=action is not None and is_legal(action,o)
                        fallback=Action('check' if o.to_call==0 else 'fold')
                        effective=action if valid and elapsed<=1_000_000 else fallback
                        stream.write(json.dumps(dict(state_id=row['state_id'],bot=number,policy_seed=policy_seed,repeat=repeat,
                            raw_kind=action.kind if action else None,raw_amount=action.amount if action else None,
                            effective_kind=effective.kind,effective_amount=effective.amount,
                            valid=valid,error=error,elapsed_ns=elapsed,timeout=elapsed>1_000_000,
                            **wrapped.counters()),separators=(',',':'))+'\n')
    out.with_suffix(out.suffix+'.meta.json').write_text(json.dumps(dict(version=VERSION,corpus_sha256=hashlib.sha256(Path(corpus).read_bytes()).hexdigest(),
        bots=list(numbers),policy_seeds=policy_seeds,repeats=repeats,
        initialization='Fresh Agent/table state per snapshot; supplied current-hand history only. Startup outside timing.',
        timing='Raw policy action and strict1ms effective check/fold reported separately. Same-seed repeats measure execution variability.'),indent=2))


def agreement_metrics(items):
    """Pure metric core; items are (kindA,kindB,amountA,amountB,observation)."""
    n=len(items)
    if not n:return dict(count=0)
    same=sum(a==b for a,b,_,_,_ in items)
    ca,cb=Counter(x[0] for x in items),Counter(x[1] for x in items)
    chance=sum(ca[k]*cb[k] for k in ca)/(n*n)
    contested=[x for x in items if x[4]['to_call']>0]
    informative=[x for x in items if not (x[0]==x[1] and x[0] in ('fold','check'))]
    raises=[x for x in items if x[0]==x[1]=='raise']
    diffs=[abs(x-y)/max(1,o['pot']+min(o['to_call'],o['stacks'][o['seat']])) for _,_,x,y,o in raises]
    return dict(count=n,action_agreement=same/n,marginal_chance_agreement=chance,
                chance_corrected_agreement=(same/n-chance)/(1-chance) if chance<1 else None,
                both_fold=sum(a==b=='fold' for a,b,_,_,_ in items)/n,
                both_check=sum(a==b=='check' for a,b,_,_,_ in items)/n,
                contested_count=len(contested),contested_action_agreement=sum(x[0]==x[1] for x in contested)/len(contested) if contested else None,
                nontrivial_count=len(informative),nontrivial_agreement=sum(x[0]==x[1] for x in informative)/len(informative) if informative else None,
                both_raise_count=len(raises),mean_raise_difference_pot_units=sum(diffs)/len(diffs) if diffs else None,
                raises_within_5pct_pot_or_1bb=sum(abs(x-y)<=max(o['bb'],.05*(o['pot']+min(o['to_call'],o['stacks'][o['seat']]))) for _,_,x,y,o in raises)/len(raises) if raises else None)


def analyze(corpus,replays,bootstrap=1000):
    import numpy as np
    corpus_rows=read_jsonl(corpus);states={r['state_id']:r for r in corpus_rows}
    decisions={};faults=Counter();timing=Counter();seed_set=set();repeat_set=set()
    for path in replays:
        for d in read_jsonl(path):
            key=(d['bot'],d['state_id'],d['policy_seed'],d['repeat'])
            if key in decisions:raise ValueError(f'duplicate replay {key}')
            if d['state_id'] not in states:raise ValueError('replay references unknown state')
            decisions[key]=d;seed_set.add(d['policy_seed']);repeat_set.add(d['repeat'])
            faults[d['bot']]+=bool(d['error'] or not d['valid'] or d['internal_errors'])
            timing[d['bot']]+=d['timeout']
    expected=len(states)*20*len(seed_set)*len(repeat_set)
    if len(decisions)!=expected:raise ValueError(f'incomplete replay: {len(decisions)} vs {expected}')
    clusters=sorted({r['cluster'] for r in corpus_rows});cluster_index={c:i for i,c in enumerate(clusters)}
    rng=np.random.default_rng(290119)
    weights=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=bootstrap) if bootstrap else None
    pairs=[]
    for a,b in itertools.combinations(range(1,21),2):
        raw=[];effective=[];strata=defaultdict(list)
        cluster_num=np.zeros(len(clusters));cluster_den=np.zeros(len(clusters));excluded=0
        for sid,row in states.items():
            for seed in sorted(seed_set):
                for repeat in sorted(repeat_set):
                    x,y=(decisions[(n,sid,seed,repeat)] for n in (a,b))
                    if x['error'] or y['error'] or not x['valid'] or not y['valid'] or x['internal_errors'] or y['internal_errors']:
                        excluded+=1;continue
                    item=(x['raw_kind'],y['raw_kind'],x['raw_amount'],y['raw_amount'],row['observation'])
                    raw.append(item)
                    effective.append((x['effective_kind'],y['effective_kind'],x['effective_amount'],y['effective_amount'],row['observation']))
                    for field in ('street','depth','pressure','payout','choice','relative_button','active_players'):
                        strata[f'{field}:{row["stratum"][field]}'].append(item)
                    ci=cluster_index[row['cluster']];cluster_num[ci]+=item[0]==item[1];cluster_den[ci]+=1
        confidence=None
        if bootstrap and raw:
            denominator=weights@cluster_den;numerator=weights@cluster_num
            values=numerator[denominator>0]/denominator[denominator>0]
            confidence=[float(x) for x in np.quantile(values,[.025,.975])]
        pairs.append(dict(a=a,b=b,raw=agreement_metrics(raw),strict_1ms=agreement_metrics(effective),
                          raw_agreement_cluster95=confidence,excluded_faulty_decisions=excluded,
                          strata={k:agreement_metrics(v) for k,v in strata.items()}))
    stability={}
    for n in range(1,21):
        comparisons=[]
        for sid,row in states.items():
            for seed in seed_set:
                selected=[decisions[(n,sid,seed,r)] for r in sorted(repeat_set)]
                for x,y in itertools.combinations(selected,2):
                    comparisons.append((x['raw_kind'],y['raw_kind'],x['raw_amount'],y['raw_amount'],row['observation']))
        stability[n]=agreement_metrics(comparisons)
    pairs.sort(key=lambda p:(p['raw'].get('chance_corrected_agreement') or -1),reverse=True)
    return dict(version=VERSION,corpus_sha256=hashlib.sha256(Path(corpus).read_bytes()).hexdigest(),
                states=len(states),clusters=len(clusters),policy_seeds=sorted(seed_set),same_seed_repeats=sorted(repeat_set),
                bootstrap=bootstrap,faults=dict(faults),timeouts=dict(timing),within_bot_same_seed_stability=stability,pairs=pairs,
                limitations=['Probe distribution is scripted and stratified, not the tournament visitation distribution.',
                  'Fresh-state probes omit adaptation accumulated across hands; current-hand history is supplied.',
                  'Intervals resample source table/hand clusters, retaining repeated policy seeds and same-seed repeats together; they are exploratory, not adjusted for190 pair comparisons.',
                  'Identical folds/checks and common legal constraints are weak evidence: use chance-corrected, contested, nontrivial and raise-size measures together.',
                  'Hardware scheduling and internal time budgets can change actions despite controlled policy seeds.',
                  'Implementation resemblance and behavioral resemblance are distinct and do not establish copying.'])


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('source');s.add_argument('--output',required=True)
    c=sub.add_parser('corpus');c.add_argument('--output',required=True);c.add_argument('--max-states',type=int,default=768);c.add_argument('--tables-per-profile',type=int,default=4)
    r=sub.add_parser('replay');r.add_argument('--corpus',required=True);r.add_argument('--output',required=True);r.add_argument('--policy-seeds',type=int,default=3);r.add_argument('--repeats',type=int,default=2);r.add_argument('--bots',default=','.join(map(str,range(1,21))))
    a=sub.add_parser('analyze');a.add_argument('--corpus',required=True);a.add_argument('--replays',nargs='+',required=True);a.add_argument('--output',required=True);a.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args()
    if args.command=='corpus':result=build_corpus(args.output,args.max_states,args.tables_per_profile)
    elif args.command=='replay':return replay(args.corpus,args.output,args.policy_seeds,args.repeats,[int(x) for x in args.bots.split(',')])
    elif args.command=='source':result=source_analysis()
    else:result=analyze(args.corpus,args.replays,args.bootstrap)
    if args.command!='corpus':
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'command':args.command,'output':args.output,'complete':True}))


if __name__=='__main__':main()
