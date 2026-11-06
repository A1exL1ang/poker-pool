"""Validate the post-evaluation ICM import-hoist without changing strategy."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent import Agent
from tools.evaluated_policy import Agent as EvaluatedAgent


class RemoveImports(ast.NodeTransformer):
    def visit_Import(self,node): return None
    def visit_ImportFrom(self,node): return None


def normalized(name):
    return ast.dump(RemoveImports().visit(ast.parse(Path(name).read_text())),include_attributes=False)


assert normalized('agent.py')==normalized('tools/evaluated_policy.py')
data=json.loads(Path('results/final_capture.json').read_text())
observations=[o for row in data['rows'] for o in row['observations']]
old=EvaluatedAgent(98010);new=Agent(98010)
for o in observations:
    assert old.act(o)==new.act(o)
assert old.failures==new.failures==0
report={'evaluated_agent_sha256':hashlib.sha256(Path('tools/evaluated_policy.py').read_bytes()).hexdigest(),
        'final_agent_sha256':hashlib.sha256(Path('agent.py').read_bytes()).hexdigest(),
        'import_stripped_ast_identical':True,'observations_with_identical_actions':len(observations),
        'internal_fallbacks':0,'change':'Only from icm import utility moved from first-use decision path to module initialization; decision logic and assets unchanged.'}
Path('results/final_equivalence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
