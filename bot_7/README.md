# Bot 7

One independent, offline Python bot for standard no-limit Texas Hold'em, designed for two through eight remaining players. Runtime uses only the Python standard library. Nothing was submitted, published, committed, or pushed.

`RULES.txt` was read in full. The official Agent interface, finishing-point schedule, later blind levels, and sandbox hardware were not supplied. **This implements a local interface, not verified official compatibility.** Adapt the boundary in `agent.py` after those details are released; see [INTERFACE.md](INTERFACE.md).

The policy combines position/depth-aware preflop ranges, independently generated preflop equity tables, exact 5–7-card evaluation, range-conditioned postflop sampling, table-local opponent statistics, selective value betting/bluffing, and side-pot-weighted equity. Actual payouts, when supplied, enable exact ICM utility calculations for large heads-up calls. With no payouts, decisions use chip equity. Ranges, realization, bet sizing, and shove responses remain heuristics; this is not a solved equilibrium strategy.

Run these commands from this directory with Python 3.10+; development was tested on Python 3.11.9 for Windows:

```powershell
Get-Content examples/observation.jsonl | python agent.py
python -m unittest discover -s tests -v
python -m tools.exhaustive_evaluator
python -m tools.equity_audit
python -m tools.latency --iterations 30000 --output benchmarks/latency_final.json
python -m tools.benchmark --tables 500 --field mixed --seed 2700 --budget-us 0 --output benchmarks/mixed_final.json
python -m tools.report
python -m tools.package
```

The example returns `{"action":"raise","amount":43}`. On POSIX, use `python agent.py < examples/observation.jsonl` instead of `Get-Content`.

In process:

```python
from agent import Agent, Observation
bot = Agent()                       # Construct once per table.
state = Observation.from_dict(data) # Map/validate outside the timed call.
action = bot.act(state)             # Action("raise", total_street_bet), etc.
```

The final validation results are recorded in `benchmarks/RESULTS.md`. Raw reports include configuration, seeds, runtime file hashes, timing distributions, errors, and per-table results. The suite checks all 2,598,960 five-card categories, 7,500 independent slow-evaluator comparisons, ICM against permutation and subset-DP references, legal raises, cumulative short-all-in reopening, side pots, short blind edge cases, and chip conservation.

The default sampler targets 48 opponent comparisons, with a minimum eight multiway runouts and a **450 microsecond soft decision budget**. It checks the clock every two runouts. This is not a hard real-time guarantee: Python, garbage collection, and OS scheduling can still exceed 1 ms, and measured overruns are retained in the reports. Module loading takes much longer than 1 ms and must occur before timed decisions. The unknown official adapter and sandbox must be benchmarked before use. JSON parsing and dictionary normalization are not part of `Agent.act`; dictionary-boundary timings are reported separately.

Tournament benchmarks use **synthetic** blinds `20,30,50,80,120,200,320,500`, advancing every 25 hands, 2,000 starting chips, and a 200-hand cap. Reports show winner-take-all ICM share and separately labeled hypothetical `8,7,6,5,4,3,2,1` finishing points. Neither is claimed to be the official point schedule. Test opponents are deliberately simple local fixtures; success against them does not predict contest ranking. `--budget-us 0` disables clock-dependent early stopping for reproducible strategy experiments. `--enforce-timeout` models the rule's fold/check fallback. With the soft budget enabled, exact trajectories depend on machine load.

Additional reproducible runs:

```powershell
python -m tools.benchmark --tables 300 --field equity --seed 1700 --budget-us 0 --output benchmarks/equity_final.json
python -m tools.benchmark --tables 300 --field self --seed 9700 --budget-us 0 --output benchmarks/self_final.json
python -m tools.benchmark --tables 200 --field mixed --seed 3700 --enforce-timeout --output benchmarks/timeouts_final.json
python -m tools.benchmark --tables 100 --field mixed --seed 4700 --payouts 50,30,20,0,0,0,0,0 --output benchmarks/payouts_final.json
```

Assets are included. To regenerate them locally (C++ is needed only for the offline preflop generator):

```powershell
python tools/generate_evaluator.py
g++ -O3 -std=c++17 tools/generate_preflop.cpp -o tools/generate_preflop.exe
./tools/generate_preflop.exe 40000
```

The preflop asset covers all 169 classes against seven ranges, using 40,000 seeded trials per class/range, plus random multiway estimates. `eval.bin.gz` holds 73,775 rank-multiset scores and 8,192 flush-mask scores. Asset generation and evaluation contain no copied competitor code. The raw `.bin` and native executable are development tools; the portable runtime bundle excludes both.

`python -m tools.report` checks that final benchmark hashes match the runtime and creates the validation summary. `python -m tools.package` creates `build/bot7.zip`, a manifest, and an isolated extraction/JSON smoke test. The archive includes the Python runtime, required assets, documentation, example, and final validation evidence; the development harness and generators remain in this workspace. No network calls occur in gameplay. All work stayed in this workspace. Remote compute was unnecessary; no Modal or Engaging jobs were started, and no user allocations were touched.

Limitations: opponent ranges and future betting are approximate; Monte Carlo uncertainty is material near thresholds; the short-stack shove model primarily assumes one caller; preflop multiway equity uses an approximation; ICM call analysis assumes a heads-up pot awarded to the opponent on a fold and ignores future betting; the simulator has no ante schedule or official integration. Earlier exploratory benchmark files may use older policy or simulator versions; use the final files listed in `benchmarks/RESULTS.md`.
