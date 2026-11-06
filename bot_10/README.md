# bot_10

Independent, offline, Python-standard-library no-limit Texas Hold'em bot. This is a **local experiment**, not an official contest submission or an implementation of a verified official Agent interface. The orchestrator must adapt the documented observations/actions to the actual starter kit. Nothing was pushed, published, or submitted.

## Run

Python 3.11+:

These verification/benchmark commands run from the full workspace. `dist/bot_10.zip` contains only runtime code, selected assets, this README, and a SHA-256 manifest; tests, generators, archived candidates, and result files remain in the workspace. Use `python main.py` or import `Agent` from the extracted runtime bundle.

```powershell
python -m unittest discover -s tests -v
python main.py
python benchmark.py --tables 100 --seed 60000 --lineup self --output results/example.json
python tools/latency.py --rounds 10 --output results/example_latency.json
python tools/package.py
```

For direct integration, construct `agent.Agent(seed=10)` once per table, call `act(observation)` on each turn, and call `reset()` at a table boundary. `PokerBot` is an alias. No network, subprocess, model service, native extension, NumPy, or sklearn is used in gameplay. JSON-lines `main.py` is a diagnostic transport: one observation per line produces one action per line; `{"reset":true}` resets the table. Process startup and JSON transport are not the in-process decision benchmark.

## Local observation contract

- `hole`: two distinct integer cards. `board`: zero, three, four, or five distinct cards, disjoint from hole cards. Card ID is `4*rank+suit`, rank 0=2 through 12=A, suits 0=clubs, 1=diamonds, 2=hearts, 3=spades. `cards.parse_card('As')` converts display strings.
- `hero`, `button`: zero-based seats. `players` is a fixed seat-indexed list. Every player has `seat`, remaining `stack`, current-street `bet`, total-hand `committed`, `folded`, and `all_in`. Eliminated seats remain present with zero stacks. Chips are nonnegative integers.
- `street`: 0 preflop, 1 flop, 2 turn, 3 river. `hand_id` changes every hand. `history` is the full public action history of the current hand: mappings with `seat`, `street`, `action`, `amount`. Blind postings use `small_blind`/`big_blind`; calls have incremental amounts and raises have total current-street targets. Do not include unrevealed cards.
- `small_blind`, `big_blind`; `pot` includes all prior/current contributions. `to_call` is the actual incremental call cost capped by hero stack, excluding unmatched overbets hero need not match. The policy separately caps eligible pot layers when hero is short.
- `can_raise` incorporates reopening restrictions and dry side pots. `min_raise_to` is the minimum full legal total current-street target. `max_raise_to` is hero's current bet plus remaining stack. A target below minimum is legal only when it equals maximum (short all-in).
- Optional `payouts`: descending **remaining-survivor** awards from first place onward, omitting already-paid tail places; omitted remaining places pay zero. No payout vector is assumed when absent. Exact ICM adjusts isolated heads-up all-in call thresholds, up to eight surviving seats. The policy otherwise uses chip equity and does not solve full ICM tournament strategy.

Return values are `{"action":"fold"}`, `{"action":"check"}`, `{"action":"call"}`, or `{"action":"raise","amount":TOTAL_STREET_TARGET}`. An all-in is a call or raise with the appropriate bounds, never a special action name. Malformed mappings receive a conservative check/fold and increment `Agent.failures`; invalid non-mappings receive fold. The official adapter should validate its own types and units.

## Strategy and assets

The policy combines position-sensitive opening ranges, value three-bets, pot-odds calls, short-stack mixed strategies, and postflop value betting/checking/bluffing. Anonymized seat statistics are strongly shrunk toward priors and reset between tables; only observed actions count as opportunities, since hero cannot see later actions after folding. Postflop ranges depend on preflop activity and subsequent betting evidence. Eligible pots, guaranteed shared-nut boards, and optional all-in ICM are treated separately from estimated equities.

`assets/preflop_equity.json` has 169 hand classes against uniform, top35%, top15%, and top5% ranges ranked by sampled uniform equity: 500,000 runouts per class/profile, 338 million sampled showdowns. Worst-case per-estimate standard error is 0.000707. Whole hand classes at cutoffs make actual range sizes differ slightly from nominal percentages.

`assets/equity_model.json` is an eight-tree, depth-16 forest trained offline on 180,000 random flop/turn/river states, each with 96 samples against four preflop-range profiles (uniform and heuristic top60/30/12%). Runtime traverses ordinary Python arrays; no Monte Carlo occurs during decisions. Features and native teacher were independently crosschecked. The 12,000-state model-selection set used 384 samples/profile; 8,000 fresh states with 512 samples/profile are reserved for final calibration. These ranges are modeling assumptions; real betting-conditioned ranges can differ substantially. A geometric multiway approximation and action-strength adjustment are heuristic, not exact multiway equities.

`assets/pushfold.json` is an offline mixed-strategy first-caller abstraction at 2–15 BB and one to seven seats behind. Its 169×169 payoff matrix has 10,000 samples per unordered class pair and exact opponent-card blocker counts. The model permits one shove and the first call, forcing later players to fold; it excludes overcalls, unequal stacks, limps, antes, ICM, and postflop play. Five hundred fictitious-play iterations have maximum unilateral best-response gain 0.00697 BB/dealt hand **within that abstraction**; this is not a full-game equilibrium or optimality claim. Runtime use is restricted to first-in short-stack shoves and clean, nearly equal-stack heads-up shove calls.

## Rules and simulation assumptions

The complete supplied `RULES.txt` was read. It specifies Python, offline play, a 1 ms decision limit, a 100 MB file limit, eight starting seats with 2,000 chips, initial 10/20 blinds, increases every 25 hands, and a 200-hand cap. It does not supply an official API, subsequent blind levels, antes, payout vector, simultaneous-elimination convention, or exact timing hardware. The user separately specified standard no-limit Hold'em and authorized unlimited useful development time.

`engine.py` is our independent simulator: standard best-five-of-seven showdown, burns, heads-up button/SB action order, full and short raises, cumulative short-raise reopening, short blinds, dry side pots, uncalled-wager refunds, side pots, tied pots, and odd chips clockwise after the button. There are no antes. Buttons advance to the next funded seat. Simultaneous eliminations are ranked by hand-start stack, splitting awards among equal stacks; this is a local convention to be replaced if official rules differ.

`benchmark.py` uses three clearly **synthetic** blind schedules, all increasing every 25 hands, and three normalized scoring sensitivities: winner-only, top-three 50/30/20, and descending linear points. These are not official contest payouts. Survivors at hand 200 receive exact ICM allocation. Synthetic opponents include loose/tight/passive/aggressive policies and a frozen checkpoint of our own earlier policy (`tools/reference_policy.py`). They cannot establish performance against unknown contest bots.

## Validation and reproducibility

Results, seeds, configurations, and raw per-table outcomes are in `results/`. Development/tuning seeds are separate from final evaluation seeds. Timing measures wall-clock call duration, including scheduler/GC pauses, excluding one-time import/model loading. Occasional measured overruns are retained in reports; Python on a shared host cannot provide a hard real-time guarantee. A contest timeout would trigger the rules' check/fold fallback. One isolated local startup measurement took about 0.32 seconds to import/load models and 51 microseconds to construct the Agent; initialize outside the decision clock and verify the actual host policy.

The frozen candidate was evaluated in **10,000 held-out tables paired with the earlier reference**, spanning three blind schedules and eight opponent/schedule cohorts. Aggregate normalized winner-only score improved by 0.0272 (approximate paired 95% interval 0.0187–0.0357), and top-three score by 0.00577 (0.00195–0.00959). Linear score changed by -0.000797 (-0.00212–0.000524), which is inconclusive. These are synthetic scoring sensitivities, not contest payouts or contest-win predictions. The 2,000-table normal-schedule frozen-reference cohort had winner-only score 0.1668 versus 0.1220 for the earlier policy. All 10,000 candidate tables had zero invalid actions/exceptions or internal policy fallbacks; strict timeout effects are evaluated separately. Full per-cohort outcomes, uncertainty, and frozen source/model hashes are in `results/final_evaluation.json`.

Final Linux CPython 3.12 replay used 14,702 captured observations over 20 passes plus 2,000 explicit ICM decisions: **296,040 calls**, p99 30.1µs preflop / 80.6µs postflop / 81.6µs ICM, **five wall-clock overruns above 1ms**, maximum 47.2ms. No illegal actions or internal fallbacks occurred. Timing includes OS scheduling and GC; the long-tail cause was not separately instrumented. In a separate **1,000-table strict-timeout run**, all 1,242,249 all-player decisions were below 1ms (maximum 731.9µs), with zero timeout/exception fallbacks. These are measured samples, not a guarantee for future or official hardware. See `results/final_latency_linux.json`, `results/final_latency_windows.json`, and `results/final_timeout_enforced.json`.

Final Windows CPython 3.11 replay made **149,020 calls**: p99 75.2µs preflop / 236.5µs postflop / 232.3µs ICM, **17 wall-clock overruns**, maximum 102.6ms; zero illegal actions or internal fallbacks. The Windows results reinforce that sub-millisecond typical computation does not imply a hard wall-clock deadline guarantee.

After tournament evaluation, the only runtime change moved the ICM import from first use to module initialization, keeping startup work outside decisions. The evaluated source is preserved in `tools/evaluated_policy.py`; `results/final_equivalence.json` confirms identical import-stripped syntax trees and identical actions on all 14,702 captured observations. Assets and strategy logic are unchanged. A standalone first-ever ICM decision, without importing the simulator/benchmark beforehand, took 240.1µs locally after this fix (`results/cold_icm.json`).

The test suite includes exhaustive standard five-card category frequencies (optional), independent best-five comparisons, native/Python feature checks (optional compiler required), 10,000 randomized engine hands, 10,000 collected legal candidate actions, side-pot and reopening regressions, exact ICM against independent permutations, shared-nut boards, and malformed-observation fallback.

```powershell
$env:TEST_EXHAUSTIVE='1'; python -m unittest discover -s tests -p test_cards.py
$env:VERIFY_CPP='1'; python -m unittest discover -s tests -p test_features.py
python tools/build_equity.py --samples 500000 --seed 9173248901
python tools/build_matchups.py --samples 10000 --seed 550109173
python tools/solve_pushfold.py --iterations 500
```

Asset generation needs GCC/G++ only offline. Forest generation/training additionally needs NumPy and scikit-learn; scripts `tools/generate_equity.cpp`, `tools/train_equity.py`, and the bounded Slurm scripts document the settings. `results/training_requirements.txt` pins the training packages. Run these on your own compute allocation, never heavy work on a login node. `tools/tune.py 600` and `tools/refine.py 1500` use preserved development policy behavior and archived models; `python tools/verify_reproduction.py` checks their replay against saved outcomes. `tools/final_evaluation.py` records the frozen final evaluation.

Public research informed the choice of offline abstraction and mixed strategies, rather than an attempted millisecond full-game solve: Brown and Sandholm, [Superhuman AI for multiplayer poker](https://doi.org/10.1126/science.aay2400), and [Libratus: The Superhuman AI for No-Limit Poker](https://ijcai.org/proceedings/2017/772). All strategy/evaluator/simulator code and sampled assets here were developed independently. No other bot workspace or private poker code was inspected.

## Limitations

No official interface compatibility is claimed. Unknown payouts and blind levels prevent contest-specific tournament optimization. No complete-game exploitability bound exists. Range inference, implied odds, postflop multiway equity, and short-stack abstractions are approximate. Training and synthetic opponents do not model every adversarial strategy. Startup costs and timing on official hardware must be checked by the collector. The bundle contains the best tested local candidate; further adaptation to the actual starter kit remains necessary.
