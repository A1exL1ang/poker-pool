# Independent local poker laboratory

This environment was written specifically for bot_13 from the stated Hold'em
brief. It is **not** the contest starter kit, and its Python contract is not a
claim of official interface compatibility. `RULES.txt` does not supply official
later blind levels, an Agent schema, or placement points. Callers supply those
settings. Named `gentle`, `doubling`, and `fixed` schedules in `benchmark.py`
are local sensitivity scenarios; `fixed` deliberately tests a non-escalating
control and does not model the rules' increasing blinds.

## Observation and action contract

`policy.act(observation)` returns a mapping with `action` equal to `fold`,
`check`, `call`, or `raise`. A raise has an integer `amount` representing the
**new total street bet**, including chips already posted. For an opening bet,
use `raise` with the new bet amount. `call` is legal only when facing a bet;
`check` is legal only when the call cost is zero.

Cards are integers `4 * rank + suit`, with ranks 0..12 meaning 2..A and suits
0..3 meaning clubs, diamonds, hearts, spades. The observation contains only
hero's private cards, public board cards, public actions, and public chip
counts. `HandResult.holes` is a laboratory-only record and is never passed to
policies. There is no automatic showdown data transfer to the next hand.

`pot` is the sum of **all commitments**, including the current street's bets.
`players[].stack` excludes committed chips. `players[].bet` is the current
street total; `committed` is the entire hand's total. `to_call` is capped by
hero's stack. `min_raise_to` reflects the last full raise increment, and
`max_raise_to` is hero's street bet plus remaining stack. A short all-in raise
is allowed at `max_raise_to` even when this is below `min_raise_to`.

`button` is an original seat index and can name an eliminated seat under the
dead-button rule. `small_blind_seat` is `None` for a dead SB;
`big_blind_seat` always names a funded seat. `seat_count` and the original
player list remain fixed, retaining eliminated seats with zero chips. All
seat order computations are circular over original indices.

Each history event supplies `seat`, `street`, `action`, new street total
`amount`, `amount_paid`, `pot_before`, `to_call_before`, `stack_before`,
`current_bet_before`, `is_all_in`, `full_raise`, and `can_raise_before`.
`paid` and `to_call` are aliases retained for local consumers. Forced blind
posts and uncalled refunds are not decision events.

## Rule conventions and tests

The local model has no antes, rake, rebuys, straddles, or table changes. It
uses one standard 52-card deck, two deal rounds, and a burn before each board
street. The button posts the SB and acts first preflop and last postflop in
heads-up play. In multiway play, BB progression is authoritative: each next
BB is the next surviving seat after the prior BB. A busted prior BB creates
a dead SB; the button advances through the nominal SB position. Transition
to heads-up adjusts the button so no survivor posts consecutive BBs.

Minimum raises use the last **full** bet or raise increment. Short all-ins
can cumulatively reopen action, tested separately for each player's last
action level. A postflop opening all-in of 20 with minimum bet 100 permits
a new raiser to raise to at least 120; completing to 100 is not a no-limit
operation. A player who checked before that short opening bet has not yet
faced a full reopening bet. A short BB still requires a nominal full BB call
while multiple funded players can bet; when every opponent is all-in, the
sole funded player only needs to match the actual opposing wager.

These choices follow the [2026 Poker TDA rules, especially rules 21, 34,
36, 45, and 49](https://www.pokertda.com/view-poker-tda-rules/) and the TDA
administrator's [Robert's Rules cross-check for a subminimum opening
all-in](https://www.pokertda.com/forum/index.php?topic=1456.0). Contest-specific
engine behavior still requires confirmation against the eventual starter kit.

Uncalled bets return at street completion. Folded contributions remain in
pots. At showdown, pots are separated by commitment level and only eligible
nonfolded contributors can win each pot. Tied pots split independently, with
odd chips awarded clockwise left of the button. Chip conservation is asserted
throughout. Simultaneous busts are ordered by start-of-hand stacks; equal
start stacks split the affected placement points equally. That payout tie
convention is an explicit local assumption because the supplied contest rules
do not specify it. At the hand cap, exact subset-DP ICM allocates only the
caller-supplied remaining top-rank payout values to survivors.

Run the rule checks with:

```powershell
python -m unittest tests.test_simulator -v
```

The suite includes ordering, privacy, raise-to semantics, individual and
cumulative reopening, short opening bets, short blinds, dead buttons/SBs,
heads-up transitions, side pots, uncalled refunds, odd chips, all-in runouts,
ICM, invalid/exception/deadline fallbacks, and random hand conservation.

## Benchmarks

```powershell
python -m lab.benchmark --candidate bot13.agent:Agent --tables 64 --seed-start 1000 --lineup mixed --blind-profile gentle --payouts '[50,30,20,0,0,0,0,0]' --output measurements/tuning.json
```

The payout vector in that command is illustrative, not official. Omit
`--payouts` to report chips only. Use `--blind-schedule` for explicit JSON
levels, `--opponents` for seven specified styles, `--config` for a candidate
configuration, and `--reference-config` with `reference` opponents to compare
another configuration of the same local bot. `--observations file.jsonl.gz`
saves only candidate observations and returned decisions for replay/timing.
Candidate seats cycle through all eight positions; each table has a distinct
seed. Separate tuning and held-out seed ranges externally.

Timing covers only the policy call. Import, construction, observation creation,
and engine work are excluded. `--enforce-deadline` replaces a decision over
1 ms with check when free and fold otherwise. Exceptions and invalid decisions
use the same fallback. Without that flag, overruns are measured but actions
remain in force. Runtime scheduling noise is included and must be reported;
no local result guarantees the target sandbox's 1 ms behavior.

Baselines are original heuristic opponents (`tight`, `station`, `aggressive`,
`random`, `checkfold`) plus a `equity` reference that samples uniform opposing
hands and does not infer ranges. They provide diversity and exploitable
styles. Good performance against them does not establish contest strength or
game-theoretic optimality. Output intervals are unadjusted normal 95%
intervals across seeded tables, and small batches have substantial variance.

Additional original stress styles broaden adversarial coverage:

- `shovewide` open-shoves approximately the top 60% of combinations in the
  bot's own offline preflop equity ordering, then calls raises much tighter.
- `premiumnit` voluntarily enters only QQ+/AK and calls very large deep
  preflop bets only with KK+. It tests overcalling against rare strong ranges.
- `minpressure` uses wide positional minimum raises, small continuation bets,
  blocker bluffs, and private value. It tests overfolding to cheap aggression.

`--lineup stress` mixes these with the uniform-equity reference. Each style
also has its own homogeneous lineup. These are targeted test opponents with
intentional weaknesses, not competing deliverables or externally copied bots.
