# Similarity across the 20 frozen bots

Bots 5 and 15 are the clearest behavioral pair when both action choice and bet size matter. Bots 3 and 11 agree slightly more often on action type, but their raise sizes differ substantially. These are similarities, not evidence of copying or interchangeable policies.

## Behavioral comparison

The shared corpus contains 768 legal observations from 508 table/hand clusters, spanning streets, positions, stack depths, pressure and payout scenarios. Each bot received three policy seeds and two repeats per seed: 92,160 decisions total. There were zero exceptions or illegal actions and seven timing overruns. Agents were initialized afresh per observation with current-hand history, so this does not measure accumulated opponent adaptation.

| Pair | Action agreement (95% CI) | Chance-corrected kappa | Agreement on nontrivial states | Similar raise size when both raise |
|---|---:|---:|---:|---:|
| 3 / 11 | 90.32% (88.36–92.28%) | 0.861 | 69.82% | 60.70% |
| 5 / 15 | 90.23% (88.22–92.15%) | 0.858 | 68.58% | 91.67% |
| 14 / 19 | 90.02% | 0.852 | 63.72% | 72.88% |

Similar raise size means amounts differ by at most the greater of 5% of the pot and one big blind. Universal checks/folds inflate raw agreement; nontrivial-state agreement and sizing make the limits visible. Confidence intervals bootstrap table/hand clusters, with 1,000 replicates; they are exploratory and not adjusted for 190 pair comparisons. Same-seed repeats agreed exactly for all bots except bot 8 (99.9566%).

## Implementation comparison

A separate policy-only analysis uses normalized Python AST seven-token shingles. It excludes packaged duplicate copies, generated tables, simulators, schemas, exact ICM helpers and mechanical input/output boilerplate. Constants and attribute names remain, so this is a structural resemblance score, not a percentage of identical code.

| Pair | Policy shingle Jaccard |
|---|---:|
| 7 / 8 | 0.1619 |
| 4 / 8 | 0.1615 |
| 2 / 8 | 0.1576 |
| 4 / 7 | 0.1530 |
| 1 / 3 | 0.1503 |

The strongest structural pairs differ from the strongest behavioral pairs. Bots 7 and 8 share risk-adjustment and action-dispatch structure; bots 4 and 8 share action dispatch, opponent observation and preflop structure. Their leading normalized function matches are only partial; the top pairs have no exact normalized function matches. Similar standard poker logic can emerge independently.

This analysis deliberately covers selected policy functions, not complete programs. Separately implemented equity/model modules can hide substantive differences. The earlier source filter admitted simulator and generated ICM code; its rankings were discarded and replaced by the explicit policy allowlist in `similarity_source_v2.py`. Behavioral measurements were unaffected.

## Reproducible evidence

- [Corrected source analysis](similarity_source_v2.json), including exact selected files, hashes, excluded functions and strongest function matches.
- [Behavior analysis](similarity_behavior.json), including all pairs, strata and uncertainty.
- [Raw corpus and replay archive](similarity_raw_results.tar.gz).
- [Frozen bot file manifest](frozen_bots20.json).

Run source analysis with `PYTHONPATH=evaluation python evaluation/similarity_source_v2.py --output evaluation/similarity_source_v2.json`. Use the corpus/replay and behavioral-analysis commands in `similarity20.py` for replay. All substantial analysis ran on new task-owned Engaging allocations.
