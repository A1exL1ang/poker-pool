"""Build a concise report only when final benchmark hashes match the runtime."""
from datetime import datetime, timezone
import json
from pathlib import Path
from .evidence import ROOT, fingerprint


def main():
    directory = ROOT / "benchmarks"
    names = ["mixed_final", "equity_final", "self_final", "timeouts_final", "payouts_final"]
    reports = [json.loads((directory / (name + ".json")).read_text()) for name in names]
    latency = json.loads((directory / "latency_final.json").read_text())
    current = fingerprint()
    for report in reports + [latency]:
        assert report["runtime_fingerprint"] == current, "Rerun final benchmarks after runtime changes"
        assert report["agent_errors"] == 0
    audit = json.loads((directory / "equity_audit.json").read_text())
    exhaustive = json.loads((directory / "exhaustive_evaluator.json").read_text())
    assert exhaustive["passed"]
    wall = latency["direct_call_ns"]
    boundary = latency["dict_boundary_ns"]
    lines = ["# Bot 7 validation", "", f"Generated {datetime.now(timezone.utc).isoformat()}.", "",
             f"All final tournament and latency reports match runtime SHA-256 `{current['sha256']}`. No official environment or competitors were available.", "",
             "23 unit tests passed. The independent evaluator audit checked all 2,598,960 five-card category counts; 7,500 random five-, six-, and seven-card hands matched a separate slow best-five evaluator, including tie breakers. Exact ICM was checked against permutations, independent eight-player subset DP, symmetry, and utility conservation. Engine regressions cover side pots, short all-ins, reopening, heads-up blinds, and a short-big-blind phantom-call edge case.", "",
             f"On {latency['platform']}, Python {latency['python']}, {latency['iterations']:,} direct calls over {latency['corpus_states']:,} recorded/stress states measured:", "",
             "| Boundary | Median | p95 | p99 | Maximum |", "|---|---:|---:|---:|---:|"]
    for label, timing in (("Agent.act", wall), ("Dictionary conversion + act", boundary)):
        lines.append(f"| {label} | {timing['median']/1e6:.3f} ms | {timing['p95']/1e6:.3f} ms | {timing['p99']/1e6:.3f} ms | {timing['max']/1e6:.3f} ms |")
    stress_over = sum(s["over_1ms"] for s in latency["stress"])
    stress_n = sum(s["ns"]["count"] for s in latency["stress"])
    lines += ["", f"**The strict 1 ms wall-clock requirement is not guaranteed:** {latency['over_1ms']:,}/{latency['iterations']:,} direct calls exceeded it ({latency['over_1ms']/latency['iterations']:.3%}); the separate adversarial/stress repetitions had {stress_over:,}/{stress_n:,} overruns. The 450 microsecond soft budget reduces sampling work but cannot prevent scheduler/GC delays or bound the official adapter. All raw outliers are retained. Aggregate thread CPU usage, including harness overhead, averaged {latency['aggregate_thread_cpu_ns_per_call']/1e6:.3f} ms/call. Per-call CPU percentiles are not reported because Windows accounting was too coarse.", "",
              "Timing excludes cold import. A separate local cold-import observation was approximately 0.261 s; assets must load before timed decisions if the official environment permits that lifecycle.", "",
              "All tournaments below are local synthetic tests: eight initial seats, 2,000 chips, at most 200 hands, and big blinds 20/30/50/80/120/200/320/500, changing every 25 hands. No ante. Finishing tie convention uses chips at hand start. These are test choices, not the missing official progression or points.", "",
              "Win-equity is final chip fraction at the cap, equivalently winner-take-all ICM utility; it is not the observed frequency of outright victories. The linear-point column uses hypothetical payouts 8/7/6/5/4/3/2/1. A symmetric reference is 12.5% win-equity and 4.5 linear points. SE is the standard error across seeded tables, not a bound on performance against unknown competition bots.", "",
              "| Fixture | Tables | Mean win-equity ± SE | Hypothetical linear points ± SE | Hero errors |", "|---|---:|---:|---:|---:|"]
    labels = ["Mixed simple opponents", "Equity-based simple opponent", "Identical-policy clones", "Mixed, timeout fallback enforced", "Mixed, hypothetical 50/30/20 payout input"]
    for name, label, report in zip(names, labels, reports):
        lines.append(f"| [{label}]({name}.json) | {report['configuration']['tables']} | {report['mean_win_equity']:.3%} ± {report['win_equity_standard_error']:.3%} | {report['mean_hypothetical_linear_points']:.3f} ± {report['linear_points_standard_error']:.3f} | {report['agent_errors']} |")
    lines += ["", f"The final runs total {sum(r['configuration']['tables'] for r in reports):,} tables, {sum(r['hands_played'] for r in reports):,} hands, and {sum(r['hero_decision_ns']['count'] for r in reports):,} timed hero actions. Chip conservation and action legality are asserted throughout. The first three strategy experiments disable the soft clock budget for deterministic seeded decisions. The timeout and payout-input runs use production defaults, so timings can change their trajectories. Timeout fallback applies to every seat. These opponent fixtures are weak and share utility/evaluation helpers with the test harness; they are not independent contest-strength rivals.", "",
              f"River sampling was compared with exact enumeration of the declared heuristic ranges on {audit['cases']} states and {audit['estimates_per_case']} seeds per state: signed bias {audit['mean_signed_error']:+.5f}, single-decision RMSE {audit['pooled_single_decision_rmse']:.5f}, and mean absolute error of the 80-estimate averages {audit['mean_absolute_error_of_80_trial_means']:.5f}. This measures sampling error, not correctness of opponent range assumptions. See [equity_audit.json](equity_audit.json).", "",
              "The expanded tournament validation found a local simulator error when a short big blind left only one player able to act: it could demand a nonexistent full-blind completion above every opposing all-in. This was fixed, directly regression-tested, and excluded from final results by rerunning the final benchmarks. Older exploratory files are retained but are not the final evidence.", "",
              "The local package is generated and isolated-smoke-tested by `python -m tools.package`; its size and hashes are in `../build/manifest.json`. No external compute jobs, submissions, network gameplay, commits, or publication were performed. The unknown official interface, payout/blind schedule, startup accounting, and target-hardware latency remain integration limitations. See [README](../README.md) and [interface contract](../INTERFACE.md).", ""]
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote benchmarks/RESULTS.md; all final hashes match {current['sha256']}")


if __name__ == "__main__":
    main()
