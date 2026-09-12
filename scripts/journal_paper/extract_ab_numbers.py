"""Extract verified A/B numbers (baseline vs MutationFallback) from run archives.

Walks AutoRed-Final/results/benchmark/<victim>/ for merged_summary.json files
and prints one line per run: victim, dir, n, success_rate, fallback counters.
Read-only; no mutation of any artifact.
"""
import json
import os
import sys

ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "AutoRed-Final", "results", "benchmark",
)


def find_merged(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "merged_summary.json" in filenames:
            yield os.path.join(dirpath, "merged_summary.json")


def main():
    rows = []
    for victim in sorted(os.listdir(ROOT)):
        vroot = os.path.join(ROOT, victim)
        if not os.path.isdir(vroot):
            continue
        for p in find_merged(vroot):
            with open(p) as f:
                d = json.load(f)
            md = d.get("metadata", {})
            diag = d.get("mutation_fallback_diagnostics") or {}
            rows.append(
                {
                    "victim": victim.replace("--", "/"),
                    "dir": os.path.basename(os.path.dirname(os.path.dirname(p))),
                    "n": md.get("n_rounds"),
                    "sr": d.get("success_rate"),
                    "avg_att": d.get("avg_attempts_on_success"),
                    "verified": d.get("verified_success"),
                    "fb_trig": d.get("mutation_fallback_triggered", 0),
                    "fb_win": d.get("mutation_fallback_successes", 0),
                    "coop": md.get("cooperative_seeding"),
                    "coop_n": md.get("cooperative_n"),
                    "fb_rounds": md.get("max_fallback_rounds"),
                    "noop_rate": diag.get("no_op_rate"),
                    "mutators": diag.get("mutator_counts"),
                    "winners": diag.get("winning_mutator_counts"),
                }
            )
    for r in rows:
        sr = f"{r['sr']:.4f}" if r["sr"] is not None else "None"
        print(
            f"{r['victim']:44s} | n={r['n']} | sr={sr} | avg_att={r['avg_att']} "
            f"| verified={r['verified']} | fb_trig={r['fb_trig']} | fb_win={r['fb_win']}"
        )
        if r["mutators"]:
            print(f"    mutators={r['mutators']}")
        if r["winners"]:
            print(f"    winners={r['winners']} noop_rate={r['noop_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
