"""Run the whole TUI e2e regression harness.

    python .claude/skills/tui-regression/run_all.py [--emit DIR]

Writes the synthetic data files (optionally to --emit DIR, else a temp dir) so
the real ``load_*`` paths are exercised, then drives all five TUIs via Pilot
(none of them calls a model). Prints a combined PASS/FAIL report and exits non-zero if
any scenario fails — suitable for CI or a pre-merge gate.
"""

import argparse
import asyncio
import sys
import tempfile

# _e2e puts repo root + this dir on sys.path.
import _e2e  # noqa: F401
import drive_edit_tui
import drive_newsletter_grade
import drive_newsletter_label
import drive_newsletter_review
import drive_review
import synth_data
from _e2e import report, run_scenarios

DRIVERS = [
    ("newsletter_label", drive_newsletter_label),
    ("newsletter_grade", drive_newsletter_grade),
    ("review", drive_review),
    ("edit_tui", drive_edit_tui),
    ("newsletter_review", drive_newsletter_review),
]


async def _run() -> int:
    rc = 0
    grand_pass = grand_total = 0
    for name, mod in DRIVERS:
        results = await run_scenarios(mod.scenarios())
        rc |= report(name, results)
        grand_pass += sum(1 for _, ok, _ in results if ok)
        grand_total += len(results)
    print(f"\n########## TOTAL: {grand_pass}/{grand_total} scenarios passed ##########")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="TUI e2e regression harness")
    ap.add_argument("--emit", metavar="DIR",
                    help="Write the synthetic data JSONL files to DIR (default: a temp dir)")
    args = ap.parse_args()

    target = args.emit or tempfile.mkdtemp(prefix="tui_e2e_")
    paths = synth_data.write_all(target)
    print("Synthetic data written (loads through the real load_* paths):")
    print(f"  {len(synth_data.newsletters())} newsletters -> {paths['newsletters']}")
    print(f"  {len(synth_data.threads())} threads      -> {paths['threads']}")
    print(f"  {len(synth_data.assessment_records())} assessments  -> {paths['assessments']}")

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
