# Reviewer guide

For anyone, human or agent, reviewing a change to this repo. It points at the
documents that hold the facts (decision D7: every fact has one home) and adds
what a reviewer needs that those documents do not say: what a change here can
break, what counts as an interface, and what reviews of this repo tend to miss.

## Before flagging anything

1. Read the **Review Charter** in `CLAUDE.md`, then `docs/decisions.md`. A
   behaviour recorded in the registry is a decision, not a finding. A finding
   that contradicts an entry has to name the entry and argue the reversal.
2. Read the **Failure Model** and **Privacy Posture** sections of `CLAUDE.md`.
   Most serious findings in this repo are a departure from one of the two.

## What a change here can break

| Area | What goes wrong | Where the rule lives |
|---|---|---|
| Gmail writes | A label or archive applied to real mail from a failed or partial classification | D5 Rule 1; `labeler.py`; `tests/test_daemon.py` |
| Privacy routing | A person-classified body sent to the cloud tier | D2, D3, D4; `tests/test_privacy.py` |
| Blame and give-up | An innocent thread struck out to `agent/attempted`, or a provider fault left quiet | D5 Rule 2, D19, D22 |
| Function independence | A newsletter fault stopping email triage, or the reverse | D1, D5 scope |
| Newsletter grades | A grade whose meaning drifts from the rubric, or a record the review TUI cannot read | D23 and `docs/newsletter-rubric.md`; D12, D18 |
| Eval validity | A comparison that looks like a result but is not one (see below) | D21, D23; `evals/README-technical.md` |
| Real content in git | Email or newsletter text, names or ids in a tracked file or fixture | `.gitignore`; fixtures are written from scratch with synthetic values |

## What counts as an interface

A change to any of these needs its documentation and its guard test updated in
the same change:

- **`config.toml` keys and prompt sections**: real-config tests in
  `tests/test_daemon.py`. Comments there are the design record for sizing (D7).
- **Environment variables**: `tests/test_env_var_docs.py`,
  `tests/test_env_example_docs.py`, the table in `README-technical.md`.
- **Eval CLI flags and modules**: `tests/test_eval_cli_docs.py`,
  `tests/test_eval_newsletter_cli_docs.py`, `tests/test_newsletter_eval_docs.py`.
- **TUI hotkeys and usage**: `tests/test_tui_docs.py`.
- **The assessments JSONL schema** (`schema_version`, D12) and the golden-set
  JSONL schemas in `evals/schemas.py` and `evals/newsletter_schemas.py`.
- **Gmail label names**: the daemon verifies them at startup and exits if one
  is missing, so a renamed label is a deployment step, not only a code change.

## What reviews here tend to miss

- **A prompt edit is a behaviour change.** It alters grades or labels on real
  mail and changes `prompt_hash`, so eval runs from before and after stop being
  prompt-comparable. Check it against the rubric it implements (D21, D23).
- **The tests mock the external services** (`CLAUDE.md`, Testing). Green tests
  say nothing about a live provider: request shapes, error bodies and timeouts
  are assumptions until probed.
- **Tests that share the implementation's assumption.** Ask whether a new test
  was written from the caller's shape and whether it was seen to fail.
- **Eval comparisons that cannot carry the claim.** Check the size of the
  golden set against the difference being claimed, whether the two runs differ
  in more than one thing, and whether the response cache could have served one
  run's answers to the other (its key includes the model but not the endpoint).
- **Restated values.** A number or a rule copied into prose instead of pointed
  at will drift; the authority map in `CLAUDE.md` says where each fact lives.
- **Absolute wording in docs** ("never", "always", "only") without a test that
  demonstrates it. The privacy guarantee in particular is stated as best-effort
  on purpose (D2).
- **Frozen history edited as if live.** Files under `docs/plans/` carry a status
  header and are records, not instructions (D8).

## Commands

```bash
uv sync --extra dev
uv run --extra dev pytest tests/ -v
uv run --extra dev ruff check .
```

CI runs the last two (`.github/workflows/ci.yml`).
