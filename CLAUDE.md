# Email Labeler

One system, two independent functions sharing a daemon and chassis:

1. **Email triage** — polls Gmail, classifies each thread with a two-tier LLM
   pipeline, applies inbox/archive labels.
2. **Newsletter grading** — grades ministry-newsletter stories for writing
   quality and thematic alignment, appending assessments to a durable JSONL.

Both functions normally run in one daemon process — hybrid operation is the
default deployment. Newsletter grading is enabled by the presence of
`[newsletter]` in config.toml, **not** by `NEWSLETTER_ONLY`, which is a filter
that skips non-newsletter threads (for running the newsletter function alone).
A symmetric `INBOX_ONLY` filter (email triage alone) would be legitimate — the
asymmetry is an accident of history, not a decision (registry D1).
Neither function depends on the other, and design values are scoped
per-function. The failure model (D5) scopes faults per-function as well: a
provider-balance halt stops the affected function and leaves the other running
(registry D5/D19).

Non-goals: this is a single-owner deployment, not a generic multi-user
product; org-specific content (the Ends Statement themes, the newsletter
recipient) is configuration, not something to abstract away.

## Documentation

- `README.md` — Human-oriented overview: privacy model, architecture, setup instructions, running commands
- `README-technical.md` — Agent/reference: project structure, config key reference, environment variables, test coverage
- `evals/README.md` — Human-oriented eval suite guide: pipeline stages, common workflows, key commands
- `evals/README-technical.md` — Agent/reference: complete CLI flags for all eval tools, LLM cache internals, chain-of-thought capture format
- `docs/decisions.md` — Decisions registry: adjudicated tradeoffs reviews must not re-litigate (see Review Charter)
- `docs/labeling-rubric.md` — Labeling rubric: what each label means, golden-set labeling and exclusion procedure, decided vs proposed rules; the prompts in `config.toml` implement it (D21)
- `docs/plans/` — Frozen history: every file carries a status header; superseded plans are records, not instructions
- `docs/runbook-agent-attempted-recovery.md` — Owner-run manual sweep of threads dropped to `agent/attempted` by the issue-#64 bug; time-sensitive (cleanest while the daemon is stopped, before fresh give-ups mix into the label), and not to be executed by an agent with Gmail write access
- `scripts/migrate_assessments.py` — Convert pre-#53 records in an assessments JSONL (list themes → graded dicts, 1-5 scores → Poor/OK/Good) so the review TUI can open a file with old history; tiers are preserved verbatim. Dry-run by default, `--in-place` applies (keeps `.bak`)

Authority map (decision D7): every fact has one home. CLAUDE.md holds
principles (no literal values). config.toml is authoritative for operational
values and their rationale — its comments are the design record for sizing.
README-technical.md is the operational reference (env vars, config key
meanings, procedures, test coverage). README.md is the human overview.
docs/decisions.md is the decisions registry. docs/plans/ is frozen history
(status header on every file). When editing docs: point, don't restate.

## Privacy Posture (email-triage function)

Bodies of threads **classified as person** are processed only by the local
LLM; the cloud LLM sees metadata (sender, subject, snippet) for Stage 1
routing. This is best-effort routing, stated honestly (decision D2): Stage 1
can misclassify, and unparseable Stage 1 output deliberately defaults to
SERVICE. The residual misroute risk is measured — the eval suite reports a
privacy-violation rate — not denied. Do not restate this guarantee in
absolute ("never") form, and do not "fix" the SERVICE default to PERSON.

Ownership rule (D3): a thread with any message To/Cc-addressed to the
configured newsletter recipient belongs to the newsletter function and is
organizational content — its full transcript, including person-written
replies, goes to the cloud by design, without person/service routing.

Public-API stand-ins for the local endpoint (`MLX_API_KEY`, e.g. Novita.ai)
are non-production/eval-only (D4): pointing the local tier at a public API
sends person bodies off-network.

## Failure Model (decision D5)

**Rule 1 — Outcomes only come from successes.** A committed outcome (label,
archive, grade record, `agent/processed`) is only ever produced by a
successful classification. Failures never commit anything: they defer, set
aside findably (`agent/attempted`), or stop the affected function loudly.

**Rule 2 — Blame by correlation.** When processing fails, ask whether sibling
threads are succeeding. Others succeed → the thread is the problem: bounded
strikes, then set aside under `agent/attempted` (findable, never silent).
Others failing the same way → the thread is innocent (provider, proxy, disk,
or our own config/code): no strikes for anyone, get loud, keep the backlog.

**Scope — functions fail independently.** A fault that disables one function
(e.g. its LLM provider's balance) stops that function loudly; the other
function continues.

The registry entry D5 lists the corollaries — all implemented (Wave 2), each
naming its commit. The code obeys this model; a deviation from it is a bug,
not a documented gap. Current-behavior descriptions live in README.md
(Resilience) and README-technical (Health Checking, write-before-label) and
stay accurate to the code, not the model.

## Architecture

```
Poll loop → find unprocessed emails
  → Stage 1: sender+subject+snippet → Cloud LLM → person or service?
  → Stage 2a (service): full body → Cloud LLM → classify
  → Stage 2b (person): full body → Local MLX → classify
  → Apply label + action via api-proxy → Gmail
```

Per-cycle processing is bounded by the cloud/local/fetch/write semaphores —
sizing rationale lives in config.toml `[daemon]` comments (authoritative).
`local_parallel` deliberately defaults to serial; see README-technical
"Local Model Serving & Memory" and the config.toml `[daemon]` comments
before raising it.

## Newsletter Classification

Whenever `[newsletter]` is configured (it is in the shipped config), threads
To/Cc-matching the recipient are routed to the newsletter pipeline before
person/service routing. `NEWSLETTER_ONLY=1` additionally skips all
non-newsletter threads.

```
Poll loop → find unprocessed newsletters (To/Cc matches config recipient)
  → Extract individual stories from newsletter body (Cloud LLM)
  → Score each story on 4 quality dimensions (simple, concrete, personal, dynamic) as Poor/OK/Good (Cloud LLM)
  → Grade each story against Ends Statement themes as Absent/Present/Emphasized (Cloud LLM)
  → Compute each story's tier from its averaged dimension scores; overall tier = best story's tier (bands: registry D15 / config.toml)
  → Append assessment record to JSONL file  ← before labeling (registry D18)
  → Apply tier + theme labels via api-proxy → Gmail
```

Newsletter grading uses its own `[newsletter.llm]` config — model and
endpoint independent of the email-classification tiers (see config.toml and
README-technical's environment variables).

### Newsletter Review TUI

```bash
python -m newsletter_review
```

Browses the assessments JSONL. Usage, filters, and hotkeys: README.md
(guarded by test_tui_docs).

### Newsletter evaluation

Distinct from the read-only `newsletter_review/` package (which only *browses*
production assessments), the newsletter eval harness under `evals/` *measures* the
grading pipeline against hand-labeled ground truth so the prompts can be iterated.
It mirrors the email eval's 4 stages with `newsletter_`-prefixed modules:
`newsletter_harvest → newsletter_label → newsletter_run → newsletter_report`
(schemas in `evals/newsletter_schemas.py`). Quality/theme scoring uses **fixed
golden stories** so it is decoupled from extraction variability; the shared cache
(`evals/cache/llm_cache.jsonl`) and each run's `prompt_hash` make prompt A/Bs
cheap and self-identifying. See `evals/README.md` (workflows) and
`evals/README-technical.md` (every CLI flag, cache/prompt_hash details).

## Review Charter

Before flagging any finding, reviewers (human or agent) check
`docs/decisions.md`. A behavior or tradeoff recorded there is a decision, not
a finding — do not re-litigate it. A finding that contradicts a registry entry
is valid only if it names the entry and argues the reversal explicitly. A
change that reverses a registry decision must update that entry in the same
change.

## Package Management

Uses `uv` for dependency management. No pip.

```bash
uv sync --extra dev          # Install all deps including dev
uv run --extra dev pytest    # Run tests
uv run --extra dev ruff check .  # Lint
```

## Testing

**Always build using red/green TDD.** Every behavior change — new feature, bug fix,
or refactor — starts with a failing test:

1. **RED** — Write one minimal test for the new behavior. Run it and *watch it fail
   for the expected reason* (behavior missing, not a typo/import error). A test you
   never saw fail proves nothing.
2. **GREEN** — Write the minimal production code to make it pass. Run the test; confirm
   it passes and the rest of the suite stays green.
3. **REFACTOR** — Clean up with the tests green. Add no behavior without a new failing
   test first.

**No production code without a failing test first.** If you wrote code before the test,
delete it and start over from the test.

**Proving after-the-fact tests with mutation.** When a test is written *after* the code
(e.g. covering a fix already applied, or legacy code), it can't be red/green-verified —
so prove it instead: deliberately break the production behavior it targets (a "mutation")
and confirm the test fails, then revert. A test that stays green when you break the code
under it is not testing that code. Prefer the mutation that reproduces the exact bug the
test guards against.

**Always run the full test suite before declaring any task complete.**

```bash
uv run --extra dev pytest tests/ -v
```

All tests use mocks — no external services needed. Per-module test coverage:
README-technical.md "Test Coverage by Module" (authoritative).

## Where Things Live

- Project structure and per-module test coverage: README-technical.md
  (authoritative).
- Environment variables: README-technical.md (authoritative, test-guarded by
  test_env_var_docs).
- Label names and inbox/archive actions: config.toml `[labels]`,
  `[labels.actions]`, `[newsletter.labels]` (authoritative). All labels must
  be pre-created in Gmail; the daemon verifies at startup and exits if any are
  missing. Markers: `agent/processed` (success), `agent/attempted` (give-up,
  findable), both excluded from `gmail_query`.
- TUI usage and hotkeys: README.md and evals/README.md (guarded by
  test_tui_docs). Shared TUI conventions: README-technical.md "TUI
  Conventions" + `tui_common.py`.
- Operational values and their rationale: config.toml comments.
- Adjudicated tradeoffs: docs/decisions.md.
