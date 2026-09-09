# Evaluation Suite

A 4-stage pipeline for measuring classification accuracy against human-reviewed threads.

```
harvest → review → run_eval → report
```

> For complete CLI flag references, cache internals, and chain-of-thought capture details, see [README-technical.md](README-technical.md).

## Setup

The eval tools run outside Docker and need access to the same environment variables as the daemon. If you haven't already, symlink to `agent-stack/.env`:

```bash
ln -s ../agent-stack/.env .env
```

Since the symlinked `.env` may contain Docker-internal hostnames (e.g. `PROXY_URL=http://api-proxy:8000`), use `--proxy-url` to point at the proxy's host-accessible address:

```bash
uv run python -m evals.harvest --proxy-url http://localhost:8000 --max-threads 200
```

**Stop the daemon before running evals** if your local MLX server only supports one model at a time. The daemon and eval runner may request different models, causing the server to swap models mid-request and return errors.

## 1. Harvest — Build a golden set from production data

Pulls threads already labeled by the daemon, infers ground truth from their Gmail labels, and appends to JSONL.

Harvest always **appends** to the output file, deduplicating by thread ID. It never overwrites an existing golden set — that file also holds your manual review state (confirmed labels, exclusions, notes). To start fresh, delete the file manually.

```bash
# Harvest up to 200 processed threads (appends new ones, dedupes automatically)
uv run python -m evals.harvest --proxy-url http://localhost:8000 --max-threads 200

# Filter by sender type or label
uv run python -m evals.harvest --proxy-url http://localhost:8000 --sender-type person
uv run python -m evals.harvest --proxy-url http://localhost:8000 --label needs_response
```

`--label` takes the config **key** (`needs_response`, `fyi`, `low_priority`) — not the Gmail label name. Harvest translates the key to the actual Gmail label via the `[labels]` section of `config.toml` (e.g. `needs_response` → `agent/needs-response`), so pass `--label needs_response`, not `--label agent/needs-response`.

`--label` is ANDed into the Gmail query, so the fetch targets matching threads
directly rather than filtering the recent processed window after the fact. This
is the way to boost a rare class (e.g. `needs_response`) in the golden set:
harvested threads still carry their inferred labels into `evals.review` for
manual confirmation, so the manual classification step is not bypassed.

**Hand-picking threads.** Sometimes the threads you want are ones the daemon
got wrong — cold sales pitches it filed as `service`/`low_priority`, say — so
no `--sender-type`/`--label` combination will find them. Instead, pick them in
Gmail: create a label (e.g. `eval/harvest`), apply it to threads the daemon
has already processed, then harvest exactly those and correct their labels:

```bash
uv run python -m evals.harvest --proxy-url http://localhost:8000 --gmail-label eval/harvest
uv run python -m evals.review --unreviewed-only
```

The new rows carry the daemon's inferred labels as ground truth and a note
naming the Gmail label, so they are recognizable in `evals.review`. Leave the
Gmail label in place: a re-run skips threads already in the golden set before
fetching them. Flag semantics, and what the harvest can and cannot reach:
[README-technical.md](README-technical.md#harvest).

## 2. Review — Manually verify ground truth labels

Interactive CLI for reviewing and correcting labels in the golden set. Saves atomically after each session. Press `z` at any prompt to undo the last classification — undo works as a stack, walking back through previous decisions.

**Setting an email aside.** Some threads aren't useful as test cases. Two options:

- **Skip** (`k`) — render no judgment. The thread is left unreviewed and resurfaces in a later review session.
- **Exclude** (`e`) — permanently set the thread aside. It is dropped from the review queue and never evaluated. Excluded threads are flagged with an `X` in the `--edit` list view. In `--edit`, `e` is a **toggle**: press it on an included thread to exclude it, or on an excluded one to bring it back.

Hotkeys: sender type `p`/`s` (person/service); label `r`/`f`/`l` (needs_response/fyi/low_priority); `n` notes; `z` undo; `k` skip; `e` exclude; `q` quit.

```bash
# Review all threads (blind mode by default)
uv run python -m evals.review

# Show existing labels while reviewing
uv run python -m evals.review --show-labels

# Textual TUI (evals/edit_tui.py) for editing reviewed threads
# (also where you toggle exclude/un-exclude)
uv run python -m evals.review --edit

# Review only sender classification (stage 1) or label classification (stage 2)
uv run python -m evals.review --stage 1
uv run python -m evals.review --stage 2

# Only review unreviewed threads
uv run python -m evals.review --unreviewed-only

# Drain the unreviewed backlog toward a thin cell in the crosstab, e.g.
# person threads only (--sender-type also combines with --filter-label)
uv run python -m evals.review --unreviewed-only --sender-type person

# Where does the set stand? Composition dashboard (read-only, no TUI):
# total / excluded / pending counts plus a sender × label crosstab of the
# reviewed & unexcluded threads — the set run_eval actually scores.
uv run python -m evals.review --stats
```

Use `--stats` to steer curation: the crosstab makes thin cells (e.g. few
`person`/`needs_response` threads) obvious, so you can target harvesting and
reviewing at the gaps instead of guessing.

## 3. Run — Replay golden set through the classifier

Sends each golden thread through the real `EmailClassifier` with live LLM endpoints. Results are written to timestamped JSONL files in `evals/results/`.

```bash
# Full evaluation (both stages)
uv run python -m evals.run_eval

# Evaluate only Stage 1 (sender classification)
uv run python -m evals.run_eval --stages stage1_only

# Evaluate only Stage 2 (uses expected sender type as input)
uv run python -m evals.run_eval --stages stage2_only

# Override model without editing config.toml
uv run python -m evals.run_eval --local-model mlx-community/Qwen3.6-27B-8bit --tag qwen3.6-27b

# Per-endpoint request overrides, also without touching config.toml:
# a longer timeout for slow prefills, or arbitrary request-body JSON
uv run python -m evals.run_eval --local-timeout 300
uv run python -m evals.run_eval --cloud-extra-body '{"top_p": 0.9}'

# Disable reasoning/thinking for a run (see the A/B workflow below)
uv run python -m evals.run_eval --local-no-think

# Dry run — show what would be evaluated
uv run python -m evals.run_eval --dry-run
```

Results include an LLM response cache so re-runs with the same config are instant. Pass `--no-cache` to bypass it.

## 4. Report — Compute metrics and compare runs

Generates accuracy, confusion matrix, per-class precision/recall/F1, and privacy violation reports.

```bash
# Single run report
uv run python -m evals.report --results evals/results/run.jsonl

# Verbose — show per-thread disagreements
uv run python -m evals.report --results evals/results/run.jsonl --verbose

# Compare two runs side by side
uv run python -m evals.report --compare evals/results/run_a.jsonl evals/results/run_b.jsonl

# Trend view across all runs
uv run python -m evals.report --results-dir evals/results/
```

## Newsletter evaluation

The daemon's newsletter pipeline (active under `NEWSLETTER_ONLY=1`) grades
ministry-newsletter stories: it **extracts** stories from a body, **scores** each
on 4 quality dimensions (simple/concrete/personal/dynamic) as **Poor/OK/Good** → a
**tier** (excellent/good/fair/poor, from the averaged scores), and grades **themes**
(scripture/christlikeness/church/vocation-family/disciple-making) as
**Absent/Present/Emphasized**. This harness measures whether that grading is any
good, so you can **iterate on the prompts** and see the effect of each change.

It mirrors the email eval's 4 stages, with newsletter-specific modules prefixed
`newsletter_` (distinct from the top-level read-only `newsletter_review/` TUI):

```
newsletter_harvest → newsletter_label → newsletter_run → newsletter_report
```

- **harvest** — Pull candidate newsletters from Gmail into
  `evals/newsletter_golden_set.jsonl`. Each is guarded by `is_newsletter(...)` and
  its `body` is built exactly like production input. **No ground truth is inferred** —
  newsletters land unlabeled with an empty story list. Re-runs skip
  already-harvested threads instead of re-fetching them.
- **label** — A Textual tool to build ground truth by hand (quality is
  subjective, so there are no auto-labels). The detail screen shows the
  newsletter **body with each story highlighted in place** plus a story strip
  and a mode bar. Phase A curates the story list by hand (issue #59 removed the
  LLM auto-seeding — it never sped curation up), defining story boundaries in
  two modes — in **browse mode** pick a story with
  `n`/`p` or a number key; in **span mode** (`a` for a new story, `e` to
  re-bound the selected one) mark the first line then the last line to define a
  story from that verbatim body slice. `d` deletes, `C` clears all,
  `c` accepts the list and advances to the next newsletter, `z` undoes, `k`
  skips. Phase B (`l`) assigns per-story dimension scores + themes; the tier is
  auto-derived from scores. `X` excludes a whole newsletter from the queue and
  eval runs (toggle; relaunch with `--include-excluded` to see and restore
  excluded ones). No LLM endpoint is needed.
- **run** — Replay the golden set through the real classifier (LLM-cached) and
  write timestamped results.
- **report** — Compute tier/dimension/theme/extraction metrics, compare two runs,
  or show a trend across runs.

**Environment.** `newsletter_run` calls the `[newsletter.llm]` endpoint resolved
from `NEWSLETTER_LLM_URL` / `NEWSLETTER_LLM_API_KEY`, falling back to
`CLOUD_LLM_URL` / `CLOUD_LLM_API_KEY`; `newsletter_harvest` needs
`PROXY_API_KEY` (and `PROXY_URL` or `--proxy-url`). Both tools fail fast with a
message naming the missing variable (`newsletter_label` needs no LLM). Default data
paths are CWD-relative, so run the tools from the repo root (only `--config`
defaults to the repo-root `config.toml` regardless of CWD).

**Fixed golden stories.** Extraction is inherently variable, so quality and theme
scoring is *decoupled* from it: one golden set holds both. Extraction is scored at
the newsletter-body level (raw body → predicted stories → matched against the
newsletter's golden story list). Quality + themes are scored on the **fixed golden
story text** of each reviewed story, independent of what extraction produced —
so a prompt tweak that only affects scoring is measured cleanly.

```bash
# 1. Harvest a few newsletters (no ground truth yet)
uv run python -m evals.newsletter_harvest --proxy-url http://localhost:8000 --max-threads 50

# 2. Label them: curate stories, then score dimensions + themes
uv run python -m evals.newsletter_label
uv run python -m evals.newsletter_label --unreviewed-only   # only unlabeled

# 3. Run the whole pipeline and print a report
uv run python -m evals.newsletter_run --mode all --tag baseline --report

# 4. Report on a single run, or a trend
uv run python -m evals.newsletter_report --results evals/newsletter_results/<run>.jsonl
uv run python -m evals.newsletter_report --results-dir evals/newsletter_results/
```

**Build a golden set:**

```bash
uv run python -m evals.newsletter_harvest --proxy-url http://localhost:8000
uv run python -m evals.newsletter_label --unreviewed-only
```

**Iterate a prompt variant:**

```bash
uv run python -m evals.newsletter_run --tag baseline
# Edit a [newsletter.prompts.*] block in config.toml (or use --prompts alt.toml), then:
uv run python -m evals.newsletter_run --prompts alt.toml --tag variant
```

The shared cache (`evals/cache/llm_cache.jsonl`) is keyed by prompt content, so
unchanged stages are reused and only the changed prompt is re-run. Each run records
a `prompt_hash`, so variants are self-identifying.

**Compare two runs:**

```bash
uv run python -m evals.newsletter_report \
    --compare evals/newsletter_results/*baseline*.jsonl evals/newsletter_results/*variant*.jsonl --verbose
```

The globs may also match each run's `.cot.jsonl` chain-of-thought sidecar —
`--compare` ignores sidecars (with a stderr note), so this works as long as each
glob matches exactly one real results file. The comparison renders
tier/dimension/theme/extraction deltas and prints each run's `prompt_hash` + tag
+ mode (warning when the modes differ); `--verbose` lists per-story flips
(per-newsletter extraction diffs appear only in single-run `--results` reports).

## Typical Workflows

**Evaluate a new local model (fast path):**

The local model is only used for Stage 2 on person bodies, so `--local-only`
(= `--stages stage2_only --sender-type person`) isolates it — and needs no cloud
credentials. The one-command wrapper sets the model, auto-tags the run by the
model name, preflights the endpoint, prints a report, and optionally compares to
a prior run:

```bash
# Prereq: mlx_lm.server is already serving <hf-id>, and MLX_URL points at it.
python scripts/eval_model.py qwen/qwen3-14b                 # run + report
python scripts/eval_model.py qwen/qwen3-14b qwen3-8b        # ...also compare vs the newest qwen3-8b run
```

Equivalent explicit form (the wrapper just chains these):

```bash
uv run python -m evals.run_eval --local-only --local-model qwen/qwen3-14b --report
```

`--local-only` errors if the local endpoint is unreachable or `MLX_MODEL` doesn't
match the served model (a 404), instead of producing a whole run of per-thread
errors. The check waits up to the local model's request timeout, so a server that
loads the model on demand has time to cold-load it; tune with `--preflight-timeout
SECONDS`, or skip it entirely with `--skip-preflight`.

**Prompt A/B test:**

```bash
uv run python -m evals.run_eval --tag baseline
# Edit prompts in config.toml, then:
uv run python -m evals.run_eval --tag new-prompts
uv run python -m evals.report --compare evals/results/*baseline*.jsonl evals/results/*new-prompts*.jsonl
```

**Thinking on/off A/B (reasoning models):**

The shipped `config.toml` disables the local model's native thinking (issues
#10, #64), so a plain run is already the thinking-off arm. To measure what
thinking ON does to accuracy, re-enable it explicitly for one run and compare:

```bash
uv run python -m evals.run_eval --stages stage2_only --sender-type person --tag think-off
uv run python -m evals.run_eval --stages stage2_only --sender-type person \
    --local-extra-body '{"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": true}}' \
    --tag think-on
uv run python -m evals.report --compare evals/results/*think-off*.jsonl evals/results/*think-on*.jsonl
```

`--local-no-think` / `--cloud-no-think` emit both known disable dialects —
`chat_template_kwargs.enable_thinking=false` (mlx_lm.server / LM Studio) and
`reasoning_effort="none"` (Ollama; the only form its OpenAI-compat endpoint
honors) — and also natively disable thinking on GLM-style cloud models. If
your server expects a different flag, pass it explicitly via
`--local-extra-body` / `--cloud-extra-body`. The extra body is part of the LLM
cache key, so the two runs cache independently — no `--no-cache` needed.

**Model swap:**

```bash
uv run python -m evals.run_eval --tag deepseek-v3
uv run python -m evals.run_eval --config config_v2.toml --tag gpt-4o
uv run python -m evals.report --compare evals/results/*deepseek*.jsonl evals/results/*gpt-4o*.jsonl
```

**Ongoing monitoring:**

```bash
uv run python -m evals.harvest --proxy-url http://localhost:8000
uv run python -m evals.review --unreviewed-only
uv run python -m evals.run_eval --tag weekly
uv run python -m evals.report --results-dir evals/results/
```
