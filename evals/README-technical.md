# Evaluation Suite — Technical Reference

Complete CLI flag references, cache internals, and chain-of-thought capture details for the evaluation suite. For an overview and common workflows, see [README.md](README.md).

## CLI Reference

### harvest

| Flag | Description |
|---|---|
| `--output` | Output JSONL path (default: `evals/golden_set.jsonl`) |
| `--max-threads` | Max threads to fetch (default: `200`) |
| `--sender-type` | Filter: `person` or `service` |
| `--label` | Filter by config **key**: `needs_response`, `fyi`, `low_priority` (not the Gmail label name — the key is mapped via `[labels]` in config.toml, e.g. `needs_response` → `agent/needs-response`). ANDed into the Gmail query so the fetch returns a dense pool of matching threads — useful for boosting a rare class like `needs_response` in the golden set. A key with no `[labels]` mapping is an error |
| `--gmail-label` | Only harvest threads that ALSO carry this Gmail label (e.g. `eval/harvest`). Takes the Gmail label **name** (any user label — not a config key); it must exist in the account (checked before fetching, case-insensitively) and is passed quoted in the query, ANDed with `agent/processed` and `--label`. Only threads the daemon has already labeled (`agent/processed` plus sender-type and classification labels) are found — picks under `agent/attempted` or not yet processed are absent, and every per-thread skip is reported on stderr. Ground truth is still inferred from the daemon's labels; each row's `notes` names the Gmail label, and the labels are corrected in `evals.review` |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--proxy-url` | API proxy URL (overrides `PROXY_URL` env var) |

Harvest always appends to `--output`, deduplicating by thread ID: threads already in the file are skipped before they are fetched and do not count against `--max-threads`. There is no overwrite mode: the golden set also stores manual review state (confirmed labels, exclusions, notes), so harvest never truncates it. To rebuild from scratch, delete the file manually.

### review

| Flag | Description |
|---|---|
| `--golden-set` | Path to golden set JSONL (default: `evals/golden_set.jsonl`) |
| `--show-labels` | Show existing labels (default is blind mode) |
| `--edit` | Textual TUI for editing reviewed threads (auto-saves on each change) |
| `--stage` | Review only stage 1 (sender) or stage 2 (label) |
| `--unreviewed-only` | Show only threads not yet reviewed |
| `--filter-label` | Show only threads with this label |
| `--sender-type` | Show only threads with this expected sender type (`person` or `service`). In `--edit` mode, like `--filter-label`, an explicit filter replaces the reviewed-only default |
| `--start-at` | Start at thread index (0-based) |
| `--stats` | Print a composition summary and exit (read-only, no TUI): total / excluded / unreviewed-pending / reviewed-&-unexcluded counts, plus a sender × label crosstab of the reviewed-&-unexcluded set (what `run_eval` scores) |

Review hotkeys: `p`/`s` sender (person/service); `r`/`f`/`l` label (needs_response/fyi/low_priority); `n` notes; `z` undo; `k` skip; `e` exclude; `q` quit. **Skip** (`k`) leaves the thread unreviewed so it reappears later. **Exclude** (`e`) sets `excluded=True` (also marks reviewed): excluded threads are dropped from the review queue here and from `run_eval` entirely. In the `--edit` TUI detail view, `e` is a symmetric **toggle** (exclude an included thread / un-exclude an excluded one; `reviewed` is left untouched). The `excluded` field is persisted in the golden set JSONL; legacy records using the old `skipped` key are still read as excluded.

### run_eval

| Flag | Description |
|---|---|
| `--golden-set` | Path to golden set JSONL (default: `evals/golden_set.jsonl`) |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--output-dir` | Output directory for results (default: `evals/results/`) |
| `--stages` | `full`, `stage1_only`, or `stage2_only` (default: `full`) |
| `--parallelism` | Concurrent evaluations (default: `cloud_parallel` from config) |
| `--include-unreviewed` | Also evaluate threads not yet reviewed (default: reviewed only) |
| `--dry-run` | Show what would be evaluated without calling LLMs |
| `--tag` | Tag for the results filename (e.g. `new-prompts`) |
| `--no-cache` | Disable LLM response cache (default: cache enabled) |
| `--sender-type` | Only evaluate threads with this expected sender type (`person` or `service`) |
| `--max-threads` | Max threads to evaluate (default: all) |
| `--local-only` | Shortcut for `--stages stage2_only --sender-type person` — evaluate only the local classifier (no cloud creds needed). Errors if combined with a conflicting `--stages`/`--sender-type` |
| `--skip-preflight` | Skip the endpoint reachability check that runs before evaluation |
| `--preflight-timeout` | Seconds to wait for the pre-run endpoint check, overriding both probes (default: each endpoint's own request timeout — generous for a local cold-load, fast-failing for the cloud). Raise it for a server that cold-loads a large model on demand |
| `--report` | Print a metrics report (accuracy, confusion matrix, per-class P/R/F1) for this run after it completes |
| `--compare-to` | After the run, print a side-by-side comparison against this prior results JSONL file |
| `--cloud-model` | Override cloud LLM model name from config |
| `--local-model` | Override local LLM model name from config |
| `--cloud-temperature` | Override cloud LLM temperature (alias: `--cloud-temp`) |
| `--local-temperature` | Override local LLM temperature (alias: `--local-temp`) |
| `--cloud-max-tokens` | Override cloud LLM max tokens from config |
| `--local-max-tokens` | Override local LLM max tokens from config |
| `--cloud-timeout` | Override cloud LLM request timeout in seconds (for slow prefills on long inputs) |
| `--local-timeout` | Override local LLM request timeout in seconds (for slow prefills on long inputs) |
| `--cloud-extra-body` | JSON object merged into every cloud LLM request body (e.g. `'{"top_p": 0.9}'`) |
| `--local-extra-body` | JSON object merged into every local LLM request body (e.g. `'{"chat_template_kwargs": {"enable_thinking": false}}'`) |
| `--cloud-no-think` | Disable thinking for the cloud LLM (sets both `chat_template_kwargs.enable_thinking=false` and `reasoning_effort="none"`; see below) |
| `--local-no-think` | Disable thinking for the local LLM (sets both `chat_template_kwargs.enable_thinking=false` and `reasoning_effort="none"`; see below) |

#### A/B testing thinking on vs. off

The shipped `config.toml` already disables native thinking on the local tier
(issues #10, #64), so a plain run **is** the thinking-off arm. To measure the
accuracy impact of thinking ON (e.g. re-validating the #10 result on a new
backend), re-enable it explicitly for one run and compare:

```bash
# Thinking off — the shipped config; person threads only
uv run python -m evals.run_eval --stages stage2_only --sender-type person --tag think-off

# Thinking on: override both disable dialects that config.toml carries.
# reasoning_effort "low" maps to the model default on Ollama — i.e. thinking
# back on (only "none" disables there); enable_thinking=true re-enables under
# mlx_lm.server / LM Studio.
uv run python -m evals.run_eval --stages stage2_only --sender-type person \
    --local-extra-body '{"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": true}}' \
    --tag think-on

uv run python -m evals.report --compare evals/results/<think-off>.jsonl evals/results/<think-on>.jsonl
```

Because `extra_body` is part of the cache key, the two runs are cached
independently — no `--no-cache` needed. The exact disable payload is
server-specific and **a server silently ignores dialects it doesn't know** —
which is how an A/B can quietly compare think-on to think-on (issue #64:
the flags used to emit only the `chat_template_kwargs` form, which Ollama
ignores). `--local-no-think` / `--cloud-no-think` therefore emit BOTH known
dialects: `chat_template_kwargs.enable_thinking = false` (mlx_lm.server /
LM Studio) and top-level `reasoning_effort = "none"` (Ollama, verified on
0.32.5 — `"low"` is a silent no-op and `think: false` is ignored on its
OpenAI-compat endpoint). If your server expects something else, pass it
explicitly via `--local-extra-body`. Caveat: a strictly-validating provider
(e.g. OpenAI for non-reasoning models, or one that enforces the
low/medium/high enum) may reject the top-level `reasoning_effort = "none"`
with a 400 rather than ignoring it — every row then errors instead of
producing a think-off arm. Against such a backend, skip the flag and pass only
the dialect it accepts via `--extra-body`/`--local-extra-body`.

### report

| Flag | Description |
|---|---|
| `--results` | Path to a single results JSONL file |
| `--compare` | Two result file paths for side-by-side comparison |
| `--results-dir` | Directory of results for trend view |
| `--verbose` | Show per-thread disagreements |
| `--format` | `table` (default) or `json` |

## Newsletter Evaluation CLI Reference

The newsletter eval harness mirrors the email eval's 4-stage shape
(`newsletter_harvest → newsletter_label → newsletter_run → newsletter_report`) but
scores the newsletter grading pipeline (story extraction, per-story quality
scores/tier, and themes). Its modules live under `evals/` prefixed `newsletter_`
to avoid colliding with the top-level read-only `newsletter_review/` package.

**Path defaults are CWD-relative** (except `--config`): the default `--output` /
`--golden-set` / `--output-dir` strings (`evals/newsletter_golden_set.jsonl`,
`evals/newsletter_results/`) are resolved against the current working directory,
so the four tools only chain on their defaults when run from the repo root. The
one exception is `--config`, whose default is the **repo-root** `config.toml`
regardless of CWD.

**Environment variables.** The tools run outside Docker and read the same env as
the daemon (see the `.env` symlink note in [README.md](README.md)):

| Variable | Used by | Purpose |
|---|---|---|
| `NEWSLETTER_LLM_URL` / `NEWSLETTER_LLM_API_KEY` | `newsletter_run` | The `[newsletter.llm]` endpoint. The override is atomic: once the URL is set, the key comes only from `NEWSLETTER_LLM_API_KEY` |
| `CLOUD_LLM_URL` / `CLOUD_LLM_API_KEY` | same | Fallback endpoint when `NEWSLETTER_LLM_URL` is unset |
| `PROXY_URL` / `PROXY_API_KEY` | `newsletter_harvest` | Gmail api-proxy address + auth. `PROXY_API_KEY` must be non-empty; harvest exits with a one-line error when it is missing |

`newsletter_run` fails fast before doing any work when no LLM endpoint is
configured, and its error messages name the resolved URL and which env var
supplied it. (`newsletter_label` needs no LLM at all — curation is fully manual
since issue #59 removed auto-seeding.)

### newsletter_harvest

| Flag | Description |
|---|---|
| `--output` | Output golden-set JSONL path (default: `evals/newsletter_golden_set.jsonl`, CWD-relative — run from the repo root so it chains with the other tools) |
| `--max-threads` | Max threads to fetch (default: `50`) |
| `--recipient` | Newsletter recipient to query (`to:<recipient>`); default from `config["newsletter"]["recipient"]` |
| `--config` | Path to config.toml (default: the repo-root `config.toml`, regardless of CWD) |
| `--proxy-url` | API proxy URL (overrides `PROXY_URL` env var) |

Harvest guards each candidate with `is_newsletter(...)`, builds `body` via
`daemon.format_thread_transcript(...)` (identical to production input), and seeds
each `GoldenNewsletter` with `stories=[]` and `reviewed=False` — **no ground truth
is inferred**. It always appends, deduplicating by `thread_id` — threads already
present in `--output` are skipped **before** fetching (no `get_thread` call), with
a post-fetch dedup as backstop, so re-runs don't re-download known threads. It
tolerates malformed existing lines. To rebuild from scratch, delete the file
manually.

A successful run ends with a next-step hint pointing at
`uv run python -m evals.newsletter_label --golden-set <output>`. Per-request
httpx/httpcore INFO logs are silenced via the shared `daemon.quiet_http_logging()`
helper (each entry point calls it explicitly — importing daemon alone silences
nothing), and a missing `PROXY_API_KEY` exits 1 with a one-line message instead
of a traceback.

### newsletter_label

| Flag | Description |
|---|---|
| `--golden-set` | Path to newsletter golden set JSONL (default: `evals/newsletter_golden_set.jsonl`) |
| `--unreviewed-only` | Show only newsletters not yet reviewed |
| `--include-excluded` | Also queue excluded newsletters, so they can be inspected or restored with the `X` hotkey (default: excluded newsletters are skipped) |

Curation is fully manual and needs no LLM endpoint (issue #59 removed the
Phase-A auto-seeding and its `--edit`/`--config` flags — the seed never sped
curation up, so manual-only became the only mode).

The detail screen is **body-centric**: the newsletter body fills the screen with
each story's span highlighted **in place** (a colored gutter bar + tint, and a
`▶ Story N` marker row at the span's first line), a compact **story strip** above
the body listing each story (`▶N. L<lo>-<hi> [scores] excerpt`, `⚠ not found` if
the text can't be located), and a **mode bar** that always names the active mode
and its keys. A story's line span is **derived at runtime** by matching its text
against the body (`locate_story_span`) — the golden set stores only the story
text, never line numbers.

Two phases per newsletter. **Phase A** curates the story list (extraction truth)
in two explicit modes:

- **Browse mode** (default): `n`/`p` or a number key (`1`–`9`) selects a story
  (emphasized in the body and the strip); `Enter` on a story's body row selects
  it. `a` starts a new story, `e` edits the selected story's boundaries, `d`
  deletes it (a `y/N` prompt guards a labeled story; deletes echo what was
  removed), `C` clears **all** stories after a `y/N` confirm, `u` toggles the
  selected story's exclusion, `l` labels it. `c` **accepts** the story list
  (`confirm_story_list` → `reviewed=True`, warning first if stories are
  unlabeled) and advances to the next newsletter; `k` skips without marking
  reviewed.
- **Span mode** (entered by `a` for a new story or `e` for the selected one):
  the working span live-highlights as the cursor moves. A new story is two
  presses of `Enter` — mark the first line, then the last line; `s`/`e`
  fine-adjust the start/end to the cursor; `Enter` commits and `Esc` cancels. A
  committed story's text is the **verbatim inclusive body slice**; editing an
  existing story replaces its text but preserves its `story_id`, labels, themes,
  notes, and excluded flag. If a story can't be located in the body, `e` falls
  back to a one-line text prompt.

Story ids are **stable**: assigned at creation (max existing numeric suffix + 1,
so delete-then-add never collides) and preserved by accept, which only repairs
duplicates — a delete leaves an id gap by design, so cross-run comparisons keyed
on `story_id` stay valid.

**Phase B** labels the selected story (`l`): the 4 dimensions (simple, concrete,
personal, dynamic) each graded Poor/OK/Good (keys `1`/`2`/`3`); themes graded
Absent/Present/Emphasized (each key `s`/`c`/`h`/`v`/`d` cycles through the three);
notes (`N`), undo (`z`); `expected_tier` is auto-derived via `compute_tier` (mean
of the 1/2/3 dimension scores) on save and `story.reviewed` is set. Saves are atomic (temp-file + rename). `excluded` stories are kept as
extraction truth but skipped from quality/theme scoring.

A **whole newsletter** can be excluded with `X` in the detail view: it is
dropped from the labeling queue and from eval runs entirely (the run's load
summary counts it under "excluded"). `X` toggles, the change is undoable (`z`),
and `reviewed` is left untouched — a confirmed story list stays valid if the
newsletter is later restored. Excluded newsletters vanish from the default
queue on the next launch; relaunch with `--include-excluded` to see them
(marked `X` in the list's flag column) and press `X` again to restore.

TUI details:

- **Undo** (`z`) is a multi-level stack (up to 100 snapshots per newsletter). A
  snapshot is pushed only when a mutation actually happens — cancelled prompts
  never consume an undo level.
- **Esc** cancels an in-progress span edit; otherwise it acts as "back" to the
  list. (Textual handles Esc natively — the old curses `ESCDELAY` workaround is
  gone.)
- **Row cache**: body rows + located spans are cached and rebuilt only on
  mutation or resize; selection and span-highlight changes re-style the cached
  rows without a rebuild, and a resize rewraps the body to the new width.
- **List view**: columns are `R Lbl Sender Subject` where `Lbl` is
  labeled/total stories and the flag column shows `Y` (reviewed) or `X`
  (excluded, which takes precedence); the list supports `PgUp`/`PgDn`.

> Story **titles were removed** from the pipeline (the extraction prompt now emits
> `STORY:` blocks and quality/theme prompts take text only), so stories are
> identified everywhere by a first-words text excerpt. This changed the
> extraction/quality/theme `prompt_hash`, so runs recorded before the change are
> no longer prompt-comparable.

### newsletter_run

| Flag | Description |
|---|---|
| `--golden-set` | Path to newsletter golden set JSONL (default: `evals/newsletter_golden_set.jsonl`) |
| `--config` | Path to config.toml (default: the repo-root `config.toml`, regardless of CWD) |
| `--output-dir` | Output directory for results (default: `evals/newsletter_results/`) |
| `--mode` | Which outputs to evaluate: `extraction`, `quality`, `themes`, or `all` (default: `all`). `quality` and `themes` fire **only their own** LLM calls; the skipped side's fields are `null` in the results rows |
| `--tag` | Tag for the results filename (e.g. `baseline`) |
| `--no-cache` | Disable the LLM response cache (default: cache enabled) |
| `--parallelism` | Concurrent evaluations (default: `1`) |
| `--include-unreviewed` | Also evaluate newsletters not yet reviewed (default: reviewed only) |
| `--prompts` | Path to a TOML file whose `[newsletter.prompts.*]` blocks are deep-merged over the base config (changes `prompt_hash`) |
| `--model` | Override the newsletter LLM model name from config |
| `--report` | Print a metrics report for this run after it completes |
| `--compare-to` | After the run, print a comparison against this prior results JSONL file. No longer always-verbose: per-story detail follows `--verbose` |
| `--verbose` | With `--report`/`--compare-to`: per-story diffs, parse-failure raws, extraction detail |
| `--match-threshold` | With `--report`/`--compare-to`: `SequenceMatcher` ratio threshold forwarded to the embedded report/comparison (default: `0.6`) |
| `--skip-preflight` | Skip the endpoint reachability check before evaluating |

Run replays the golden set through the real `NewsletterClassifier` using the
`[newsletter.llm]` endpoint (resolved via `resolve_newsletter_llm_endpoint()`).
Extraction mode feeds each `body` through `extract_stories`; quality/theme mode
scores the **fixed golden story text** of every reviewed, non-excluded story
(independent of what extraction produced) and derives the predicted tier via
`compute_tier`. Story modes skip stories with `reviewed=False` (Phase-A-confirmed
but never Phase-B-labeled) and print a skip count; `meta.story_count` counts what
the run actually evaluated — labeled, non-excluded stories in story modes, but
**all** golden stories of the evaluated newsletters in `--mode extraction`
(extraction matches against the full confirmed story list). `--parallelism`
must be ≥ 1 (validated at the CLI). In a results row, `scores_raw`/`themes_raw = null`
means that call was **never attempted** (mode skipped it), while a captured raw
string with a `null` parsed field means **attempted but unparsed**. Outputs are
timestamped `<ts>_<mode>_<tag>_<runid8>.jsonl` (`NewsletterRunMeta` first line,
then `StoryPrediction` rows in quality/theme mode and `ExtractionPrediction` rows
in extraction mode) plus a `.cot.jsonl` sidecar of `NewsletterThinkingEntry` rows
(written only when at least one entry captured thinking). The sidecar holds two
row shapes: story-level rows (`story_id` + `quality_cot`/`theme_cot`) and
newsletter-level extraction rows (`thread_id` + `extraction_cot`) capturing the
extractor's segmentation reasoning per newsletter — `story_id` is optional in the
schema.

Run output: the golden-set load prints a kept/unreviewed/excluded breakdown (with
actionable hints when nothing is evaluable); `[k/N]` progress lines print during
evaluation; the end-of-run summary line is
`Rows: N (X errors, Y quality parse failures, Z theme parse failures, W theme
near-misses)` plus optional `Unrecognized theme labels dropped by the parser:`
and `Theme lines the parser skipped (near-misses):` detail lines; a
`Chain-of-thought written to <path>.cot.jsonl` line appears when the sidecar is
written; httpx request logging is quieted to WARNING. Missing/malformed config,
`--prompts`, and golden-set files exit 1 with one-line errors, and preflight and
per-row LLM errors name the resolved endpoint URL and its source env var —
preflight failures also append `probe()`'s captured detail (HTTP status or
exception text, plus a model-mismatch hint on 404).

### newsletter_report

| Flag | Description |
|---|---|
| `--results` | Path to a single results JSONL file |
| `--compare` | Two result file paths for side-by-side comparison. `.cot.jsonl` sidecars matched by a glob are ignored (with a stderr note), so `*<tag>*.jsonl` globs work |
| `--results-dir` | Directory of results for trend view (`.cot.jsonl` sidecars skipped) |
| `--verbose` | Show per-story disagreements/flips and, in single-run mode only, per-newsletter extraction diffs (no effect with `--results-dir`; a stderr note says so) |
| `--format` | `table` (default) or `json`. JSON works in all three modes: single run (`{meta, metrics}`), `--compare` (`{"run_a": {meta, metrics}, "run_b": {...}}`), and `--results-dir` (array of summary rows: file, run_id, timestamp, mode, tag, prompt_hash, tier_accuracy, theme_micro_f1, mae_simple, extraction_micro_f1 — `null` for absent sections) |
| `--match-threshold` | `SequenceMatcher` ratio threshold for the extraction one-to-one match (default: `0.6`). Applies to **text** similarity (see below) |

Report computes tier accuracy + confusion matrix + P/R/F1 (excellent/good/fair/
poor), per-dimension MAE (0–2) and exact-match agreement (within-1 was dropped in
the 3-value migration — degenerate on a 1–3 scale), theme multi-label metrics, and
extraction precision/recall/F1 from a greedy one-to-one `SequenceMatcher` match.

Theme metrics are computed at two thresholds (issue #53): the **primary**
(`metrics["themes"]`) is **Emphasized-positive** — exactly what the daemon labels
in Gmail, so it is production-aligned — reported as per-theme + micro/macro F1 +
exact-set-match; a **secondary** detection metric (`metrics["themes_detection"]`,
positive = Present-or-Emphasized) reports a single micro-F1 "did we notice the
theme at all". `theme_micro_f1` in the trend view is the primary (Emphasized) one.

Metric semantics:

- **Extraction matching compares story text** (whitespace-collapsed, lowercased)
  — golden text is the curated ground truth. `--match-threshold` applies to that
  text similarity.
- A newsletter with 0 predicted and 0 golden stories scores
  precision = recall = 1.0 (correct abstention), not 0.0.
- **Theme metrics** include a story iff its own theme call succeeded (row has no
  error and `themes_raw` is present, or — for legacy rows without `themes_raw` —
  a non-empty parsed prediction). A quality-parse failure does not drop a story
  from theme P/R/F1 or exact-set match.
- **Tier metrics** count a row as a parse error only when quality was actually
  attempted (an error, or `scores_raw` captured) and `predicted_scores` is
  `None`. The run's `RunMeta.mode` disambiguates errored rows: in a
  `--mode themes` results file an errored row is a *theme*-call failure, so it
  never counts as a quality/tier error and no tier section is rendered — a
  themes-only results file cannot resurrect a tier section full of fake errors.
- The report's theme list is derived from `newsletter._VALID_THEMES` (single
  source) — themes added or removed upstream propagate automatically.
- Aggregates (theme micro/macro F1, exact-set match, extraction micro/macro
  P/R/F1) are `None`/`N/A` when a run has zero scored rows, never a fake `0.0%`.
  Single-run reports omit the Quality Dimensions and Themes sections entirely
  when the run contains no story predictions (mirroring the tier section); JSON
  output serializes these as `null`.

Single-run report output: the header shows `Golden set:`; the tier section
pluralizes correctly and lists `Failed stories (quality parse/network): <ids>`;
the themes section shows a `Parse anomalies: N` line when theme responses
contained invalid tokens, near-miss lines the parser skipped, or unparseable
output; the extraction section shows
`(<N> newsletters, <M> errors, match threshold <t>)` and renders even when every
extraction call errored (count 0), listing the failed newsletters — extraction
metrics carry `errors`/`error_threads` alongside the scored counts. Bad
`--results`/`--compare` paths (missing file, a directory, permission denied —
any `OSError`) exit with a one-line error instead of a traceback.

`--verbose` detail: per-story flips in `--compare` gate the tier diff on both
runs having quality scores but compare theme grades (a present -> emphasized
change counts as a flip) whenever both rows' theme call succeeded — two
`--mode themes` runs show their theme flips. Story
Disagreements lines append a story text excerpt (best-effort
lookup from the run's recorded `golden_set_path`; degrades to bare ids if the
file moved) and include stories whose quality parse failed but whose themes
disagree; a `Parse/Network Failures` section prints each failed story with its
raw quality response (`scores_raw`) or error; a `Theme Parse Anomalies` section
prints `themes_raw` for stories whose theme output was invalid
(`invalid tokens dropped: ...`), contained near-miss lines the parser skipped
(`near-miss lines skipped: ...`), or was unparseable (`parsed to []`); Extraction Diffs
honor `--match-threshold` (header says `threshold <t>`) and, for any newsletter
that isn't a perfect match, list matched pairs with their `SequenceMatcher` ratio
plus each unmatched predicted / unmatched golden story by a text excerpt.

`--compare` prints a Run A/Run B/Delta table; the header shows each run's
`prompt_hash` + `tag` + `mode=...`, and a WARNING line prints when the two runs'
modes differ. Sections absent from one run render `N/A`, never `-100%` deltas.
`--results-dir` renders a trend table with Run ID, Timestamp, Mode, Tag, Prompt
(`prompt_hash[:8]`), TierAcc, ThemeF1, MAE(simple), and ExtractF1 columns; rows
are sorted chronologically by `run_meta.timestamp` (not filename), and the Tag
column shows `-` when a run has no tag.

Nonexistent or meta-less results files produce a one-line `Error: ...` on stderr
and exit code 1 (all three modes) — and when the offending path is a
`.cot.jsonl`, a hint points at the main results file. Public helpers:
`match_stories_detailed()`, `theme_parse_anomalies()`, `load_story_excerpts()`,
`build_trend_rows()`, `comparison_as_json()`.

### Newsletter shared cache & prompt separation

The newsletter runner reuses the **same** disk cache as the email eval —
`evals/cache/llm_cache.jsonl` via `CachedLLMClient`. The
cache key `[model, temperature, max_tokens, extra_body, system_prompt,
user_content]` already **separates prompt variants by content**: editing a
`[newsletter.prompts.*]` string changes the `system_prompt` (and thus the key), so
A/B runs are cached independently and unchanged stages are reused. Each run's
`NewsletterRunMeta` records a **`prompt_hash`** =
`sha256(json.dumps(config["newsletter"]["prompts"], sort_keys=True))[:16]` plus the
model/temperature/max_tokens/extra_body/tag/counts and the three system prompts
verbatim, so prompt A/Bs are self-identifying and comparable. `--prompts` (and
`--model`) are applied **before** `prompt_hash` is computed, so an override yields a
distinct hash.

## LLM Response Cache

The eval suite includes a disk-backed LLM response cache that avoids redundant LLM calls across repeated evaluation runs. This is especially useful during prompt A/B testing or model swaps — once a thread has been evaluated with a given configuration, re-running produces instant results from cache.

**Location:** `evals/cache/llm_cache.jsonl`

**Cache key:** A SHA-256 hash of a JSON array containing these fields (from `llm_cache.py`):

```
[model, temperature, max_tokens, extra_body, system_prompt, user_content]
```

Changing any of these — the model name, temperature, inference parameters, `extra_body` config, or the prompt content — produces a different cache key, ensuring stale hits don't occur across configurations. The flip side: a config-only change (e.g. raising a tier's `max_tokens`) invalidates every cached entry for that tier and re-bills the next run, **without changing `prompt_hash`** — the hash covers prompts only, so two runs that differ only in inference parameters are not distinguishable by hash. Note such changes alongside any A/B comparison they straddle.

**Behavior:**

- On startup, the entire cache file is loaded into memory.
- Cache hits return instantly without contacting the LLM. An entry whose stored
  `thinking` is `""` **and** which carries the `thinking_complete` marker was
  written by post-#64 any-model capture: the response carried no
  separately-capturable reasoning — no separate reasoning field
  (`reasoning_content` / `reasoning`) and no inline `<think>` block. It is a
  normal hit, not re-fetched.
  **`""` is NOT proof the model didn't reason** (issue #64): with native
  thinking disabled, the model reasons as untagged prose in the content
  channel, which is preserved in the result's `label_raw`, not in `thinking`.
- Entries with unknown thinking are backfilled with one LLM call the first time
  chain-of-thought is requested for them, then re-written with the
  `thinking_complete` marker and cached permanently. "Unknown" covers both
  legacy entries (no `thinking` key at all) and entries whose `""` predates the
  any-model capture fix (no `thinking_complete` marker): those were written
  when capture was GLM-only, so reasoning a backend routed to a separate
  non-GLM field — e.g. Ollama's `reasoning` — was dropped as `""` even though
  the model reasoned. If the backfill call itself returns no usable answer
  (`LLMContentError`, e.g. the re-decode truncates at the `max_tokens` budget),
  the cached response is still served — never discarded over a thinking
  backfill — and `""` is frozen with the marker so the doomed re-fetch isn't
  re-billed every run.
- Cache misses call the real LLM, then store the response in memory.
- New entries are appended to disk when `flush()` is called (at the end of each run).
- Hit/miss statistics are printed at the end of every evaluation run.

**Disabling the cache:**

Pass `--no-cache` to `run_eval` to bypass the cache entirely and call the LLM for every thread:

```bash
uv run python -m evals.run_eval --no-cache
```

## Chain-of-Thought Capture

When running evaluations with the LLM cache enabled (the default), chain-of-thought content is automatically captured and stored in sidecar files alongside results. Capture reads, in order: the response's separate reasoning field for any model (`reasoning_content` — GLM; `reasoning` — Ollama), then inline `<think>...</think>` blocks in content. With native thinking disabled (the shipped local config), neither exists — the model's step-by-step reasoning is the untagged content itself, preserved in `label_raw`; that's where to look when inspecting a thinking-off run's reasoning (e.g. issue #14).

**Sidecar format:** `evals/results/<run>.cot.jsonl` — one JSON line per thread with `stage1_thinking` and `stage2_thinking` fields.
