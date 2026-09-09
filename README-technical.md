# Email Labeler — Technical Reference

Detailed configuration, environment variables, project structure, and test coverage for the email-labeler daemon. For an overview of the system, see [README.md](README.md).

## Project Structure

```
email-labeler/
├── daemon.py           Main entry point: polling loop and orchestration
├── classifier.py       Two-tier classification logic and LLM output parsing
├── labeler.py          Gmail label verification and application
├── newsletter.py       Newsletter story extraction, quality scoring, and theme classification
├── llm_client.py       LLM abstraction for cloud and local endpoints
├── proxy_client.py     Gmail API proxy client (forked from email-agent, Feb 2026;
│                       independently maintained here — no sync obligation, decision D9)
├── gmail_utils.py      Email header and body parsing (forked from email-agent, Feb 2026;
│                       independently maintained here — no sync obligation, decision D9)
├── config_utils.py     Config loading and env var substitution
├── tui_common.py       Shared Textual widgets/screens for the TUIs
├── config.toml         Label definitions, prompts, and operational parameters
├── pyproject.toml      Python project metadata and dependencies
├── Dockerfile          Container image definition
├── .env.example        Environment variable template
├── evals/              Classification evaluation suite
│   ├── schemas.py      Dataclasses: GoldenThread, PredictionResult, RunMeta
│   ├── harvest.py      Pull processed threads → golden set JSONL
│   ├── review.py       Interactive CLI for label review and correction
│   ├── run_eval.py     Replay golden set through real classifier
│   ├── report.py       Metrics: accuracy, confusion matrix, P/R/F1, privacy
│   ├── newsletter_schemas.py  Newsletter golden-set and result dataclasses
│   ├── newsletter_harvest.py  Pull candidate newsletters → golden set (unlabeled)
│   ├── newsletter_label.py    Hand-label story quality scores and themes
│   ├── newsletter_run.py      Replay golden stories through the newsletter classifier
│   ├── newsletter_report.py   Newsletter tier/dimension/theme/extraction metrics
│   └── results/        Timestamped result files from evaluation runs
├── newsletter_review/  Textual TUI for browsing newsletter assessments
│   ├── __main__.py     CLI entry point (python -m newsletter_review)
│   └── tui.py          Pure data helpers + Textual app
├── docs/
│   └── runbook-agent-attempted-recovery.md  Owner-run manual sweep of threads
│                       dropped to agent/attempted by issue #64 (time-sensitive:
│                       cleanest while the daemon is stopped)
├── scripts/
│   ├── eval_model.py           One-command-per-model eval wrapper
│   ├── migrate_assessments.py  Convert pre-#53 records in an assessments JSONL
│   └── smoke_concurrency.py    Local-serving concurrency smoke test
└── tests/
    ├── conftest.py          Shared fixtures and sample Gmail data
    ├── test_llm_client.py   LLM client tests
    ├── test_classifier.py   Classifier tests
    ├── test_labeler.py      Label manager tests
    ├── test_daemon.py       Daemon orchestration tests
    ├── test_config_utils.py Config loading tests
    ├── test_newsletter.py   Newsletter pipeline tests
    ├── test_eval_schemas.py Golden set and result serialization tests
    ├── test_eval_harvest.py Ground truth inference and deduplication tests
    └── test_eval_report.py  Metrics computation and report formatting tests
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PROXY_API_KEY` | Yes | — | API key for the api-proxy server |
| `PROXY_URL` | No | `http://host.docker.internal:8000` | URL of the api-proxy server |
| `CLOUD_LLM_URL` | Yes | — | Cloud LLM chat completion endpoint (any OpenAI-compatible API) |
| `CLOUD_LLM_API_KEY` | Yes | — | API key for the cloud LLM |
| `NEWSLETTER_LLM_URL` | No | `CLOUD_LLM_URL` | Endpoint for the newsletter grading LLM (`[newsletter.llm]` in `config.toml`). Set when that model needs a different provider than the cloud classifier — e.g. a Claude model via Anthropic's OpenAI-compatible endpoint. |
| `NEWSLETTER_LLM_API_KEY` | No | `CLOUD_LLM_API_KEY` | API key for the newsletter LLM endpoint. The override is atomic: once `NEWSLETTER_LLM_URL` is set, the key comes only from this var (never the cloud key), so set both together. |
| `MLX_URL` | No | — | Local MLX LLM chat completion endpoint. If unset or unreachable, person emails are skipped. |
| `MLX_MODEL` | No | — | Local LLM model name. Shared with email-agent so both services use the same model. Referenced in `config.toml` as `{env.MLX_MODEL}`. |
| `MLX_API_KEY` | No | — | Local LLM API key. Empty for real MLX. Setting it for a public API stand-in (e.g. Novita.ai) is **non-production/eval use only** — it sends person email bodies off-network (decision D4). |
| `USER_NAME` | No | — | User's display name, substituted into classification prompts via `{env.USER_NAME}` in `config.toml`. |
| `VIP_SENDERS` | No | — | Comma-separated email addresses of VIP senders. VIP threads skip the sender classification LLM call. |
| `EMAIL_LABELER_API_KEY` | No | — | Fallback API key for the api-proxy server, used when `PROXY_API_KEY` is not set. |
| `NEWSLETTER_ONLY` | No | — | Set to `1`, `true`, or `yes` to run only the newsletter function: threads not matching the newsletter recipient are skipped. Newsletter grading itself is enabled by `[newsletter]` in config.toml, with or without this flag. |
| `LOCAL_PARALLEL` | No | `1` (from `config.toml`) | Max concurrent local MLX requests, overriding `local_parallel` in `config.toml`. Modern MLX servers batch these (shared weights), so concurrency mostly costs KV cache. Keep ≤ 8 — mlx-lm has a KV-cache cross-contamination bug at 16+. |
| `MAX_EMAILS_PER_CYCLE` | No | `10` (from `config.toml`) | Max threads processed per poll cycle, overriding `max_emails_per_cycle` in `config.toml`. Raise temporarily to drain a large backlog faster. |
| `WRITE_PARALLEL` | No | `4` (from `config.toml`) | Max concurrent label-application writes (`modify_message`), overriding `write_parallel` in `config.toml`. Bounds the proxy-write burst when `max_emails_per_cycle` is large. Sized separately from reads because writes may block on human approval (`WRITE_TIMEOUT`, 300s). |
| `GIT_SHA` | No | `unknown` | Git commit SHA of the running build, logged once at daemon startup (decision D11). Stamped by the image build (Dockerfile `ARG`/`ENV`); not an operator knob. |
| `MAX_FAILURES` | No | `5` (from `config.toml`) | Strikes a thread takes before it is set aside under `agent/attempted`, overriding `max_failures` in `config.toml`. Only failures the cycle-level attribution blames on the thread count (decision D5 Rule 2). Also sets the masquerade escalation threshold. |
| `HALT_PROBE_INTERVAL_SECONDS` | No | `3600` (from `config.toml`) | How often a halted function re-probes its LLM provider and resumes if it answers, overriding `halt_probe_interval_seconds` in `config.toml` (decision D22). |
| `NTFY_URL` | No | — | Full ntfy topic URL for halt/resume push notifications (decision D22). Unset (or `NTFY_TOKEN` unset): notifications are disabled, one WARNING at startup, otherwise no change in behaviour. |
| `NTFY_TOKEN` | No | — | Bearer token for `NTFY_URL`. Mint one for the labeler alone — do not reuse another service's. Secret: never logged. |

Note: The cloud LLM **model name** is configured in `config.toml` under `[llm.cloud]`, not in `.env`. The local LLM **model name** is set via the `MLX_MODEL` environment variable (shared with email-agent) and referenced in `config.toml` as `{env.MLX_MODEL}`. This keeps secrets (keys, URLs) in `.env` while operational parameters (temperature, prompts) stay in version-controlled `config.toml`.

## Configuration

All operational parameters are in `config.toml`. The daemon reads this file on startup.

### Daemon settings

`[daemon]` keys (values and rationale live in config.toml — authoritative):
`poll_interval_seconds` (poll cadence) · `status_interval_seconds` (idle
heartbeat cadence) · `max_emails_per_cycle` (per-cycle thread cap; env
override `MAX_EMAILS_PER_CYCLE`) · `gmail_query` (the unprocessed-thread
search; excludes the `agent/processed` and `agent/attempted` markers) ·
`max_thread_chars` (transcript cap — see below) · `cloud_parallel` /
`local_parallel` / `fetch_parallel` / `write_parallel` (concurrency
semaphores; env overrides `LOCAL_PARALLEL` / `WRITE_PARALLEL`) ·
`max_failures` (strikes before `agent/attempted`, and the masquerade
escalation threshold; env override `MAX_FAILURES`) · `healthcheck_file`
(heartbeat path).

Threads found in a poll cycle are processed concurrently, bounded by the
`cloud_parallel` and `local_parallel` semaphores. **`local_parallel` defaults to 1**:
modern MLX servers do batch concurrent requests (weights loaded once, shared), but
each concurrent request still needs its own KV cache, and long email transcripts
make those caches multi-GB. On a memory-constrained Mac, a few concurrent
long-transcript requests can exceed the GPU's Metal working set (~75% of unified
memory) and OOM-crash the server. Raise `local_parallel` (via `LOCAL_PARALLEL`)
only once you've confirmed the model plus N concurrent KV caches fit the GPU
working set — tune the serving side too (`--prompt-cache-size`, and on macOS
`sudo sysctl iogpu.wired_limit_mb=...`). Keep it ≤ 8 regardless (mlx-lm KV-cache
cross-contamination bug at 16+). See [Local Model Serving & Memory](#local-model-serving--memory).

`max_thread_chars` caps the transcript fed to the classifier. It's deliberately
modest: the local model prefills the entire transcript before answering, so a
50k-char thread can take minutes and exceed the local request timeout — which, on
a stateless daemon, means the thread errors and is retried every cycle forever.
`max_emails_per_cycle`, `local_parallel`, and (via config) the LLM `timeout` are
the levers that keep a single huge thread from stalling the loop.

### LLM settings

`[llm.cloud]` and `[llm.local]` configure model, `max_tokens`, `temperature`,
and `timeout` per tier. **config.toml is authoritative for the values and —
via its comments — for their rationale**; they are deliberate calibrations
(issue #64), not free knobs: the token budgets are coupled to the
thinking-disable decision (registry D16/D17), and `finish_reason: length`
raises rather than silently truncating. Change them only with the config
comments in view. URL and API key come from `.env` (secrets); model names and
inference parameters stay in version-controlled config.toml.

### Extra request body fields

Each `[llm.*]` section supports an optional `extra_body` table. Any key-value pairs defined here are merged into every API request body sent to that endpoint. This is useful for provider-specific parameters that aren't part of the standard OpenAI chat completion format.

**Disabling thinking for reasoning models** — Models like Qwen3, DeepSeek-R1, and GLM-4.5 generate chain-of-thought reasoning before answering — inline in `<think>` tags or in a separate response field, depending on the backend. While the daemon strips inline tags from responses, you can disable thinking entirely to save tokens and reduce latency.

The shipped `config.toml` **disables native thinking on the local person-email classifier**. A paired `stage2_only --sender-type person` eval (n=20) found native thinking strictly worse here: the model spent its `max_tokens` budget reasoning in its thinking channel and emitted no label on some threads, while think-off was ≥ think-on on every thread (85% vs 78% accuracy, 0 vs 2 errors). Issue #64 later reproduced exactly that failure in production under Ollama. The classification prompt already drives step-by-step reasoning into the *content* channel, so disabling native thinking preserves reasoning quality without the budget-split failure.

**The disable dialect is backend-specific, and a backend silently ignores dialects it doesn't understand** — a wrong-dialect disable is indistinguishable from a working one until the errors start (that was issue #64). The shipped config sends both known forms:

For **Ollama** (verified on 0.32.5), the only working field on its OpenAI-compatible endpoint is a top-level `reasoning_effort = "none"`. Only `"none"` disables — `"low"` is a silent no-op, and `think = false` / `chat_template_kwargs` are ignored there:

```toml
[llm.local.extra_body]
reasoning_effort = "none"
```

For **LM Studio and `mlx_lm.server`** with models that use `chat_template_kwargs` (e.g. Qwen3) — where a top-level `enable_thinking` is ignored:

```toml
[llm.local.extra_body.chat_template_kwargs]
enable_thinking = false
```

Some providers accept a top-level `enable_thinking` flag instead (Novita.ai, various OpenAI-compatible APIs):

```toml
[llm.local.extra_body]
enable_thinking = false
```

Do **not** reach for Qwen's `/no_think` prompt switch: under Ollama it emitted a stray closing think-tag into content with no opening tag, which the tag-stripping regex cannot remove (issue #64).

You can put any provider-specific fields in `extra_body` — it is not limited to thinking controls. For example:

```toml
[llm.cloud.extra_body]
top_p = 0.9
frequency_penalty = 0.5
```

### Newsletter settings

`[newsletter]` configures `recipient` (the To/Cc match that routes a thread
to the newsletter function) and `output_file` (the assessment JSONL sink);
`[newsletter.llm]` configures the grading model and its `max_tokens` /
`temperature` / `timeout` — values and sizing rationale in config.toml
(authoritative; the budget/timeout comments there are deliberate issue-#64
calibration).

**`output_file` is relative to the process working directory** and the daemon
`mkdir`s it silently if missing. In Docker (`WORKDIR /app`) that means
`/app/data/newsletter_assessments.jsonl` **inside the container's writable
layer** — no error, no warning, and the records are destroyed the next time the
container is recreated. Running in Docker therefore **requires a volume mount**
for the data directory, pointed at the host path you browse with
`python -m newsletter_review`:

```yaml
services:
  email-labeler:
    volumes:
      - ./data:/app/data
```

#### Sink preflight (startup)

Every way this sink fails is silent in normal operation — an unmounted path
accepts writes right up until the container is recreated — so the daemon
preflights it before grading anything and reports what it found:

```
INFO  Newsletter assessments append to: /app/data/newsletter_assessments.jsonl (412 existing record(s))
INFO  Assessments are persisted by the ext4 mount at /app/data (source /home/you/stack/data,
      relative to that filesystem's root) — confirm that is the directory you review
ERROR Newsletter classification is enabled but [newsletter] output_file is not set in
      config.toml — newsletters will be graded and labeled, but no assessment records
      will be written
ERROR Newsletter assessments resolve to /app/data/newsletter_assessments.jsonl, which no
      volume covers — it is inside the container's writable layer. Writes will appear to
      succeed, but the records are invisible on the host and are DESTROYED when the
      container is recreated. Mount the directory you review, e.g. '- ./data:/app/data'
      under the service's volumes.
ERROR Newsletter assessments sink is not writable: /app/data. Newsletters will be left
      unprocessed (retried every cycle until this is fixed, never abandoned) rather
      than graded into a record that cannot be saved.
```

* **The record count** is the tell for a *misdirected* sink: a daemon that has
  been grading for weeks against a path holding `0 existing record(s)` is not
  appending to the file you review.
* **The persistence check** compares the resolved path against
  `/proc/self/mountinfo`: it fires only inside a container (`/.dockerenv` or
  `/run/.containerenv`), and only when the nearest mount enclosing the path is
  the container root itself *or* is an ephemeral filesystem (`tmpfs`, `ramfs`,
  another `overlay`) — a tmpfs over the directory looks like a volume to a
  mount-point check but dies with the container just the same. A durable bind
  mount over the directory *or* over the file silences it; an unreadable
  `mountinfo` (non-Linux) means no evidence, so no warning. Note this only
  detects containers that drop one of those two marker files — Docker and Podman
  do, containerd under Kubernetes does not.
* **The mount source and fstype** are logged when a real mount *does* hold the
  sink, because no check can know which host directory you meant: a volume aimed
  at the wrong one fails as silently as no volume at all. The source is
  `mountinfo` field 4 — the path within the mount's own filesystem
  (`/var/lib/docker/volumes/<name>/_data` for a named volume). It is *relative to
  that filesystem's root*, not to the host's `/`: if the host keeps `/srv` on its
  own filesystem, a bind of `/srv/stack/data` is reported as `/stack/data`, so
  compare the tail rather than expecting a path that resolves on the host.
* **A missing `output_file`** — `[newsletter]` configured with no sink at all —
  is its own ERROR: grading and labeling proceed, nothing is ever recorded.
* **The writability check** uses the nearest existing ancestor when the file
  doesn't exist yet (`write_assessment` creates missing parents), and rejects an
  `output_file` that names a directory — fatal for every grading, yet `os.access`
  calls a directory writable.
* **An unreadable sink** (the count comes back unknown) is an ERROR too, not a
  footnote on the INFO line: nothing below it can vouch for the path.

#### Assessment record schema

One JSON object per line, appended by `write_assessment` (newsletter.py). The
documented shape is `schema_version: 1`:

| Field | Meaning |
|---|---|
| `timestamp` | *Processed* time, ISO-8601 UTC. Always present. |
| `schema_version` | `1` (int) for this shape. Absent on pre-versioning records — see version semantics below. |
| `message_id`, `thread_id` | Gmail message/thread ids of the graded newsletter. |
| `from` | Sender. The JSON key is the reserved word `from`, not `sender` (the Python parameter's name). |
| `subject` | Subject header value. |
| `send_date` | The email's own send date (email-intrinsic), ISO-8601 UTC or null. |
| `model` | Grading model identifier, or null. |
| `overall_tier` | Best story's tier as a string, or null. |
| `stories[]` | One object per extracted story: `text`; `scores` (dict of dimension → 1\|2\|3, or null when grading failed); `average_score` (float or null); `tier` (string or null); `themes` (dict, theme → grade); `quality_cot`; `theme_cot`. |

Two semantics the shape alone does not convey:

* A theme **absent** from `stories[].themes` graded Absent —
  absence-by-omission (decisions D14/D15); only `present`/`emphasized` grades
  are recorded.
* `migrated_from: "pre-#53"` marks a record converted by
  `scripts/migrate_assessments.py` — what conversion preserves is the
  migration table in the next section.

Version semantics:

* `schema_version: 1` is the shape above, with one carve-out: records stamped
  v1 by the migration script (they bear `migrated_from`) may lack
  `send_date`/`model` entirely — those keys postdate the records being
  migrated, and the migration deliberately does not fabricate them.
* **Absence of `schema_version` means a pre-versioning record**, of which two
  shapes exist: post-#53 current-shape (possibly lacking `send_date`/`model`,
  which arrived without a bump) and pre-#53 legacy (list-shaped themes, 1-5
  scores) — the latter readable only after `scripts/migrate_assessments.py`
  (next section).
* Files may mix versions: concatenating a rescued copy onto the host file is
  an expected operation, deduped on read by newest `timestamp` per
  `thread_id` (decision D18; the read-side rule is documented under
  write-before-label ordering below). Readers keep `.get()` tolerance.

#### Migrating pre-#53 records

Issue #53 replaced the 1-5 dimension scores with the 3-value Poor/OK/Good rubric
and list-shaped story themes with theme->grade dicts, and the readers were then
made to **reject** the old shapes (`70c1a02`) on the premise that affected data
would be regenerated. That premise held for the eval golden set. It does not hold
for this file: it is append-only and is the only copy of every grading ever made,
so a single pre-#53 record at the top makes the whole file unopenable in the
review TUI. Convert it in place:

```bash
python -m scripts.migrate_assessments data/newsletter_assessments.jsonl             # dry run: counts
python -m scripts.migrate_assessments data/newsletter_assessments.jsonl --in-place  # apply (keeps .bak)
```

Run this **on the host**, against the host copy of the file: the image is built
with `COPY *.py ./`, so `scripts/` is not in it and `python -m
scripts.migrate_assessments` has nothing to import inside the container. If the
records are still stranded in the container layer, `docker compose cp` them out
first (see [README.md](README.md)).

Stop the daemon first — it appends to this file. Every line is parsed and
converted before anything is written, so a malformed file (or a dimension score
outside the old 1-5 rubric) aborts with the original untouched, and the rewrite
goes through a temp file + `os.replace`. An `--in-place` run with nothing left to
migrate is a no-op — it does not rewrite the file, and does not overwrite the
`.bak` the first run left.

What the conversion can and cannot preserve:

| Field | Treatment |
|---|---|
| `stories[].themes` | List -> `{theme: "present"}`. Exact in meaning: the old list asserted presence with no emphasis judgment. Nothing becomes `emphasized`, so no migrated record implies a theme label — matching the labels those emails carry. |
| `stories[].scores` | Bucketed 1-2 -> Poor(1), 3 -> OK(2), 4-5 -> Good(3). Lossy by construction — 5 values onto 3. |
| `stories[].average_score` | Recomputed from the bucketed scores so it agrees with the labels the detail view renders. |
| `overall_tier`, `stories[].tier` | **Preserved verbatim, never recomputed.** The tier is what the grader concluded under the old rubric and what was applied to the email as a Gmail label; deriving a new one from re-bucketed dimensions would leave the record disagreeing with the message. |
| `migrated_from` | Added (`"pre-#53"`). The review TUI's detail view surfaces it, so re-bucketed scores are never mistaken for what the grader emitted. |
| `schema_version` | Added (`1`) — a converted record is the documented v1 shape (with the carve-out under version semantics above: it may still lack `send_date`/`model`, which the migration does not fabricate). The `1` is frozen in the script, not tied to whatever version `write_assessment` stamps today. Records already in the new scheme pass through byte-identical, so an unversioned post-#53 record stays unversioned. |

#### Write-before-label ordering

The assessment record is written **before** `apply_newsletter_classification`
commits the Gmail labels. The JSONL is the only durable copy of a grading —
Gmail keeps just the coarse tier/theme labels — and those labels include
`agent/processed`, which drops the thread out of `gmail_query` permanently.
Labeling first and writing after (with the write error swallowed) turned any
sink fault into silent, unrecoverable loss: labels applied, grading gone, thread
never re-graded. Writing first inverts the failure mode:

| Fault | Outcome |
|---|---|
| Sink write fails | `OSError` logged at ERROR naming the resolved path, re-raised as `AssessmentSinkError` and caught by its own arm: thread left unprocessed → retried next cycle → retried *forever*. A sink fault is shared-cause (decision D5), so it never counts toward give-up and the newsletter is never abandoned to `agent/attempted`; the per-cycle ERROR is the loudness, and the ResultCache means each retry within the session re-attempts only the write |
| Labels fail after a successful write | Thread unprocessed → retried next cycle from the daemon's session `ResultCache` (issue #29): the cached grading is reused and the JSONL append is skipped, so only the labels write is re-attempted — no LLM re-run and no second record for the same thread content **within a daemon session**. The cache is in-memory and pruned to each cycle's `max_emails_per_cycle` page, so three things re-grade and re-append: a changed fingerprint (a new message in the thread), a daemon restart mid-retry, and a backlog large enough to push the pending thread off a page. For all of them `load_assessments` keeps the newest record per thread — by the record's own `timestamp`, not by file position, so merging a rescued copy of the file cannot resurrect an older grading — and the review TUI still shows one row |

### Prompt templates

The `[prompts.sender_classification]` and `[prompts.email_classification]` sections contain the system prompts and user message templates used for each classification stage. Templates use Python format strings with `{sender}`, `{subject}`, `{snippet}`, and `{body}` placeholders.

### Label configuration

All label names and their inbox/archive actions are defined in `config.toml` under `[labels]` and `[labels.actions]`. Newsletter labels live under `[newsletter.labels]`.

## Local Model Serving & Memory

Person-email bodies are classified by a local MLX model (`MLX_URL`), typically
`mlx_lm.server`. On a memory-constrained Mac this is the most failure-prone part
of the stack, and the failures are non-obvious, so they're documented here.

### Starting the server

```bash
mlx_lm.server --model <mlx-model> --host 0.0.0.0 --port 8080 --temp 0 \
  --decode-concurrency 8 --prompt-cache-size 2
```

- `--host 0.0.0.0` — required to reach the server from another machine (e.g. over Tailscale). The default `127.0.0.1` accepts only localhost; a remote connection to a localhost-bound server is **reset**, surfacing in the daemon as a connection error.
- The request's `model` field must match the loaded `--model`, or mlx_lm.server returns `404 Not Found`. So `MLX_MODEL` must name the served model.
- `--prompt-cache-size N` bounds how many past request KV caches are retained for reuse. The default (10) accumulates several GB of KV across distinct emails (which share no prefix) and can exhaust GPU memory; **2 is recommended** for this workload.
- `--decode-concurrency` sizes the continuous-batching slots; it only matters when the daemon sends concurrent requests (`local_parallel` > 1).
- To verify continuous batching actually engages, run `scripts/smoke_concurrency.py` (stdlib-only) against the server: it times N concurrent requests vs. one — with batching working, N concurrent finish in roughly the wall time of a single request; ~1× throughput means requests are serializing.

### The GPU memory ceiling (why it OOM-crashes)

Apple Silicon caps the GPU's Metal working set at roughly **75% of unified memory** (~48 GB on a 64 GB Mac). Model weights + every live KV cache + prefill activation buffers must fit under that ceiling, not the full RAM. Exceeding it aborts the server:

```
[METAL] Command buffer execution failed: Insufficient Memory
  (kIOGPUCommandBufferCallbackErrorOutOfMemory)  ... SIGABRT
```

A 27B model at 8-bit is ~34 GB, leaving only ~14 GB of headroom. Long transcripts have multi-GB KV caches, so a few concurrent long-transcript requests (or a large retained prompt cache) blow the ceiling. Note that `[llm.local] max_tokens` was raised 1024 → 4096 (issue #64), so each in-flight request's KV cache can now grow up to 3,072 tokens further during decode than when the `local_parallel` ≤ 8 guidance was calibrated — re-confirm the model + N KV caches fit under the ceiling before raising `LOCAL_PARALLEL`. Mitigations, in order of preference:

1. **Free system RAM** — leaked/idle processes shrink what the GPU can wire.
2. **`local_parallel = 1`** (the default) — one live KV cache at a time.
3. **`--prompt-cache-size 2`** — stop the retained cache piling up.
4. **`max_thread_chars`** — cap prefill size (also bounds latency).
5. **Raise the ceiling**: `sudo sysctl iogpu.wired_limit_mb=53248` (~52 GB on a 64 GB Mac; resets on reboot; don't starve macOS).

### Prefill latency and the request timeout

The model prefills the whole transcript before emitting a label, at ~100–200 tokens/sec on consumer hardware. A 50k-char (~17k-token) thread can take minutes — longer than `[llm.local] timeout` (default 180s) — so the client times out and the thread errors. On the stateless daemon that thread would otherwise be retried every cycle forever; **`max_thread_chars`** (cap the input) is the first guard and the **`FailureTracker`** (see `daemon.py`) is the second. The second one is conditional: a timeout is a strike candidate, so it is counted only when the cycle-level attribution blames the thread (decision D5 Rule 2) — one oversized thread timing out while its siblings classify does converge to `agent/attempted`, but several timing out together share a signature and are held as shared cause, deferred every cycle with the backlog kept. Capping the input is what keeps that case from being the norm.

## Health Checking

The daemon writes a timestamp to `/tmp/healthcheck` after each successful poll cycle. The Dockerfile includes a `HEALTHCHECK` instruction that verifies this file was updated within the last 180 seconds:

```dockerfile
HEALTHCHECK --interval=120s --timeout=5s --retries=3 \
    CMD test -f /tmp/healthcheck && \
        test $(($(date +%s) - $(stat -c %Y /tmp/healthcheck))) -lt 180
```

Check container health with:

```bash
docker inspect --format='{{.State.Health.Status}}' agent-stack-email-labeler-1
```

### Out-of-funds halt (healthy but halted)

When an LLM provider reports the account is out of funds (HTTP 402, or a
400/403 whose body carries a balance signature such as Novita's
`NOT_ENOUGH_BALANCE` or Anthropic's "credit balance is too low" — raised as
`LLMBalanceError`), the **function whose provider it is** stops: the fault is
account-wide, so per-thread retries would only re-fail that function's whole
backlog every cycle against a provider that cannot answer any of it. The halt
is per-function (decisions D5, D19): a newsletter-tier balance fault halts
newsletter grading while email triage keeps
classifying, and vice versa; when the two share one client (`[newsletter.llm]`
absent) a shared-provider fault halts both within a cycle or two. HTTP 429 never
halts, even with quota phrasing — a
per-minute rate limit is worded identically to hard quota exhaustion, and a
wrong restart-only halt is worse than treating a rare 429-signaled
out-of-funds as provider unavailability: deferred and retried each cycle,
never a strike (decision D5). Note what that costs in visibility. Neither
loud-failure ERROR covers it: the shared-cause line is emitted only over
*candidate* (thread-attributable) failures, and the masquerade escalation
requires exactly one affected thread while siblings succeed. An account-wide
429 fails every thread, so a sustained one is visible as the per-thread
`LLM unavailable processing thread …` WARNING each cycle and a flat
`Processed 0/N threads` summary — no ERROR. The triggering thread is left
unprocessed. The halt is in-memory only; **restarting the daemon is the only
reset**.

While **one** function is halted the loop keeps polling for the other, and logs
this line at ERROR once per poll interval. If email triage is the halted one and
newsletter grading is enabled, the Gmail query also gains a `to:<recipient>`
clause for the rest of the session, so the halted function's backlog neither
costs a thread fetch per cycle nor crowds newsletter threads out of the
`max_emails_per_cycle` page:

```
Function halted — email triage: <reason>. Add funds to the provider account, then restart the daemon to resume it; the other function keeps running.
```

Once **every enabled** function is halted (newsletter grading counts as enabled
iff `[newsletter]` is configured, email triage iff `NEWSLETTER_ONLY` is unset)
the daemon stops polling altogether, logging this line at ERROR once per poll
interval and keeping the healthcheck timestamp fresh (deliberately halted, not
hung — the container stays healthy):

```
Daemon halted — every enabled function stopped (<function: reason>; …). Add funds to the provider account, then restart the daemon to resume processing.
```

## Release Identity

The image build stamps the git SHA into `GIT_SHA` (see the env table and README.md's Docker section), and the daemon logs `email-labeler starting — build <sha>` once at startup. Notable releases get a lightweight tag, applied manually by the owner — no semver, no changelog (decision D11): `git tag deploy-YYYY-MM-DD <sha>`.

## TUI Conventions (Textual)

All interactive terminal UIs use [Textual](https://textual.textualize.io/) (a runtime
dependency; framework choice evaluated in issue #40, migration tracked in issue #43).
Conventions shared by every TUI:

- **Pure-helper split**: data transforms (filtering, row/detail formatting, state
  transitions, persistence) live in pure functions with direct unit tests; only the
  widget/screen layer is Textual-specific. UI behavior is tested with Textual's
  `Pilot` driver — real key presses in, widget state and rendered content out.
- **Shared widgets/screens** live in `tui_common.py` (e.g. `KeyMenuScreen`, the
  single-keypress menu that replaces the curses "press one key, anything else
  cancels" prompt idiom, and its `CANCEL` sentinel — distinct from a chosen value
  of `None`, which means "clear").
- **`markup=False` for all record-derived text**: bracketed content like
  `[f]ilter` or user text is otherwise parsed as Rich markup and silently
  swallowed (found via pty smoke test during the issue #40 spike).
- **Pilot test patterns**: `async with app.run_test(size=(100, 30)) as pilot:`,
  drive with `await pilot.press(...)`, assert on widget state
  (`app.query_one(...)`) and rendered content (`str(widget.render())`).
  `asyncio_mode = "auto"` is set, so Pilot tests are plain `async def` tests.
- **Async**: Textual apps are asyncio-native. Long-running work inside a UI
  (e.g. LLM calls) must be awaited or run in a Textual worker — never
  `asyncio.run()` inside an app, which raises `RuntimeError` in a running loop.
- **Every TUI must be documented in a README**, enforced by
  `tests/test_tui_docs.py`. That test discovers TUIs from disk (any non-test
  module defining a Textual `App` subclass) and checks the *nearest* `README.md`
  walking up from the module — `evals/README.md` for eval tools, the root
  `README.md` for everything else. A TUI launchable as `python -m <target>`
  (package with a `__main__.py`, or module with an `if __name__ == "__main__":`
  guard) needs that command shown in a fenced code block; a TUI reached only
  through another tool's flag (e.g. `evals/edit_tui.py` via
  `evals.review --edit`) needs the README to name the module. A new TUI fails
  those tests until documented, with no test edit required.

## Test Coverage by Module

| Test file | Module | What's covered |
|---|---|---|
| `test_llm_client.py` | `llm_client.py` | Request format, auth headers, `<think>` tag stripping, separate reasoning-field capture (`reasoning`/`reasoning_content`), `finish_reason: length` handling, error handling, out-of-funds (`LLMBalanceError`) detection including the 429 exclusion asserted against balance-signature bodies (decision D19), availability checks |
| `test_classifier.py` | `classifier.py` | `parse_sender` formats, `parse_sender_type` edge cases and its SERVICE default, `parse_email_label` edge cases and its keyword-free raise, cloud/local routing, full pipeline |
| `test_labeler.py` | `labeler.py` | Label verification (all present, partial, none), label ID mapping, inbox/archive actions, single API call per email, per-write semaphore bound on every write path (LabelManager-owned `write_sem`, slot released between messages — classification, markers, and newsletter writes; issue #33) |
| `test_daemon.py` | `daemon.py` | Service email path, person email path, MLX-unavailable skip, error isolation, per-function out-of-funds halt (`FunctionHalts`: enabled-slot arithmetic, the `NEWSLETTER_ONLY` stand-down), config loading, assessment-sink preflight + write-before-label durability, classification result reuse across write-retry cycles (`ResultCache`, issue #29 — reuse, fingerprint invalidation, clear/prune, and the poll-loop session wiring), cycle-level failure attribution (correlation strikes, timeout candidates, marking from this cycle's strikes only, count cleared beside a landed marker, adjudicated singleton/zero-success edges, deferral-only threads excluded from the correlation denominator — decision D5 Rule 2), masquerade bookkeeping and escalation (`MasqueradeTracker`: success-clear, prune, single-suspect + success increment condition, throttle reset), halt deferrals that record no failure and commit nothing, the `MAX_FAILURES` knob |
| `test_privacy.py` | `classifier.py`, `daemon.py` | Negative-form privacy tests (registry D2/D3): person-classified bodies reach only the local tier (classifier and daemon level), Stage 1 whole-call payload discipline, unparseable-Stage-1 SERVICE-default pin, VIP short-circuit, newsletter ownership bypass, no cloud fallback on local failure, metadata-shape allowlist |
| `test_config_utils.py` | `config_utils.py` | Config loading, `{env.VAR}` substitution |
| `test_env_var_docs.py` | env-var docs (meta-test) | Every env var referenced by daemon sources or `config.toml` `{env.VAR}` is documented in this file's Environment Variables table |
| `test_env_example_docs.py` | `.env.example` (meta-test) | Every var the example declares is documented in the env table; every Required var has an active line in the example |
| `test_newsletter.py` | `newsletter.py` | Newsletter story extraction, quality scoring, theme classification, assessment record writing (incl. the `schema_version` stamp), sink persistence/writability diagnostics, and the Rule-1 raises that keep `no-stories` to a successful zero-story extraction (unparseable extraction reply; every story failing to grade — decisions D5/D20) |
| `test_eval_schemas.py` | `evals/schemas.py` | GoldenThread/PredictionResult/RunMeta serialization round-trips |
| `test_eval_harvest.py` | `evals/harvest.py` | Ground truth inference from labels, deduplication |
| `test_eval_report.py` | `evals/report.py` | Confusion matrix, precision/recall/F1, accuracy, privacy violation metrics |
| `test_eval_newsletter_schemas.py` | `evals/newsletter_schemas.py` | Golden-set/result dataclass round-trips, missing-key tolerance |
| `test_eval_newsletter_harvest.py` | `evals/newsletter_harvest.py` | Newsletter filtering, body build, dedup, no ground-truth inference |
| `test_eval_newsletter_label.py` | `evals/newsletter_label.py` | Story curation + per-story scoring/theme pure functions, tier derivation, undo + Pilot UI tests (seed guard, undo stack, delete/label flows, selection, skip-through, autosave) |
| `test_eval_newsletter_run.py` | `evals/newsletter_run.py` | `prompt_hash`, cache reuse, extraction vs quality/theme modes |
| `test_eval_newsletter_report.py` | `evals/newsletter_report.py` | `match_stories`, tier/dimension/theme metrics, comparison deltas |
| `test_newsletter_review.py` | `newsletter_review/tui.py` | Pure helpers (loading, per-thread dedup, filtering, formatting, source line, migrated-record note) + Pilot UI tests (navigation, drill-down, tier/theme/sender filters, source header, quit) |
| `test_migrate_assessments.py` | `scripts/migrate_assessments.py` | Old-scheme detection, theme/score conversion, tier preservation, `schema_version` stamped on converted records only, atomic in-place rewrite + abort-on-malformed, round-trip through `load_assessments` |

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`, authoritative) runs dependency
sync, lint, and the full mocked suite on every PR and push to main; no
secrets are required (registry D13).
