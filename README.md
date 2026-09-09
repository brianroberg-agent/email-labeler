# Email Labeler

A background daemon housing two independent functions: **email triage** (classifies unclassified Gmail threads with a two-tier LLM system and applies labels autonomously) and **newsletter grading** (grades ministry-newsletter stories for writing quality and thematic alignment). Both normally run in one daemon.

> For detailed configuration reference, environment variables, project structure, and test coverage, see [README-technical.md](README-technical.md).

## Privacy Model

The email-triage function keeps person-email bodies local, best-effort:

1. **Stage 1 (Cloud LLM)** sees only metadata — sender, subject, snippet —
   and classifies the sender as person or service. Bodies are never sent in
   this stage.
2. **Stage 2** routes by that result: service-classified bodies go to the
   cloud LLM; person-classified bodies go to the local MLX instance and do
   not leave the local network. If the local LLM is unavailable, person
   emails are skipped and retried — never sent to the cloud as a fallback.

This is best-effort routing, not an absolute guarantee: Stage 1 classifies
from metadata and can be wrong, and unparseable Stage 1 output deliberately
defaults to SERVICE. The residual misroute risk is measured — the eval suite
reports a privacy-violation rate (person threads that were routed as
service).

**Newsletter threads are exempt by rule** (decision D3): a thread addressed
(To/Cc) to the configured newsletter recipient is organizational content, and
its full transcript — including person-written replies in the thread — is
graded by the cloud LLM without person/service routing.

## Label Taxonomy

The daemon classifies emails into three categories and applies the corresponding Gmail label:

| Label | Meaning | Action |
|---|---|---|
| `agent/needs-response` | Requires a reply or action from you | Stays in inbox |
| `agent/fyi` | Worth reading, no action needed | Stays in inbox |
| `agent/low-priority` | Routine notifications, newsletters, spam, unwanted | Archived |

Additional marker labels (`agent/processed`, `agent/attempted`, `agent/personal`, `agent/non-personal`) track processing state and routing decisions. The newsletter pipeline adds its own labels under `agent/newsletter/`.

Newsletter labels (see [Newsletter Classification](#newsletter-classification)):

| Label | Purpose |
|---|---|
| `agent/newsletter` | Marker applied to all newsletter emails |
| `agent/newsletter/excellent` | Best story averaged >= 2.75 (dimensions scored Poor/OK/Good as 1/2/3) |
| `agent/newsletter/good` | Best story averaged >= 2.25 |
| `agent/newsletter/fair` | Best story averaged >= 1.75 |
| `agent/newsletter/poor` | Best story averaged < 1.75 |
| `agent/newsletter/no-stories` | Extraction succeeded and found no stories to grade — only that (registry D5/D20). An extraction reply that can't be parsed, or a newsletter none of whose stories graded, is a failure: nothing is labeled or recorded, and the thread is retried next cycle — one that keeps failing while other threads succeed ends findably under `agent/attempted` (see [Resilience](#resilience)) |
| `agent/newsletter/theme/*` | Theme tags: `scripture`, `christlikeness`, `church`, `vocation-family`, `disciple-making` — applied only when a story grades the theme Emphasized (merely-Present themes are recorded in the assessment JSONL but not labeled) |

## Architecture

```
                          +-----------+
                          | Gmail API |
                          +-----+-----+
                                |
                          +-----+-----+
                          | api-proxy |  (human-in-the-loop controls)
                          +-----+-----+
                                |
              +-----------------+-----------------+
              |           email-labeler            |
              |                                    |
              |  Poll loop (every 60s)             |
              |    │                               |
              |    ├─ list unprocessed emails       |
              |    │                               |
              |    ├─ For each thread:             |
              |    │   │                           |
              |    │   ├─ Newsletter? (To: check)  |
              |    │   │   YES ──► Cloud LLM ×3   |
              |    │   │   (extract, score, theme) |
              |    │   │   └─► label + JSONL       |
              |    │   │                           |
              |    │   ├─ Stage 1 ──► Cloud LLM   |
              |    │   │  (metadata only)          |
              |    │   │    └─► PERSON or SERVICE  |
              |    │   │                           |
              |    │   ├─ Stage 2a (service)       |
              |    │   │    └─► Cloud LLM          |
              |    │   │         (full body)       |
              |    │   │                           |
              |    │   ├─ Stage 2b (person)        |
              |    │   │    └─► Local MLX          |
              |    │   │         (full body)       |
              |    │   │                           |
              |    │   └─ Apply label + action     |
              |    │                               |
              |    └─ Write healthcheck file       |
              +------------------------------------+
```

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- Access to an [api-proxy](../api-proxy) instance with a valid API key
- A cloud LLM endpoint (any OpenAI-compatible chat completion API)
- A local MLX LLM endpoint for person email classification (optional but recommended)
- All Gmail labels created manually (see [Label Setup](#label-setup))

## Setup

### 1. Install dependencies

```bash
uv sync --extra dev
```

### 2. Create environment file

If running as part of the `agent-stack` setup (recommended), symlink to the shared `.env`:

```bash
ln -s ../agent-stack/.env .env
```

Otherwise, copy the example and fill in your values:

```bash
cp .env.example .env
```

At minimum you need:

```env
PROXY_API_KEY=aproxy_your_key_here
CLOUD_LLM_URL=https://your-llm-provider.com/v1/chat/completions
CLOUD_LLM_API_KEY=your_api_key_here
MLX_URL=http://macbook:8080/v1/chat/completions
MLX_MODEL=mlx-community/Qwen3.6-27B-8bit
```

See [README-technical.md](README-technical.md#environment-variables) for the full variable reference.

### 3. Label Setup

The api-proxy blocks programmatic label creation, so all labels must be created manually in Gmail before the daemon starts.

In Gmail, go to **Settings > Labels > Create new label** and create each of these:

```
agent/needs-response
agent/fyi
agent/low-priority
agent/processed
agent/attempted
agent/personal
agent/non-personal
agent/newsletter
agent/newsletter/excellent
agent/newsletter/good
agent/newsletter/fair
agent/newsletter/poor
agent/newsletter/no-stories
agent/newsletter/theme/scripture
agent/newsletter/theme/christlikeness
agent/newsletter/theme/church
agent/newsletter/theme/vocation-family
agent/newsletter/theme/disciple-making
```

Gmail will treat the `/` as a label hierarchy separator, nesting them under an `agent` parent. The daemon verifies all labels exist on startup and exits with an error if any are missing.

> **Upgrading an existing deployment:** because startup validation is strict, a release that adds a new required label will exit (and, under Docker, crash-loop) until you create it. Create any newly-required labels in Gmail **before** deploying the new version. This release adds **`agent/attempted`** (applied to threads abandoned after repeated failures); create it first, then upgrade.

## Running

### Local development

```bash
uv run python daemon.py
```

The daemon will:
1. Verify all Gmail labels exist (exits if any are missing)
2. Enter the poll loop, querying Gmail every 60 seconds
3. Classify and label each unprocessed email
4. Write a healthcheck timestamp to `/tmp/healthcheck`

### Docker (via agent-stack)

```bash
docker compose build email-labeler
docker compose up email-labeler
```

Pass `--build-arg GIT_SHA=$(git rev-parse --short HEAD)` to the build to stamp
the image with its commit — the daemon logs it once at startup, so the logs
answer "what is deployed?" (decision D11). Builds without the arg default to
`unknown`.

To run only the newsletter function (skip email triage):

```bash
NEWSLETTER_ONLY=1 docker compose up email-labeler
```

**⚠️ Newsletter assessments need a volume mount.** The daemon appends newsletter
grading records to `data/newsletter_assessments.jsonl` — a path relative to the
container's working directory (`/app`). Without a bind mount the records land in
the container's writable layer and are **destroyed the next time the container is
recreated** (`docker compose up -d` with a new image). Mount the host directory
you browse with `python -m newsletter_review`:

```yaml
services:
  email-labeler:
    volumes:
      - ./data:/app/data   # or an absolute host path
```

The daemon checks this for you at startup. It logs the resolved absolute path
along with how many records that file already holds
(`Newsletter assessments append to: /app/data/newsletter_assessments.jsonl (412
existing record(s))`) — a long-running daemon reporting `0 existing record(s)` is
writing somewhere other than the file you browse. If nothing is mounted over the
path, it says so as an ERROR on the next line.

If you have already been running without the mount, the records so far are still
inside the *running* container and will be destroyed the moment it is recreated.
Rescue them before your next `docker compose up -d`:

```bash
docker compose cp email-labeler:/app/data/newsletter_assessments.jsonl ./recovered.jsonl
```

Then append `recovered.jsonl` to your host file (the review TUI keeps the newest
record per newsletter, decided by each record's own timestamp rather than by
where it landed in the file, so overlapping entries are harmless in either
order).

### Newsletter review TUI

Newsletter grading writes one assessment record per newsletter to
`data/newsletter_assessments.jsonl`. To read them, browse the file in a terminal
UI instead of by hand:

```bash
uv run python -m newsletter_review
```

The listing shows one row per newsletter, newest send-date first. Press `Enter`
to open a newsletter and see its stories with their dimension scores and themes,
`Esc` to go back, `f` to open the filter menu, and `q` to quit.

The header names the file it read, how many newsletters are in it, and the newest
send date — e.g. `/srv/stack/data/newsletter_assessments.jsonl — 253 newsletters,
newest sent 2026-07-29`. That is deliberate: a stale copy of the file, or a copy
from a path the daemon no longer writes to, looks exactly like a working one until
you notice its newest newsletter is weeks old. Cross-check it against the path the
daemon logs at startup.

Filters are also available up front, so you can open straight into a slice:

```bash
uv run python -m newsletter_review --tier poor       # excellent | good | fair | poor
uv run python -m newsletter_review --theme scripture # scripture, christlikeness, church,
                                                     # vocation-family, disciple-making
uv run python -m newsletter_review --sender dm.org   # substring match on the sender
uv run python -m newsletter_review --since 2026-01-01  # sends on/after a local date
uv run python -m newsletter_review --file path/to/file.jsonl  # a JSONL somewhere else
```

If you ran the daemon in Docker, point `--file` at the host path you mounted (or
run the command from the directory holding `data/`) — the default path is
relative to the current directory.

#### "old-scheme record" on startup

```
Error loading …/newsletter_assessments.jsonl: …:1: old-scheme record — story themes
are a list (['vocation_family']), not a theme->grade dict.
```

Records written before the scoring-scheme change (July 2026) store story themes
as a plain list and dimension scores on a 1-5 scale. The reader rejects those
outright, so one old record at the top of the file blocks the whole file. Convert
it — with the daemon stopped, since it appends there:

```bash
python -m scripts.migrate_assessments path/to/newsletter_assessments.jsonl             # counts only
python -m scripts.migrate_assessments path/to/newsletter_assessments.jsonl --in-place  # apply
```

Run it from a checkout on the host — `scripts/` is not inside the Docker image.
A `.bak` copy is kept (a second run changes nothing and leaves that copy alone). Old themes become `present`, old scores are bucketed into
Poor/OK/Good, and each record's tier is left exactly as graded — so it still
matches the label on the email. Migrated records say so in their detail view.
See [README-technical.md](README-technical.md#migrating-pre-53-records) for what
the conversion can and cannot preserve.

## Resilience

The daemon is designed to run unattended and recover from transient failures:

- **Exponential backoff**: If a poll cycle fails, the sleep interval doubles (up to 10x the base interval), then resets on the next successful cycle.
- **Per-email error isolation**: If one email fails to classify, the error is logged and the loop continues with the next email.
- **Provider-shaped vs. thread-attributable failures** (decision D5): A provider that can't serve anyone — an unreachable LLM endpoint or api-proxy, a dropped connection, an exhausted 429 or any 5xx from either — is never one thread's blame: the thread is deferred and retried next cycle, and nothing counts toward give-up. Thread-attributable failures (a request timeout on an oversized transcript, an unparseable reply, a request-specific 4xx) are strike candidates, counted only when cycle-level correlation blames the thread: the failure's signature is unique in its cycle and a sibling thread succeeded (a lone attempting thread counts too). Only threads that actually attempted work — succeeded, or failed — are weighed as siblings; ones that merely deferred (a halted function, the local LLM offline, a rejected write, an unwritable assessment sink) are not evidence either way. Several threads failing the same way — or a cycle where nothing succeeds — reads as a shared cause instead: no strikes, one ERROR line, backlog kept.
- **Stuck-thread give-up**: A thread the correlation keeps blaming is abandoned after repeated strikes (default 5) and marked `agent/attempted`, so one poison thread can't be retried forever — it stays findable under that label. The per-cycle summary reports abandonments distinctly: `Processed N/M threads (K abandoned after repeated failures: [...])`.
- **Masquerade escalation**: A thread that keeps failing with provider-shaped errors while its siblings succeed is never abandoned — it retries forever, and after repeated qualifying cycles the daemon escalates with a distinct ERROR line, repeated on the status heartbeat, telling the operator to investigate the thread or the provider route.
- **Rejected writes**: A label write the api-proxy rejects or blocks (HTTP 403 — an operator's "not now") is a human answer, not a failure: the thread is simply re-offered next cycle, never counted toward give-up and never abandoned.
- **Per-function halt, self-healing** (decisions D5, D22): An LLM provider reporting the account out of funds on consecutive requests (the count is `balance_halt_strikes` in `config.toml` `[daemon]`) stops only the function it serves — a newsletter-tier balance fault halts newsletter grading while email triage keeps classifying, and vice versa. The halted function's threads are left untouched (nothing labeled, nothing abandoned), an ERROR naming it repeats every cycle, and — if `NTFY_URL`/`NTFY_TOKEN` are configured — a push goes out once. While halted, the function re-probes its provider hourly with one request shaped like its real ones (a fixed one-word prompt, no email content) and resumes on its own when the provider answers, with a second push saying how long it was down. Only when every enabled function is halted does the daemon stand down entirely (probes excepted). See [README-technical.md](README-technical.md#out-of-funds-halt-healthy-but-halted).
- **MLX graceful degradation**: If the local MLX server is unreachable, person emails are skipped (retried next cycle) while service emails continue to be classified via the cloud LLM. Because the local machine is expected to be offline for stretches, an outage is treated as routine: one summary line per cycle (`Local LLM offline — deferred N person email thread(s) this cycle`), not a warning per email.
- **No outcome from an unusable answer** (decision D5): An unrecognizable *classification* commits nothing — the reply carries no answer, so the thread is deferred and retried rather than labeled and archived on a guess (it is a thread-attributable failure, so a thread whose replies stay unusable ends findably under `agent/attempted`). An unrecognizable *sender type* still routes to SERVICE: that is a routing default, not an outcome, and it is deliberately best-effort (see [Privacy Model](#privacy-model)).
- **Startup validation**: The daemon verifies all required Gmail labels exist before entering the poll loop.

## Evaluation Suite

The `evals/` directory provides a 4-stage pipeline (`harvest → review → run_eval → report`) for measuring classification accuracy against a golden set of human-reviewed threads.

See [`evals/README.md`](evals/README.md) for full documentation and CLI reference.

## Testing

All tests use mocks and require no external services.

```bash
# Run all tests
uv run --extra dev pytest tests/ -v

# Run a specific test file
uv run --extra dev pytest tests/test_classifier.py -v

# Lint
uv run --extra dev ruff check .
```

See [README-technical.md](README-technical.md#test-coverage-by-module) for per-module coverage details.
