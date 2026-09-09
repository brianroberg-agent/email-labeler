# Decisions Registry

Adjudicated tradeoffs and product decisions. Per the Review Charter in
CLAUDE.md, reviewers check this registry before flagging: a behavior recorded
here is a decision, not a finding. Reversing a decision requires updating its
entry in the same change that reverses it.

Entry format — one `##` per decision:
`## D<n> — <title> (<date decided>)` followed by **Status**, the decision, and
what it forecloses. Status parts use: `implemented` · `implementation pending
(<wave/issue>)` · `reversed by D<m>`; a split decision combines them (e.g.
"docs implemented; enforcement pending (Wave 1)").

---

## D1 — One system, two independent functions (2026-07-30)

**Status:** implemented (docs, Wave 0).

The project is one system housing two independent functions — email triage and
newsletter grading — sharing a daemon and chassis. It is not two products and
will not be split into two repos; the shared chassis stays shared. Hybrid
operation (both functions in one daemon process) is the normal deployment.
Newsletter grading is enabled by the presence of `[newsletter]` in config.toml;
`NEWSLETTER_ONLY` is a *filter* that skips non-newsletter threads, not a mode
switch. A symmetric `INBOX_ONLY` filter (email triage alone) would be
legitimate — the asymmetry is an accident of history, not a decision. Neither
function depends on the other.
Forecloses: repo splits; treating newsletter grading as a test-only mode;
fixes that assume one function's values govern the other.

## D2 — Privacy posture: best-effort routing, stated honestly (2026-07-30)

**Status:** implemented (docs, Wave 0; test enforcement, Wave 1).

The email-triage privacy guarantee is: **bodies of threads classified as
person are processed only by the local LLM**. This is best-effort routing, not
an absolute guarantee — Stage 1 classifies from metadata and can be wrong, and
unparseable Stage 1 output deliberately defaults to SERVICE (availability
first). The residual misroute risk is *measured, not denied*: the eval reports
a privacy-violation rate. Docs must not state the invariant in absolute
("NEVER") form.
Forecloses: "hardening" fixes that flip the unknown-sender default to PERSON;
findings that treat the SERVICE default as a privacy bug.

## D3 — Newsletter ownership rule (2026-07-30)

**Status:** implemented (docs Wave 0; exact-address matching Wave 2).

A thread with any message To/Cc-addressed to the configured newsletter
recipient belongs to the newsletter function and is organizational content —
its full transcript, **including person-written replies**, goes to the cloud
LLM by design, before and instead of person/service routing. The recipient
match is an exact address comparison.
Forecloses: routing newsletter threads through Stage 1 "to restore the
invariant"; privacy findings about person replies inside newsletter threads.

## D4 — Public-API local stand-ins are non-production only (2026-07-30)

**Status:** implemented (docs, Wave 0).

Pointing `MLX_URL`/`MLX_API_KEY` at a public API (e.g. Novita.ai) sends person
email bodies off-network and is sanctioned **only for non-production/eval
use**. Docs describing the stand-in carry this warning.
Forecloses: presenting a public stand-in as a routine production configuration.

## D5 — Failure model: two rules and a scope (2026-07-30)

**Status:** implemented (model Wave 0; corollaries Wave 2); restart-only-reset
corollary reversed by D22 (2026-09-09).

**Rule 1 — Outcomes only come from successes.** A committed outcome (label,
archive, grade record, `agent/processed`) is only ever produced by a
successful classification. Failures never commit anything: they defer, set
aside findably (`agent/attempted`), or stop loudly.
**Rule 2 — Blame by correlation.** Siblings succeeding while one thread fails →
thread-scoped: bounded strikes, then `agent/attempted`. Siblings failing the
same way → shared cause (provider, proxy, disk, or our config/code): count no
strikes, get loud, keep the backlog intact.
**Scope — functions fail independently.** A fault that disables one function
(e.g. its provider's balance) fails that function loudly without stopping the
other.

Corollaries — all landed in Wave 2; each names the commit that implemented it,
and the code no longer deviates from the model anywhere the entry once said it
did:
- Exhausted 429/5xx (LLM and proxy) never count toward give-up — implemented
  (Wave 2 T8, `957ba96`). **This deliberately reverses issue #26.**
  proxy_client's transient-error docstrings are now true; llm_client raises
  LLMUnavailableError for an exhausted 429 or any 5xx (other non-200s stay
  RuntimeError — request-specific strike candidates). A 429-signaled
  out-of-funds is likewise retried as unavailability, never a halt (D19) and
  never give-up.
- Correlation is the attribution mechanism — implemented (Wave 2 T8,
  `957ba96`). Strikes are decided post-gather, per cycle
  (`attribute_cycle_failures`): a candidate
  failure (Timeout / RuntimeError incl. LLMContentError / unexpected
  Exception) counts iff its signature is unique among the cycle's candidate
  failures and, when the cycle has more than one *attempting* thread, at least
  one sibling was handled successfully; marking derives only from the cycle's
  own strikes, never raw tracker counts. **The correlation denominator is the
  threads that attempted work** — those handled successfully or that recorded a
  `CycleFailure` — not every thread the cycle fetched. A thread that only
  DEFERRED (its function halted, the local tier offline, a `NEWSLETTER_ONLY`
  skip, a 403-rejected write, an assessment-sink fault) tried nothing and
  committed nothing, so it is evidence neither of blame nor of innocence and
  leaves the denominator; it stays in the cycle summary and in both prunes,
  since it is still pending. (Refinement found in the Wave 2 review of T8's
  literal wording, "the cycle contained other threads": counting deferral-only
  threads let a single permanently-deferred sibling — a halted function
  re-fetches and re-defers its backlog every cycle — make every cycle look
  multi-thread-and-zero-success, so a genuinely poisoned thread never struck
  and never reached `agent/attempted`, silently voiding Rule 1's set-aside
  guarantee.) Adjudicated edges, unchanged: **singleton cycles count** (no
  attempting siblings to correlate against; the poison thread typically *is* a
  singleton — residual: a code bug failing the only pending thread accrues
  strikes, accepted); **zero-success multi-thread cycles count no strikes**
  (all shared-cause, one ERROR line, backlog kept); **N same-signature poison
  threads shield each other** while they co-fail (the shared-cause ERROR is
  the loudness). The single-thread masquerade (provider-shaped errors,
  siblings succeeding) retries forever, never abandoned: at `max_failures`
  qualifying cycles it becomes a suspect and a distinct ERROR repeats at most
  once per status interval (`MasqueradeTracker`); singleton and zero-success
  cycles never increment the counter, and nothing but a thread's own success
  resets it — a lone thread that *succeeds* clears its count like any other,
  since a success is unambiguous whatever the cycle's shape. Local-tier LLM
  unavailability is excluded entirely (the deliberately-offline MLX host makes
  person-thread deferral routine, issue #24). The masquerade half needs no
  denominator change: it moves only on positive evidence (a succeeding thread,
  and provider-shaped entries that are attempts by construction), which a
  deferral-only thread can never supply — the blame rule is the asymmetric one,
  because its no-sibling fallback is to blame.
- The halt is per-function — implemented (Wave 2 T9, `da368e6`).
  `FunctionHalts` holds one first-tripper-wins `DaemonHalt` slot per function;
  the newsletter branch traps
  `LLMBalanceError` at its `classify_newsletter` call site (the error carries no
  function provenance, and the newsletter client is also `tier="cloud"`, so the
  *call site* is what tells the functions apart) while the outer arm serves the
  email pipeline. A thread whose own function is halted defers below routing —
  no strike, no marker, no `CycleFailure`. The poll loop stands down only when
  every *enabled* function is halted (enabled: newsletter iff configured, email
  iff not `NEWSLETTER_ONLY`); a partial halt keeps polling, names the halted
  function at ERROR each cycle, and — email halted, newsletter running — narrows
  the Gmail query with a `to:{recipient}` clause until restart (the mirror
  direction accepts the fetch-and-skip churn: the query cannot express "not
  to:recipient"). A shared client ([newsletter.llm] absent) trips both slots
  within a cycle or two, which is correct: the fault disables both functions.
  **Reset superseded by D22 (2026-09-09):** "until restart" / "restart-only
  reset" was this corollary's wording as implemented in Wave 2 and is kept
  here as the record; the halt now trips on the `balance_halt_strikes`-th
  consecutive balance fault (config.toml `[daemon]`), re-probes its provider
  hourly, clears itself when the provider
  answers (undoing the query narrowing with it), and pushes a notification at
  halt and at resume. A restart still clears the in-memory state but is no
  longer the only way out.
- A keyword-free label reply raises instead of silently defaulting to
  LOW_PRIORITY→archive — implemented (Wave 2 T10, `ee3958d`).
  `parse_email_label` raises
  `LLMContentError` (quoting the reply) when no keyword survives its three
  passes, so an unusable answer commits nothing (Rule 1) and is a strike
  candidate under the correlation attribution; this completes the issue-#64
  fail-loud direction, which had closed only the empty/truncated-reply half.
  The unknown-sender→SERVICE default *stays* (D2). llm_client's content guard
  stays too — it names the budget as the cause and covers Stage 1, where the
  SERVICE default still applies. Evals degrade honestly: run_eval turns the
  raise into an error row instead of a silent LOW_PRIORITY prediction.
- `agent/newsletter/no-stories` may only result from a successful extraction
  that found zero stories — implemented (Wave 2 T11, `a4005ea`), closing **two**
  false paths. `parse_stories` returns `[]` only for an explicit `NO_STORIES` reply
  and raises `LLMContentError` for any other reply that yields no story
  (empty/whitespace included — unreachable behind llm_client's content guard,
  but the parser's contract must not lie about it). `classify_newsletter`
  raises when stories were extracted yet not one produced scores — issue #30's
  remaining parse-to-None route, whose most common instance is a single-story
  newsletter whose only story fails to grade. Per-story isolation survives for
  the partial case (a story that fails while a sibling grades). Both raises are
  pipeline-wide and are strike candidates under the correlation attribution, so
  a poison newsletter converges to a findable `agent/attempted` rather than a
  false no-stories record; the eval harness, which shares the parser, degrades
  to error rows.
- Assessment-sink faults are shared-cause: never counted, retried forever —
  implemented (Wave 2 T12, `1a9d1fd`). The daemon re-raises the sink `OSError`
  as a dedicated `AssessmentSinkError` (newsletter.py) at the `write_assessment`
  call site and catches it in its own arm ahead of the candidate arms: no
  strike, no `CycleFailure` (the fault never reaches attribution at all), no
  marker — the newsletter is retried every cycle until the operator fixes the
  sink. A dedicated class rather than `except OSError`, which would also
  swallow `TimeoutError` (an OSError subclass, and a strike candidate). The
  per-cycle ERROR naming the resolved path is the loudness, alongside the
  startup preflight; with the ResultCache (T6) each retry *within a daemon
  session* re-attempts only the JSONL write, so "forever" costs no LLM spend —
  the cache is in-memory and pruned to each cycle's page, so a restart, or a
  backlog that pushes the thread off a page, re-grades it once (refined in the
  Wave 2 review: the original wording claimed the record could only repeat on a
  fingerprint change, which the session scope and the page-sized prune both
  contradict; D18's newest-timestamp dedup on read is the backstop).
  README-technical's write-before-label table and the sink-preflight warning
  text, which both documented the give-up ending, changed with it.
- `max_failures` (the strike bound) is an operator knob — implemented (Wave 2
  T13, `a40c2c2`). It lives in config.toml `[daemon] max_failures` with its sizing
  rationale (the authoritative home, D7), is overridable per run with
  `MAX_FAILURES` via `resolve_int_env` (the `WRITE_PARALLEL` precedent), and
  is documented in README-technical's env table and `[daemon]` key list. The
  same number is the masquerade escalation threshold, so one knob moves both
  bounds together.
Forecloses: per-cell relitigating of the failure table; new error paths that
commit outcomes on failure; give-up counting for provider-shaped faults.

## D6 — Proxy 403 is a human answer, not a failure (2026-07-09, reaffirmed 2026-07-30)

**Status:** implemented (Wave 2; issue #28 Option A).

A 403 on a gated write means an operator said "not now": log one clean line,
count nothing, re-offer next cycle. A rejection can never cause
`agent/attempted`. (Also a corollary of D5, kept separate for the #28 trail.)

## D7 — Documentation authority map + altitude rule (2026-07-30)

**Status:** implemented for the restated settings blocks (Wave 0); the
remaining altitude/duplication sweep (env-table defaults, README literals and
quoted log lines) lands with Wave 3.

Every fact has exactly one home; other docs point rather than restate.
**CLAUDE.md** = constitution: identity, privacy posture, failure model, review
charter — principles, no literal values. **config.toml** = authoritative for
every operational value *and its rationale* (its comments). **README-technical**
= operational reference (env vars, config key meanings/constraints, procedures,
test coverage); it does not restate config.toml values. **README.md** = human
overview. Altitude rule: prose never quotes exact log strings or literal counts
outside runbooks.
Forecloses: duplicated fact tables; doc fixes that add a second home.

## D8 — docs/plans/ is frozen history (2026-07-30)

**Status:** implemented (Wave 0).

Plan/design docs are historical artifacts. Every file carries a one-line
status header (`proposed` / `executed` / `partially executed` / `superseded`,
with date); superseded plans' agent-directives are neutralized. The 2026-07-08
issue roadmap is frozen with a banner; GitHub issues are the living tracker.
New plans start with the status line.

## D9 — proxy_client.py / gmail_utils.py are a declared fork (2026-07-30)

**Status:** implemented (docs, Wave 0).

Both files were copied from email-agent (Feb 2026) and have since diverged
deliberately (error taxonomy, retry layer, body cleaning). They are
**independently maintained here with no sync obligation in either direction**,
and no re-convergence is intended. Never "sync" them with email-agent.

## D10 — Remove the eval web app (2026-07-30)

**Status:** implemented (Wave 2).

`evals/web_app.py`, `web_auth.py`, `web_data.py`, `run_web.py`,
`evals/templates/`, and evals/README.md §5 are removed; `fastapi`, `jinja2`,
`uvicorn`, and `python-multipart` leave the runtime dependencies
(`python-multipart` was web-only in fact — FastAPI form parsing — though
this entry originally omitted it; the `multipart` strings in gmail_utils.py
are MIME types, not the package). Coupled removals the implementer
must include or the suite fails: drop `evals.run_web` from
`tests/test_eval_cli_docs.py`'s `_CLI_MODULES` (it imports the module), and
delete the `### run_web` section of evals/README-technical.md. Implementation
found two doc edits beyond this list: the "plus a web UI" clause in
evals/README.md's intro and the web-UI sentence in README-technical's
Chain-of-Thought section (the tui-regression skill's venv pip line also
dropped the four packages). Rationale:
forgotten by the owner, workflows CLI-covered, zero tests, and it contradicted
the no-web-server posture at the dependency level. CoT capture/sidecars are
unaffected.

## D11 — Minimal release identity (2026-07-30)

**Status:** implemented (Wave 2).

The Dockerfile bakes the git SHA (build-arg), the daemon logs it once at
startup, and notable releases get a lightweight git tag. No semver, no
changelog — just enough that "what is deployed?" has an answer in the logs.

## D12 — Assessments JSONL: documented schema + version field (2026-07-30)

**Status:** implemented (Wave 2).

The assessments JSONL (the only durable copy of gradings) gets a schema
document (home: README-technical) and a `schema_version` field on each record,
so the next schema evolution is a version bump + migration, not shape-sniffing.

## D13 — Adopt CI (2026-07-30)

**Status:** implemented (Wave 1).

A minimal GitHub Actions workflow (uv sync, pytest, ruff) on PRs and pushes to
main. The suite is fully mocked; no secrets. Its absence was not deliberate.

---

*Backfilled entries — decisions adjudicated before this registry existed:*

## D14 — Theme labels are Emphasized-only (issue #53, 2026-07-08)

**Status:** implemented.

`agent/newsletter/theme/*` is applied only when a theme grades Emphasized;
Present is recorded in the JSONL but not labeled; Absent is omitted from the
record. Deliberately changed the label's meaning from "present" to
"emphasized".

## D15 — Newsletter rubric: Poor/OK/Good + Absent/Present/Emphasized (issue #53, 2026-07-08)

**Status:** implemented.

Dimensions score as Poor/OK/Good stored as ints 1/2/3; tier bands ≥2.75 /
≥2.25 / ≥1.75 on the four-dimension mean. Themes as theme→grade dicts. No
backward compatibility in readers (production JSONL is migrated via
`scripts/migrate_assessments.py`, which preserves tiers verbatim).

## D16 — Local tier: native thinking disabled, max_tokens 4096, coupled (issue #64)

**Status:** implemented.

Thinking is disabled on the local classifier (both dialect forms sent);
`finish_reason: length` raises loudly; local `max_tokens` = 4096 sized to the
honest scaffold demand. These three are one coupled decision — see
config.toml's `[llm.local]` comments (authoritative).

## D17 — Cloud tier: native thinking enabled, max_tokens 1024 (issue #64 era)

**Status:** implemented.

Deliberately asymmetric with D16: glm-5's accuracy rides its native reasoning
channel; budget kept at 1024 with the loud length-finish as backstop. See
config.toml's `[llm.cloud]` comments (authoritative). Disable only behind an
eval gate.

## D18 — Assessment record before labels (PR #65)

**Status:** implemented. (An instance of D5 Rule 1 before it was named.)

The JSONL record is written before Gmail labels commit; a sink fault leaves
the thread unprocessed rather than labeled-but-lost. Dedup on read: newest
`timestamp` per `thread_id` wins.

## D19 — HTTP 429 never halts the daemon (PR #61)

**Status:** implemented.

Rate-limit phrasing is indistinguishable from quota exhaustion; a wrong
restart-only halt is worse than retry. Balance-signature 402/400/403 halts the
function whose provider reported it — per-function under D5, implemented in
Wave 2 T9 (was daemon-wide). The halt was restart-only when this was decided;
since D22 it self-heals, which bounds the cost of a wrong halt to about an
hour plus a push — still worse than retrying a rate limit, so the 429
exclusion stands.

## D20 — Content-less grading is a failure, not an outcome (issue #30, 2026-07-08)

**Status:** implemented — the exception path with issue #30, the parse-to-None
path with D5's corollary in Wave 2 T11.

Stories-exist-but-every-grade-errored raises rather than recording an outcome;
so do stories-exist-but-every-grade-*unparseable* and an extraction reply that
parses to no stories without saying `NO_STORIES` (T11). The raise is a strike
candidate under D5's cycle-level attribution — counted only when correlation
blames the newsletter, so one that keeps failing while its siblings grade ends
findably under `agent/attempted`, while several failing the same way are held
as shared cause. A successful zero-story extraction remains a valid
`no-stories` outcome — the only one.

## D22 — Halts self-heal by re-probe; notification on halt and resume (issue #73, 2026-09-09)

**Status:** implemented (PR for issue #73).

Supersedes the "restart is the only reset" corollary of D5 (its text is kept
there as the record). Background: on 2026-08-25 the cloud provider answered a
single 403 with a balance signature while the account had funds; the daemon
halted on that one response and, with a restart as the only reset and nothing
reporting the halt, stayed halted for fourteen days. Four decisions, taken
together:

1. **Re-probe while halted.** A halted function sends one chat completion with
   its own request shape (the client's real `max_tokens`, `temperature`,
   `extra_body` and, for GLM, the thinking field — so a rejection that depends
   on the request fails the probe too) and a fixed innocuous prompt (no email
   content) through *its own* `LLMClient` — the client that raised, so email
   and newsletter probe their own providers — once per
   `halt_probe_interval_seconds` (config.toml `[daemon]`, authoritative; env
   override `HALT_PROBE_INTERVAL_SECONDS`; validated at startup). Due probes
   run concurrently with a short timeout (`HALT_REPROBE_TIMEOUT`), so the loop
   head stalls for at most one probe and the heartbeat stays fresh. A 200
   clears the halt, undoes any halt-time state (the email-only query
   narrowing) and resumes normal processing in the same cycle (the re-probe
   runs at the loop head, ahead of the poll), at INFO; anything else stays
   halted and logs below ERROR. The email function re-probes a cloud-tier
   client if any fault in the streak came from the cloud tier; the local tier
   is a paid provider only under D4's eval-only stand-in, and there the count
   is per function rather than per client — a cloud answer at Stage 1 resets
   a streak of local-tier faults. Accepted as a D4-only limitation.
2. **Consecutive balance faults trip the halt, not one.** The count is
   `balance_halt_strikes` in config.toml `[daemon]` (authoritative, with its
   rationale — D7's one-home rule; env override `BALANCE_HALT_STRIKES`;
   validated at startup). Any request one of the function's LLM clients
   answers *and the pipeline parses* resets the count — Stage 1 of the email pipeline included, and
   each LLM call of a newsletter grading. Per function. Two consequences,
   accepted: the count is per observed outcome, not per time, so faults
   arriving back-to-back within a single poll cycle (a few-second provider
   blip under `cloud_parallel` concurrency) can reach it and trip — the cost
   is bounded to one probe interval, after which the re-probe resumes the
   function; and a balance-shaped 400/403 specific to one thread is counted
   like any other, so while sibling threads answer it defers each cycle at
   one ERROR line without halting or striking, and alone in its cycles it
   reaches the count as a real outage would.
3. **Notification is opt-in by two env vars**, `NTFY_URL` (full topic URL)
   and `NTFY_TOKEN` (a bearer token minted for the labeler, not shared with
   another service). Either unset: one WARNING at startup, then the daemon
   behaves as before. A notification failure is logged and swallowed
   (test-guarded), so the poll loop continues.
4. **Push on halt and on resume.** Once per halt — not repeated hourly once it
   has landed; a push that fails to send is re-attempted on the probe cadence
   until one does — with provider tier, model, HTTP status, the matched
   balance signature (the short recognised phrase, e.g. `NOT_ENOUGH_BALANCE`)
   and the time; once per resume, with the downtime. The daemon adds no email
   content and no credentials; the provider's response body is not forwarded
   (a 400 can echo the rejected request) — the HTTP status and the matched
   phrase are.

Forecloses: reinstating restart-only halts; halting on a single balance
response; retrying the halted function's backlog on the poll cadence; an hourly ERROR or an
hourly push for an unchanged halt; forwarding the provider's response body in
a push; a balance-*endpoint* check replacing the completion probe (a separate issue records it as a possible complement — the
2026-08-25 fault was a completion 403 with funds present, which a balance
check would have called fine).
