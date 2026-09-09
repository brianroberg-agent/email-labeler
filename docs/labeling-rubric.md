# Email labeling rubric

**Status:** living document. This is the **specification** of what each label means and how
golden-set threads are labeled. The prompts in `config.toml` are an **implementation** of
this rubric for a particular model; when they disagree, the rubric wins and the prompt is
what changes. Owner: @brianroberg. Rules marked **Decided** carry the date of his ruling.
Rules marked **Proposed** have not been ruled on and bind nothing.

Started 2026-09-09, consolidating decisions that were scattered across PR #74, PR #75 and
the assistant's session notes of 2026-09-08.

---

## 1. Why a rubric comes first

- **The golden set is labeled by this rubric, before any model sees the thread.** With ~100
  labeled person threads there is no volume to learn rules implicitly (as a Bayesian filter
  would); the stated rules *are* the training signal. An unstated rule shows up as scatter
  across models, not as a model defect.
- **Two kinds of failure, kept separate.** (a) The rubric does not determine the label →
  only a better rule fixes it. (b) The rule is clear and a model still fails to apply it →
  a prompt or model problem; wording, worked examples and model choice are the levers.
  Rules first, so that every remaining miss is identifiably kind (b).
- **The prompt can be no better than the rubric.** That is a ceiling, not a claim that
  prompt work is worthless below it. (2026-09-08: the "known person reasoned down to
  LOW_PRIORITY by elimination" loss was kind (b) — the intent existed, the prompt let every
  model route around it — and one sentence closed it.)
- **Labels are a function of the email and the sender relationship only** — never of
  Brian's state at the time (what is already on his calendar, whether he already did the
  thing). Otherwise the set encodes noise no model can learn. **Decided 2026-09-08.**

## 2. The categories

| label | meaning | production effect |
|---|---|---|
| **NEEDS_RESPONSE** | An **obligation that Brian's office must discharge** — Brian himself or his assistant. Not "a reply is owed": the test is *"would doing nothing about this be a defect?"* | stays in the inbox, in the separated needs-response view |
| **FYI** | Worth reading; doing nothing about it is not a defect. | stays in the inbox |
| **LOW_PRIORITY** | Archive unread. Test: *"would I be content never to have seen this?"* | archived |

Definitions **Decided 2026-09-08** (frame) and confirmed 2026-09-09.

## 3. Error costs are asymmetric — score by class, never by one number

Worst first (**Decided 2026-09-09**):

1. Person mail sent to LOW_PRIORITY — hidden, archived unread.
2. needs_response read as FYI — drops out of the focused view; action missed or late.
3. FYI read as needs_response — clutter in the focused view.
4. A cold pitch kept in the inbox.

Evaluation reports lead with (1) and (2). A single accuracy figure hides the only errors that
cost anything. **Proposed:** when a thread is genuinely undecidable, the *production* labeler
should choose the cheaper error — keep it, and lean toward needs_response.

## 4. Stage 1 — sender type

- **Decided 2026-09-08 (PR #74):** calendar invitations and other notifications a service
  sends on a person's behalf (Google Calendar invitations, updates, cancellations; document
  share and comment notifications) are **SERVICE** even though From carries the person's
  name — the person chose to send it, the service wrote the words. Golden-set convention
  already 20/20.

## 5. Stage 2 — label rules for PERSON senders

**Decided:**

- **P1 (2026-09-08, PR #75).** A person's mail is at least FYI unless it is unsolicited sales,
  partnership or cold outreach from someone with no existing relationship, which is
  LOW_PRIORITY. There is no other route to LOW_PRIORITY for a person, however dull the mail.
- **P2 (2026-09-08).** An ask addressed to Brian individually is NEEDS_RESPONSE even if he
  will decline.
- **P3 (2026-09-08).** A **broadcast conditional ask** ("could anyone give me a ride?") is
  FYI. The distinction is *"each of you"* (everyone is expected to act → obligation) versus
  *"any of you"* (a volunteer is sought → no obligation on Brian).
- **P4 (2026-09-08).** **Information Brian must act on**, with no reply needed beyond
  thanks, is NEEDS_RESPONSE when the action goes beyond updating his own awareness.
  A calendar tweak = FYI; a quote he must evaluate = NEEDS_RESPONSE.
- **P5 (2026-09-08).** A calendar **invitation** is NEEDS_RESPONSE — the obligation exists
  and is *delegated* to the assistant. **Updates and cancellations of already-accepted
  events are FYI.** (Invitations are SERVICE at stage 1, so this reaches the service path;
  recorded here because it defines the category.)

- **P6 (2026-09-09).** Group mail that asks **every** recipient to act (RSVP, vote, complete
  an evaluation, register) is NEEDS_RESPONSE — the "each of you" half of P3 made explicit.
  Golden-set consequence: "[DM Staff] Fall Conference Registration: A Few Clarifications"
  ("please register yourself ASAP") flips fyi → needs_response, matching the conference
  eval thread.
- **P7 (2026-09-09).** A request addressed to someone else with Brian copied is FYI.
- **P8 (2026-09-09).** A known-person **newsletter or prayer letter** (family, pastor,
  supported missionary) is FYI, never LOW_PRIORITY — a named instance of P1, because models
  still read "newsletter" as a LOW_PRIORITY trigger. Brian wants this paired with a **list of
  recognised newsletter senders** so the model can tell a known relationship's newsletter
  from a stranger's — the same "preference as data" mechanism as S2 below.
- **P9 (2026-09-09).** **Survey and feedback requests** are NEEDS_RESPONSE when they come
  from DiscipleMakers or Oakwood; from just about anyone else they are LOW_PRIORITY.
  Relationship-based — a candidate for the same known-senders list.

**Proposed, not yet ruled:** none open on the person side as of 2026-09-09.

## 6. Stage 2 — label rules for SERVICE senders

**Current prompt (step 4), unchanged:** if no action is needed, is it worth reading for
awareness? Payment receipts, transaction confirmations and curated editorial newsletters
are FYI; automated product updates, account alerts, marketing and digests are LOW_PRIORITY.
Cold outreach is LOW_PRIORITY even when it contains a question or a call to action.

**Open problem (Brian, 2026-09-09):** this is the squishiest boundary in the scheme. For
persons, P1 made LOW_PRIORITY crisp; for services, "worth reading for awareness" reduces to
how much Brian wants to see a given email's content, which is a preference, not a rule a
model can infer from the thread.

**Decided 2026-09-09 (voice, walked through as decision questions so the work is ready when
the cloud-tier run can happen):**

- **S1 — "about me" versus "about them".** If the email *reports something that happened
  to Brian* — a charge, an order, a delivery, a statement, a security or account change, a
  deadline on something he holds — it is FYI. If it exists to *draw his attention to the
  sender's own content* — newsletter, digest, product update, promotion, re-engagement — it
  is LOW_PRIORITY. Brian accepts the consequence that security/account-change alerts about
  him, LOW_PRIORITY in today's prompt, become FYI.
- **S1a — three-way split.** "About me" *and requiring action* (renewal, signature, verify,
  failed payment) is NEEDS_RESPONSE (the service analogue of P4); "about me" record-only is
  FYI; "about them" is LOW_PRIORITY unless the sender is on the keep-list (S2).
- **S1b — tiebreaker against sales dressed as alerts** (accepted to try; judge by the
  service-slice run). "About me" must name a **specific completed event or a specific
  consequence to something Brian holds** — an amount, an order number, a date, a device,
  an expiry. Generic "review your account / manage your licenses / unlock…" names nothing
  and is "about them". (Sales reps at vendor addresses — Adobe was the example — are
  PERSON at stage 1 and fall to P1 cold outreach already.)
- **S2 — the keep-list is a deterministic code LOOKUP, not prompt text.** Sender on the
  list → FYI, no model call. Brian chose determinism over the model's ability to generalise
  to a new address; making the list cheap to update is the follow-on once it exists.
- **Golden set, service side:** do **not** relabel the 273 reviewed service threads
  wholesale. Run the models (cloud tier — a docker-3 job), then adjudicate only the
  model-versus-label disagreements, exactly as on the person side. Brian: "the bulk of the
  labels are going to be fine; isolate those at the boundaries."

**Proposed:** none open on the service side as of 2026-09-09; S1b is provisional pending
evidence.

## 7. Golden-set procedure

- **Label by the rubric, blind to model output.** Use `evals.review` (blind mode is the
  default). Record the rule applied, or the doubt, in the thread's notes when it is not
  obvious.
- **Exclusion.** A thread is excluded when it stays ambiguous *after* a good-faith attempt
  to state the rule: the rule that would decide it is arbitrary, or the answer depends on
  something not in the thread. Hard-but-decidable threads stay in — they are where models
  diverge and are the most valuable records. **Track the excluded fraction and report it**;
  if it grows large, the category scheme is the finding. **Decided 2026-09-09** (approach
  agreed in conversation; the wording here is the assistant's).
- **Proposed:** keep excluded-as-undecidable person threads in a separate bucket scored
  only on the safety rule of §3 (not archived; not dropped from needs_response when
  needs_response was plausible) rather than on exact label.
- **Adjudication when models disagree with a label.** For each disputed thread, rule one
  of three ways: the label was wrong (fix it); the label is right and the rule can be
  stated in one sentence that also covers the neighbouring cases (add it to §5/§6); or it
  is a coin flip (exclude). Never write a rule aimed at making the model pick the label —
  write the rule you were applying. Whether models follow it is the eval's job.
- **Mechanics.** Edit labels with `evals.review --edit --sender-type person` (`l` label,
  `e` toggle exclude, `s` sender type; notes only in first-pass review mode). The atomic save
  resets the file to mode 600 root — `chmod o+r` if another user must read it. Relabeling
  costs nothing to re-score: the LLM response cache is keyed on model + prompt + params, so
  re-running baselines against the edited set is instant.
- **Held-out split.** Prompt candidates are designed on the *tune* half only and judged on
  the *validation* half. The split is by thread id and survives relabels; excluded threads
  drop out. Current split: 64 tune / 32 validation of 96 person threads (2026-09-08).
- **Noise floor.** At n≈96, one figure carries ≈±4.5 points; differences under ~8 threads
  overall or ~4 on a 32-thread half are not distinguishable from chance.

## 8. Change log

- **2026-09-09 (later)** — P6–P8 ruled Decided, P9 added, S1/S1a/S1b/S2 ruled Decided with
  the service golden-set procedure, all by Brian by voice on 2026-09-09 morning.
- **2026-09-09** — document created; §1–§7 consolidate PR #74, PR #75, the 2026-09-08
  review rulings and the 2026-09-09 eval discussion. Eval evidence for the current state:
  the assistant's `docs/labeler-eval-report-2026-09-09.md` (workspace repo).
