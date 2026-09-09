# Newsletter grading rubric

**Status:** living document, started 2026-09-09. This is the **specification** of what the newsletter
grader's outputs mean — story extraction, the four storytelling dimensions, the tier, and the five
Ends-Statement themes — and of how golden-set stories are labeled. The prompts in
`config.toml [newsletter.prompts.*]` are an **implementation** of this rubric; when they disagree, the
rubric wins and the prompt is what changes (the same relationship `docs/labeling-rubric.md` states for
email triage; registry entries D21 and D23). Owner: @brianroberg. Every rule is marked **Decided**
(with the date of his ruling), **Implicit** (what the code and prompts do today, transcribed from source,
not yet ruled on), or **Proposed** (binds nothing).

Sources for the Implicit material: `config.toml:173-290` at `f7245f9`; `docs/decisions.md` D3, D14,
D15, D18, D20; `newsletter.py` (`compute_tier`, `aggregate_theme_grades`); `daemon.py:834-839`
(best-story tier); `labeler.py:200-235` (Emphasized-only labels); commits `e962234` (2026-02-19),
`7759dea` (2026-03-06), `08f56a6` (2026-07-07), `f7a1cbf` (2026-07-08, issue #53); issue #8.

---

## 1. Purpose — Decided 2026-09-09

In Brian's words:

> Helping me identify stories that are candidates for the DM newsletter is just one slice of it. More
> generally, I want to help our staff grow in their storytelling, including alignment of the stories they
> select with the subpoints of our Board Ends. Classifying newsletters in these two dimensions will help
> me in that teaching/training task that's part of my Comms role.

Two uses, one primary:

1. **Coaching (primary).** Per-story grades on four storytelling dimensions and on five Ends subpoints
   are the raw material for Brian's training of staff: what a story does well, what it lacks, and whether
   what it is *about* lands on an Ends subpoint.
2. **Shortlisting (a slice).** Tier and Emphasized-theme labels let Brian filter a year's newsletters for
   DM-newsletter candidates (`python -m newsletter_review --tier/--theme`).

**Audience — Decided 2026-09-09.** Brian, who may share data with Peter or other staff trainers; not
published to staff. Label names may stay in the grader's shorthand, but every reason sentence (§6) must
stand alone for a peer trainer reading it cold: name the test applied, quote the evidence. Revisit if
output is ever shown to staff directly.

### What follows from the purpose

- **Unit of analysis is the story.** Coaching is per story. The newsletter-level Gmail label (§4) is a
  shortlist filter, not a grade of the newsletter.
- **Dimension grades matter more than the tier.** The tier is a band over the mean of four grades; for
  coaching the four grades are the signal. Evaluation leads with per-dimension agreement.
- **Present vs Emphasized is the coaching distinction.** "You touched Scripture; make it the point" is the
  feedback. The boundary needs a test (§5) and its confusions need their own column in the eval (#55).
- **Error costs — Decided 2026-09-09: coaching governs.** Worst first:
  1. A grade that would mis-teach: GOOD given to a story that lacks the dimension; EMPHASIZED given where
     the theme is merely present. Over-grading is the costly direction.
  2. Under-grading a genuinely strong story on a dimension (discouraging; loses a shortlist candidate).
  3. A missed theme (Present/Emphasized graded Absent).
  4. Tier-only errors with the dimensions right (a summary artefact).
  5. Poor↔OK swaps at the bottom.
  The shortlist-only view (under-tiering worst) is subordinate: a strong story under-graded one step still
  lands in good/excellent, and the shortlist is read by a person.
- **When torn, grade down and say why — Decided 2026-09-09.** A lower grade with a stated reason is
  coachable; an inflated grade is invisible as an error. Themes already resolve every undecidable path to
  no label (D14), which is the same rule.

## 2. Extraction — what is a story

**Implicit (`config.toml:221-236`):** *"A 'story' is a narrative segment about people, events, or ministry
activities."* Skipped as non-story: headers, footers, signatures; donation appeals or fundraising asks;
event calendar listings; administrative announcements; contact information. No stories → `NO_STORIES`.

- **Decided 2026-07-07 (`08f56a6`):** stories are identified by their text; the model does not invent a
  title, because a headline swayed the score.
- **Decided (D20, 2026-07-08):** a successful zero-story extraction is a valid `no-stories` outcome — the
  only content-less outcome that commits.
- **Gap:** nothing says how to split adjacent narrative segments; the golden-set curation disagreements
  recorded in `docs/newsletter-label-ux-redesign.md` are about exactly this.
- **Proposed — splitting rule:** one story = one person's or group's arc; a change of protagonist or of
  time-frame starts a new story; a paragraph that only comments on another story belongs to it.
  Extraction truth is the golden set's fixed input: a wrong split is corrected in the golden set, not
  tuned around in grading.

## 3. Storytelling dimensions

**Decided 2026-07-08 (#53, Brian's wording; `config.toml:242-246`).** The grade answers *"How well does
this story demonstrate this particular dimension of storytelling?"*:

| grade | anchor |
|---|---|
| POOR | The story does not display this dimension of storytelling in any meaningful way. |
| OK | The story displays this dimension, but only partially or inconsistently. There is significant room for improvement. |
| GOOD | The story consistently displays this dimension. While not necessarily perfect, it could serve as an example of what this dimension means. |

**Implicit (`config.toml:249-252`):**

| dimension | question |
|---|---|
| SIMPLE | Does the story focus on one key idea or progression, with no tangents or extraneous details? |
| CONCRETE | Does the story narrate particular events involving particular people at particular times and places, with vivid specifics rather than abstractions? |
| PERSONAL | Does the story center on one person or a few people, such that what matters to them is what matters to the story — rather than events, numbers, or ideas disconnected from people? |
| DYNAMIC | Does the story describe how a person changes over time — a clear arc of transformation rather than a static account with no before-and-after? |

- **Gaps:** the anchors are generic (no per-dimension statement of what OK vs GOOD looks like); "vivid",
  "meaningful", "significant" and "could serve as an example" are judgment words without tests; GOOD is
  defined by exemplar and no prompt carries an exemplar.
- **Decided 2026-09-09 — two kinds of boundary, two instruments.** *Qualification* (does the story
  display the dimension at all: is there a particular moment, is there a before-and-after) is shown by
  **short excerpt examples**; minimal pairs — the same sentence with and without the qualifying feature —
  **may be written** rather than found. *Degree* (OK vs GOOD) is decided by a **test**, not an example.
- **Proposed — degree test:** subtraction — *GOOD if the dimension carries the story (remove it and the
  story loses its effect); OK if the dimension is displayed but the story would read much the same without
  it.* Brian's ruling: this is *a* test, not yet *the* test; others may be added beside it.
- **Proposed — qualification sentences,** in the shape CHURCH already has for themes:
  - CONCRETE qualifies when the passage names a particular person, place and moment; a passage of only
    summary or abstraction does not.
  - DYNAMIC qualifies when the passage states or shows a before and an after for a person; a description
    of a state, however detailed, does not.
  - PERSONAL qualifies when the story's stakes are a named person's stakes; a passage whose stakes are a
    program's, a number's or an idea's does not.
  - SIMPLE qualifies when every paragraph serves one progression; a paragraph that could be deleted
    without loss is a tangent.

## 4. Tier

**Decided (D15, 2026-07-08).** POOR/OK/GOOD → 1/2/3; tier = band over the mean of the four dimensions:
excellent ≥ 2.75 · good ≥ 2.25 · fair ≥ 1.75 · else poor (`newsletter.py:202-215`). With four grades the
reachable means make this: excellent ⇔ sum ≥ 11 (at most one OK, no POOR); good ⇔ sum ≥ 9; fair ⇔
sum ≥ 7. Rationale on record: "all-OK = 2.0 → fair; one Poor among three Good ≈ 2.5 → good."

**Decided 2026-09-09.** The newsletter's Gmail tier is its **best** story's tier and its theme labels the
strongest grade seen across its stories (`daemon.py:834-839`, `newsletter.py:222-227`, D14). This is a
**shortlist filter, not a grade of the newsletter**; the coaching view is the per-story record, read
through `newsletter_review`. A newsletter with no stories is `no-stories`.

- **Note:** "Poor" names both a tier and a dimension grade — same word, different scales.

## 5. Ends-Statement themes

**Decided 2026-07-08 (#53, Brian's wording; `config.toml:265-269`, `:279`).** ABSENT — the theme is
not present in the story at all · PRESENT — present but underdeveloped · EMPHASIZED — well-developed and
highlighted / a substantial part of what the story is about.

**Decided (D14, 2026-07-08).** Gmail theme labels only for EMPHASIZED; PRESENT is recorded, not labeled;
ABSENT omitted. **Decided 2026-09-09:** D14 is a cost rule — a wrong theme label costs more than a
missed one — and its registry text should say so.

| theme | current text | boundary test? | provenance |
|---|---|---|---|
| SCRIPTURE | Study the Scriptures correctly, apply them to all of life, and teach them to others | **none** — the bare Ends bullet; issue #8 documents reading-vs-handling confusion | Implicit since 2026-02-19 |
| CHRISTLIKENESS | Exhibit increasing Christlikeness in response to the Gospel. Look for a specific person growing in character, attitude, or behavior in a way that reflects Christ — not merely faith or spiritual activity mentioned in passing. | yes | **Decided 2026-03-06** (`7759dea`); "must" → "Look for" 2026-07-08 |
| CHURCH | A college student actively participates in or serves at a local church (not just a campus fellowship). Look for substantial involvement in a local church community during college; a passing mention of attending church or a church name does not qualify. | yes; its positive example was removed in #53 and the softened wording not re-validated (the 2026-03-06 validation on 65 stories ran with the example present) | **Decided 2026-03-06** (`7759dea`) |
| VOCATION_FAMILY | Honor God in their vocation and family relationships | **none** | Implicit since 2026-02-19 |
| DISCIPLE_MAKING | Continue to make disciples wherever God takes them | **none** | Implicit since 2026-02-19 |

- **Gap:** the PRESENT/EMPHASIZED boundary is worded two ways in the same prompt ("well-developed and
  highlighted" vs "a substantial part of what the story is about") and neither is tested.
- **Decided 2026-09-09 — instruments, as in §3.** *Qualification* (does this passage count as the theme
  at all — #8's reading vs handling) is shown by short excerpt examples, minimal pairs may be written.
  *Degree* (PRESENT vs EMPHASIZED) is decided by a **test, not an example**: any PRESENT excerpt is a
  substring of some EMPHASIZED story, so an example cannot carry the boundary.
- **Proposed — degree test (a test, not the test):** *EMPHASIZED if removing the theme leaves the story
  without its point; PRESENT if the story stands without it.*
- **Proposed — qualification for the three bare bullets,** SCRIPTURE first (closes #8): *qualifies when a
  person's handling of Scripture changes — they read, apply or teach it differently; Scripture appearing
  as an activity ("we read Romans together") does not qualify.* VOCATION_FAMILY and DISCIPLE_MAKING need
  Brian's sentence.
- **Scarcity is a finding, not a defect — Decided 2026-09-09.** Some themes will have few or no
  EMPHASIZED stories in a year of staff newsletters. The eval reports those cells as "no support", never
  0%; the review browser should show EMPHASIZED counts per theme per year, because that count *is* the
  coaching agenda. Exemplars of GOOD and EMPHASIZED may come from stories Brian selected and edited for
  the DM newsletter; for a theme with none, Brian may author the exemplar (teacher-written exemplar =
  coaching material), or the theme carries a qualification example and a degree test with no exemplar.
- **Proposed:** single-source the theme list — it exists in three copies today (the prompt,
  `_VALID_THEMES` in `newsletter.py`, `[newsletter.labels.themes]`).

## 6. Rationale — Decided 2026-09-09

The grader returns **one sentence per dimension and per theme naming the test it applied**, stored in
the assessment record alongside the grade (D18 already writes the record before labels). For coaching,
the reason is as much the product as the grade; for adjudication (§7) it is the evidence. Written for a
peer trainer reading cold (§1 audience).

## 7. Golden-set procedure

- **Label by the rubric, blind to model output.** `evals.newsletter_label` shows no predictions (Phase A
  seeding was removed, #59). Label the four dimensions, the five themes, and — **Decided 2026-09-09** — a
  one-line note per story stating the rule applied where it is not obvious (`GoldenStory.notes`; the TUI
  needs a hotkey for it).
- **Exclusion.** A story is excluded (`u`) when it stays ambiguous *after* a good-faith attempt to state
  the rule, or when it has been used as a prompt example (so the eval never scores text the model was
  shown — note it "prompt example"). Hard-but-decidable stories stay in. The excluded count is tracked
  and reported. A newsletter is excluded (`X`) when it is not a staff newsletter at all.
- **Adjudication.** When a model's grade disagrees with a label, rule one of three ways: the label was
  wrong (fix it); the label is right and the rule can be stated in a sentence that also covers the
  neighbouring stories (add it to §3/§5); coin flip (exclude). Never write a rule aimed at making the model
  pick the label. Fixed golden stories make this cheap: a relabel never touches extraction truth, and the
  response cache makes re-scoring free.
- **Size and noise (measured 2026-09-09):** 50 newsletters harvested, 15 reviewed, **19 graded
  stories**, 9 EMPHASIZED labels across five themes. At n≈19 one figure carries ≈±10 points and per-class
  cells cannot support a metric: no tune/validation split yet, no prompt A/B distinguishable from noise.
  The 35 harvested, unreviewed newsletters are the next labeling pass and should be labeled against this
  rubric. A split, when the set is large enough (≥150 stories), is by newsletter `thread_id`, not story —
  stories in one newsletter share author and style.
- **Evaluation** leads with per-dimension agreement and per-theme direction (EMPHASIZED→PRESENT vs
  →ABSENT vs PRESENT→EMPHASIZED, #55), prints `n` beside every figure and `N/A` where a denominator is
  zero, and records `prompt_hash` (already in `NewsletterRunMeta`).

## 8. Change log

- **2026-09-09** — document created: purpose, audience, cost order, grade-down rule, best-story label as
  shortlist filter, rationale, example/test instruments, scarcity ruling, golden-set procedure (Brian's
  rulings of 2026-09-09); §2–§5 transcribed from the prompts, D14/D15/D20 and commit history with
  provenance.
