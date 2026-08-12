# Coherency and voice review: `colm26_normative-simulacra`

Read pass over the five body sections (`01_intro`, `02_rw`, `03_methods`,
`04_results`, `05_discussion`) on 2026-08-08, extended on 2026-08-09 to the
appendices `A_additional-methods`, `B_additional-results`, and
`B_reranker_judge_ablation` (§6 below). Read against `academic-writing-mini.md`
and the `no-ai-slop` pattern set. `C`, `D`, and `E` are excluded by request.

This is the half of the copyedit that `scripts/copyedit.py` cannot do. A linter
reads one sentence at a time, so it cannot see that a term is spelled three ways
across three files, that a promise made in §1 is never paid off, or that a
sentence is grammatical and still says the wrong thing. Everything below was
verified by grep against the source; each item names the pattern, quotes the
line, and gives the fix in a few words. Nothing here has been rewritten for you.

Line numbers are from the 2026-08-08 working tree and drift with edits.

---

## 1. Blocking: wrong word in a defined term

- [x] **FIXED 2026-08-09.** **`01_intro.tex:37` — GRPO was expanded wrongly.**
  > SFT, Group Relative **Privacy** Optimization (GRPO) \cite{shao2024deepseekmath}

  Every other expansion in the paper reads *Policy*: `02_rw.tex:28`,
  `03_methods.tex:57`, `B_additional-results.tex:398`. This is the acronym's
  first expansion in the paper, so the one place a reader learns what GRPO
  stands for is the one place it is wrong.
  **Fix:** `Privacy` → `Policy`.

- [x] **FIXED 2026-08-09.** **`03_methods.tex:26` — set braces did not render.**
  > `$a\in{\mathrm{appr.},\mathrm{inappr.}}$`

  In math mode `{...}` is a grouping, not a set. This typesets as
  *a ∈ appr., inappr.* with no braces, which reads as a malformed expression
  right where the paper defines its central object.
  **Fix:** `\{` and `\}`.

## 2. Terminology drift

The same object under more than one name. Each of these was confirmed by grep.

- [x] **FIXED 2026-08-09** (standardized on `fiction10`). **The 10-novel corpus was spelled three ways.** `fiction-10`
  (`03_methods.tex:9`, where it is *introduced*), `fiction 10`
  (`04_results.tex:10`), and `fiction10` (everywhere else, ~40 occurrences in
  `B_additional-results.tex`).
  **Fix:** pick one, almost certainly `\textit{fiction10}`, and change the two
  outliers. The introduction site is the one that currently disagrees with the
  convention.

- [x] **FIXED 2026-08-09** (standardized on `CI-RL Vignettes`). **The CIRL benchmark was spelled four ways.** `CI-RL Vignettes`
  (`03_methods.tex:76`, `A_additional-methods.tex:23`), `CI-RL model`
  (`04_results.tex:18`), `CIRL` (`04_results.tex:20`), `CIRL-Vignettes`
  (`05_discussion.tex:38`).
  **Fix:** one spelling for the benchmark and one for the model, used
  consistently. Note these are two different things, so the fix is not a single
  search-and-replace.

- [x] **FIXED 2026-08-09** (standardized on `Group-Relative`). **`Group Relative` vs `Group-Relative`.** Unhyphenated at `02_rw.tex:28`,
  hyphenated at `03_methods.tex:57` and `B_additional-results.tex:398`.
  **Fix:** match the DeepSeekMath paper's own styling and use it throughout.

- [x] **WON'T FIX** (author ruling 2026-08-09: two separate codebases are intended). **Two artifact repositories.**
  `04_results.tex:10` links `github.com/pendulating/normative-simulacra`;
  `E_prompts.tex:3` links `github.com/pendulating/trawler`.
  **Fix:** if both are real, say what each holds; if not, correct one.

- [x] **FIXED 2026-08-09.** **`five benchmarks` vs `the 5 benchmarks`.** Words at `01_intro.tex:45`
  and `03_methods.tex:76`, numeral at `04_results.tex:35`.
  **Fix:** spell out small counts throughout.

## 3. Claims that do not line up across sections

The three still-open items here were reviewed on 2026-08-09 and judged not worth
addressing for the camera-ready.

- [ ] **Two different "four of five" claims.** `04_results.tex:20` says four of
  five benchmarks have a metric where no model can be isolated (a *noise*
  claim). `05_discussion.tex:28` says four of five carry a flagship flaw (a
  *design* claim). Same numerator, same denominator, different propositions,
  ~20 pages apart. A reader will read the second as a restatement of the first.
  **Fix:** signpost that these are different counts, or drop one framing.

- [ ] **The §5 benchmark critique enumerates four items but never numbers
  them,** and one of the four (`05_discussion.tex:33`, CI-CoT) is a prompting
  method applied *to* PrivacyLens, not a fifth benchmark. VLM-GeoPrivacy, the
  benchmark that is presumably the unflawed one, is never named as such.
  **Fix:** the house guide's own `First,… Second,… Finally,` signposting, and
  one sentence saying which benchmark is exempt and why.

- [x] **FIXED 2026-08-09.** **`04_results.tex:18` referenced the same table twice in one paragraph
  with different framing.**
  > \autoref{tab:benchmark_results} compares the canonical model set … The full
  > cross-model evaluation table can be viewed at \autoref{tab:benchmark_results}

  **Fix:** delete the second mention.

- [ ] **Three appendix figures are never referenced from any text**
  (`fig:corpus-composition`, `fig:corpus-norm-attributes`, `fig:weight-deltas`),
  so LaTeX places them wherever it likes and no sentence tells the reader what
  to look at. `fig:weight-deltas` is the evidence for the §4.4 weight claim.
  Reported mechanically as `UNREFERENCED_FLOAT` in `COPYEDIT_REVIEW.md`.

## 4. Grammar the parser missed

**ALL FIXED 2026-08-09.** Long-distance agreement, which LanguageTool does not
reliably catch. Kept here as the record of what changed.

- [x] **`01_intro.tex:2` — subject-verb.**
  > The emergence of personified, persistent-state instantiations of Large
  > Language Models (LLMs), often described as ``agents''~\citep{...}, **raise**
  > important questions

  The subject is *emergence*. **Fix:** `raise` → `raises`.

- [x] **`05_discussion.tex:9` — subject-verb.**
  > Debates on norms **have**, unsurprisingly, **intensifies** when attempting…

  **Fix:** `intensifies` → `intensified`. The trailing clause also dangles
  (*attempting* has no subject).

- [x] **`01_intro.tex:18` — number.** "fulfill its users' need" → `needs`.

- [x] **`05_discussion.tex:11` — misplaced terminal period.**
  > to maintain this relationship (invariant, as it were**.)**

  **Fix:** `.)` → `).`

- [x] **`05_discussion.tex:11` — missing Oxford comma**, which the house guide
  requires: "ends, purposes and values" → "ends, purposes, and values".

- [x] **`05_discussion.tex:4` — broken list punctuation.**
  > directly prescriptive media are more obvious sources, **including,** legal
  > briefs, written law, …

  **Fix:** delete the comma after *including*.

- [x] **`01_intro.tex:44` — "in lieu of" means *instead of*, not *in the absence
  of*.**
  > In lieu of advancements in privacy law, norms regulate

  **Fix:** "Absent advances in privacy law…" or "Until privacy law catches
  up…". Keep the punchy `Why norms?` before it: see "Deliberately not flagged"
  in §5.

- [x] **`02_rw.tex` vs `03_methods.tex` — `\paragraph` heads are punctuated
  inconsistently.** `02_rw.tex` ends every one with a colon
  ("Behavioral alignment in agents:"); `03_methods.tex:9` ends with a period
  ("Fiction novels."). **Fix:** pick one.

## 5. AI-slop patterns

**ALL FIXED 2026-08-09** (applied by a subagent running the `no-ai-slop` skill;
edits verified against the source afterwards). Scored against the `no-ai-slop`
set. The manuscript is largely clean: it is specific, numerate, and opinionated,
which is the opposite of slop. Five hits.

- [x] **Fake-profound kicker** — `05_discussion.tex:11`, final sentence.
  > In demonstrating the systematic relationship between good contextual norms
  > and contextual values the wildly different microcosms of fictional worlds
  > hold wisdom for the real one.

  A closing metaphor in place of a claim, and it is also a dangling modifier
  with a missing comma. **Fix:** delete it and end on the *1984*/Darcy sentence
  before it, which is concrete and does the real work.

- [x] **Restated non-sequitur** — `05_discussion.tex:4`.
  > literary fiction offered distinctive and intriguing opportunities. **Where
  > norms are articulated and debated; they should be examined.**

  This repeats "where norms are articulated and debated" from the clause
  immediately before it, misuses the semicolon, and asserts nothing the
  paragraph does not already say. **Fix:** cut the sentence.

- [x] **Empty importance-puffery adjectives** — `05_discussion.tex:4`,
  "distinctive and intriguing opportunities". The actual reason arrives in the
  next sentence (fiction builds self-contained social worlds).
  **Fix:** lead with the reason and drop the adjectives.

- [x] **Vague throat-clearing topic sentence** — `04_results.tex:3`.
  > We now present layered results around the norm distribution of our two
  > corpora and…

  "layered results around" says nothing. **Fix:** state the finding, per the
  guide's "lead each paragraph with a topic sentence stating the claim".

- [x] **`04_results.tex:25` — "yields significant increase"** is both missing an
  article and using *significant* non-statistically, five lines after a
  paragraph about statistical validity and noise floors. Given how carefully
  this paper gates its claims, this word will read as a statistical claim.
  **Fix:** "yields a marked increase", plus the effect size.

### Deliberately not flagged

These match `academic-writing-mini.md` and are Matt's voice, not slop. Left alone
on purpose so a later pass does not "fix" them:

- `01_intro.tex:44` **"Why norms?"** — the guide explicitly calls for
  "occasionally a one-word 'Why?'". Keep.
- `01_intro.tex:30` **"Our central insight:"** — a colon reveal, but the guide
  prescribes "Here, we propose/show" pivots and colon-introduced elaborations.
  Keep.
- `05_discussion.tex:43` **"First,… Second,… Third,… Fourth,… Fifth,"** — the
  guide's prescribed signposting. Keep. (There is a stray double space after
  "Fifth,".)
- `05_discussion.tex:49` **"To conclude, we contribute…"** — `no-ai-slop` warns
  against summary-recap endings, but a short contribution restatement is a
  conference-paper convention and the guide asks for a bolder close. Keep.

## 6. Appendices A, B, and B-reranker

Read 2026-08-09. These are the strongest prose in the manuscript: dense,
numerate, and unusually careful about what the evidence does and does not
license. I re-derived the arithmetic and it holds throughout, including the
weight-delta learning-rate budgets (GRPO $450/540 = 0.83$, KTO
$5{\times}10^{-6}{\cdot}627 / 2{\times}10^{-5}{\cdot}540 = 0.29$), the KTO step
count against its preference-set size ($20{,}059 / 32 = 627$), the pooled norm
count ($10{,}034 + 53{,}492 = 63{,}526$), and the $\kappa$ deltas in
`B:645-653` against the numbers in §4.3. The findings below are defects of
consistency and wording, not of substance.

### Definitions that contradict each other

**ALL FIXED 2026-08-09.** The CI tuple is five fields (author ruling); the
quality-screen figure in §3 was raised to "fewer than 5\%" to match Appendix B.

- [x] **"eleven model families" should be "eleven models".** The paper has
  **11 models across 5 families** (`01_intro.tex:45`). Four places call the 11
  "model families":
  `A_additional-methods.tex:66`, `B_additional-results.tex:343`,
  `B_additional-results.tex:371`, and `tables/sft_training_summary.tex:5`.

  Two of them are self-contradicting as written. `B:371` reads
  > diagnostics for all eleven **model families** … Colour encodes backbone
  > **family** and line style distinguishes **variants within a family**

  which cannot both be true. And `sft_training_summary.tex:5` says "one row per
  model **family**" and "Ten of the eleven **models**" in the same caption.
  **Fix:** "models" in all four, keeping "family" only for the five backbones.

- [x] **The CI tuple is five fields, but `A:96` retrieves over six.**
  `03_methods.tex:18` and `02_rw.tex:11` define the I-tuple as $(s,r,u,a,t)$,
  and `A_additional-methods.tex:87` requires "all **five** non-empty CI tuple
  components". Nine lines later, `A:96` retrieves "by cosine similarity over
  the flow's **six** CI fields".
  **Fix:** if a sixth field is genuinely embedded (the flow description, say),
  name it; otherwise correct to five.

- [x] **The quality-screen pass rate disagrees between §3 and Appendix B.**
  `03_methods.tex:36` says the screen flags "fewer than 2\%" of norms or flows.
  `B_additional-results.tex:167` says it "passes $95.6\%$ and $99.5\%$ of norms
  respectively", i.e. **4.4%** flagged on `fiction10`, more than twice the §3
  claim.
  **Fix:** reconcile. I cannot tell from the text which is right, and the two
  may be measuring different objects (norms vs flows, pre- vs post-abstraction);
  if so, say which.

### Terminology drift inside the appendices

- [x] **The same novel is called both *1984* and *Nineteen Eighty-Four*,
  sometimes 25 lines apart.** `B:24` and `B:26` use *Nineteen Eighty-Four*;
  `B:49` and `B:64` use *1984*. The body uses *1984* (`03_methods.tex:10`,
  `01_intro.tex:32`, `05_discussion.tex:11`) and `04_results.tex:56` uses
  *Nineteen Eighty-Four*. **Fix:** one form; `03_methods.tex:10` establishes
  *1984* as the corpus name, so that is the one to keep.

- [x] **`B:50` shortens *Alice's Adventures in Wonderland* to *Alice in
  Wonderland*,** which is a different (and non-canonical) title. `03_methods.tex:10`
  has the full one.

- [x] **The incumbent judge is "Qwen3-32B" in the body and "Qwen3-32B-AWQ" in
  the appendix.** `04_results.tex:35` vs `B:878`, `B:890`, `B:899`. The
  quantization matters to reproducibility, so the appendix form is probably the
  right one. Note `CONGRUENCE.md` already tracks judge naming as an open item.

### Wording

**ALL RESOLVED 2026-08-09.**

- [x] **`B:664` — a "Second," with no "First,".**
  > Two measurements appear to be new. GRPO and KTO updates trained from a
  > byte-identical base: ours are mutually near-orthogonal, yet …
  > … Second, prior work relates weight travel to task accuracy …

  The first item is a fragment with no main verb and no enumerator.
  **Fix:** "First, GRPO and KTO updates trained from a byte-identical base have
  not been compared: ours are …".

- [x] **`B:522` — "significantly below chance" with no test.** Same problem as
  `04_results.tex:25`: this paper reports confidence intervals everywhere, so
  "significantly" will be read as a statistical claim. The very next sentences
  give the interval, so the word is doing no work. **Fix:** delete it.

- [x] **`B:640` — "roughly thirty times below its 95th percentile" is
  ambiguous.** "N times below" has no agreed meaning. **Fix:** "roughly a
  thirtieth of its 95th percentile".

- [x] **`B:138` — "${\approx}\,87\%$ in both" is slightly off for one corpus.**
  From `B:136-137`, `fiction10` is $54.0 + 34.3 = 88.3\%$ and `top100` is
  $53.4 + 33.8 = 87.2\%$. **Fix:** "87–88%", or give both.

- [x] **`B:762-763` — the `\label` for `app:privacylens-noise` hangs off a
  `\paragraph`, not a `\subsection`.** It resolves, so nothing prints `??`, but
  `04_results.tex:20` cites it as if it were a section. **Fix:** promote to a
  `\subsection`, or reword the citing sentence.

- [x] **VERIFIED, no change needed.** `04_results.tex:56` uses `15.5` for two different quantities in
  adjacent clauses** — the Hodges-Lehmann shift ($+15.5$ points) and *Nineteen
  Eighty-Four*'s per-direction flip rate ($15.5\%$ each). `B:27` independently
  corroborates the second. Checked against the generating notebook
  (`notebooks/colm-camera-ready/norm_grounding_disagreement.py`): it computes the
  shift as `hl_shift * 100` and prints `p = 0.004`, `+15.5` points under CI
  coding on 9/10 novels, which is exactly what the manuscript reports. The two
  15.5 values are independently sourced; the coincidence is genuine.

### Mechanically caught after this read

The read-through surfaced mixed British/American spelling, concentrated in
`B_additional-results.tex`. Rather than list it here, I added a
`SPELLING_VARIETY` rule to `scripts/tex_grammar.py`; it infers the house variety
from the document and reports the minority side. It finds **11 instances against
147**, including three the manual read missed (`grey`, `travelled` twice). See
`COPYEDIT_REVIEW.md`.

### Not a copyedit finding, but worth a check

`A:44-45` and `B:90-91` both state that both corpora were extracted "under the
same **fiction** reasoning and extraction prompts". Whether the runs actually
used the fiction prompt variants is a factual question about the run records,
not about the prose, and it sits with `CONGRUENCE.md` and
`MANUSCRIPT_STALE_AUDIT.md` rather than here. Flagging only because the claim is
load-bearing for the corpus comparison: every difference is attributed to
corpus composition on the strength of the extractor being held fixed.

## 7. Structural check, passed

Recorded so a later reader does not redo the work:

- **Promise and payoff.** Every artifact §1 promises (two corpora, three
  post-training methods, five benchmarks, 11 models) appears in §3 and §4.
- **Reading-order sequence.** §1's `\S\ref` roadmap at `01_intro.tex:55` matches
  the actual section order.
- **Cross-references.** All resolve; nothing renders as `??`. See
  `COPYEDIT_REVIEW.md`.
