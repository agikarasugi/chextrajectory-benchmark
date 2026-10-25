# CheXTrajectory benchmark protocol

This protocol specifies the implemented single-finding temporal reasoning
benchmark. The primary population is `temporal-reasoning-gold-pilot-v1`.
The executable contracts are in
[`protocol/contracts.py`](../experiments/temporal_reasoning_single_finding/protocol/contracts.py),
[`protocol/task_rules.py`](../experiments/temporal_reasoning_single_finding/protocol/task_rules.py),
and [`protocol/task_selection.py`](../experiments/temporal_reasoning_single_finding/protocol/task_selection.py).

## Scope and unit of evaluation

A trajectory comprises one target finding and three or four distinct frontal
images from one patient, ordered chronologically. Three timepoints (3TP) produce
two adjacent intervals, and four timepoints (4TP) produce three. Chronological
order does not require inclusion of every intervening clinical study.

The primary Gold pilot retains 335 3TP and 178 4TP trajectories, with one eligible
task assigned to each. All six settings receive the same 513 assigned questions.
Eligibility for all four tasks is retained separately from assignment. Questions
and sequence lengths can share patients and images. Descriptive question-level
scores do not establish independent patient-level effects.

Construction enforces these population counts and task quotas.

## Source validation and label semantics

Construction reads two strict-order single-finding JSON exports and an audit
Parquet sidecar. The JSONs supply content. The sidecar validates source agreement,
unambiguity, and image membership. See [Required data](DATA_INPUTS.md) for the
file schemas and locations.

Candidates must have valid identities, supported finding and label names,
distinct frontal images, safe relative image paths, strict chronology, matching
source metadata, and consistent finding-presence states across shared visits.
Quality exclusions are counted. Malformed records and conflicting identities or
references stop construction. The Gold pilot retains all valid Gold candidates.

| Source label | Reference change label | Finding presence before and after |
| --- | --- | --- |
| New | deterioration | absent, present |
| Worse | deterioration | present, present |
| Stable | unchanged | present, present |
| Improved | improvement | present, present |
| Resolved | improvement | present, absent |

The finding-presence state at a shared visit must agree between its adjacent
source labels. These states are source annotation semantics, not absolute
severity values. The benchmark does not assume equal magnitudes for different
improvement or deterioration intervals.

## Four tasks

1. **Trajectory Shape.** Remove unchanged intervals and collapse repeated
   consecutive directions. No remaining direction gives `constant`. A single
   direction gives `monotonic_worsening` or `monotonic_improvement`. Exactly one
   switch gives `single_peak` for deterioration followed by improvement, or
   `single_trough` for the reverse. Two switches give `fluctuating`, possible
   only with 4TP.
2. **Turning Regions.** Identify every switch between opposing active
   directions. A deterioration-to-improvement switch is a peak, and the reverse
   is a trough. Intervening unchanged visits extend the turning region into a
   plateau, for example `peak:T2-T3`. Return all regions chronologically,
   separated by commas, or `none`.
3. **Most Severe Visit or Plateau.** Return the uniquely identifiable maximal
   visit or contiguous plateau implied by the relative labels. Constant
   trajectories return the entire sequence. Monotonic trajectories return the
   appropriate endpoint plateau, and single peaks return the peak region.
   Single troughs and fluctuating trajectories have incomparable candidate
   maxima, so this task is ineligible for those reference trajectories.
4. **First Improvement Between Visits.** Return the earliest interval labelled
   improvement, for example `T2-T3`, or `none` if no interval improves.

Answer strings are case-sensitive and use the exact vocabularies returned by
`answer_choices()` in `protocol/contracts.py`. The independent synthetic oracle
covers all 9 possible 3TP and 27 possible 4TP change vectors across the four
tasks, including undefined Most Severe outcomes.

## Deterministic task assignment

| Assigned task | 3TP | 4TP |
| --- | ---: | ---: |
| Trajectory Shape | 84 | 44 |
| Turning Regions | 84 | 44 |
| Most Severe Visit or Plateau | 83 | 45 |
| First Improvement Between Visits | 84 | 45 |
| Total | 335 | 178 |

For each length, sort by trajectory ID and shuffle with an independent
`random.Random(42)`. Assign Most Severe first from eligible candidates in this
order. Shuffle the remaining items with the same generator, then allocate
Shape, Turning Regions, and First Improvement in that order. Write items in
trajectory-ID order. Insufficient eligibility or a population/quotas mismatch
fails construction.

The manifest records `most-severe-first-shuffle-v1` and the selection contract
`one-eligible-task-v1`. Reference Check verifies all four task entries per
trajectory, including ineligible entries. Inference and final-task scoring use
only the assigned eligible task. The 2,052 reference-check entries therefore
have a different denominator from the 513 selected questions.

## Six settings

FC means fixed code. LC means a language-model call over change labels.

| Setting | Input | Final answer mechanism |
| --- | --- | --- |
| SEQ-DIR | Entire image sequence and target finding | Model answers the assigned task directly |
| PAIR-FC | Separately predicted adjacent-pair change labels | Deterministic task rules |
| SEQ-FC | Change labels predicted jointly from the entire sequence | Deterministic task rules |
| REF-LC | Reference change labels | Model answers the assigned task |
| PAIR-LC | PAIR-predicted change labels | Model answers the assigned task |
| SEQ-LC | SEQ-predicted change labels | Model answers the assigned task |

PAIR classifies each adjacent image pair independently. Identical ordered pairs
for the same finding and run contract are reused. SEQ predicts every adjacent
label in one call for a trajectory. Both intermediates are shared by their FC
and LC settings. FC requires no additional inference. Label-only prompts do not
reveal whether labels are reference or predicted.

REF-LC isolates reasoning from reference labels. PAIR/SEQ interval diagnostics
cover all trajectories, independently of the assigned final task.

## Current LC inference policy

New runs record `lc_inference_policy: predicted-labels-v1`. PAIR-LC and SEQ-LC
receive valid upstream labels even when their fixed-code answer is wrong.
They also receive valid label vectors for which fixed code cannot identify a
unique Most Severe answer. They must still return an allowed task answer.
Unusable upstream predictions produce skip records instead of downstream calls.

The compatibility policy `reference-supported-answer-v1` gates LC calls on the
fixed-code answer matching the reference. Such runs preserve their policy in
the manifest and record unsupported answers as `upstream_answer_mismatch`.
Absent policy fields retain legacy behavior. These populations of submitted LC
calls differ, so comparisons should identify the saved policy for each run.
Migration tooling supports legacy records.

## Prompts, parsing, and request identity

Each call has one user message, starting with the radiologist-support role
instruction, followed by the finding, ordered images or labels, and one task.
No system or developer message is sent. All prompts request a 1-3 sentence
explanation followed by exactly one JSON object. Explanations are saved but
not scored. This instruction is separate from optional provider reasoning.

The [prompt templates](PROMPTS.md) and 19 golden fingerprints fix wording,
answer choices, and image positions. The parser accepts a schema-valid JSON
object alone or as the single valid object embedded in prose or a code fence.
It rejects multiple objects, duplicate keys, malformed JSON, unexpected fields,
invalid values, and wrong capitalization. It does not repair answers. Final
correctness is exact agreement with the deterministic reference answer.

PAIR returns `change_label`, SEQ returns `change_labels` with exact adjacent
interval keys, and final tasks return `answer`. Prompt versions remain
`pair-change-v1`, `sequence-change-v1`, and `final-tasks-v1`, with response schema
`v1`. Exact content fingerprints supplement these identifiers. Benchmark,
model, prompt, task, image/label, and request-option identities prevent mixing
incompatible prepared inputs or responses within a run.

Optional `reasoning_effort` and `max_completion_tokens` are forwarded unchanged.
Their omission uses endpoint defaults. Changing explicit values requires a new
run name. Request timeout is an execution option. Endpoint equivalence is not
proved by matching an advertised model name.

## Execution, failure handling, and scoring

Preflight verifies benchmark checksums, references, assignments, prompt
fingerprints, image-file existence, endpoint model availability, and the call
plan. Runtime resolves PNGs by replacing only the source path's final
`.jpg` suffix. Prepared records contain image references rather than payloads.

The primary population's planned call ceiling is 492 PAIR calls, 513 SEQ calls,
513 SEQ-DIR calls, and 513 calls for each LC setting, totaling 3,057 before
retries. Of these, 1,518 contain images and 1,539 are text-only. Failed upstream
predictions can reduce downstream calls. Retries add attempts and cost.

Retryable technical failures have at most three attempts per logical call.
Invalid parsed answers are terminal. Resume reuses completed compatible records.
Explicit `--retry-failed` reopens technical failures and repairs dependent
results, invalidating stale coverage and evaluation summaries.

The offline evaluator validates complete coverage and consistency before writing
`evaluation.json`. Current output uses evaluator `2.0.0` and schema `4`, while
respecting the LC policy stored in each run. Report 3TP and 4TP separately and
keep Gold and Silver cohorts separate when using compatibility populations.

The evaluator reports:

- Exact final-answer accuracy by task, setting, and cohort. Every selected
  question counts equally. Invalid responses, technical failures, and skips
  remain incorrect in this denominator.
- Balanced accuracy as mean recall over reference classes with positive support.
- PAIR/SEQ interval accuracy, per-class precision/recall/F1, complete-trajectory
  accuracy, and error counts. Reused pairs count once per containing trajectory
  for these diagnostics, without duplication by final task or setting.
- LC accuracy conditional on the corresponding FC answer being correct or
  wrong, and FC/LC answer agreement. Agreement uses the full selected-case
  denominator, with missing or invalid answers counted as non-agreement.
- Submitted-call failure rates, downstream failures, population counts, and
  reference distributions. Skipped calls do not count as submitted calls.

Zero denominators yield JSON `null`. Results are descriptive and question-weighted.
Task assignment balances task totals, while reference-answer frequencies can
differ. Interpret accuracy alongside reference distributions and balanced
accuracy.

## Compatibility and reproducibility

The package also retains the mixed pilot, full-task Gold, and smoke constructors
for compatibility. Their manifests define their own populations and selection
contracts. Use separate benchmark versions or run names when changing the
population, assignment, prompts, or request settings. Benchmark publication is
atomic and conflicting frozen outputs are rejected.
