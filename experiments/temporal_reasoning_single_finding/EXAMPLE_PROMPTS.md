# Full model-facing prompts with short explanations

This reference is rendered from [protocol/prompts.py](protocol/prompts.py) and the answer choices in [protocol/contracts.py](protocol/contracts.py). Each fenced block contains one complete user message. Each request consists of that single message.

`{finding}` is replaced by the target finding display name. `[IMAGE: Tn]` marks an image content part, not literal prompt text. The runner attaches the corresponding images at submission time. Text and image parts appear in the order shown, with line breaks around image markers for readability.

The label-only examples use `deterioration`, `unchanged`, and (for four visits) `improvement` as illustrative input labels. Actual calls substitute the reference or predicted labels for that case. Each final-task call contains exactly one task.

| Setting or stage | Input and model request |
| --- | --- |
| SEQ-DIR | Complete image sequence and one final task. |
| PAIR | Earlier and later images, requesting one change label. Used upstream of PAIR-FC and PAIR-LC. |
| SEQ | Complete image sequence, requesting all adjacent change labels. Used upstream of SEQ-FC and SEQ-LC. |
| REF-LC | Reference change labels and one final task, with no images. |
| PAIR-LC | PAIR-predicted change labels and one final task, with no images. |
| SEQ-LC | SEQ-predicted change labels and one final task, with no images. |
| PAIR-FC / SEQ-FC | Fixed code derives final answers from the corresponding predicted labels. No additional final-task model prompt. |

REF-LC, PAIR-LC, and SEQ-LC use identical prompt wording for the same finding, labels, task, and sequence length. The setting name and label provenance are not included in the message.

Every prompt requests a short explanation in 1-3 sentences before the required JSON answer. Configure provider reasoning separately through the optional `reasoning_effort` request parameter.

## Contents

- [PAIR: adjacent-image change classification](#pair-adjacent-image-change-classification)
- [3-visit sequences](#3-visit-sequences)
- [4-visit sequences](#4-visit-sequences)

### PAIR: adjacent-image change classification

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following two images are ordered chronologically.

Earlier image:
[IMAGE: earlier image]
Later image:
[IMAGE: later image]
Classify how the target finding changes from the earlier image to the later image.

Change-label definitions:
- deterioration: The target finding newly appears or worsens.
- unchanged: The target finding remains stable.
- improvement: The target finding improves or resolves.

Choose exactly one change label: deterioration, unchanged, or improvement.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"change_label":"VALUE"}
```

### 3-visit sequences

#### SEQ: all adjacent change labels

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images are ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
Classify how the target finding changes between every pair of consecutive images.

Change-label definitions:
- deterioration: The target finding newly appears or worsens.
- unchanged: The target finding remains stable.
- improvement: The target finding improves or resolves.

Choose exactly one change label for each interval: deterioration, unchanged, or improvement.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly the fields shown below:
{"change_labels":{"T1_to_T2":"VALUE","T2_to_T3":"VALUE"}}
```

#### SEQ-DIR: trajectory_shape

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images represent visits ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
Task: Across the complete sequence, classify the trajectory shape of the target finding.

Task definition:

Ignore unchanged intervals when counting direction switches. A direction switch occurs when two successive remaining intervals have opposite directions. Consecutive intervals with the same direction do not count as a switch.

- constant: Every interval is unchanged.

- monotonic_worsening: After ignoring unchanged intervals, at least one interval remains and every remaining interval is deterioration.

- monotonic_improvement: After ignoring unchanged intervals, at least one interval remains and every remaining interval is improvement.

- single_peak: After ignoring unchanged intervals, there is exactly one direction switch: deterioration to improvement. There is no subsequent switch back.

- single_trough: After ignoring unchanged intervals, there is exactly one direction switch: improvement to deterioration. There is no subsequent switch back.

Answer choices:
- constant
- monotonic_worsening
- monotonic_improvement
- single_peak
- single_trough

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### SEQ-DIR: turning_regions

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images represent visits ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
Task: Identify every turning region for the target finding across the complete sequence.

Task definition:
- none: There is no switch between deterioration and improvement.
- peak:Tn: The finding changes from deterioration before visit Tn to improvement after visit Tn.
- trough:Tn: The finding changes from improvement before visit Tn to deterioration after visit Tn.
- peak:Tn-Tm: The finding changes from deterioration to improvement across the consecutive unchanged visits from Tn through Tm.
- trough:Tn-Tm: The finding changes from improvement to deterioration across the consecutive unchanged visits from Tn through Tm.
- If there is more than one turning region, list every region in chronological order, separated by commas.
- An unchanged interval by itself is not a turning region.
- Consecutive changes in the same direction do not form a turning region.
- An unchanged region is included only when it lies between deterioration and improvement in opposing directions.

Answer choices:
- none
- peak:T2
- trough:T2

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### SEQ-DIR: most_severe_visit_or_plateau

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images represent visits ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
Task: At which visit, or across which consecutive visits, does the target finding appear most severe?

Task definition:
- Most severe refers only to the appearance of the target finding. It does not refer to the patient's overall health, symptoms, or other findings.
- Tn: The target finding appears more severe at visit Tn than at every other visit.
- Tn-Tm: The target finding has the same most-severe appearance at every consecutive visit from Tn through Tm, and appears less severe at every visit outside that range, if any.
- A plateau is two or more consecutive visits during which the target finding remains unchanged.
- If the target finding remains unchanged throughout the complete sequence, the complete sequence is the most-severe plateau.
- Exactly one listed answer is correct.

Answer choices:
- T1
- T1-T2
- T1-T3
- T2
- T2-T3
- T3

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### SEQ-DIR: first_improvement_between_visits

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images represent visits ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
Task: Between which two consecutive visits does the target finding first improve, or does it never improve during the sequence?

Task definition:
- The target finding improves when it appears less severe at the later visit than at the immediately preceding visit.
- Select the first interval in which improvement occurs.
- Later deterioration, unchanged appearance, or further improvement does not change the answer.
- none: The target finding does not improve between any consecutive visits.

Answer choices:
- T1-T2
- T2-T3
- none

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### REF-LC / PAIR-LC / SEQ-LC: trajectory_shape

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: {finding}

T1 to T2: deterioration
T2 to T3: unchanged

Task: Across the complete sequence, classify the trajectory shape of the target finding.

Task definition:

Ignore unchanged intervals when counting direction switches. A direction switch occurs when two successive remaining intervals have opposite directions. Consecutive intervals with the same direction do not count as a switch.

- constant: Every interval is unchanged.

- monotonic_worsening: After ignoring unchanged intervals, at least one interval remains and every remaining interval is deterioration.

- monotonic_improvement: After ignoring unchanged intervals, at least one interval remains and every remaining interval is improvement.

- single_peak: After ignoring unchanged intervals, there is exactly one direction switch: deterioration to improvement. There is no subsequent switch back.

- single_trough: After ignoring unchanged intervals, there is exactly one direction switch: improvement to deterioration. There is no subsequent switch back.

Answer choices:
- constant
- monotonic_worsening
- monotonic_improvement
- single_peak
- single_trough

Use the change labels exactly as written. Do not change or re-predict them.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### REF-LC / PAIR-LC / SEQ-LC: turning_regions

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: {finding}

T1 to T2: deterioration
T2 to T3: unchanged

Task: Identify every turning region for the target finding across the complete sequence.

Task definition:
- none: There is no switch between deterioration and improvement.
- peak:Tn: The finding changes from deterioration before visit Tn to improvement after visit Tn.
- trough:Tn: The finding changes from improvement before visit Tn to deterioration after visit Tn.
- peak:Tn-Tm: The finding changes from deterioration to improvement across the consecutive unchanged visits from Tn through Tm.
- trough:Tn-Tm: The finding changes from improvement to deterioration across the consecutive unchanged visits from Tn through Tm.
- If there is more than one turning region, list every region in chronological order, separated by commas.
- An unchanged interval by itself is not a turning region.
- Consecutive changes in the same direction do not form a turning region.
- An unchanged region is included only when it lies between deterioration and improvement in opposing directions.

Answer choices:
- none
- peak:T2
- trough:T2

Use the change labels exactly as written. Do not change or re-predict them.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### REF-LC / PAIR-LC / SEQ-LC: most_severe_visit_or_plateau

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: {finding}

T1 to T2: deterioration
T2 to T3: unchanged

Task: At which visit, or across which consecutive visits, does the target finding appear most severe?

Task definition:
- Most severe refers only to the appearance of the target finding. It does not refer to the patient's overall health, symptoms, or other findings.
- Tn: The target finding appears more severe at visit Tn than at every other visit.
- Tn-Tm: The target finding has the same most-severe appearance at every consecutive visit from Tn through Tm, and appears less severe at every visit outside that range, if any.
- A plateau is two or more consecutive visits during which the target finding remains unchanged.
- If the target finding remains unchanged throughout the complete sequence, the complete sequence is the most-severe plateau.
- Exactly one listed answer is correct.

Answer choices:
- T1
- T1-T2
- T1-T3
- T2
- T2-T3
- T3

Use the change labels exactly as written. Do not change or re-predict them.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### REF-LC / PAIR-LC / SEQ-LC: first_improvement_between_visits

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: {finding}

T1 to T2: deterioration
T2 to T3: unchanged

Task: Between which two consecutive visits does the target finding first improve, or does it never improve during the sequence?

Task definition:
- The target finding improves when it appears less severe at the later visit than at the immediately preceding visit.
- Select the first interval in which improvement occurs.
- Later deterioration, unchanged appearance, or further improvement does not change the answer.
- none: The target finding does not improve between any consecutive visits.

Answer choices:
- T1-T2
- T2-T3
- none

Use the change labels exactly as written. Do not change or re-predict them.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

### 4-visit sequences

#### SEQ: all adjacent change labels

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images are ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
T4:
[IMAGE: T4]
Classify how the target finding changes between every pair of consecutive images.

Change-label definitions:
- deterioration: The target finding newly appears or worsens.
- unchanged: The target finding remains stable.
- improvement: The target finding improves or resolves.

Choose exactly one change label for each interval: deterioration, unchanged, or improvement.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly the fields shown below:
{"change_labels":{"T1_to_T2":"VALUE","T2_to_T3":"VALUE","T3_to_T4":"VALUE"}}
```

#### SEQ-DIR: trajectory_shape

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images represent visits ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
T4:
[IMAGE: T4]
Task: Across the complete sequence, classify the trajectory shape of the target finding.

Task definition:

Ignore unchanged intervals when counting direction switches. A direction switch occurs when two successive remaining intervals have opposite directions. Consecutive intervals with the same direction do not count as a switch.

- constant: Every interval is unchanged.

- monotonic_worsening: After ignoring unchanged intervals, at least one interval remains and every remaining interval is deterioration.

- monotonic_improvement: After ignoring unchanged intervals, at least one interval remains and every remaining interval is improvement.

- single_peak: After ignoring unchanged intervals, there is exactly one direction switch: deterioration to improvement. There is no subsequent switch back.

- single_trough: After ignoring unchanged intervals, there is exactly one direction switch: improvement to deterioration. There is no subsequent switch back.

- fluctuating: After ignoring unchanged intervals, there are two or more direction switches.

Answer choices:
- constant
- monotonic_worsening
- monotonic_improvement
- single_peak
- single_trough
- fluctuating

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### SEQ-DIR: turning_regions

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images represent visits ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
T4:
[IMAGE: T4]
Task: Identify every turning region for the target finding across the complete sequence.

Task definition:
- none: There is no switch between deterioration and improvement.
- peak:Tn: The finding changes from deterioration before visit Tn to improvement after visit Tn.
- trough:Tn: The finding changes from improvement before visit Tn to deterioration after visit Tn.
- peak:Tn-Tm: The finding changes from deterioration to improvement across the consecutive unchanged visits from Tn through Tm.
- trough:Tn-Tm: The finding changes from improvement to deterioration across the consecutive unchanged visits from Tn through Tm.
- If there is more than one turning region, list every region in chronological order, separated by commas.
- An unchanged interval by itself is not a turning region.
- Consecutive changes in the same direction do not form a turning region.
- An unchanged region is included only when it lies between deterioration and improvement in opposing directions.

Answer choices:
- none
- peak:T2
- peak:T2,trough:T3
- peak:T2-T3
- peak:T3
- trough:T2
- trough:T2,peak:T3
- trough:T2-T3
- trough:T3

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### SEQ-DIR: most_severe_visit_or_plateau

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images represent visits ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
T4:
[IMAGE: T4]
Task: At which visit, or across which consecutive visits, does the target finding appear most severe?

Task definition:
- Most severe refers only to the appearance of the target finding. It does not refer to the patient's overall health, symptoms, or other findings.
- Tn: The target finding appears more severe at visit Tn than at every other visit.
- Tn-Tm: The target finding has the same most-severe appearance at every consecutive visit from Tn through Tm, and appears less severe at every visit outside that range, if any.
- A plateau is two or more consecutive visits during which the target finding remains unchanged.
- If the target finding remains unchanged throughout the complete sequence, the complete sequence is the most-severe plateau.
- Exactly one listed answer is correct.

Answer choices:
- T1
- T1-T2
- T1-T3
- T1-T4
- T2
- T2-T3
- T2-T4
- T3
- T3-T4
- T4

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### SEQ-DIR: first_improvement_between_visits

```text
You are an AI assistant supporting radiologists.

Target finding: {finding}

The following images represent visits ordered chronologically.

T1:
[IMAGE: T1]
T2:
[IMAGE: T2]
T3:
[IMAGE: T3]
T4:
[IMAGE: T4]
Task: Between which two consecutive visits does the target finding first improve, or does it never improve during the sequence?

Task definition:
- The target finding improves when it appears less severe at the later visit than at the immediately preceding visit.
- Select the first interval in which improvement occurs.
- Later deterioration, unchanged appearance, or further improvement does not change the answer.
- none: The target finding does not improve between any consecutive visits.

Answer choices:
- T1-T2
- T2-T3
- T3-T4
- none

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### REF-LC / PAIR-LC / SEQ-LC: trajectory_shape

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: {finding}

T1 to T2: deterioration
T2 to T3: unchanged
T3 to T4: improvement

Task: Across the complete sequence, classify the trajectory shape of the target finding.

Task definition:

Ignore unchanged intervals when counting direction switches. A direction switch occurs when two successive remaining intervals have opposite directions. Consecutive intervals with the same direction do not count as a switch.

- constant: Every interval is unchanged.

- monotonic_worsening: After ignoring unchanged intervals, at least one interval remains and every remaining interval is deterioration.

- monotonic_improvement: After ignoring unchanged intervals, at least one interval remains and every remaining interval is improvement.

- single_peak: After ignoring unchanged intervals, there is exactly one direction switch: deterioration to improvement. There is no subsequent switch back.

- single_trough: After ignoring unchanged intervals, there is exactly one direction switch: improvement to deterioration. There is no subsequent switch back.

- fluctuating: After ignoring unchanged intervals, there are two or more direction switches.

Answer choices:
- constant
- monotonic_worsening
- monotonic_improvement
- single_peak
- single_trough
- fluctuating

Use the change labels exactly as written. Do not change or re-predict them.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### REF-LC / PAIR-LC / SEQ-LC: turning_regions

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: {finding}

T1 to T2: deterioration
T2 to T3: unchanged
T3 to T4: improvement

Task: Identify every turning region for the target finding across the complete sequence.

Task definition:
- none: There is no switch between deterioration and improvement.
- peak:Tn: The finding changes from deterioration before visit Tn to improvement after visit Tn.
- trough:Tn: The finding changes from improvement before visit Tn to deterioration after visit Tn.
- peak:Tn-Tm: The finding changes from deterioration to improvement across the consecutive unchanged visits from Tn through Tm.
- trough:Tn-Tm: The finding changes from improvement to deterioration across the consecutive unchanged visits from Tn through Tm.
- If there is more than one turning region, list every region in chronological order, separated by commas.
- An unchanged interval by itself is not a turning region.
- Consecutive changes in the same direction do not form a turning region.
- An unchanged region is included only when it lies between deterioration and improvement in opposing directions.

Answer choices:
- none
- peak:T2
- peak:T2,trough:T3
- peak:T2-T3
- peak:T3
- trough:T2
- trough:T2,peak:T3
- trough:T2-T3
- trough:T3

Use the change labels exactly as written. Do not change or re-predict them.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### REF-LC / PAIR-LC / SEQ-LC: most_severe_visit_or_plateau

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: {finding}

T1 to T2: deterioration
T2 to T3: unchanged
T3 to T4: improvement

Task: At which visit, or across which consecutive visits, does the target finding appear most severe?

Task definition:
- Most severe refers only to the appearance of the target finding. It does not refer to the patient's overall health, symptoms, or other findings.
- Tn: The target finding appears more severe at visit Tn than at every other visit.
- Tn-Tm: The target finding has the same most-severe appearance at every consecutive visit from Tn through Tm, and appears less severe at every visit outside that range, if any.
- A plateau is two or more consecutive visits during which the target finding remains unchanged.
- If the target finding remains unchanged throughout the complete sequence, the complete sequence is the most-severe plateau.
- Exactly one listed answer is correct.

Answer choices:
- T1
- T1-T2
- T1-T3
- T1-T4
- T2
- T2-T3
- T2-T4
- T3
- T3-T4
- T4

Use the change labels exactly as written. Do not change or re-predict them.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```

#### REF-LC / PAIR-LC / SEQ-LC: first_improvement_between_visits

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: {finding}

T1 to T2: deterioration
T2 to T3: unchanged
T3 to T4: improvement

Task: Between which two consecutive visits does the target finding first improve, or does it never improve during the sequence?

Task definition:
- The target finding improves when it appears less severe at the later visit than at the immediately preceding visit.
- Select the first interval in which improvement occurs.
- Later deterioration, unchanged appearance, or further improvement does not change the answer.
- none: The target finding does not improve between any consecutive visits.

Answer choices:
- T1-T2
- T2-T3
- T3-T4
- none

Use the change labels exactly as written. Do not change or re-predict them.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"answer":"VALUE"}
```
