# CheXTrajectory benchmark prompts

This reference shows the complete prompt templates used by the benchmark.
Text in square brackets is replaced before inference. Lines such as `[T1 IMAGE ATTACHMENT]`
show the position of a multimodal image content part and are not literal text
sent to the model.

Every call uses a single user message. Its very first line is:

```text
You are an AI assistant supporting radiologists.
```

The benchmark sends no system or developer message. For multimodal inputs,
this instruction is the start of the first text part, before any image.

## Short explanations

All prompts request a short explanation in 1-3 sentences before the required
JSON object. For SEQ, the sentence limit covers the whole response. Scoring
uses the JSON answer. Provider reasoning is configured separately.

## PAIR change-label prompt

The following is one complete PAIR user message:

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following two images are ordered chronologically.

Earlier image:
[EARLIER IMAGE ATTACHMENT]
Later image:
[LATER IMAGE ATTACHMENT]
Classify how the target finding changes from the earlier image to the later image.

Change-label definitions:
- deterioration: The target finding newly appears or worsens.
- unchanged: The target finding remains stable.
- improvement: The target finding improves or resolves.

Choose exactly one change label: deterioration, unchanged, or improvement.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly this field:
{"change_label":"VALUE"}
```

The image attachment markers represent separate image content parts. The text
before and after each marker remains in the displayed order.

## SEQ change-label prompt: 3TP

The following is one complete 3TP SEQ user message:

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images are ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
Classify how the target finding changes between every pair of consecutive images.

Change-label definitions:
- deterioration: The target finding newly appears or worsens.
- unchanged: The target finding remains stable.
- improvement: The target finding improves or resolves.

Choose exactly one change label for each interval: deterioration, unchanged, or improvement.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly the fields shown below:
{"change_labels":{"T1_to_T2":"VALUE","T2_to_T3":"VALUE"}}
```

## SEQ change-label prompt: 4TP

The following is one complete 4TP SEQ user message:

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images are ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
T4:
[T4 IMAGE ATTACHMENT]
Classify how the target finding changes between every pair of consecutive images.

Change-label definitions:
- deterioration: The target finding newly appears or worsens.
- unchanged: The target finding remains stable.
- improvement: The target finding improves or resolves.

Choose exactly one change label for each interval: deterioration, unchanged, or improvement.

Write a short explanation of your classification in 1-3 sentences before the answer. Then end your response with exactly one JSON object using exactly the fields shown below:
{"change_labels":{"T1_to_T2":"VALUE","T2_to_T3":"VALUE","T3_to_T4":"VALUE"}}
```

## Final-answer prompt construction

Each section below shows a complete SEQ-DIR prompt and a complete shared LC
prompt. REF-LC, PAIR-LC, and SEQ-LC use the same LC template and make separate
fresh calls. Only the substituted change-label values can differ.

## Trajectory Shape: 3TP SEQ-DIR

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images represent visits ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
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

## Trajectory Shape: 3TP shared LC

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: [FINDING]

T1 to T2: [CHANGE LABEL]
T2 to T3: [CHANGE LABEL]

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

## Trajectory Shape: 4TP SEQ-DIR

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images represent visits ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
T4:
[T4 IMAGE ATTACHMENT]
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

## Trajectory Shape: 4TP shared LC

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: [FINDING]

T1 to T2: [CHANGE LABEL]
T2 to T3: [CHANGE LABEL]
T3 to T4: [CHANGE LABEL]

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

## Turning Regions: 3TP SEQ-DIR

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images represent visits ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
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

## Turning Regions: 3TP shared LC

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: [FINDING]

T1 to T2: [CHANGE LABEL]
T2 to T3: [CHANGE LABEL]

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

## Turning Regions: 4TP SEQ-DIR

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images represent visits ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
T4:
[T4 IMAGE ATTACHMENT]
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

## Turning Regions: 4TP shared LC

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: [FINDING]

T1 to T2: [CHANGE LABEL]
T2 to T3: [CHANGE LABEL]
T3 to T4: [CHANGE LABEL]

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

## Most Severe Visit or Plateau: 3TP SEQ-DIR

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images represent visits ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
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

## Most Severe Visit or Plateau: 3TP shared LC

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: [FINDING]

T1 to T2: [CHANGE LABEL]
T2 to T3: [CHANGE LABEL]

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

## Most Severe Visit or Plateau: 4TP SEQ-DIR

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images represent visits ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
T4:
[T4 IMAGE ATTACHMENT]
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

## Most Severe Visit or Plateau: 4TP shared LC

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: [FINDING]

T1 to T2: [CHANGE LABEL]
T2 to T3: [CHANGE LABEL]
T3 to T4: [CHANGE LABEL]

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

## First Improvement Between Visits: 3TP SEQ-DIR

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images represent visits ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
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

## First Improvement Between Visits: 3TP shared LC

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: [FINDING]

T1 to T2: [CHANGE LABEL]
T2 to T3: [CHANGE LABEL]

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

## First Improvement Between Visits: 4TP SEQ-DIR

```text
You are an AI assistant supporting radiologists.

Target finding: [FINDING]

The following images represent visits ordered chronologically.

T1:
[T1 IMAGE ATTACHMENT]
T2:
[T2 IMAGE ATTACHMENT]
T3:
[T3 IMAGE ATTACHMENT]
T4:
[T4 IMAGE ATTACHMENT]
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

## First Improvement Between Visits: 4TP shared LC

```text
You are an AI assistant supporting radiologists.

Change labels between consecutive images:

Target finding: [FINDING]

T1 to T2: [CHANGE LABEL]
T2 to T3: [CHANGE LABEL]
T3 to T4: [CHANGE LABEL]

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
