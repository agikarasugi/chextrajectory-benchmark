"""Exact approved prompt renderers with safe image references."""

from collections.abc import Sequence
from typing import Any

from .contracts import (
    answer_choices,
    validate_change_labels,
    validate_task,
    validate_timepoint_count,
)


ROLE_INSTRUCTION = "You are an AI assistant supporting radiologists."

SHORT_EXPLANATION_PREFIX = (
    "Write a short explanation of your classification in 1-3 sentences "
    "before the answer. Then end your response with exactly one JSON "
    "object "
)
SHORT_EXPLANATION_INSTRUCTION = (
    SHORT_EXPLANATION_PREFIX + 'using exactly this field:\n{"answer":"VALUE"}'
)


CHANGE_DEFINITIONS = """Change-label definitions:
- deterioration: The target finding newly appears or worsens.
- unchanged: The target finding remains stable.
- improvement: The target finding improves or resolves.

Choose exactly one change label: deterioration, unchanged, or improvement.

Return exactly one JSON object using exactly this field:
{"change_label":"VALUE"}"""

SEQ_CHANGE_DEFINITIONS = "\n".join(
    (
        "Classify how the target finding changes between every pair of "
        "consecutive images.",
        "",
        "Change-label definitions:",
        "- deterioration: The target finding newly appears or worsens.",
        "- unchanged: The target finding remains stable.",
        "- improvement: The target finding improves or resolves.",
        "",
        "Choose exactly one change label for each interval: deterioration, "
        "unchanged, or improvement.",
        "",
        "Return exactly one JSON object using exactly the fields shown below:",
        "{schema}",
    )
)

TASK_QUESTIONS = {
    "trajectory_shape": (
        "Task: Across the complete sequence, classify the trajectory shape of "
        "the target finding."
    ),
    "turning_regions": (
        "Task: Identify every turning region for the target finding across the "
        "complete sequence."
    ),
    "most_severe_visit_or_plateau": (
        "Task: At which visit, or across which consecutive visits, does the "
        "target finding appear most severe?"
    ),
    "first_improvement_between_visits": (
        "Task: Between which two consecutive visits does the target finding "
        "first improve, or does it never improve during the sequence?"
    ),
}

TASK_DEFINITIONS = {
    "trajectory_shape": "\n\n".join(
        (
            "Task definition:",
            "Ignore unchanged intervals when counting direction switches. A direction "
            "switch occurs when two successive remaining intervals have opposite "
            "directions. Consecutive intervals with the same direction do not count "
            "as a switch.",
            "- constant: Every interval is unchanged.",
            "- monotonic_worsening: After ignoring unchanged intervals, at least one "
            "interval remains and every remaining interval is deterioration.",
            "- monotonic_improvement: After ignoring unchanged intervals, at least "
            "one interval remains and every remaining interval is improvement.",
            "- single_peak: After ignoring unchanged intervals, there is exactly one "
            "direction switch: deterioration to improvement. There is no subsequent "
            "switch back.",
            "- single_trough: After ignoring unchanged intervals, there is exactly "
            "one direction switch: improvement to deterioration. There is no "
            "subsequent switch back.",
        )
    ),
    "turning_regions": "\n".join(
        (
            "Task definition:",
            "- none: There is no switch between deterioration and improvement.",
            "- peak:Tn: The finding changes from deterioration before visit Tn to "
            "improvement after visit Tn.",
            "- trough:Tn: The finding changes from improvement before visit Tn to "
            "deterioration after visit Tn.",
            "- peak:Tn-Tm: The finding changes from deterioration to improvement "
            "across the consecutive unchanged visits from Tn through Tm.",
            "- trough:Tn-Tm: The finding changes from improvement to deterioration "
            "across the consecutive unchanged visits from Tn through Tm.",
            "- If there is more than one turning region, list every region in "
            "chronological order, separated by commas.",
            "- An unchanged interval by itself is not a turning region.",
            "- Consecutive changes in the same direction do not form a turning "
            "region.",
            "- An unchanged region is included only when it lies between "
            "deterioration and improvement in opposing directions.",
        )
    ),
    "most_severe_visit_or_plateau": "\n".join(
        (
            "Task definition:",
            "- Most severe refers only to the appearance of the target finding. "
            "It does not refer to the patient's overall health, symptoms, or other "
            "findings.",
            "- Tn: The target finding appears more severe at visit Tn than at every "
            "other visit.",
            "- Tn-Tm: The target finding has the same most-severe appearance at "
            "every consecutive visit from Tn through Tm, and appears less severe "
            "at every visit outside that range, if any.",
            "- A plateau is two or more consecutive visits during which the target "
            "finding remains unchanged.",
            "- If the target finding remains unchanged throughout the complete "
            "sequence, the complete sequence is the most-severe plateau.",
            "- Exactly one listed answer is correct.",
        )
    ),
    "first_improvement_between_visits": "\n".join(
        (
            "Task definition:",
            "- The target finding improves when it appears less severe at the "
            "later visit than at the immediately preceding visit.",
            "- Select the first interval in which improvement occurs.",
            "- Later deterioration, unchanged appearance, or further improvement "
            "does not change the answer.",
            "- none: The target finding does not improve between any consecutive "
            "visits.",
        )
    ),
}


def _user_message(content: str | list[dict[str, str]]) -> dict[str, Any]:
    """Put the role instruction first without requiring a system-role message."""
    prefix = ROLE_INSTRUCTION + "\n\n"
    if isinstance(content, str):
        user_content = prefix + content
    else:
        user_content = [
            {**content[0], "text": prefix + content[0]["text"]},
            *content[1:],
        ]
    return {"role": "user", "content": user_content}


def _image_ref(image_id: str) -> dict[str, str]:
    """Create a safe model-input image marker without image content."""
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a non-empty string")
    return {"type": "image_ref", "image_id": image_id}


def _answers_block(task: str, timepoint_count: int) -> str:
    """Render the exact ordered answer choices."""
    choices = "\n".join(
        f"- {choice}" for choice in answer_choices(task, timepoint_count)
    )
    return f"Answer choices:\n{choices}"


def _task_block(
    task: str,
    timepoint_count: int,
    *,
    lc: bool,
) -> str:
    """Render one frozen task definition and answer contract."""
    validate_task(task)
    validate_timepoint_count(timepoint_count)
    definition = TASK_DEFINITIONS[task]
    if task == "trajectory_shape" and timepoint_count == 4:
        definition += (
            "\n\n- fluctuating: After ignoring unchanged intervals, there are two "
            "or more direction switches."
        )
    block = (
        f"{TASK_QUESTIONS[task]}\n\n{definition}\n\n"
        f"{_answers_block(task, timepoint_count)}"
    )
    if lc:
        block += (
            "\n\nUse the change labels exactly as written. Do not change or "
            "re-predict them."
        )
    instruction = SHORT_EXPLANATION_INSTRUCTION
    return f"{block}\n\n{instruction}"


def _change_output_instruction(block: str) -> str:
    """Change only the output instruction, preserving the label JSON schema."""
    return block.replace("Return exactly one JSON object ", SHORT_EXPLANATION_PREFIX)


def render_pair_messages(
    finding: str,
    earlier_image_id: str,
    later_image_id: str,
) -> list[dict[str, Any]]:
    """Render the exact PAIR instruction and images in one user message."""
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                f"Target finding: {finding}\n\n"
                "The following two images are ordered chronologically.\n\n"
                "Earlier image:\n"
            ),
        },
        _image_ref(earlier_image_id),
        {"type": "text", "text": "Later image:\n"},
        _image_ref(later_image_id),
        {
            "type": "text",
            "text": (
                "Classify how the target finding changes from the earlier image "
                "to the later image.\n\n"
                f"{_change_output_instruction(CHANGE_DEFINITIONS)}"
            ),
        },
    ]
    return [_user_message(content)]


def render_seq_messages(
    finding: str,
    image_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Render the exact complete-sequence change-label messages."""
    timepoint_count = len(image_ids)
    validate_timepoint_count(timepoint_count)
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                f"Target finding: {finding}\n\n"
                "The following images are ordered chronologically.\n\nT1:\n"
            ),
        }
    ]
    for index, image_id in enumerate(image_ids, start=1):
        if index > 1:
            content.append({"type": "text", "text": f"T{index}:\n"})
        content.append(_image_ref(image_id))
    fields = ",".join(f'"{key}":"VALUE"' for key in _interval_keys(timepoint_count))
    content.append(
        {
            "type": "text",
            "text": _change_output_instruction(
                SEQ_CHANGE_DEFINITIONS.format(
                    schema=f'{{"change_labels":{{{fields}}}}}'
                ),
            ),
        }
    )
    return [_user_message(content)]


def render_seq_dir_messages(
    finding: str,
    image_ids: Sequence[str],
    task: str,
) -> list[dict[str, Any]]:
    """Render one exact complete-image-sequence final-task request."""
    timepoint_count = len(image_ids)
    validate_timepoint_count(timepoint_count)
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                f"Target finding: {finding}\n\n"
                "The following images represent visits ordered "
                "chronologically.\n\nT1:\n"
            ),
        }
    ]
    for index, image_id in enumerate(image_ids, start=1):
        if index > 1:
            content.append({"type": "text", "text": f"T{index}:\n"})
        content.append(_image_ref(image_id))
    content.append(
        {
            "type": "text",
            "text": _task_block(
                task,
                timepoint_count,
                lc=False,
            ),
        }
    )
    return [_user_message(content)]


def render_lc_messages(
    finding: str,
    change_labels: Sequence[str],
    task: str,
) -> list[dict[str, Any]]:
    """Render the shared fresh text-only LC request."""
    labels = validate_change_labels(change_labels)
    timepoint_count = len(labels) + 1
    validate_timepoint_count(timepoint_count)
    intervals = "\n".join(
        f"T{index} to T{index + 1}: {label}"
        for index, label in enumerate(labels, start=1)
    )
    task_block = _task_block(
        task,
        timepoint_count,
        lc=True,
    )
    user = (
        "Change labels between consecutive images:\n\n"
        f"Target finding: {finding}\n\n"
        f"{intervals}\n\n"
        f"{task_block}"
    )
    return [_user_message(user)]


def _interval_keys(timepoint_count: int) -> tuple[str, ...]:
    """Avoid importing mutable prompt state when rendering SEQ keys."""
    return tuple(
        f"T{index}_to_T{index + 1}"
        for index in range(1, timepoint_count)
    )
