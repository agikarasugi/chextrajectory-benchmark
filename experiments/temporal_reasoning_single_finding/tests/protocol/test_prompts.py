"""Golden tests for all 19 approved prompt variants."""

import hashlib
import json
import re
import unittest
from pathlib import Path
from typing import Any

from experiments.temporal_reasoning_single_finding.protocol.contracts import TASKS
from experiments.temporal_reasoning_single_finding.protocol.prompts import (
    ROLE_INSTRUCTION,
    render_lc_messages,
    render_pair_messages,
    render_seq_dir_messages,
    render_seq_messages,
)


ROOT = Path(__file__).resolve().parents[4]
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "golden_prompts"
    / "sha256.json"
)
APPROVED = ROOT / "docs" / "PROMPTS.md"

TASK_TITLES = {
    "trajectory_shape": "Trajectory Shape",
    "turning_regions": "Turning Regions",
    "most_severe_visit_or_plateau": "Most Severe Visit or Plateau",
    "first_improvement_between_visits": "First Improvement Between Visits",
}


def _logical_user(messages: list[dict[str, Any]], markers: list[str]) -> str:
    """Replace safe image parts with the approved document markers."""
    content = messages[0]["content"]
    if isinstance(content, str):
        return re.sub(
            r"(T\d to T\d: )(deterioration|unchanged|improvement)",
            r"\1[CHANGE LABEL]",
            content,
        )
    parts: list[str] = []
    marker_iterator = iter(markers)
    for part in content:
        if part["type"] == "text":
            parts.append(part["text"])
        else:
            parts.append(next(marker_iterator) + "\n")
    return "".join(parts)


def _rendered_variants() -> dict[str, str]:
    """Render all approved variants with their logical placeholders."""
    variants = {
        "PAIR change-label prompt": _logical_user(
            render_pair_messages("[FINDING]", "earlier", "later"),
            ["[EARLIER IMAGE ATTACHMENT]", "[LATER IMAGE ATTACHMENT]"],
        )
    }
    for count in (3, 4):
        image_ids = [f"image-{index}" for index in range(1, count + 1)]
        markers = [f"[T{index} IMAGE ATTACHMENT]" for index in range(1, count + 1)]
        variants[f"SEQ change-label prompt: {count}TP"] = _logical_user(
            render_seq_messages("[FINDING]", image_ids), markers
        )
        for task in TASKS:
            title = TASK_TITLES[task]
            variants[f"{title}: {count}TP SEQ-DIR"] = _logical_user(
                render_seq_dir_messages("[FINDING]", image_ids, task), markers
            )
            variants[f"{title}: {count}TP shared LC"] = _logical_user(
                render_lc_messages(
                    "[FINDING]",
                    ["deterioration"] * (count - 1),
                    task,
                ),
                [],
            )
    return variants


class PromptTests(unittest.TestCase):
    """Compare renderers with independent approved-text fingerprints."""

    def test_all_nineteen_golden_variants(self) -> None:
        """Complete wording, whitespace, answers, and image positions are fixed."""
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        rendered = _rendered_variants()
        self.assertEqual(len(rendered), 19)
        actual = {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in rendered.items()
        }
        self.assertEqual(actual, expected)

    def test_fixture_is_anchored_to_approved_review_document(self) -> None:
        """Golden hashes match the independent prompt-review code blocks."""
        document = APPROVED.read_text(encoding="utf-8")
        headings = list(re.finditer(r"^## (.+)$", document, re.MULTILINE))
        approved: dict[str, str] = {}
        for index, heading in enumerate(headings):
            end = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(document)
            )
            match = re.search(
                r"```text\n(.*?)\n```",
                document[heading.end() : end],
                re.DOTALL,
            )
            if match:
                approved[heading.group(1)] = hashlib.sha256(
                    match.group(1).encode("utf-8")
                ).hexdigest()
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(approved, expected)

    def test_prompt_privacy_and_lc_equivalence(self) -> None:
        """Safe prompts contain no payloads and LC origin cannot alter text."""
        labels = ["deterioration", "improvement"]
        messages = render_lc_messages("edema", labels, "trajectory_shape")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0]["content"].startswith(ROLE_INSTRUCTION + "\n\n"))
        self.assertNotIn("PAIR-LC", messages[0]["content"])
        self.assertNotIn("SEQ-LC", messages[0]["content"])
        self.assertNotIn(
            "data:image",
            json.dumps(render_seq_messages("edema", ["a", "b", "c"])),
        )
        self.assertNotIn("Do not use Markdown fences", json.dumps(_rendered_variants()))

    def test_shape_requires_total_switch_count_and_nonempty_monotonic_case(self) -> None:
        """Alternating directions cannot satisfy either single-turn definition."""
        for labels in (
            ["deterioration", "improvement", "deterioration"],
            ["improvement", "deterioration", "improvement"],
            ["unchanged", "unchanged", "unchanged"],
        ):
            text = render_lc_messages(
                "edema", labels, "trajectory_shape",
            )[0]["content"]
            self.assertIn("Ignore unchanged intervals when counting direction switches.", text)
            self.assertEqual(text.count("There is no subsequent switch back."), 2)
            self.assertEqual(text.count("at least one interval remains"), 2)
            self.assertIn(
                "- fluctuating: After ignoring unchanged intervals, there are "
                "two or more direction switches.", text,
            )
            self.assertNotIn("Unchanged throughout.", text)
        text = render_lc_messages(
            "edema", ["unchanged", "unchanged"], "trajectory_shape"
        )[0]["content"]
        self.assertNotIn("fluctuating", text)


    def test_all_requests_are_user_only_with_instruction_first(self) -> None:
        """Every prompt supports servers without system roles."""
        variants = [render_pair_messages("edema", "earlier", "later")]
        for n in (3, 4):
            images = [f"image-{i}" for i in range(n)]
            variants.append(render_seq_messages("edema", images))
            for task in TASKS:
                variants.extend([
                    render_seq_dir_messages(
                        "edema", images, task,
                    ),
                    render_lc_messages(
                        "edema", ["unchanged"] * (n - 1), task,
                    ),
                ])
        for messages in variants:
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "user")
            content = messages[0]["content"]
            first_text = content if isinstance(content, str) else content[0]["text"]
            self.assertEqual(first_text.splitlines()[0], ROLE_INSTRUCTION)
            self.assertEqual(first_text.count(ROLE_INSTRUCTION), 1)


class HistoricalPromptTests(unittest.TestCase):
    """Anchor offline compatibility to the exact previous answer-only snapshots."""

    def test_legacy_reconstruction_matches_frozen_snapshots(self) -> None:
        """The offline adapter reproduces all 19 historical logical messages."""
        from functools import partial
        from unittest.mock import patch
        from experiments.temporal_reasoning_single_finding.tests.protocol import test_prompts
        from experiments.temporal_reasoning_single_finding.validation.legacy_prompts import render_answer_only
        from experiments.temporal_reasoning_single_finding.protocol.prompt_compatibility import LEGACY_FIXTURE_SHA256

        fixture = FIXTURE.with_name('legacy_sha256.json')
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), LEGACY_FIXTURE_SHA256)
        renderers = {
            name: partial(render_answer_only, getattr(test_prompts, name))
            for name in ('render_pair_messages', 'render_seq_messages', 'render_seq_dir_messages', 'render_lc_messages')
        }
        with patch.multiple(test_prompts, **renderers):
            rendered = test_prompts._rendered_variants()
        self.assertEqual({
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in rendered.items()
        }, json.loads(fixture.read_text()))


if __name__ == "__main__":
    unittest.main()
