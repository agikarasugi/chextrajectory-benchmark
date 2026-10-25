"""Tests for canonical identities and reuse boundaries."""

import unittest

from experiments.temporal_reasoning_single_finding.protocol.contracts import SETTINGS
from experiments.temporal_reasoning_single_finding.protocol.identities import (
    final_case_id,
    model_input_id,
    pair_prediction_id,
    result_id,
    seq_prediction_id,
)


class IdentityTests(unittest.TestCase):
    """Prove required reuse and forbidden identity sharing."""

    def _pair(self, annotation_source: str = "silver") -> str:
        """Create a representative PAIR identity."""
        return pair_prediction_id(
            dataset="chexpert_plus",
            annotation_source=annotation_source,
            earlier_image_id="train/p/study1/view1_frontal.jpg",
            later_image_id="train/p/study2/view1_frontal.jpg",
            target_finding="edema",
            model="model",
            pair_prompt_version="pair-change-v1",
            json_schema_version="v1",
            benchmark_version="benchmark-v1",
        )

    def test_pair_reuse_and_tier_separation(self) -> None:
        """PAIR omits consumers but separates annotation sources."""
        self.assertEqual(self._pair(), self._pair())
        self.assertNotEqual(self._pair("silver"), self._pair("gold"))

    def test_seq_uses_only_namespaced_source_sequence(self) -> None:
        """SEQ identity changes with its source sequence and ignores metadata."""
        arguments = {
            "source_sequence_id": "chexpert_plus:silver-sequence",
            "model": "model",
            "seq_prompt_version": "sequence-change-v1",
            "json_schema_version": "v1",
            "benchmark_version": "benchmark-v1",
        }
        identity = seq_prediction_id(**arguments)
        self.assertEqual(identity, seq_prediction_id(**arguments))
        arguments["source_sequence_id"] = "chexpert_plus:gold-sequence"
        self.assertNotEqual(identity, seq_prediction_id(**arguments))
        self.assertNotEqual(identity, self._pair())

    def test_uniform_six_setting_results(self) -> None:
        """Every setting has a distinct ID from only case and setting."""
        case = final_case_id(
            source_sequence_id="chexpert_plus:sequence",
            task="trajectory_shape",
        )
        identifiers = {result_id(case_id=case, setting=setting) for setting in SETTINGS}
        self.assertEqual(len(identifiers), 6)

    def test_model_input_fingerprints_safe_messages_not_result(self) -> None:
        """Identical LC input can retain one fingerprint across setting results."""
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "same LC text"},
        ]
        first = model_input_id(
            safe_model_input=messages,
            model="model",
            user_prompt_version="final-tasks-v1",
            json_schema_version="v1",
            benchmark_version="benchmark-v1",
        )
        second = model_input_id(
            safe_model_input=messages,
            model="model",
            user_prompt_version="final-tasks-v1",
            json_schema_version="v1",
            benchmark_version="benchmark-v1",
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
