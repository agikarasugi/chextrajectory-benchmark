"""Tests for exact and recovered frozen JSON responses."""

import unittest

from experiments.temporal_reasoning_single_finding.inference.json_recovery import (
    parse_final_response,
    parse_pair_response,
    parse_response,
    parse_seq_response,
)


class JsonRecoveryTests(unittest.TestCase):
    """Exercise every approved recovery mode and broad failure reason."""

    def test_exact_pair_response(self) -> None:
        """Accept one exact object with exact spelling and case."""
        result = parse_pair_response('{"change_label":"deterioration"}')
        self.assertEqual(
            result,
            {
                "valid": True,
                "parsed_output": {"change_label": "deterioration"},
                "parsing_mode": "exact",
                "failure_reason": None,
            },
        )

    def test_fenced_and_surrounding_objects(self) -> None:
        """Recover one complete object from either allowed wrapper."""
        fenced = parse_pair_response(
            "```json\n{\"change_label\":\"unchanged\"}\n```"
        )
        self.assertTrue(fenced["valid"])
        self.assertEqual(fenced["parsing_mode"], "fenced")

        prose = "Analysis before. " + ("long prose " * 100)
        surrounded = parse_pair_response(
            prose + '{"change_label":"improvement"} trailing prose.'
        )
        self.assertTrue(surrounded["valid"])
        self.assertEqual(surrounded["parsing_mode"], "surrounding_text")

    def test_balanced_scanner_respects_nested_strings_and_objects(self) -> None:
        """Do not split a candidate at braces inside its JSON strings."""
        result = parse_pair_response(
            'prefix {"change_label":"unchanged","note":"a } { b"} suffix'
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["failure_reason"], "schema_mismatch")

    def test_zero_malformed_and_truncated_objects(self) -> None:
        """Distinguish absent objects from malformed object attempts."""
        self.assertEqual(
            parse_pair_response("ordinary prose")["failure_reason"],
            "no_json_object",
        )
        for text in (
            '{"change_label":}',
            '{"change_label":"unchanged"',
            '{"change_label":NaN}',
            '{"change_label":Infinity}',
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    parse_pair_response(text)["failure_reason"],
                    "malformed_json",
                )

    def test_duplicate_and_multiple_objects(self) -> None:
        """Reject duplicate keys and more than one complete object."""
        duplicate = parse_pair_response(
            '{"change_label":"unchanged","change_label":"improvement"}'
        )
        self.assertEqual(duplicate["failure_reason"], "duplicate_json_key")
        multiple = parse_pair_response(
            '{"change_label":"unchanged"} '
            '{"change_label":"improvement"}'
        )
        self.assertEqual(multiple["failure_reason"], "multiple_json_objects")

    def test_pair_schema_is_exact(self) -> None:
        """Reject wrong fields, types, values, and capitalization."""
        for text in (
            '{"answer":"unchanged"}',
            '{"change_label":"unchanged","extra":1}',
            '{"change_label":1}',
            '{"change_label":"Unchanged"}',
            '[{"change_label":"unchanged"}]',
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    parse_pair_response(text)["failure_reason"],
                    "schema_mismatch",
                )

    def test_seq_schema_requires_exact_interval_keys(self) -> None:
        """Accept 3TP and 4TP vectors and reject interval mismatches."""
        three = parse_seq_response(
            '{"change_labels":{"T1_to_T2":"deterioration",'
            '"T2_to_T3":"improvement"}}',
            3,
        )
        self.assertTrue(three["valid"])
        four = parse_seq_response(
            '{"change_labels":{"T3_to_T4":"unchanged",'
            '"T1_to_T2":"improvement","T2_to_T3":"deterioration"}}',
            4,
        )
        self.assertTrue(four["valid"])

        invalid = (
            '{"change_labels":{"T1_to_T2":"unchanged"}}',
            '{"change_labels":{"T1_to_T2":"unchanged",'
            '"T2_to_T3":"improvement","T3_to_T4":"unchanged"}}',
            '{"change_labels":{"T1_to_T2":"stable",'
            '"T2_to_T3":"improvement"}}',
            '{"change_labels":["unchanged","improvement"]}',
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(
                    parse_seq_response(text, 3)["failure_reason"],
                    "schema_mismatch",
                )

    def test_final_schema_uses_task_and_length_vocabulary(self) -> None:
        """Accept only exact task answers supported at that sequence length."""
        accepted = parse_final_response(
            '{"answer":"peak:T2,trough:T3"}',
            "turning_regions",
            4,
        )
        self.assertTrue(accepted["valid"])
        wrong_case = parse_final_response(
            '{"answer":"Peak:T2,trough:T3"}',
            "turning_regions",
            4,
        )
        self.assertEqual(wrong_case["failure_reason"], "schema_mismatch")
        wrong_length = parse_final_response(
            '{"answer":"fluctuating"}',
            "trajectory_shape",
            3,
        )
        self.assertEqual(wrong_length["failure_reason"], "schema_mismatch")
        wrong_key = parse_final_response(
            '{"change_label":"constant"}',
            "trajectory_shape",
            3,
        )
        self.assertEqual(wrong_key["failure_reason"], "schema_mismatch")

    def test_complete_object_plus_truncated_object_is_malformed(self) -> None:
        """Reject a valid candidate when another object attempt is truncated."""
        result = parse_pair_response(
            '{"change_label":"unchanged"} trailing {"change_label":'
        )
        self.assertEqual(result["failure_reason"], "malformed_json")

    def test_generic_parser_returns_saved_record_fields(self) -> None:
        """Expose valid and invalid statuses without transport-specific fields."""
        valid = parse_response(
            '{"answer":"none"}',
            "final",
            task="first_improvement_between_visits",
            timepoint_count=3,
        )
        self.assertEqual(valid["status"], "valid")
        self.assertEqual(valid["parsed_output"], {"answer": "none"})
        self.assertEqual(valid["parsing_mode"], "exact")
        self.assertIsNone(valid["failure_reason"])

        invalid = parse_response("no object", "pair")
        self.assertEqual(invalid["status"], "invalid_response")
        self.assertEqual(invalid["failure_reason"], "no_json_object")
        with self.assertRaises(ValueError):
            parse_response("{}", "seq")
        with self.assertRaises(ValueError):
            parse_response("{}", "unsupported")


if __name__ == "__main__":
    unittest.main()
