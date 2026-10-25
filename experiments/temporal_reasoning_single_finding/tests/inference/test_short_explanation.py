"""Verify permanent explanations and compatibility with interrupted saved runs."""

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json, read_jsonl, write_json, write_jsonl,
)
from experiments.temporal_reasoning_single_finding.inference import (
    run_change_labels, run_lc, run_pipeline, run_seq_dir,
)
from experiments.temporal_reasoning_single_finding.inference.derive_fixed_code import (
    build_fixed_code_records,
)
from experiments.temporal_reasoning_single_finding.inference.json_recovery import parse_final_response
from experiments.temporal_reasoning_single_finding.inference.prepare_run import (
    _prompt_preview, build_registries, prepare_run,
)
from experiments.temporal_reasoning_single_finding.protocol.prompts import (
    SHORT_EXPLANATION_INSTRUCTION, SHORT_EXPLANATION_PREFIX,
)
from experiments.temporal_reasoning_single_finding.protocol.prompt_compatibility import (
    FIXTURE_KEY, LEGACY_FIXTURE_SHA256,
)
from experiments.temporal_reasoning_single_finding.tests.inference.test_resume import (
    _item, _manifest,
)
from experiments.temporal_reasoning_single_finding.validation import preflight
from experiments.temporal_reasoning_single_finding.validation.legacy_prompts import (
    build_answer_only_registries,
)


class ShortExplanationTests(unittest.TestCase):
    """Preserve existing prompt identities and completed responses across removal."""

    def test_matches_identities_captured_before_removing_flag(self) -> None:
        """All safe inputs, IDs, and preview match the previous enabled behavior."""
        fixture = Path(__file__).resolve().parents[1] / "fixtures/golden_prompts/short_explanation_identity.json"
        # Captured from e117041 with enable_short_explanation=True.
        expected = read_json(fixture)
        actual = {str(n): build_registries(
            {"model": "model"}, _manifest(),
            [_item("one", n, ["deterioration", "unchanged", "improvement"][:n - 1])],
        ) for n in (3, 4)}
        actual["preview"] = _prompt_preview()
        self.assertEqual({
            key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            for key, value in actual.items()
        }, expected)

    def test_preparation_and_all_final_specs_agree(self) -> None:
        """All final stages submit the same permanent prompts prepared for them."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest()
            manifest_path = root / 'benchmark.json'
            write_json(manifest_path, manifest)
            experiment = {
                'run_name': 'explanation-test', 'model': 'model',
                'base_url': 'http://unused/v1',
                'benchmark_path': str(manifest_path),
            }
            items = [_item('one', 4, ['deterioration', 'unchanged', 'improvement'])]
            with patch(
                'experiments.temporal_reasoning_single_finding.inference.'
                'prepare_run.repository_root', return_value=root,
            ):
                prepared = prepare_run(
                    experiment, manifest_path, manifest, items,
                )
            run_dir = prepared['run_dir']
            cases = read_json(run_dir / 'prepared/final_cases.json')
            selected = build_registries(
                experiment, manifest, items
            )
            self.assertNotIn(
                'enable_short_explanation',
                (run_dir / 'run_manifest.json').read_text(),
            )
            preview = read_json(run_dir / 'prepared/prompt_preview.json')
            for name, messages in preview.items():
                self.assertEqual(
                    SHORT_EXPLANATION_PREFIX in str(messages).replace('\\n', '\n'),
                    True,
                )
            direct = run_seq_dir.build_seq_dir_specs(
                run_dir, cases, 'model'
            )
            reference = run_lc.build_ref_lc_specs(
                run_dir, cases, 'model'
            )
            pairs = {
                r['prediction_id']: {
                    'prediction_id': r['prediction_id'], 'status': 'valid',
                    'predicted_change_label': r['reference_change_label'],
                } for r in selected['pair_inputs']
            }
            seqs = {
                r['prediction_id']: {
                    'prediction_id': r['prediction_id'], 'status': 'valid',
                    'predicted_change_labels': r['reference_change_labels'],
                } for r in selected['seq_inputs']
            }
            pair_fc, seq_fc = build_fixed_code_records(cases, pairs, seqs)
            write_json(run_dir / 'derived/pair_fc.json', pair_fc)
            write_json(run_dir / 'derived/seq_fc.json', seq_fc)
            write_jsonl(run_dir / 'responses/pair.jsonl', list(pairs.values()))
            write_jsonl(run_dir / 'responses/seq.jsonl', list(seqs.values()))
            all_specs = direct + reference
            for source in ('PAIR', 'SEQ'):
                specs, _ = run_lc.build_predicted_lc_specs(
                    run_dir, cases, 'model', source,
                )
                all_specs.extend(specs)
                self.assertEqual(
                    [r['safe_model_input'] for r in specs],
                    [r['safe_model_input'] for r in reference],
                )
            for spec in direct + reference:
                self.assertEqual(
                    SHORT_EXPLANATION_INSTRUCTION in str(
                        spec['safe_model_input']
                    ).replace('\\n', '\n'), True,
                )
            answers = {
                json.dumps(spec['safe_model_input']): spec['reference_answer']
                for spec in all_specs
            }

            def submit(_client, _model, messages, _image_root):
                """Return synthetic final text without accessing image files."""
                answer = answers[json.dumps(messages)]
                text = json.dumps({'answer': answer})
                text = 'The supplied changes identify this answer.\n' + text
                return {'kind': 'text', 'raw_text': text, 'error': None}

            with ExitStack() as stack:
                for module in (run_seq_dir, run_lc):
                    stack.enter_context(patch.object(module, 'require_exact_model'))
                stack.enter_context(patch(
                    'experiments.temporal_reasoning_single_finding.inference.'
                    'run_change_labels.submit_messages', side_effect=submit,
                ))
                run_seq_dir.run_seq_dir(
                    run_dir, experiment, root / 'unused-images', client=object(),
                )
                run_lc.run_ref_lc(
                    run_dir, experiment, client=object(),
                )
                for source in ('PAIR', 'SEQ'):
                    run_lc.run_predicted_lc(
                        run_dir, experiment, source, client=object(),
                    )
            for name in ('seq_dir', 'ref_lc', 'pair_lc', 'seq_lc'):
                records = read_jsonl(run_dir / f'responses/{name}.jsonl')
                self.assertEqual(len(records), len(cases))
                self.assertTrue(all(r['correct'] for r in records))
                self.assertTrue(all(r['status'] == 'valid' for r in records))

    def test_explanation_keeps_exact_answer_validation(self) -> None:
        """Prose is accepted without relaxing canonical or ambiguity checks."""
        prefix = 'The first interval improves. Later changes do not alter it.\n'
        result = parse_final_response(
            prefix + '{"answer":"T1-T2"}', 'first_improvement_between_visits', 3
        )
        self.assertTrue(result['valid'])
        self.assertEqual(result['parsed_output'], {'answer': 'T1-T2'})
        for suffix in (
            '{"answer":"T3-T4"}',
            '{"answer":"T1-T2","explanation":"extra field"}',
            '{"answer":"T1-T2"}\n{"answer":"none"}',
        ):
            self.assertFalse(parse_final_response(
                prefix + suffix, 'first_improvement_between_visits', 3
            )['valid'])

    def test_interrupted_legacy_run_reuses_completed_requests(self) -> None:
        """The old fixture revision resumes without changing saved outputs or IDs."""
        for n in (3, 4):
            with self.subTest(timepoints=n), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = _manifest()
                manifest_path = root / 'benchmark.json'
                write_json(manifest_path, manifest)
                experiment = {
                    'run_name': 'resume-test', 'model': 'model',
                    'base_url': 'http://unused/v1', 'benchmark_path': str(manifest_path),
                }
                items = [_item('one', n, ['deterioration'] * (n - 1))]
                with patch(
                    'experiments.temporal_reasoning_single_finding.inference.'
                    'prepare_run.repository_root', return_value=root,
                ):
                    run_dir = prepare_run(experiment, manifest_path, manifest, items)['run_dir']
                    # Baseline fingerprint tests above establish that these prepared
                    # artifacts exactly match those written by the old enabled mode.
                    saved_manifest_path = run_dir / 'run_manifest.json'
                    saved_manifest = read_json(saved_manifest_path)
                    saved_manifest['prompt_fixture_checksums'] = {FIXTURE_KEY: LEGACY_FIXTURE_SHA256}
                    write_json(saved_manifest_path, saved_manifest)
                    original_manifest = saved_manifest_path.read_bytes()
                    original_prepared = {
                        path.name: path.read_bytes() for path in (run_dir / 'prepared').iterdir()
                    }
                    registries = build_registries(experiment, manifest, items)
                    answers = {}
                    for key in ('pair_inputs', 'seq_inputs'):
                        for spec in registries[key]:
                            output = {'change_label': 'deterioration'} if key == 'pair_inputs' else {
                                'change_labels': {f'T{i}_to_T{i + 1}': 'deterioration' for i in range(1, n)}
                            }
                            answers[json.dumps(spec['safe_model_input'], sort_keys=True)] = output
                    submitted = []

                    def submit(_client, _model, messages, _image_root):
                        """Interrupt after a saved response without touching image files."""
                        key = json.dumps(messages, sort_keys=True)
                        submitted.append(key)
                        if len(submitted) == 2:
                            raise RuntimeError('synthetic logic failure')
                        return {'kind': 'text', 'raw_text': 'The finding worsens.\n' + json.dumps(answers[key]), 'error': None}

                    with (
                        patch.object(run_change_labels, 'require_exact_model'),
                        patch.object(run_change_labels, 'submit_messages', side_effect=submit) as transport,
                    ):
                        with self.assertRaisesRegex(RuntimeError, 'synthetic logic failure'):
                            run_change_labels.run_change_labels(run_dir, experiment, root / 'unused', client=object())
                        response_path = run_dir / 'responses/pair.jsonl'
                        saved_response = response_path.read_bytes()
                        self.assertEqual(len(read_jsonl(response_path)), 1)
                        benchmark = preflight.ValidatedBenchmark(manifest_path, manifest, {n: items}, items)
                        preflight._check_run_target(run_dir, experiment, benchmark)
                        prepare_run(experiment, manifest_path, manifest, items)
                        result = run_change_labels.run_change_labels(run_dir, experiment, root / 'unused', client=object())
                        self.assertEqual(result['pair']['reused'], 1)
                        self.assertEqual(result['pair']['submitted'], n - 2)
                        self.assertEqual(result['seq']['submitted'], 1)
                        self.assertEqual(submitted.count(submitted[0]), 1)
                        self.assertTrue(response_path.read_bytes().startswith(saved_response))
                        self.assertEqual(saved_manifest_path.read_bytes(), original_manifest)
                        for name, value in original_prepared.items():
                            self.assertEqual((run_dir / 'prepared' / name).read_bytes(), value)
                        # Completed records must still match the current input ID.
                        rows = read_jsonl(response_path)
                        rows[0]['model_input_id'] = 'changed-input'
                        write_jsonl(response_path, rows)
                        transport.reset_mock()
                        with self.assertRaisesRegex(ValueError, 'identity conflict'):
                            run_change_labels.run_change_labels(run_dir, experiment, root / 'unused', client=object())
                        transport.assert_not_called()

    def test_answer_only_preparation_cannot_resume_inference(self) -> None:
        """Known fixture compatibility does not permit a change in actual prompts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest()
            manifest_path = root / 'benchmark.json'
            write_json(manifest_path, manifest)
            experiment = {
                'run_name': 'old-answer-only', 'model': 'model',
                'base_url': 'http://unused/v1', 'benchmark_path': str(manifest_path),
            }
            items = [_item('one', 3, ['deterioration', 'improvement'])]
            with patch(
                'experiments.temporal_reasoning_single_finding.inference.'
                'prepare_run.repository_root', return_value=root,
            ):
                run_dir = prepare_run(experiment, manifest_path, manifest, items)['run_dir']
                legacy = build_answer_only_registries(experiment, manifest, items)
                for key in ('pair_inputs', 'seq_inputs'):
                    write_json(run_dir / 'prepared' / f'{key}.json', legacy[key])
                saved = read_json(run_dir / 'run_manifest.json')
                saved['prompt_fixture_checksums'] = {FIXTURE_KEY: LEGACY_FIXTURE_SHA256}
                write_json(run_dir / 'run_manifest.json', saved)
                with self.assertRaisesRegex(ValueError, 'conflicts'):
                    prepare_run(experiment, manifest_path, manifest, items)
                with patch.object(run_change_labels, 'submit_messages') as transport:
                    with self.assertRaisesRegex(ValueError, 'conflicts'):
                        run_change_labels.run_change_labels(run_dir, experiment, root / 'unused', client=object())
                    transport.assert_not_called()

    def test_retired_cli_flag_is_rejected(self) -> None:
        """Every local entrypoint removes the option and rejects stale commands."""
        for module in (run_pipeline, run_change_labels, run_seq_dir, run_lc, preflight):
            parser = getattr(module, 'parse_args', None) or module._parse_args
            argv = ['runner', '--experiment', 'experiment.json']
            if module is not run_lc:
                argv += ['--paths', 'paths.json']
            with self.subTest(module=module.__name__), patch('sys.argv', argv):
                self.assertFalse(hasattr(parser(), 'enable_short_explanation'))
            with (
                self.subTest(module=module.__name__),
                patch('sys.argv', argv + ['--enable-short-explanation']),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                parser()
            self.assertEqual(error.exception.code, 2)

    def test_unknown_fixture_revision_is_rejected_at_every_gate(self) -> None:
        """Fixture compatibility is explicit, and never ignores an unknown digest."""
        from experiments.temporal_reasoning_single_finding.validation.validate_run import _validate_run_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest()
            manifest_path = root / 'benchmark.json'
            write_json(manifest_path, manifest)
            experiment = {
                'run_name': 'unknown-fixture', 'model': 'model',
                'base_url': 'http://unused/v1', 'benchmark_path': str(manifest_path),
            }
            items = [_item('one', 3, ['deterioration', 'improvement'])]
            with patch(
                'experiments.temporal_reasoning_single_finding.inference.'
                'prepare_run.repository_root', return_value=root,
            ):
                run_dir = prepare_run(experiment, manifest_path, manifest, items)['run_dir']
                path = run_dir / 'run_manifest.json'
                saved = read_json(path)
                benchmark = preflight.ValidatedBenchmark(manifest_path, manifest, {3: items}, items)
                for invalid in ({FIXTURE_KEY: '0' * 64}, {}, {FIXTURE_KEY: LEGACY_FIXTURE_SHA256, 'extra': 'hash'}):
                    with self.subTest(checksums=invalid):
                        write_json(path, {**saved, 'prompt_fixture_checksums': invalid})
                        with self.assertRaisesRegex(ValueError, 'prompt-fixture'):
                            prepare_run(experiment, manifest_path, manifest, items)
                        with self.assertRaisesRegex(ValueError, 'prompt-fixture'):
                            preflight._check_run_target(run_dir, experiment, benchmark)
                        with self.assertRaisesRegex(ValueError, 'prompt-fixture'):
                            _validate_run_manifest(path)
