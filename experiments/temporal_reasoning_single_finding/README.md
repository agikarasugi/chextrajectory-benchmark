# Running the CheXTrajectory benchmark

This package implements the [benchmark protocol](../../docs/PROTOCOL.md).
Prepare the [required data](../../docs/DATA_INPUTS.md), then run the commands
below from the repository root.

## Construct the primary benchmark

From the repository root, with both strict-order source JSONs and the audit
sidecar at their documented paths:

```shell
uv run --locked python -m \
  experiments.temporal_reasoning_single_finding.construction.constructor_gold_pilot
```

The constructor retains 335 Gold 3TP and 178 Gold 4TP trajectories and assigns
one eligible task per trajectory using the frozen quotas and seed. It writes
`manifest.json`, `3tp.json`, and `4tp.json` under
`artifacts/temporal_reasoning_single_finding/benchmarks/gold_pilot_v1/`.
Construction verifies every reference, records provenance, and atomically
publishes the files. Identical reruns preserve existing files. Conflicting
frozen outputs stop construction with an error.

For a smoke test, derive `smoke_v2` from the Gold pilot:

```shell
uv run --locked python -m \
  experiments.temporal_reasoning_single_finding.construction.constructor_smoke_v2
```

This retains two assigned examples per task and sequence length, or 16 questions
in total.
The mixed pilot and full-task Gold constructors remain available for
compatibility, with their own manifest versions and selection contracts.

## Configure an endpoint run

Copy the two local configuration templates:

```shell
mkdir -p config
cp experiments/temporal_reasoning_single_finding/examples/experiment.example.json config/experiment.json
cp experiments/temporal_reasoning_single_finding/examples/paths.example.json config/paths.json
```

In `config/experiment.json`, set:

- `run_name` to a new lowercase identifier matching `[a-z0-9][a-z0-9._-]*`.
- `model` to the exact identifier advertised by the intended endpoint.
- `base_url` to its OpenAI-compatible API root.
- `benchmark_path` to
  `artifacts/temporal_reasoning_single_finding/benchmarks/gold_pilot_v1/manifest.json`,
  or the chosen compatible smoke manifest.

Set `chexpert_plus_image_root` in `config/paths.json` to the absolute PNG root.
Set `OPENAI_API_KEY` in your environment. The `config/` directory is ignored
by Git.

Optional experiment fields are `reasoning_effort`, `max_completion_tokens`, and
`request_timeout_seconds`. Omit them to use the implementation/endpoint defaults.
Use a new run name when changing explicit reasoning effort or completion limits.
Every prompt requests a short explanation. See [complete rendered prompts](EXAMPLE_PROMPTS.md).

## Preflight and inference

Preflight checks the frozen benchmark, reference answers, prompt fingerprints,
required image files, endpoint model availability, and potential call counts:

```shell
uv run --locked python -m \
  experiments.temporal_reasoning_single_finding.validation.preflight \
  --experiment config/experiment.json --paths config/paths.json
```

After preflight succeeds, start inference:

```shell
uv run --locked python -m \
  experiments.temporal_reasoning_single_finding.inference.run_pipeline \
  --experiment config/experiment.json --paths config/paths.json \
  --workers 1 --progress
```

New runs use `predicted-labels-v1`: valid predicted labels proceed to LC even
when the corresponding fixed-code answer is wrong or undefined. The run manifest
records the policy used for evaluation.

The runner saves prepared inputs, responses, fixed-code results, and the run
manifest under `artifacts/temporal_reasoning_single_finding/runs/<run_name>/`.
Resume with the same model deployment and compatible settings to reuse completed
responses.

Add `--retry-failed` only to reopen terminal technical failures in an existing
run. Parser-invalid answers remain terminal. Retry preparation repairs dependent
results and invalidates stale coverage and evaluation summaries. If interrupted,
resume without the retry flag.

Response records retain endpoint-provided token usage when available and
client-observed request timing. Token counts describe the returned attempt,
and request timing includes network and endpoint queue time.

## Offline evaluation

Select one complete saved run by its exact name:

```shell
uv run --locked python -m \
  experiments.temporal_reasoning_single_finding.evaluation.evaluate \
  --run-name your-run-name
```

The evaluator requires the matching frozen benchmark and complete saved records.
It verifies coverage and identity before writing `evaluation.json`, using
evaluator `2.0.0` and output schema `4`, with the saved LC policy. Repeating
evaluation replaces the summary after validation succeeds.

Scores include task/setting accuracy, balanced accuracy, interval diagnostics,
FC/LC agreement, conditional LC accuracy, and failure rates. See the protocol
for denominators and interpretation.

## Optional cloud execution

The [Modal usage guide](MODAL_README.md) covers GPU-hosted vLLM and CPU runners
connected to an external endpoint. The [technical guide](MODAL_GUIDE.md) describes
staging, execution, durability, and recovery. Install the `modal` dependency
group before using these adapters. The guide covers compute, transfer, and
storage costs for each execution mode.

## Tests

```shell
uv run --locked --group modal python -m unittest discover \
  -s experiments/temporal_reasoning_single_finding/tests -t . -v
uv run --locked python -m compileall -q experiments
uv lock --check
```

The suite covers task rules, prompt fingerprints, construction, inference,
evaluation, and cloud adapters. Runtime preflight also reads the prompt
fingerprints in `tests/fixtures/`.
