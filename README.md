# CheXTrajectory Benchmark

CheXTrajectory evaluates how multimodal language models reason about a single
finding across three or four chronologically ordered chest radiographs. Four
tasks test trajectory shape, turning regions, the most severe visit or plateau,
and the first interval of improvement. Six evaluation settings separate direct
image-based answers, change-label prediction, and reasoning from labels.

This repository provides the benchmark code, protocol, prompts, configuration
examples, and tests.

- [Benchmark protocol](docs/PROTOCOL.md)
- [Prompt templates](docs/PROMPTS.md)
- [Required data](docs/DATA_INPUTS.md)
- [Running the benchmark](experiments/temporal_reasoning_single_finding/README.md)
- [Running on Modal](experiments/temporal_reasoning_single_finding/MODAL_README.md)

## Installation

Use Python 3.11 or later. From the repository root, install dependencies with
`uv`:

```shell
uv sync --locked
```

For Modal execution, include the optional dependency group:

```shell
uv sync --locked --group modal
```

## Data setup

Obtain the source inputs and radiographs separately under their dataset access
terms. Follow [Required data](docs/DATA_INPUTS.md) to prepare the files for
construction, inference, or evaluation, then use the
[run guide](experiments/temporal_reasoning_single_finding/README.md).

## Tests

Run the full test suite, including the Modal adapters:

```shell
uv run --locked --group modal python -m unittest discover \
  -s experiments/temporal_reasoning_single_finding/tests -t . -v
```

The tests use generated records and mocked services and can run without the
datasets or an inference endpoint.

## Repository layout

```text
docs/                                  Protocol, prompts, and data requirements
experiments/temporal_reasoning_single_finding/
  construction/                        Benchmark construction
  protocol/                            Task rules, prompts, and identities
  inference/                           API inference, resume, and retry handling
  validation/                          Preflight, reference and coverage checks
  evaluation/                          Offline evaluation
  modal/                               Cloud execution adapters
  examples/                            Configuration templates
  tests/                               Fixtures and regression tests
```
