# Run the experiment on Modal

Use this document for setup, submission, recovery, and result retrieval. Run all
commands from the repository root. The [experiment README](README.md) defines
benchmark construction, experiment options, and offline evaluation. The
[technical guide](MODAL_GUIDE.md) explains the implementation.

Choose one of the two execution modes:

- [With vLLM serving](#with-vllm-serving): run your checkpoint and the experiment
  in the same Modal GPU container.
- [Endpoint](#endpoint): run the experiment in a Modal CPU container against an
  existing Shared or Dedicated Endpoint.

Both modes reuse staged benchmark files and images. Start with `smoke_v1` before
running the full benchmark.

## With vLLM serving

This worked example uses Qwen3.5-9B on one L40S with vLLM v0.28.0 and the
two-trajectory `smoke_v1` benchmark. Run all commands from the repository root.
If the smoke benchmark does not exist yet, follow
[Construct the benchmarks](README.md#construct-the-primary-benchmark) first. You also need the
corresponding CheXpert Plus PNGs locally for the initial upload. You can select
the model checkpoint and vLLM settings after staging the data. The JSONs below
provide a starting configuration. Use the smoke run to check model loading,
inference, and result retrieval.

Steps 1 and 2 prepare the reusable local setup and cloud data. For subsequent
runs on the same benchmark, start at Step 3. Step 4 is needed only when the
selected checkpoint is not already staged.

### 1. One-time setup: install Modal and create configuration files

```shell
uv sync --group modal
uv run --group modal modal setup

mkdir -p tmp/modal
```

`modal setup` connects the CLI to your account. Create the JSON files shown
below in `tmp/modal/`, which is ignored by Git. Reuse them for subsequent runs.

### 2. Stage data once per benchmark and Volume prefix

Complete this step once for each benchmark in the chosen set of Volumes.
Skip it for later runs using the same staged benchmark, including runs with
different models or serving settings. Repeat it only for a new benchmark or
Volume prefix, or to recover an incomplete upload.

First, edit `tmp/modal/paths.json`: set `chexpert_plus_image_root` to your local
CheXpert Plus PNG directory. For example:

```json
{
  "chexpert_plus_image_root": "/Volumes/Untitled/chexpert/PNG"
}
```

This path is machine-specific; replace it if your PNG directory differs.
Use `temporal-single` as the Volume prefix and `smoke_v1` as the benchmark.
Model selection, the other configuration
files, and a resource profile are not needed for staging.

Inspect the inventory before uploading:

```shell
uv run --group modal python -m experiments.temporal_reasoning_single_finding.modal.stage \
  --benchmark artifacts/temporal_reasoning_single_finding/benchmarks/smoke_v1/manifest.json \
  --paths tmp/modal/paths.json \
  --volume-prefix temporal-single
```

The command hashes the selected local images and reports counts and bytes
without uploading. Even repeated inventory commands read those image files. To transfer the inputs:

```shell
uv run --group modal python -m experiments.temporal_reasoning_single_finding.modal.stage \
  --benchmark artifacts/temporal_reasoning_single_finding/benchmarks/smoke_v1/manifest.json \
  --paths tmp/modal/paths.json \
  --volume-prefix temporal-single \
  --upload
```

Staging uploads the frozen benchmark and its referenced unique PNGs. Matching
staged inputs are reused across runs, including runs with different models.
The `--upload` command transfers files in this order:

1. Upload missing images to `temporal-single-images`, saving the image index
   after each completed batch.
2. Upload `manifest.json` and its referenced benchmark JSON files to
   `temporal-single-benchmarks`. Existing matching files are reused.
3. Publish `modal_images.json` beside the remote manifest, recording the
   benchmark checksum and required image inventory, then print `Staging complete`.

The manifest is therefore transferred during this staging step, after the
images. It is not uploaded by the offline configuration check or experiment
launcher. Selecting a different `benchmark_path` in `experiment.json` requires
that benchmark to have been staged too. For example, uploading `smoke_v1` does
not also upload `gold_pilot_v1`, although the two uploads can reuse shared images.

For `smoke_v1` with the prefix `temporal-single`, the manifest locations are:

| Location | Manifest path |
| --- | --- |
| Local repository | `artifacts/temporal_reasoning_single_finding/benchmarks/smoke_v1/manifest.json` |
| `temporal-single-benchmarks` Volume | `/smoke_v1/manifest.json` |
| Running Modal container | `/app/artifacts/temporal_reasoning_single_finding/benchmarks/smoke_v1/manifest.json` |

Use the repository-relative path in both staging's `--benchmark` option and
the experiment JSON's `benchmark_path`. The launcher mounts the benchmark
Volume at the matching container directory and runs from `/app`, so the JSON
path stays the same when running on Modal.

The terminal reports local hashing, remote checks, reused files, and confirmed
upload counts in batches of up to 16 images. Confirmed file bytes are reported
after each batch completes; they are not a live network-byte counter. Long
operations print an elapsed-time heartbeat every 10 seconds. Wait for
`Staging complete` before launching an experiment.

The image index is saved after every completed batch. After cancellation,
rerun the same command to reuse indexed files. For unindexed paths, a CPU job
on Modal checks the mounted files and returns only sizes and checksums. Image
payloads are never downloaded to your computer for recovery. This job requires
no model or resource profile and runs only when unindexed paths need checking.
An empty Volume listing can mean no batch finished.
If the command reports a leftover staging lock, confirm the previous uploader
has stopped before removing the `staging` key from `temporal-single-locks` in
Modal. Do not remove a lock held by another active upload.

### 3. Configure the model and experiment

For each new run, review these local files and choose a new `run_name`.
Use the benchmark path and Volume prefix already staged in Step 2. Save the
following contents in the three local files. These are explicit settings for
this Qwen3.5-9B smoke run, rather than launcher defaults.

| Local file | What to configure |
| --- | --- |
| `tmp/modal/profile.json` | Set a model-compatible vLLM `image` with a version tag or digest, the Modal `gpu` allocation, and job timeouts. Keep `volume_prefix` as `temporal-single` to use the commands below unchanged. |
| `tmp/modal/serving.json` | Set `model_path` to the checkpoint destination on Modal, such as `/models/checkpoints/Qwen/Qwen3.5-9B`. Choose `served_model_name` and supply the model's `vllm_args`. This path is on the remote model Volume. |
| `tmp/modal/experiment.json` | Set a unique `run_name`, make `model` equal `served_model_name`, and select the frozen `benchmark_path`. Keep `base_url` as `http://127.0.0.1:8000/v1`. This example selects `smoke_v1` and names the run `temporal-single-v1-qwen35-9b-smoke_v1-001`. |

#### `profile.json`: environment, resources, and storage

This file selects the environment and resources for the remote job. The first
six fields below are required. The launcher does not supply defaults for
missing required fields. Save this as `tmp/modal/profile.json`:

```json
{
  "image": "vllm/vllm-openai:v0.28.0",
  "gpu": "L40S",
  "timeout": 21600,
  "startup_timeout": 900,
  "commit_interval": 60,
  "volume_prefix": "temporal-single"
}
```

| Field | Purpose and accepted value | Example value |
| --- | --- | --- |
| `image` | Registry image containing a compatible Python and vLLM environment. Use an explicit version tag or SHA-256 digest. Untagged images and `:latest` are rejected. Changing this selects a different container environment. | `"vllm/vllm-openai:v0.28.0"` |
| `gpu` | GPU type and optional count for inference. For example, `"L40S"` allocates one GPU and `"L40S:2"` requests two. The local validator accepts counts from 1 to 8; Modal determines whether the requested GPU is available. | `"L40S"` |
| `timeout` | Maximum duration of the GPU job in seconds, including model startup and experiment execution. Allowed range: 1 to 86,400 seconds. | `21600` (6 hours) |
| `startup_timeout` | How long the launcher waits for vLLM to expose the configured model before failing, in seconds. Must be positive and shorter than `timeout`. Increase it if the selected model takes longer to load. | `900` (15 minutes) |
| `commit_interval` | Interval in seconds between explicit saves to the results Volume while an experiment stage runs. This controls persistence frequency, not inference speed or request concurrency. Allowed range: 1 to 86,400 seconds. | `60` |
| `volume_prefix` | Shared prefix for the persistent `-images`, `-benchmarks`, `-models`, and `-results` Volumes. Use 1 to 40 lowercase letters, digits, or hyphens, beginning with a letter or digit. Keep it consistent with staging commands. Changing it selects a separate set of stores. | `"temporal-single"` |
| `region` (optional) | Compute region passed to Modal for inference and checkpoint download. Omit it to leave placement to Modal. This field does not configure Volume storage residency. | Omitted |
| `pip_packages` (optional) | Additional Python dependencies installed into the inference image. Every entry must have an exact version pin, such as `"package==1.2.3"`. Omit it or use `[]` when the selected image needs no additions. Changing these dependencies changes the image build. | Omitted |

#### `serving.json`: checkpoint and vLLM settings

The three model fields are required. The logging field `vllm_env` is optional.
This file selects what vLLM loads and how it serves the model within the
resources selected by `profile.json`. Save this
as `tmp/modal/serving.json`:

```json
{
  "model_path": "/models/checkpoints/Qwen/Qwen3.5-9B",
  "served_model_name": "Qwen/Qwen3.5-9B",
  "vllm_args": [
    "--dtype", "bfloat16",
    "--tensor-parallel-size", "1",
    "--gpu-memory-utilization", "0.92",
    "--max-model-len", "65536",
    "--max-num-seqs", "1",
    "--limit-mm-per-prompt", "{\"image\": 4}",
    "--reasoning-parser", "qwen3",
    "--override-generation-config", "{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0}"
  ],
  "vllm_env": {
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_LOG_STATS_INTERVAL": "60"
  }
}
```

| Field | Purpose and accepted value | Example value |
| --- | --- | --- |
| `model_path` | Absolute checkpoint directory on the remote model Volume, strictly below `/models`. It must match the checkpoint downloader's `--model-path` and contain a completed download. It is not a local path or a Hub repository ID. | `"/models/checkpoints/Qwen/Qwen3.5-9B"` |
| `served_model_name` | Name exposed by the model API. You choose this label; it does not select or download weights. It must exactly match `model` in `experiment.json`. | `"Qwen/Qwen3.5-9B"` |
| `vllm_args` | Ordered list of command-line argument strings passed to vLLM. Use it for model-specific serving settings supported by the selected image. An empty list `[]` uses vLLM defaults apart from launcher-managed options. | See the JSON above |
| `vllm_env` | Optional logging environment variables passed to the remote vLLM process. These may change on resume. | `{"VLLM_LOGGING_LEVEL": "INFO", "VLLM_LOG_STATS_INTERVAL": "60"}` |

In `vllm_args`, put each option and value in separate strings. For example,
`["--max-model-len", "65536", "--tensor-parallel-size", "1"]`. Use full option
names and choose values supported by your model and vLLM version. Parallelism
must fit the GPU allocation. The launcher supplies the model path, served name,
host, and port, so do not repeat those options in `vllm_args`.
Standalone flags occupy one entry. A JSON-valued option takes one escaped JSON
string as its value, as shown by `--limit-mm-per-prompt` in the JSON above.
Duplicate options and overrides of managed connection, authentication, or
logging settings are rejected. Other model-specific options pass through to
vLLM, which checks whether it supports them. Both configuration files reject
unknown top-level fields.

##### Configure vLLM logging before launch

1. Open the serving JSON passed to your launch command's `--serving` option,
   for example `tmp/modal/serving.json`.
2. Add or edit `vllm_env` at the top level, alongside `model_path`,
   `served_model_name`, and `vllm_args`. The complete example above shows the
   placement and required comma after `vllm_args`.
3. Set `VLLM_LOGGING_LEVEL` to `"INFO"` and `VLLM_LOG_STATS_INTERVAL` to the
   interval in seconds as a string. Use `"60"` for once per minute or `"120"`
   for once every two minutes. Set the level to `"WARNING"` to suppress statistics.
4. Save the file before continuing to the launch step below. When resuming an
   existing run, keep the same `run_name`. Logging changes do not invalidate
   the run and take effect when the server next starts.

For once-per-minute statistics, the field is:

```json
"vllm_env": {
  "VLLM_LOGGING_LEVEL": "INFO",
  "VLLM_LOG_STATS_INTERVAL": "60"
}
```

Only these two environment variables are accepted. Values must be strings.
The interval must be finite and positive, in seconds. Accepted levels are
`DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. Without an explicit level,
the launcher retains `WARNING`, which suppresses normal statistics. An omitted
interval inherits the container setting or vLLM's default (normally 10 seconds).
These settings apply when the server starts. Adding, changing, or removing them
is allowed when resuming, including runs created before this setting existed.
Each attempt records its settings without overwriting original provenance.
Other INFO messages are not throttled by the statistics interval.
If `--disable-log-stats` is already in `vllm_args`, statistics remain disabled.
Changing that argument still requires a new run name.

To reduce routine server output, add `--uvicorn-log-level`, `warning`,
`--disable-uvicorn-access-log`, and `--disable-log-stats` as separate entries in
`serving.json`'s `vllm_args`. vLLM, pipeline, and evaluator stdout/stderr are
forwarded to Modal logs. Dependency output and error tracebacks can still appear.
Choose these serving arguments before the first run, since resume checks their
complete ordered list.

#### `experiment.json`: name and select the smoke run

Save this as `tmp/modal/experiment.json`. Its `model` matches the serving name,
and its `benchmark_path` matches the upload in Step 2:

```json
{
  "run_name": "temporal-single-v1-qwen35-9b-smoke_v1-001",
  "model": "Qwen/Qwen3.5-9B",
  "base_url": "http://127.0.0.1:8000/v1",
  "benchmark_path": "artifacts/temporal_reasoning_single_finding/benchmarks/smoke_v1/manifest.json"
}
```

**Optional `reasoning_effort`:** add this as a top-level field in
`tmp/modal/experiment.json`, alongside `model` and `benchmark_path`. For example,
add `"reasoning_effort": "high"` to request high reasoning effort, but only if
the selected model and vLLM server support that value. The runner forwards the
value unchanged. The example above omits this optional setting, leaving the
model/server default in effect.

To change reasoning effort, edit that field. To omit the setting, remove the
field entirely. If the run has already started, use a new `run_name` when adding,
changing, or removing it. Resuming a saved run requires its original setting.

**Optional `max_completion_tokens`:** add `"max_completion_tokens": 2048` as a
top-level field in this experiment JSON, alongside `model` and `benchmark_path`.
Choose a positive integer suitable for your model. Omit the field or set it to
`null` to use the endpoint/model default.
It can be used together with `reasoning_effort`.

This caps generated tokens per API attempt, including reasoning tokens where
the endpoint counts them in completion usage. It does not limit or truncate
input text or images. The server's context window and output limits still
apply. The runner forwards the value unchanged to every model stage and stops
on a non-retryable rejection without removing or increasing the cap.

Use a new `run_name` when adding, changing, or removing an explicit limit.
A small cap can leave incomplete JSON or no answer text. Incomplete JSON keeps
its parser-invalid outcome. Empty answers can incur the existing three attempts
with the same cap. Budget for those retries separately.

#### Check the configuration

Select the resource profile and check the configuration:

```shell
uv run --group modal python -m experiments.temporal_reasoning_single_finding.modal.check \
  --profile tmp/modal/profile.json \
  --serving tmp/modal/serving.json \
  --experiment tmp/modal/experiment.json
```

This check is offline. It validates configuration consistency, but does not
load the model or check remote files. Pass `--profile` on every command.
The launch argument selects this project's resource JSON and ignores old profile
environment variables. These Python launchers keep resource selection separate from Modal's own
authentication-profile option.

### 4. Download the checkpoint once per revision

Skip this step if the selected checkpoint is already complete at `model_path`
in the current model Volume.

This example pins Qwen3.5-9B to the model repository commit
[`c202236235762e1c871ad0ccb60c8ee5ba337b9a`](https://huggingface.co/Qwen/Qwen3.5-9B/commit/c202236235762e1c871ad0ccb60c8ee5ba337b9a).
The destination matches `model_path` in the serving JSON. Pass the resource
profile explicitly when downloading:

```shell
uv run --group modal python -m experiments.temporal_reasoning_single_finding.modal.download_job \
  --profile tmp/modal/profile.json \
  --repo-id "Qwen/Qwen3.5-9B" \
  --revision "c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
  --model-path /models/checkpoints/Qwen/Qwen3.5-9B
```

`--revision` is optional. For a new, empty destination, omission selects the
current commit on `main` inside the CPU worker and records its full SHA before
downloading. For an existing destination, omission resumes or reuses the recorded
commit without checking for updates. The repository and any explicitly supplied
SHA must match the saved identity. Use a new destination to download a newer
commit, and update the serving JSON's `model_path` accordingly. Retain explicit
SHAs for controlled benchmark comparisons.

To authenticate, create a Modal Secret containing `HF_TOKEN`, for example
named `huggingface-download`, and append `--secret huggingface-download` to the
command. Use your actual Secret name if different. Keep the token itself out
of these JSON files and commands. The CPU download job transfers weights
directly from the Hub to Modal and records file paths and sizes after download
completion before marking the checkpoint ready. Wait for completion before launching inference. Later runs can reuse
this checkpoint without downloading it through your computer. Normal launches
check completion records, identities, file existence, and sizes without rehashing
model weights or benchmark images. Existing checksum-bearing markers remain
usable. Keep completed input directories unchanged, since metadata checks do
not detect same-size content changes.

The download command starts its own CPU-only Modal App. It does not build the
vLLM image or allocate the GPU from `profile.json`. The downloader validates
the explicit `--profile` and uses
its `volume_prefix` and optional `region`. The inference image is built when
you launch the experiment in Step 5.

#### Recover from `Checkpoint download is already locked`

Worker interruption can leave a lock in the persistent Modal Dict because
the worker may not reach its cleanup code. GPU runs, endpoint runs, and
checkpoint downloads now accept an existing lock when its stored FunctionCall
ID matches the current call. This lets the owning call resume after a restart,
including with a lock written by the earlier code. These entrypoints use
single-input `.spawn()` submissions. The call-level lock helper must not be
used for multiple inputs sharing a FunctionCall ID, such as `.spawn_map()`.
Modal automatically reschedules inputs after container failure, independently
of the downloader's `retries=0` setting. See
[Modal execution semantics](https://modal.com/docs/guide/functions#execution-semantics).
The worker-loss message alone does not establish the original cause. Inspect
container events and memory metrics in the Modal dashboard for that diagnosis.

A manually submitted replacement has a new FunctionCall ID and still rejects
the old owner's lock. Its error reports the key and both call IDs. Recovery
remains explicit because Dict has no atomic compare-and-delete operation:
checking the old owner and then popping the key could remove a competing
writer's newly acquired lock. No age-based lock expiry or automatic deletion
is used. See [Modal Dict operations](https://modal.com/docs/reference/modal.Dict).

For a replacement submission, run the following from the repository root.
First inspect the lock owner:

```shell
uv run --group modal modal dict get temporal-single-locks \
  'model:/models/checkpoints/google/gemma-4-31B'
```

Find the returned FunctionCall ID in the Modal dashboard. Confirm that it has
finished or been stopped and that no other download is writing to this model
directory. Keep other launchers for this destination stopped throughout
inspection and removal. Then remove only this lock:

```shell
uv run --group modal python - <<'PY'
import modal

locks = modal.Dict.from_name("temporal-single-locks")
key = "model:/models/checkpoints/google/gemma-4-31B"
owner = locks.pop(key, None)
print(f"Removed lock owned by: {owner}" if owner is not None else "Lock already absent")
PY
```

For another checkpoint, replace the path after `model:` with the exact
`--model-path` used for its download. If the profile uses a different
`volume_prefix`, replace `temporal-single-locks` with `<volume_prefix>-locks`.
Use the same Modal environment and account as the original download.
Never remove an active writer's lock or clear the whole Dict.

Rerun the original download command with the same profile, repository,
destination, revision selection, and Secret. Keep the model files, cache, and
`modal_download.json`. Removing the Dict entry does not delete those files.
Recovery uses the saved revision and whatever download progress persisted to
the Volume. A `9/10 files` progress message does not prove that all nine files
were durably saved. Wait for successful completion before launching inference.

This procedure is still needed for locks left by an ended call when submitting
a new call. Automatic restarts of the owning call can reuse their existing lock.
An already running App retains its submitted source until it is relaunched.

### 5. Launch the experiment

```shell
uv run --group modal python -m experiments.temporal_reasoning_single_finding.modal.gpu_job \
  --profile tmp/modal/profile.json \
  --serving tmp/modal/serving.json \
  --experiment tmp/modal/experiment.json \
  --workers 1
```

This starts the GPU job, runs preflight and inference, and evaluates the saved
responses. The Modal App name is the experiment's `run_name`, so each run is
identifiable in the dashboard. Follow progress in the terminal or Modal dashboard. The launcher
submits with `.spawn()` and prints a FunctionCall ID before waiting. Together
with `--detach`, this lets the submitted job continue after your local connection
closes. Wait for that ID before treating submission as confirmed. All prompts request
short explanations automatically. Use a new run name when changing the model,
benchmark, or model inputs.

#### Resume requirements

For a completed run with terminal `response_error` outcomes, add
`--retry-failed` to the same `modal.gpu_job` or `modal.endpoint_job` command.
The option reaches the existing pipeline under the run lock. It backs up and
reopens technical failures, repairs affected downstream results, and refreshes
evaluation after inference. It does not change the experiment identity or
the per-call retry limit. See the [retry behavior and interruption guidance](README.md#preflight-and-inference).
Normal resource allocation still applies, including when no failures need retrying.

Repeat the launch command using the existing `run_name`. The following must
remain compatible with the saved run:

| Item | What must stay the same |
| --- | --- |
| `experiment.json` | Normalized configuration values, including `run_name`, model name, benchmark path, base URL, and any generation settings such as `reasoning_effort` and `max_completion_tokens`. `request_timeout_seconds` may change. |
| `serving.json` | `model_path`, `served_model_name`, and generation/model arguments stay fixed. GPU memory utilization, context length, sequence and batch-token limits, and tensor/pipeline parallelism may change. Other arguments remain protected. |
| Model checkpoint | The saved checkpoint completion record, including its pinned revision and file inventory. Files must remain present with their recorded sizes. |
| Benchmark and prompts | Frozen benchmark identity and checksums, prompt/schema versions, a supported prompt-fixture revision, and identical prepared model inputs. A code fix is allowed, but changing these inputs can still block resume. |
| Saved responses | Existing records must still match their request identities and pass validation. Compatible completed records are reused. |

For runs created with the former `--enable-short-explanation` flag, remove the
flag when relaunching. Saved Modal provenance with
`enable_short_explanation: true` is compatible with the permanent behavior.
The prior prompt-fixture revision is accepted without rewriting the saved run
manifest or original provenance. Exact prompt text and input IDs still must
match. A saved `false` mode cannot resume inference, though its responses remain
available for offline scoring. Unknown fixture revisions are rejected.

Configuration file names and local locations may change; the launcher compares
their loaded values. Keep the saved run and original provenance intact.

`request_timeout_seconds` may increase, decrease, or be omitted to use the
300-second default when resuming. This applies to both Modal runners and
standalone inference stages. Each Modal attempt records its configured timeout
without rewriting the original run manifest or provenance. Changing the timeout
does not retry existing completed responses or exhausted technical-error records.

The following vLLM options may be added, removed, or changed on resume:
`--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`,
`--max-num-batched-tokens`, `--tensor-parallel-size`, and
`--pipeline-parallel-size`. Space-separated and `--option=value` forms are
supported. GPU allocation must still support the selected parallelism.
Context may increase or decrease, including switching from `auto` to an explicit
limit. Choose a limit that accommodates the inputs and unchanged output allowance.
When no output allowance is explicit, a different context may also change the
available generation length. Sampling, thinking/chat-template options, dtype,
quantization, generation configuration, and checkpoint identity remain protected.
Existing responses are reused and only missing requests are submitted. Original
provenance stays intact, and each attempt records the actual serving configuration.

`--workers`, code, and all profile fields may change, including GPU, timeouts,
image, optional dependencies, region, commit interval, and Volume prefix. Worker
count, source, profile, and installed package versions are recorded per attempt
without requiring equality with earlier attempts. Each container must still
match the source submitted for its current attempt.

An existing PC run without `.modal/<run_name>/provenance.json` can be imported
using the normal launch command. The launcher validates its saved experiment,
benchmark, prepared checksums, and initial prompts before recording the first
Modal configuration. An `import.json` records the imported manifest checksums.
It does not claim to verify the PC's historical checkpoint or serving settings,
which must be supplied by the operator. Normal pipeline validation still checks
each saved response before reuse. No responses are rewritten by adoption.

For a migrated run, keep its `migration.json` and the original
`.modal/<source_run_name>` directory. When that source provenance is available,
the launcher checks compatibility using the original run name and records its
path and checksum in `import.json`. It leaves the original history untouched.

Changing `volume_prefix` selects all four destination Volumes. To resume there,
first make the required inputs, existing run directory, and its
`.modal/<run_name>` history available on those Volumes. The launcher does not
copy files or find runs on other Volumes. An empty destination results Volume
starts a new run and repeats inference. GPU or image changes may affect model
outputs even when the request text and checkpoint are unchanged.
Stop the original writer before transferring or resuming on another prefix.
The destination lock does not coordinate with the original lock.

Automatic restarts can reuse their owning FunctionCall's lock. A new submission
with a different call ID still requires recovery if the old lock remains.
If a run reports a leftover lock, inspect its owning FunctionCall in the Modal
dashboard and confirm that it has stopped before removing the individual
`run:<run_name>` key from `temporal-single-locks` through the Modal SDK. Use your
actual prefix if different. Keep other launchers for this run stopped during
inspection and removal. Never clear the whole Dict or remove an active
writer's lock. A persisted `running` status is not proof that the job is active.
See [persistence and recovery](MODAL_GUIDE.md#persistence-and-recovery) for the
implementation and limits of response-tail repair.

### 6. Retrieve results and launch subsequent runs

After completion, download the run and its separate provenance/status directory.
These commands retrieve this example's smoke run from `temporal-single-results`:

```shell
uv run --group modal modal volume get temporal-single-results \
  temporal-single-v1-qwen35-9b-smoke_v1-001 \
  artifacts/temporal_reasoning_single_finding/runs/temporal-single-v1-qwen35-9b-smoke_v1-001

uv run --group modal modal volume get temporal-single-results \
  .modal/temporal-single-v1-qwen35-9b-smoke_v1-001 \
  artifacts/temporal_reasoning_single_finding/runs/.modal/temporal-single-v1-qwen35-9b-smoke_v1-001
```

Review `evaluation.json` in the downloaded run directory. After validating the
smoke run and persistence, select your intended benchmark in the experiment
JSON and assign a new run name. Stage its inputs only if that benchmark has
not already been uploaded to the selected Volumes, then launch again.
For another model, edit the serving JSON and download its checkpoint only if
it is not already present. Reuse the
resource profile when its vLLM image and GPU allocation support the new model.

## Endpoint

This mode uses an existing Shared or Dedicated Endpoint. Both use the same
commands and CPU runner. Set the connection URL and model identifier for your
chosen route. Endpoint provisioning and lifecycle management are separate from
this evaluation launcher.

### 1. Prepare the account and staged inputs

Complete [one-time setup](#1-one-time-setup-install-modal-and-create-configuration-files)
and [data staging](#2-stage-data-once-per-benchmark-and-volume-prefix) from the
vLLM section if they are not already complete. Those two steps apply to both
modes and require no model selection or serving profile. Reuse the existing
`temporal-single` images and benchmark Volumes when they contain `smoke_v1`.
There is no checkpoint download or `serving.json` for endpoint evaluation.

Use an endpoint that meets the dataset's access and retention terms. Each
image-containing request transfers encoded images from the Modal CPU container to the endpoint over HTTPS. CPU
runtime, endpoint inference, and applicable network transfer are separate costs.
Estimate transfer from image sizes, repeated requests, encoding, and retries
before a full run. Evaluation completion does not stop the endpoint.

### 2. Configure the endpoint and its Secret

Create `tmp/modal/endpoint-experiment.json` from the
[endpoint experiment template](examples/modal-endpoint-experiment.example.json):

```json
{
  "run_name": "endpoint-smoke-001",
  "model": "<exact advertised model identifier>",
  "base_url": "https://<endpoint-hostname>/v1",
  "benchmark_path": "artifacts/temporal_reasoning_single_finding/benchmarks/smoke_v1/manifest.json"
}
```

Replace both placeholders before checking or submitting the configuration:

- `base_url` is the endpoint's HTTPS API root ending in `/v1`. For a Dedicated
  Endpoint, use the API URL supplied for that deployment. For the Shared gateway,
  the integration route is `https://inference.us-west.modal.direct/v1`.
- `model` must exactly match the identifier advertised by that route's
  authenticated `/v1/models` API. The Shared gateway uses the advertised endpoint
  hostname as its model identifier. Consult the deployment's integration details
  for the pair of URL and model values.
  [Endpoint integrations](https://modal.com/docs/guide/endpoint-integrations)
- `run_name` identifies the saved experiment. Use a new name for an intentional
  change to model, endpoint deployment, benchmark, or model inputs.
- `benchmark_path` remains the repository-relative path to the staged manifest.

**Optional `reasoning_effort`:** add this as a top-level field in
`tmp/modal/endpoint-experiment.json`, alongside `model` and `benchmark_path`.
For example, add `"reasoning_effort": "high"` to request high reasoning effort,
but only if the endpoint and selected model support that value. The runner
forwards it unchanged and does not translate or remove unsupported settings.
The example above omits this optional setting, leaving the endpoint/model
default in effect.

To change reasoning effort, edit that field. To omit the setting, remove the
field entirely. If the run has already started, use a new `run_name` when adding,
changing, or removing it. Resuming a saved run requires its original setting.

**Optional `max_completion_tokens`:** add `"max_completion_tokens": 2048` as a
top-level field in this experiment JSON, alongside `model` and `benchmark_path`.
Choose a positive integer suitable for your model. Omit the field or set it to
`null` to use the endpoint/model default.
It can be used together with `reasoning_effort`.

This caps generated tokens per API attempt, including reasoning tokens where
the endpoint counts them in completion usage. It does not limit or truncate
input text or images. The server's context window and output limits still
apply. The runner forwards the value unchanged to every model stage and stops
on a non-retryable rejection without removing or increasing the cap.

Use a new `run_name` when adding, changing, or removing an explicit limit.
A small cap can leave incomplete JSON or no answer text. Incomplete JSON keeps
its parser-invalid outcome. Empty answers can incur the existing three attempts
with the same cap. Budget for those retries separately.

For a live token-limit smoke test, select `smoke_v1`, give the run a fresh name,
and add the limit to `endpoint-experiment.json`. Run the offline `--check` and
then the launch command below with `--workers 1`. Inspect saved textual answers,
parser failures, and available `usage.output_tokens` counts to choose an
appropriate output limit.

In the Modal dashboard, create a proxy token for the endpoint. In the same Modal
environment as the evaluation App, create a named Secret such as
`chextemporal-endpoint-auth` containing:

| Secret key | Value |
| --- | --- |
| `OPENAI_API_KEY` | `<proxy-token-id>.<proxy-token-secret>`, joined with one period |

Enter the actual value only in the Secret. Keep it out of experiment JSON,
resource profiles, shell arguments, and source control. The local Modal CLI
login authorizes job submission. The named Secret supplies endpoint credentials
inside the CPU container.
[Endpoint authentication](https://modal.com/docs/guide/endpoints),
[Modal Secrets](https://modal.com/docs/guide/secrets)

Create `tmp/modal/endpoint-profile.json` from the
[CPU profile template](examples/modal-endpoint-profile.example.json):

```json
{
  "volume_prefix": "temporal-single",
  "secret_name": "chextemporal-endpoint-auth",
  "cpu": 2,
  "memory": 8192,
  "timeout": 21600,
  "startup_timeout": 600
}
```

| Field | Purpose |
| --- | --- |
| `volume_prefix` | Prefix of the staged images and benchmark Volumes, results Volume, and lock Dict. |
| `secret_name` | Name of the Secret containing `OPENAI_API_KEY`. |
| `cpu` | Requested whole CPU cores, from 1 to 64. |
| `memory` | Requested memory in MiB, from 1 to 262144. |
| `timeout` | Whole job timeout in seconds, at most 86400. |
| `startup_timeout` | Readiness polling budget in seconds, shorter than `timeout`. |
| `region` (optional) | CPU compute region, independent of endpoint routing. |
| `deployment_identity` (optional) | Operator-supplied `deployment_id` and/or `checkpoint_revision` strings to retain as part of the run identity. |

For example, an optional identity is
`"deployment_identity": {"deployment_id": "recorded-deployment-id"}`. These fields
record the deployment used for the run. Keep the endpoint configuration stable
across an experiment and its resumed attempts.

The CPU profile has no `gpu`, serving configuration, or `commit_interval`.
Results use Modal background commits plus explicit commits at execution
boundaries. See [persistence and recovery](MODAL_GUIDE.md#persistence-and-recovery)
for the durability limits.

### 3. Validate locally and submit

Select the endpoint profile and check the configuration:

```shell
uv run python -m experiments.temporal_reasoning_single_finding.modal.endpoint_job \
  --profile tmp/modal/endpoint-profile.json \
  --experiment tmp/modal/endpoint-experiment.json --workers 1 --check
```

`--check` validates the configuration and fingerprints the allowlisted source
and prompt fixture locally. Replace the URL and model placeholders before
running it. Remote input and endpoint checks run when the job starts.

Once the configuration and staged inputs are ready, launch the smoke run:

```shell
uv run --group modal python -m experiments.temporal_reasoning_single_finding.modal.endpoint_job \
  --profile tmp/modal/endpoint-profile.json \
  --experiment tmp/modal/endpoint-experiment.json --workers 1
```

The command starts a detached App named after the experiment's `run_name`,
submits with `.spawn()`, prints its
FunctionCall ID, and waits for the result. Add `--no-wait` to return immediately
after submission. Interrupting the local wait does not cancel or resubmit the
cloud invocation. Keep the ID for inspection in the Modal dashboard. `--workers`
sets request concurrency inside the CPU job.

The runner checks the staged inputs, authenticates model discovery, runs
preflight and inference, and scores the saved responses. Short explanations are
requested automatically. The results Volume is created if needed. The images
and benchmark Volumes must already exist.

API failures retain HTTP status, error category, operation, model, filtered
request ID, recognized error code/parameter, and stack locations. Credentials and
raw HTTP bodies are omitted before exceptions reach Modal. A worker failure also
identifies the prepared model input by checksum. Fatal failures still stop the
job.

For example, HTTP 401 points to the Secret or endpoint token, HTTP 404 to the
route or model identifier, and HTTP 400 with `parameter=reasoning_effort` to an
unsupported request setting. For an unrecognized server error, use the request
ID to inspect endpoint logs. The diagnostic explicitly says when raw server text
was omitted. See [failure handling](MODAL_GUIDE.md#failure-handling-and-cancellation)
for details about error reporting.

### 4. Monitor and resume an interrupted run

Follow the FunctionCall in the Modal dashboard. To resume, repeat the launch
command with the same `run_name`, experiment values, frozen benchmark, and any
supplied deployment identity. Compatible completed responses are reused,
including invalid answers and exhausted technical-error records. They are not
automatically retried on resume. `--workers`, source, and CPU resource settings
may change, with each attempt recorded separately. The original provenance
remains unchanged. Stop the previous cloud invocation before resuming the same
run. Prompt and saved-request identity checks still apply.

A localhost vLLM run cannot resume through the endpoint runner under the same
name. Use a new run for a different endpoint deployment even if its public model
name remains unchanged. For historical prompt compatibility, see the vLLM
section's [resume requirements](#resume-requirements).

Keep both `<run_name>/` and `.modal/<run_name>/` on the results Volume. Changing
`volume_prefix` requires explicitly transferring these directories and the
required staged inputs to the new Volumes. Stop the original writer first. The
launcher does not find or copy prior runs automatically, and a new prefix has
an independent lock.

Automatic restarts can reuse the same FunctionCall's lock. If a new submission
is blocked by a lock left after termination, confirm in the dashboard that its owning
FunctionCall has ended before removing only `run:endpoint-smoke-001` from
`temporal-single-locks` through the SDK. Substitute your actual prefix and run
name. Keep other launchers for this run stopped during inspection and removal.
Never remove an active writer's lock or clear the whole Dict. Persisted
`running` status alone does not establish whether the invocation is active.

On handled interruption, queued requests are cancelled and active requests may
finish or time out. Their completed responses are saved while the writer remains
usable. Resume repairs an interrupted final JSONL record and skips compatible
completed records. Uncommitted or incomplete responses may be requested again,
and billed again. Hard termination can prevent cleanup.

### 5. Retrieve results and run the full experiment

Download the run and its separate Modal metadata:

```shell
uv run --group modal modal volume get temporal-single-results \
  endpoint-smoke-001 \
  artifacts/temporal_reasoning_single_finding/runs/endpoint-smoke-001

uv run --group modal modal volume get temporal-single-results \
  .modal/endpoint-smoke-001 \
  artifacts/temporal_reasoning_single_finding/runs/.modal/endpoint-smoke-001
```

Inspect `evaluation.json` in the ordinary run directory. Under
`.modal/endpoint-smoke-001/attempts/<attempt_id>/`, `endpoint.json` records the
configured and discovered models, `inference.json` contains the pipeline summary,
and provenance and status files describe that attempt.

For local rescoring, retain the matching frozen benchmark at its original
repository-relative path and follow [offline evaluation](README.md#offline-evaluation).
The evaluator uses saved text records and needs neither images nor a live endpoint.

After validating smoke-run outputs, persistence, and model compatibility, set a
new run name and the intended benchmark path in the endpoint experiment JSON.
Stage that benchmark only if it is missing from the selected Volumes. Reuse the
CPU profile and Secret when their settings remain appropriate. Monitor and stop
or scale the separately managed endpoint as appropriate after evaluation.
