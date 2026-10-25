# Required data

Prepare the files for the operation you want to run. All paths below are
relative to the repository root unless stated otherwise.

## Benchmark construction

Construction starts from prepared strict-order single-finding exports:

| File | Role |
| --- | --- |
| `artifacts/chexpert_strict_order_finding_sequences_3tp.json` | Three-timepoint source records |
| `artifacts/chexpert_strict_order_finding_sequences_4tp.json` | Four-timepoint source records |
| `artifacts/chexpert_strict_order_finding_sequences.parquet` | Source-validation sidecar |

Prepare these exports with the upstream chronology and audit pipeline before
running the constructor. The audit sidecar uses the schema below and is a
separate input from the original CheXTemporal progression-pair Parquet.

Each source JSON is an array of records with these fields:

- `sequence_id`, `image_sequence_id`, `patient_id`, `tier`, and `finding`.
- `timepoints`, containing `timepoint`, `study_id`, `image_path`,
  `patient_report_date_order_min`, and `patient_report_date_order_max` per visit.
- `change_labels`, containing one original source label per adjacent interval.

The audit sidecar requires `sequence_id`, `image_sequence_id`, `tier`,
`time_points`, `patient_id`, `finding_token`, `study_ids`, `image_paths`,
`study_order_min`, `study_order_max`, `transition_labels`, `is_unambiguous`, and
`all_images_match_csv`. The last two fields supply validation flags omitted
from the source JSONs.

[`construction/source_data.py`](../experiments/temporal_reasoning_single_finding/construction/source_data.py)
defines the schema, identity checks, chronology rules, and finding vocabulary.
Construction records checksums for both JSONs and the sidecar in the benchmark
manifest.

To use alternative source locations, call `construct_gold_pilot()` from
`construction/constructor_gold_pilot.py` with `three_tp_source`,
`four_tp_source`, `audit_sidecar`, and `output_root` as `pathlib.Path` arguments.
The command-line module uses the default paths above.

## Inference

Place the frozen benchmark files `manifest.json`, `3tp.json`, and `4tp.json`
under `artifacts/temporal_reasoning_single_finding/benchmarks/gold_pilot_v1/`.
The manifest's checksums identify the matching benchmark files. Preserve this
relative layout when moving a run between machines.

Set `chexpert_plus_image_root` in your paths configuration to the absolute
CheXpert-Plus PNG directory. Beneath that root, preserve each source image's
relative path, replacing the final `.jpg` suffix with `.png`.

Use an inference service that meets the dataset's access and retention terms.
Set credentials through environment variables and save local configuration in
`config/`.

## Offline evaluation

Keep the saved run manifest, prepared records, responses, and derived results
under `artifacts/temporal_reasoning_single_finding/runs/<run_name>/`, together
with the matching benchmark at the path recorded in the run manifest.
Evaluation reads these saved files locally.
