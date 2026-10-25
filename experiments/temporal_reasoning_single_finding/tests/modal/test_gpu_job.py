"""Offline coverage of explicit GPU submission and early validation."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from experiments.temporal_reasoning_single_finding.modal import gpu_job


class GPUJobTests(unittest.TestCase):
    """Check submission without allocating cloud resources."""

    def test_validated_profile_reaches_detached_submission(self) -> None:
        """The selected profile must match the App before starting it."""
        for matches, retry in ((True, False), (True, True), (False, True)):
            fake_app = MagicMock()
            fake_app.PROFILE = {"selected": matches}
            fake_modal = MagicMock()
            with (
                self.subTest(matches=matches),
                patch.object(sys, "argv", [
                    "gpu_job", "--profile", "selected.json",
                    "--serving", "serving.json", "--experiment", "experiment.json",
                    *(["--retry-failed"] if retry else []),
                ]),
                patch.object(gpu_job, "read_json", return_value={"selected": True}) as read,
                patch.object(gpu_job, "load_job", return_value=(
                    {}, {"run_name": "test-gpu-run"}, {"selected": True},
                )),
                patch.object(gpu_job, "source_digest", return_value="source"),
                patch.dict(sys.modules, {
                    "modal": fake_modal,
                    "experiments.temporal_reasoning_single_finding.modal.app": fake_app,
                }),
            ):
                if matches:
                    gpu_job.main()
                    fake_app.app.run.assert_called_once_with(
                        name="test-gpu-run", detach=True,
                    )
                    fake_app.main.assert_called_once_with(
                        "serving.json", "experiment.json", 1, retry_failed=retry,
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "profile changed"):
                        gpu_job.main()
                    fake_app.app.run.assert_not_called()
                self.assertEqual(str(read.call_args.args[0]), "selected.json")
