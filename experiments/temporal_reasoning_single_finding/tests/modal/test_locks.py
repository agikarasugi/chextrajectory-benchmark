"""Exercise persistent lock recovery without Modal authentication or storage."""

import unittest
from unittest.mock import Mock

from experiments.temporal_reasoning_single_finding.modal.locks import acquire_call_lock


class CallLockTests(unittest.TestCase):
    """Keep restarted calls recoverable and independent writers excluded."""

    def test_new_owner_acquires_atomically(self) -> None:
        """Normal acquisition uses insert-if-absent without inspecting other calls."""
        locks = Mock()
        locks.put.return_value = True
        acquire_call_lock(locks, "run:test", "fc-current", "Run is locked")
        locks.put.assert_called_once_with("run:test", "fc-current", skip_if_exists=True)
        locks.get.assert_not_called()
        locks.pop.assert_not_called()

    def test_restart_resumes_persisted_ownership(self) -> None:
        """A killed worker's unchanged lock permits the same call to restart."""
        locks = Mock()
        locks.put.return_value = False
        locks.get.return_value = "fc-current"
        acquire_call_lock(locks, "run:test", "fc-current", "Run is locked")
        locks.put.assert_called_once_with("run:test", "fc-current", skip_if_exists=True)
        locks.pop.assert_not_called()

    def test_other_owner_is_preserved_and_identified(self) -> None:
        """A new submission cannot delete or overwrite a different owner's lock."""
        for owner in ("fc-other", "", {"unexpected": "value"}):
            with self.subTest(owner=owner):
                locks = Mock()
                locks.put.return_value = False
                locks.get.return_value = owner
                with self.assertRaisesRegex(RuntimeError, "key='run:test', owner="):
                    acquire_call_lock(locks, "run:test", "fc-current", "Run is locked")
                self.assertEqual(locks.put.call_count, 1)
                locks.pop.assert_not_called()

    def test_released_key_requires_new_atomic_acquisition(self) -> None:
        """A release during acquisition triggers another insert-if-absent."""
        locks = Mock()
        locks.put.side_effect = [False, True]
        locks.get.return_value = None
        acquire_call_lock(locks, "run:test", "fc-current", "Run is locked")
        self.assertEqual(locks.put.call_count, 2)

    def test_competitor_wins_after_release(self) -> None:
        """A missing key does not authorize entry when another writer wins it."""
        locks = Mock()
        locks.put.return_value = False
        locks.get.side_effect = [None, "fc-winner"]
        with self.assertRaisesRegex(RuntimeError, "fc-winner"):
            acquire_call_lock(locks, "run:test", "fc-current", "Run is locked")
        locks.pop.assert_not_called()

    def test_unstable_key_fails_with_bounded_requests(self) -> None:
        """Repeated disappearance cannot create an unbounded cloud request loop."""
        locks = Mock()
        locks.put.return_value = False
        locks.get.return_value = None
        with self.assertRaisesRegex(RuntimeError, "ownership changed"):
            acquire_call_lock(locks, "run:test", "fc-current", "Run is locked")
        self.assertEqual(locks.put.call_count, 3)
        locks.pop.assert_not_called()

    def test_missing_call_identity_fails_before_dict_access(self) -> None:
        """Local or malformed identity cannot accidentally match a missing lock."""
        for owner in (None, "", 3):
            with self.subTest(owner=owner):
                locks = Mock()
                with self.assertRaisesRegex(RuntimeError, "FunctionCall ID"):
                    acquire_call_lock(locks, "run:test", owner, "Run is locked")
                locks.put.assert_not_called()

    def test_lookup_failure_never_authorizes_recovery(self) -> None:
        """A Dict service failure propagates while preserving the existing key."""
        locks = Mock()
        locks.put.return_value = False
        locks.get.side_effect = OSError("service unavailable")
        with self.assertRaisesRegex(OSError, "service unavailable"):
            acquire_call_lock(locks, "run:test", "fc-current", "Run is locked")
        locks.pop.assert_not_called()
