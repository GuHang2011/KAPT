import unittest

from kapt_lab import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_max_stops_and_stays_stopped(self):
        tracker = EarlyStopping(patience=2, min_delta=0.1)
        self.assertFalse(tracker(0.5, 0))
        self.assertFalse(tracker(0.7, 1))
        self.assertFalse(tracker(0.75, 2))
        self.assertTrue(tracker(0.6, 3))
        self.assertTrue(tracker(0.99, 4))
        self.assertEqual(tracker.best_score, 0.7)
        self.assertEqual(tracker.best_epoch, 1)
        self.assertEqual(tracker.bad_epochs, 2)

    def test_min_resets_bad_epochs_on_improvement(self):
        tracker = EarlyStopping(patience=2, mode="min")
        tracker(5.0, 1)
        tracker(6.0, 2)
        self.assertFalse(tracker(4.0, 3))
        self.assertEqual(tracker.bad_epochs, 0)
        self.assertEqual(tracker.best_epoch, 3)
        self.assertFalse(tracker(4.0, 4))
        self.assertTrue(tracker(4.0, 5))

    def test_delta_equality_is_not_an_improvement(self):
        tracker = EarlyStopping(patience=1, min_delta=0.25)
        tracker(0.5, 0)
        self.assertTrue(tracker(0.75, 1))
        self.assertEqual(tracker.best_epoch, 0)

    def test_reset_allows_a_fresh_run(self):
        tracker = EarlyStopping(patience=1)
        tracker(0.5, 0)
        tracker(0.5, 1)
        tracker.reset()
        self.assertIsNone(tracker.best_score)
        self.assertIsNone(tracker.best_epoch)
        self.assertIsNone(tracker.last_epoch)
        self.assertFalse(tracker.stopped)
        self.assertFalse(tracker(0.1, 0))

    def test_invalid_configuration(self):
        for patience in (0, -1, 1.5, True, "3"):
            with self.subTest(patience=patience), self.assertRaises(ValueError):
                EarlyStopping(patience=patience)
        for delta in (-1, float("nan"), float("inf"), True, "0", 10 ** 1000):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                EarlyStopping(min_delta=delta)
        with self.assertRaises(ValueError):
            EarlyStopping(mode="largest")

    def test_invalid_observations_do_not_mutate_state(self):
        tracker = EarlyStopping()
        tracker(0.5, 2)
        for score in (float("nan"), float("inf"), float("-inf"), True, "0.6", 10 ** 1000):
            with self.subTest(score=score), self.assertRaises(ValueError):
                tracker(score, 3)
        for epoch in (2, 1, -1, 3.5, True, "3"):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                tracker(0.6, epoch)
        self.assertEqual(tracker.last_epoch, 2)
        self.assertEqual(tracker.best_score, 0.5)


if __name__ == "__main__":
    unittest.main()
