from unittest import TestCase, main

import numpy as np
from pydantic import ValidationError

from src.Models.TimeSeries import TimeSeries


class TestTimeSeries(TestCase):
    def setUp(self):
        self.data = np.array([1.0, 2.0, 3.0])

    def test_create_TimeSeries(self):
        time_series = TimeSeries(data=self.data, window_size=2, rank=1)

        self.assertEqual(time_series.data.tolist(), self.data.tolist())
        self.assertEqual(time_series.window_size, 2)
        self.assertEqual(time_series.rank, 1)
        self.assertIsInstance(time_series.data, np.ndarray)

    def test_batch_size(self):
        time_series = TimeSeries(data=self.data, window_size=2, rank=1)
        self.assertEqual(time_series.batch_size, 1)

    def test_invalid_window_size(self):
        with self.assertRaises(ValidationError):
            TimeSeries(data=self.data, window_size=-1, rank=1)

    def test_invalid_rank(self):
        with self.assertRaises(ValidationError):
            TimeSeries(data=self.data, window_size=2, rank=3)


if __name__ == "__main__":
    main()
