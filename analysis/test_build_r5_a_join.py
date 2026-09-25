"""The R5 replay reads only admitted keys from large A3 months."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from analysis.build_r5_a import _load_joined
from stage1.features.r5 import INPUT_COLUMNS


class R5AJoinTest(unittest.TestCase):
    def test_admitted_keys_only_and_sensor_type_checked(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            month_dir = root / "year=2025" / "month=01"
            month_dir.mkdir(parents=True)
            times = pd.date_range("2025-01-01", periods=2, freq="h")
            source = pd.DataFrame({
                "channel_id": ["c", "unused"],
                "prediction_time": times,
                "sensor_type": ["gas", "gas"],
            })
            for name in INPUT_COLUMNS:
                source[name] = 0.0
            source.to_parquet(month_dir / "features.parquet", index=False)
            keys = pd.DataFrame({
                "channel_id": ["c"],
                "prediction_time": times[:1],
                "sensor_type": ["gas"],
                "split": ["validation"],
            })
            joined = _load_joined(root, "2025-01", keys)
            self.assertEqual(len(joined), 1)
            self.assertEqual(joined.channel_id.tolist(), ["c"])
            self.assertEqual(joined.sensor_type.tolist(), ["gas"])
            bad = keys.copy()
            bad["sensor_type"] = "smoke"
            with self.assertRaisesRegex(ValueError, "sensor type mismatch"):
                _load_joined(root, "2025-01", bad)


if __name__ == "__main__":
    unittest.main()
