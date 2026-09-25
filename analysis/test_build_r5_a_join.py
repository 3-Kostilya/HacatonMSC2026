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
            times = pd.date_range("2025-01-01", periods=3, freq="h")
            source = pd.DataFrame({
                "channel_id": ["c", "c", "unused"],
                "prediction_time": times,
                "sensor_type": ["gas", "gas", "gas"],
            })
            for name in INPUT_COLUMNS:
                source[name] = 0.0
            for name in ("state_transitions_1h", "state_transitions_6h",
                         "state_transitions_24h", "state_transitions_168h"):
                source[name] = pd.Series([0, 1, None], dtype="Int64")
            source.to_parquet(month_dir / "features.parquet", index=False)
            keys = pd.DataFrame({
                "channel_id": ["c", "c"],
                "prediction_time": [times[1], times[0]],
                "sensor_type": ["gas", "gas"],
                "split": ["validation", "validation"],
            })
            joined = _load_joined(root, "2025-01", keys)
            self.assertEqual(len(joined), 2)
            self.assertEqual(joined.channel_id.tolist(), ["c", "c"])
            self.assertEqual(joined.prediction_time.tolist(),
                             [times[1], times[0]])
            self.assertEqual(joined.sensor_type.tolist(), ["gas", "gas"])
            self.assertEqual(str(joined.state_transitions_1h.dtype), "float64")
            bad = keys.copy()
            bad.loc[0, "sensor_type"] = "smoke"
            with self.assertRaisesRegex(ValueError, "sensor type mismatch"):
                _load_joined(root, "2025-01", bad)


if __name__ == "__main__":
    unittest.main()
