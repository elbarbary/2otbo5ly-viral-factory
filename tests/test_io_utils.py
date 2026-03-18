from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from viral_factory.io_utils import write_json


class IoUtilsTests(unittest.TestCase):
    def test_write_json_serializes_datetime_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "payload.json"

            write_json(path, {"created_at": datetime(2026, 3, 14, tzinfo=timezone.utc)})

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["created_at"], "2026-03-14T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()