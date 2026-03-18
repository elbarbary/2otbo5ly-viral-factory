from pathlib import Path
import unittest

from viral_factory.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_reads_expected_brand(self) -> None:
        config = load_config(Path("configs/2otbo5ly.json"))
        self.assertEqual(config.brand.name, "2otbo5ly")
        self.assertTrue(config.series.enabled)
        self.assertEqual(config.series.visual_mode, "cartoon_hybrid")
        self.assertEqual(config.pipeline.script_iterations, 10)
        self.assertEqual(config.assets.video_aspect_ratio, "9:16")


if __name__ == "__main__":
    unittest.main()

