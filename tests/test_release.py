"""Checks that do not need MetaDrive or an API key."""

import json
import unittest
from pathlib import Path

from crash2scenario.llm import parse_json_response
from crash2scenario.replay import load_record, scenario_labels


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_parse_fenced_json(self):
        text = "```json\n{\"map\": \"X\", \"steps\": 10}\n```"
        self.assertEqual(parse_json_response(text)["map"], "X")

    def test_released_configs_load(self):
        paths = sorted((ROOT / "scenarios").glob("cluster_*/initial.json"))
        self.assertEqual(len(paths), 20)
        for path in paths:
            record = load_record(path)
            labels = scenario_labels(record)
            self.assertEqual(len(labels), 4)
            self.assertIn(record["config"]["map"][0], "XOTSC")

    def test_manifest_matches_table4_initial_gaps(self):
        manifest = json.loads((ROOT / "scenarios" / "manifest.json").read_text())
        gaps = [round(row["initial_min_gap_m"], 2) for row in manifest]
        expected = [
            20.00, 9.04, 30.00, 22.79, 1.30, 2.08, 3.61, 10.02, 30.74, 40.14,
            28.00, 0.10, 1.37, 0.13, 20.00, 4.31, 34.32, 34.00, 11.10, 36.26,
        ]
        self.assertEqual(gaps, expected)


if __name__ == "__main__":
    unittest.main()
