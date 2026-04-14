import json
import unittest
from pathlib import PurePosixPath

from downstream_task.report_generation_and_vqa.json_compat import make_json_compatible


class MakeJsonCompatibleTests(unittest.TestCase):
    def test_converts_paths_in_nested_structures(self):
        payload = {
            "repo_root": PurePosixPath("/tmp/repo"),
            "split": ["train", "valid"],
            "nested": {
                "config_path": PurePosixPath("/tmp/repo/config.json"),
            },
        }

        converted = make_json_compatible(payload)
        serialized = json.dumps(converted, sort_keys=True)

        self.assertEqual(converted["repo_root"], "/tmp/repo")
        self.assertEqual(converted["nested"]["config_path"], "/tmp/repo/config.json")
        self.assertIn('"repo_root": "/tmp/repo"', serialized)


if __name__ == "__main__":
    unittest.main()
