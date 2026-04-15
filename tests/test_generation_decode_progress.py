import unittest
from pathlib import Path


class GenerationDecodeProgressTests(unittest.TestCase):
    def test_generation_decode_updates_tqdm_progress_bar(self):
        source = Path(
            "downstream_task/report_generation_and_vqa/generation_decode.py"
        ).read_text(encoding="utf-8")

        self.assertIn("pbar.update(1)", source)


if __name__ == "__main__":
    unittest.main()
