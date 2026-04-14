from pathlib import Path
import unittest


class LocalBertWorkflowPathTests(unittest.TestCase):
    def test_shell_entrypoints_use_explicit_local_bert_model(self):
        run_text = Path("downstream_task/report_generation_and_vqa/run.sh").read_text(encoding="utf-8")
        test_text = Path("downstream_task/report_generation_and_vqa/test.sh").read_text(encoding="utf-8")

        self.assertIn('BERT_MODEL=${BERT_MODEL:-"/root/autodl-tmp/models/bert-base-uncased"}', run_text)
        self.assertIn('--bert_model "${BERT_MODEL}"', run_text)
        self.assertIn('BERT_MODEL=${BERT_MODEL:-"/root/autodl-tmp/models/bert-base-uncased"}', test_text)
        self.assertIn('--bert_model "${BERT_MODEL}"', test_text)

    def test_openi_workflow_examples_show_explicit_local_bert_model(self):
        workflow_text = Path("OPENI_SERVER_WORKFLOW.md").read_text(encoding="utf-8")

        self.assertIn("--bert_model /root/autodl-tmp/models/bert-base-uncased", workflow_text)


if __name__ == "__main__":
    unittest.main()
