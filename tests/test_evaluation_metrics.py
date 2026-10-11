import unittest

from src.evaluation.metrics import classification_metrics


class EvaluationMetricTests(unittest.TestCase):
    def test_metrics_and_matrix_are_deterministic(self):
        result = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], labels=[0, 1])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertIn("macro_f1", result)


if __name__ == "__main__":
    unittest.main()
