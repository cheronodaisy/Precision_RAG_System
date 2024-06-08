import unittest
from unittest.mock import patch
from app.evaluation_data_generation import EvaluationDataGenerator

class TestEvaluationDataGenerator(unittest.TestCase):
    @patch('app.evaluation_data_generation.openai.Completion.create')
    def test_generate_llm_contexts(self, mock_create):
        mock_create.return_value.choices = [{'text': 'Context 1'}, {'text': 'Context 2'}, {'text': 'Context 3'}, {'text': 'Context 4'}, {'text': 'Context 5'}]
        generator = EvaluationDataGenerator()
        contexts = generator.generate_llm_contexts("Test description")
        self.assertEqual(len(contexts), 5)
        self.assertIn('Context 1', contexts)

    def test_generate_random_test_cases(self):
        generator = EvaluationDataGenerator(additional_scenarios=['Scenario A', 'Scenario B'])
        description = "Test description"
        num_test_cases = 3
        test_cases = generator.generate_random_test_cases(description, num_test_cases)
        self.assertEqual(len(test_cases), num_test_cases)
        for case in test_cases:
            self.assertTrue(case.startswith(description))

if __name__ == '__main__':
    unittest.main()
