import unittest
from unittest.mock import patch
from app.prompt_generation import PromptGenerationSystem

class TestPromptGenerationSystem(unittest.TestCase):
    @patch('app.prompt_generation.openai.Completion.create')
    def test_generate_prompts(self, mock_create):
        # Mock response should contain 'choices' with 'text' attribute
        mock_create.return_value.choices = [
            {'text': 'Prompt 1'}, 
            {'text': 'Prompt 2'}, 
            {'text': 'Prompt 3'}, 
            {'text': 'Prompt 4'}, 
            {'text': 'Prompt 5'}
        ]
        system = PromptGenerationSystem()
        prompts = system.generate_prompts("Test description", ["Scenario 1", "Scenario 2"])
        self.assertEqual(len(prompts), 10)
        self.assertIn('Prompt 1', prompts)

    def test_evaluate_prompts(self):
        system = PromptGenerationSystem()
        description = "Test description"
        prompts = ["Prompt 1 contains Test description", "Prompt 2", "Prompt 3"]
        evaluated_prompts = system.evaluate_prompts(description, prompts)
        self.assertEqual(len(evaluated_prompts), len(prompts))
        self.assertEqual(evaluated_prompts[0][1], max([ep[1] for ep in evaluated_prompts]))

if __name__ == '__main__':
    unittest.main()
