import unittest
from unittest.mock import patch
from app.main import app

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.evaluation_data_generation.openai.Completion.create')
    @patch('app.prompt_generation.openai.Completion.create')
    def test_generate_and_rank(self, mock_create_generation, mock_create_evaluation):
        # Mock responses should contain 'choices' with 'text' attribute
        mock_create_generation.return_value.choices = [
            {'text': 'Generated Context 1'}, 
            {'text': 'Generated Context 2'}, 
            {'text': 'Generated Context 3'}, 
            {'text': 'Generated Context 4'}, 
            {'text': 'Generated Context 5'}
        ]
        mock_create_evaluation.return_value.choices = [
            {'text': 'Generated Prompt 1'}, 
            {'text': 'Generated Prompt 2'}, 
            {'text': 'Generated Prompt 3'}, 
            {'text': 'Generated Prompt 4'}, 
            {'text': 'Generated Prompt 5'}
        ]

        # Test the generate route
        response = self.app.post('/generate', data={'description': 'Test description', 'scenarios': 'Scenario 1,Scenario 2'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test description', response.data)
        self.assertIn(b'Generated Prompt 1', response.data)

        # Test the rank route
        response = self.app.post('/rank', data={'description': 'Test description', 'prompts': ['Prompt 1', 'Prompt 2', 'Prompt 3']})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prompt 1', response.data)

if __name__ == '__main__':
    unittest.main()
