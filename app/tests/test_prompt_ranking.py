import unittest
from app.prompt_ranking import PromptRanker

class TestPromptRanker(unittest.TestCase):
    def test_monte_carlo_matchmaking(self):
        ranker = PromptRanker()
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        test_cases = ["Test case 1", "Test case 2", "Test case 3"]
        ranked_prompts = ranker.monte_carlo_matchmaking(prompts, test_cases)
        self.assertEqual(len(ranked_prompts), len(prompts))
        self.assertGreaterEqual(ranked_prompts[0][1], ranked_prompts[-1][1])

if __name__ == '__main__':
    unittest.main()
