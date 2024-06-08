import os
from dotenv import load_dotenv
import openai

load_dotenv()

class PromptGenerationSystem:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_prompts(self, description, scenarios):
        prompts = []
        for scenario in scenarios:
            input_text = f"{description} {scenario}"
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=input_text,
                max_tokens=100,
                n=5
            )
            for choice in response.choices:
                prompts.append(choice['text'].strip())
        return prompts

    def evaluate_prompts(self, description, prompts):
        scores = []
        keywords = description.split()
        for prompt in prompts:
            length_score = len(prompt.split())
            keyword_score = sum(1 for word in keywords if word in prompt)
            total_score = length_score + keyword_score
            scores.append((prompt, total_score))
        return sorted(scores, key=lambda x: x[1], reverse=True)
