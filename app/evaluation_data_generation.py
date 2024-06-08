import os
import random
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class EvaluationDataGenerator:
    def __init__(self, additional_scenarios=None):
        self.additional_scenarios = additional_scenarios if additional_scenarios else []

    def generate_llm_contexts(self, description):
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"Generate contexts based on the description: {description}",
            max_tokens=50,
            n=5,
            stop=None,
            temperature=0.3
        )
        contexts = [choice['text'].strip() for choice in response.choices]
        return contexts

    def generate_random_test_cases(self, description, num_test_cases):
        llm_contexts = self.generate_llm_contexts(description)
        all_scenarios = llm_contexts + self.additional_scenarios
        test_cases = []

        for _ in range(num_test_cases):
            scenario = random.choice(all_scenarios)
            test_cases.append(f"{description} {scenario}")
        
        return test_cases
