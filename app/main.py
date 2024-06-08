from flask import request, render_template
from app import app
from app.prompt_generation import PromptGenerationSystem
from app.evaluation_data_generation import EvaluationDataGenerator
from app.prompt_ranking import PromptRanker

system = PromptGenerationSystem()
generator = EvaluationDataGenerator()
ranker = PromptRanker()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    description = request.form['description']
    scenarios = request.form['scenarios'].split(',')
    prompts = system.generate_prompts(description, scenarios)
    evaluated_prompts = system.evaluate_prompts(description, prompts)
    return render_template('results.html', prompts=evaluated_prompts, description=description)

@app.route('/rank', methods=['POST'])
def rank():
    prompts = request.form.getlist('prompts')
    description = request.form['description']
    test_cases = generator.generate_random_test_cases(description, num_test_cases=10)
    ranked_prompts = ranker.monte_carlo_matchmaking(prompts, test_cases)
    return render_template('ranking.html', ranked_prompts=ranked_prompts)
