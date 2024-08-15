import json
import os

def load_json(filename):
    """Load a JSON file."""
    if os.path.isfile(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    return []

def save_json(filename, data):
    """Save data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def update_answers(questions_file, answers_file):
    # Load the questions and answers
    questions = load_json(questions_file)
    answers = load_json(answers_file)
    
    # Create a set of existing answer IDs
    existing_ids = {item['id'] for item in answers}
    
    # Create a dictionary for easier lookup and update
    answers_dict = {item['id']: item for item in answers}
    
    # Process questions and update answers
    for question in questions:
        question_id = question.get('id')
        
        if question_id not in existing_ids:
            # Add the question ID with null answer
            answers_dict[question_id] = {
                'id': question_id,
                'answer': None
            }
            print(f"Added question ID {question_id} with null answer.")

    # Convert updated dictionary back to list
    updated_answers = list(answers_dict.values())
    
    # Save the updated answers to the file
    save_json(answers_file, updated_answers)
    print("Answers updated and saved.")

# Define file paths
questions_file = 'sch_set2_test_questions.json'
answers_file = 'answers2.json'

# Run the update function
update_answers(questions_file, answers_file)
