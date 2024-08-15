import json
import re

def filter_and_separate_questions(input_file, filtered_output_file, rest_output_file, keywords):
    try:
        # Load questions from the input file
        with open(input_file, 'r') as file:
            questions = json.load(file)
        
        # Compile regex patterns for the keywords to match only exact words
        keyword_patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]

        # Separate questions based on the presence of keywords
        filtered_questions = []
        rest_questions = []
        for question in questions:
            question_text = question.get('question', '')
            if any(pattern.search(question_text) for pattern in keyword_patterns):
                filtered_questions.append(question)
            else:
                rest_questions.append(question)
        
        # Write filtered questions to the filtered output file
        with open(filtered_output_file, 'w') as file:
            json.dump(filtered_questions, file, indent=4)
        
        # Write the rest of the questions to the rest output file
        with open(rest_output_file, 'w') as file:
            json.dump(rest_questions, file, indent=4)
        
        print(f"Filtering complete. The results are saved in '{filtered_output_file}' and '{rest_output_file}'.")

    except FileNotFoundError:
        print(f"The file {input_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {input_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define keywords related to 'works' but exclude variations like 'networks'
keywords = ["citations","citedBy","cited","citation"]

# Call the function with the input file, filtered output file, rest output file, and keywords
input_file = '/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/institution/institutions_questions.json'
filtered_output_file = 'questions_with_citations_citedBy.json'
rest_output_file = 'remaining_questions.json'

filter_and_separate_questions(input_file, filtered_output_file, rest_output_file, keywords)
