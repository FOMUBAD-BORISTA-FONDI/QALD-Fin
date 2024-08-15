import json

def sort_questions_by_text(input_file, output_file):
    try:
        # Load questions from the input file
        with open(input_file, 'r') as file:
            questions = json.load(file)
        
        # Sort questions by the 'question' field in ascending order
        sorted_questions = sorted(questions, key=lambda q: q.get('question', ''))
        
        # Write sorted questions to the output file
        with open(output_file, 'w') as file:
            json.dump(sorted_questions, file, indent=4)
        
        print(f"Questions have been sorted and written to {output_file}.")
    
    except FileNotFoundError:
        print(f"The file {input_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {input_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = '/home/borista/Desktop/test-data-breakdown/sch_set2_test_questions.json'
output_file = 'sorted_mod_sch_set2_test_questions.json'

sort_questions_by_text(input_file, output_file)
