import json

def filter_questions_with_dblp_uri_list(input_file, filtered_output_file, remaining_output_file):
    try:
        # Load questions from the input file
        with open(input_file, 'r') as file:
            questions = json.load(file)
        
        # Filter questions where 'author_dblp_uri' is a list
        filtered_questions = [q for q in questions if isinstance(q.get('author_dblp_uri'), list)]
        
        # Identify remaining questions that do not meet the filtering criteria
        remaining_questions = [q for q in questions if not isinstance(q.get('author_dblp_uri'), list)]
        
        # Write filtered questions to the filtered output file
        with open(filtered_output_file, 'w') as file:
            json.dump(filtered_questions, file, indent=4)
        
        # Write remaining questions to the remaining output file
        with open(remaining_output_file, 'w') as file:
            json.dump(remaining_questions, file, indent=4)
        
        print(f"Filtered questions have been written to {filtered_output_file}.")
        print(f"Remaining questions have been written to {remaining_output_file}.")
    
    except FileNotFoundError:
        print(f"The file {input_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {input_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = '/home/borista/Desktop/Schorlarly QALD/Code-Implementations/code/sch_set2_test_questions.json'  
filtered_output_file = 'filtered_questions.json'      
remaining_output_file = 'mod_sch_set2_test_questions.json'

filter_questions_with_dblp_uri_list(input_file, filtered_output_file, remaining_output_file)
