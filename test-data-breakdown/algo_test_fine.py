import json
import pandas as pd
import os

# Directory containing the JSON files to process
input_directory = '/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/'

# List of files to process
file_names = [
    'authors/hIndex/questions_with_hindex.json',
    'authors/2YearsMeanCitedness/questions_with_2yearsmean.json',
    'authors/citedBy/questions_with_citations_citedBy.json',
    'authors/i10index/questions_with_i10index.json',
    'authors/works/questions_with_works.json',
    'institution/citedBy/questions_with_citations_citedBy.json',
    'institution/type/questions_with_type.json',
    'institution/works/questions_with_works.json'
    ]
# Load the dictionary from the JSON file
with open('/home/borista/Desktop/Schorlarly QALD/Code-Implementations/code/Tabular/results-1/current/author_institution_info.json', 'r') as file:
    dict_question = json.load(file)

# Load the existing answers from the JSON file
try:
    with open('answers.json', 'r') as json_file:
        existing_answers = json.load(json_file)
except FileNotFoundError:
    existing_answers = []

# Create a dictionary for quick lookup of existing ids
existing_ids = {item['id'] for item in existing_answers}

# Initialize a list for new answers
new_answers = []

# Process each file
for file_name in file_names:
    file_path = os.path.join(input_directory, file_name)
    
    # Load the DataFrame from the JSON file
    df = pd.read_json(file_path)

    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        unique_id = row['id']

        # Skip if the ID already exists in the existing answers
        if unique_id in existing_ids:
            continue

        # Initialize the default answer as null
        answer_value = None

        if unique_id in dict_question:
            # Check if there's an answer available for the current key
            if dict_question[unique_id].get(row["key"]):
                answer_value = dict_question[unique_id][row["key"]][0]

        # Append the new result to the new_answers list
        new_answers.append({
            "id": unique_id,
            "answer": answer_value
        })

# Combine existing answers with new answers
all_answers = existing_answers + new_answers

# Write the updated results to the JSON file
with open('answers2.json', 'w') as json_file:
    json.dump(all_answers, json_file, indent=4)

# Print success message
print("Successfully updated 'answers2.json' with new entries from multiple files.")