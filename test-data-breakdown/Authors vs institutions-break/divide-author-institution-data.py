import json

def filter_and_separate_questions(input_file, filtered_output_file, rest_output_file, keywords):
    # Load questions from the input file
    with open(input_file, 'r') as file:
        questions = json.load(file)

    # Separate questions based on the presence of keywords
    filtered_questions = []
    rest_questions = []
    for question in questions:
        question_text = question.get('question', '').lower()
        if any(keyword.lower() in question_text for keyword in keywords):
            filtered_questions.append(question)
        else:
            rest_questions.append(question)

    # Write filtered questions to the filtered output file
    with open(filtered_output_file, 'w') as file:
        json.dump(filtered_questions, file, indent=4)

    # Write the rest of the questions to the rest output file
    with open(rest_output_file, 'w') as file:
        json.dump(rest_questions, file, indent=4)

    print("Filtering complete. The results are saved in 'filtered_questions.json' and 'rest_questions.json'.")

# Define keywords related to organizations or institutions
keywords = ["organization", "institution", "affiliation", "affiliated", "institute"]

# Call the function with the input file, filtered output file, rest output file, and keywords
filter_and_separate_questions('/home/borista/Desktop/test-data-breakdown/remove_list_uri/mod_sch_set2_test_questions.json', 'institutions_questions.json', 'authors_questions.json', keywords)