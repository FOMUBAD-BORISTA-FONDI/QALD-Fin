import json

def load_questions(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}.")
        return []

def find_personal_questions(authors_file, *output_files):
    # Load all questions from the authors file
    authors_questions = load_questions(authors_file)
    
    # Create a set of questions from all output files
    output_questions = set()
    for file in output_files:
        questions = load_questions(file)
        for question in questions:
            output_questions.add(question.get('question', '').strip().lower())
    
    # Find questions from authors file that are not in any output files
    personal_questions = []
    for question in authors_questions:
        question_text = question.get('question', '').strip().lower()
        if question_text not in output_questions:
            personal_questions.append(question)
    
    # Write the personal questions to the output file
    with open('personal_questions.json', 'w') as file:
        json.dump(personal_questions, file, indent=4)
    
    print("Personal questions have been written to 'personal_questions.json'.")

# Define file paths
authors_file = '/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/authors/authors_questions.json'
output_files = [
    '/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/authors/2YearsMeanCitedness/questions_with_2yearsmean.json',
    '/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/authors/citedBy/questions_with_citations_citedBy.json',
    '/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/authors/hIndex/questions_with_hindex.json',
    '/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/authors/i10index/questions_with_i10index.json',
    '/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/authors/works/questions_with_works.json'
]

find_personal_questions(authors_file, *output_files)
