import json

# List of dictionaries with file paths and corresponding key values
files_and_keys = [
    {
        "path": "/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/institution/acronym/questions_with_acronym.json",
        "key_value": "acronym"
    },
    {
        "path": "/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/institution/personal_questions/personal_questions.json",
        "key_value": "wikipedia_text"
    },
    {
        "path": "/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/institution/citedBy/questions_with_citations_citedBy.json",
        "key_value": "institution_citedByCount"
    },
    {
        "path": "/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/institution/type/questions_with_type.json",
        "key_value": "institution_type"
    },
    {
        "path": "/home/borista/Desktop/test-data-breakdown/Authors vs institutions-break/institution/works/questions_with_works.json",
        "key_value": "institution_worksCount"
    }
    
]

# Function to process and save JSON data
def process_file(file_info):
    file_path = file_info["path"]
    new_field_value = file_info["key_value"]
    
    # Read the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Add the new field to each JSON object
    for item in data:
        item["key"] = new_field_value
    
    # Determine the output filename from the file path
    output_filename = file_path.split("/")[-1]
    output_filename = f"{output_filename}"
    
    # Save the updated JSON data to a file
    with open(output_filename, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Updated JSON data has been saved to {output_filename}")

# Process each file
for file_info in files_and_keys:
    process_file(file_info)
