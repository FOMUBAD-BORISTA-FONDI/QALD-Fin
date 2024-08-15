import pandas as pd
import json

# Load the CSV data into a DataFrame
df = pd.read_csv('author_institution_info.csv')

# Initialize an empty dictionary to store the JSON data
data = {}

# Iterate through each row in the DataFrame
for _, row in df.iterrows():
    # Extract the unique ID
    unique_id = row['id']
    
    # Initialize a dictionary for the current ID
    if unique_id not in data:
        data[unique_id] = {
            "question": [],
            "Associated_author_uri": [],
            "id": [],
            "author_name": [],
            "hindex": [],
            "i10index": [],
            "citedByCount": [],
            "worksCount": [],
            "2YrMeanCitedness": [],
            "memberOf": [],
            "institution_name": [],
            "institution_country": [],
            "institution_type": [],
            "institution_acronym": [],
            "institution_citedByCount": [],
            "institution_worksCount": [],
            "wikipedia_text": []
        }
    
    # Append each field to the appropriate list in the dictionary
    if row["Question"] not in data[unique_id]["question"]:
        data[unique_id]["question"].append(row["Question"])

    if row["Associated_author_uri"] not in data[unique_id]["Associated_author_uri"]:
        data[unique_id]["Associated_author_uri"].append(row["Associated_author_uri"])
    
    if row["id"] not in data[unique_id]["id"]:
        data[unique_id]["id"].append(row["id"])

    if row["author_name"] not in data[unique_id]["author_name"]:
        data[unique_id]["author_name"].append(row["author_name"])

    if (row["hindex"] not in data[unique_id]["hindex"]) & (row["hindex"] != "Unknown"):
        data[unique_id]["hindex"].append(row["hindex"])

    if (row["i10index"] not in data[unique_id]["i10index"]) & (row["i10index"] != "Unknown"):
        data[unique_id]["i10index"].append(row["i10index"])

    if (row["citedByCount"] not in data[unique_id]["citedByCount"]) & (row["citedByCount"] != "Unknown"):
        data[unique_id]["citedByCount"].append(row["citedByCount"])
    
    if (row["worksCount"] not in data[unique_id]["worksCount"]) & (row["worksCount"] != "Unknown"):
        data[unique_id]["worksCount"].append(row["worksCount"])

    if (row["2YrMeanCitedness"] not in data[unique_id]["2YrMeanCitedness"]) & (row["2YrMeanCitedness"] != "Unknown"):
        data[unique_id]["2YrMeanCitedness"].append(row["2YrMeanCitedness"])

    if (row["memberOf"] not in data[unique_id]["memberOf"]) & (row["memberOf"] != "Unknown"):
        data[unique_id]["memberOf"].append(row["memberOf"])
    
    if (row["institution_name"] not in data[unique_id]["institution_name"]) & (row["institution_name"] != "Unknown"):
        data[unique_id]["institution_name"].append(row["institution_name"])

    if (row["institution_country"] not in data[unique_id]["institution_country"]) & (row["institution_country"] != "Unknown"):
        data[unique_id]["institution_country"].append(row["institution_country"])

    if (row["institution_type"] not in data[unique_id]["institution_type"]) & (row["institution_type"] != "Unknown"):
        data[unique_id]["institution_type"].append(row["institution_type"])
    
    if (row["institution_acronym"] not in data[unique_id]["institution_acronym"]) & (row["institution_acronym"] != "Unknown"):
        data[unique_id]["institution_acronym"].append(row["institution_acronym"])

    if (row["institution_citedByCount"] not in data[unique_id]["institution_citedByCount"]) & (row["institution_citedByCount"] != "Unknown"):
        data[unique_id]["institution_citedByCount"].append(row["institution_citedByCount"])

    if (row["institution_worksCount"] not in data[unique_id]["institution_worksCount"]) & (row["institution_worksCount"] != "Unknown"):
        data[unique_id]["institution_worksCount"].append(row["institution_worksCount"])

    if (row["wikipedia_text"] not in data[unique_id]["wikipedia_text"]) & (row["wikipedia_text"] != "No Wikipedia text found") :
        data[unique_id]["wikipedia_text"].append(row["wikipedia_text"])
    # data[unique_id]["id"].append(row["id"])
    # data[unique_id]["id"].append(row["id"])
    # data[unique_id]["author_name"].append(row["author_name"])
    # data[unique_id]["hindex"].append(row["hindex"])
    # data[unique_id]["i10index"].append(row["i10index"])
    # data[unique_id]["citedByCount"].append(row["citedByCount"])
    # data[unique_id]["worksCount"].append(row["worksCount"])
    # data[unique_id]["2YrMeanCitedness"].append(row["2YrMeanCitedness"])
    # data[unique_id]["memberOf"].append(row["memberOf"])
    # data[unique_id]["institution_name"].append(row["institution_name"])
    # data[unique_id]["institution_country"].append(row["institution_country"])
    # data[unique_id]["institution_type"].append(row["institution_type"])
    # data[unique_id]["institution_citedByCount"].append(row["institution_citedByCount"])
    # data[unique_id]["institution_worksCount"].append(row["institution_worksCount"])
    # data[unique_id]["wikipedia_text"].append(row["wikipedia_text"])

# Write the JSON data to a file
with open('author_institution_info.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

# Charger le contenu du fichier JSON dans la variable dict_question
# with open('author_institution_info.json', 'r') as file:
#     dict_question = json.load(file)

# # Vérifier le contenu de dict_question
# print(dict_question["6aa5cd8d-5c97-4c55-94b3-f3253440f731"])

print("CSV data has been successfully converted to JSON format.")
