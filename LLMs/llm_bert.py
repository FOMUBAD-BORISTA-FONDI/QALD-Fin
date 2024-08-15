import json
import requests
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

# Charger le modèle et le tokenizer pour le question-answering
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

# Fonction pour obtenir le nom de l'auteur à partir de DBLP
def get_author_name_from_dblp(author_dblp_uri):
    sparql_query = f"""
    PREFIX dblp: <https://dblp.org/rdf/schema#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name
    WHERE {{
        {author_dblp_uri} dblp:creatorName ?name .
    }}
    """
    endpoint = "https://dblp-april24.skynet.coypu.org/sparql"
    response = requests.post(endpoint, data={'query': sparql_query}, headers={'Accept': 'application/sparql-results+json'})

    if response.status_code == 200:
        results = response.json()
        if results['results']['bindings']:
            return results['results']['bindings'][0]['name']['value']
    print(f"Errors or no results in DBLP query: {response.text}")
    return None

# Fonction pour obtenir les informations de l'auteur à partir de SemOpenAlex
def get_author_info_from_semopenalex(author_name):
    sparql_query = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX ns2: <https://semopenalex.org/ontology/>
    PREFIX org: <http://www.w3.org/ns/org#>
    PREFIX ns3: <http://purl.org/spar/bido/>

    SELECT ?author ?name ?memberOf ?citedByCount ?worksCount ?hindex ?i10Index ?myc
    WHERE {{
        ?author foaf:name ?name .
        ?author org:memberOf ?memberOf .
        ?author ns2:citedByCount ?citedByCount .
        ?author ns2:worksCount ?worksCount .
        ?author ns3:h-index ?hindex .
        ?author ns2:2YrMeanCitedness ?myc .
        ?author ns2:i10Index ?i10Index .

        FILTER(lcase(str(?name)) = lcase("{author_name}"))
    }}
    """
    endpoint = "https://semoa.skynet.coypu.org/sparql"
    response = requests.post(endpoint, data={'query': sparql_query}, headers={'Accept': 'application/sparql-results+json'})

    if response.status_code == 200:
        results = response.json()
        if results['results']['bindings']:
            return results['results']['bindings'][0]

    return None

def extract_info(author_info, key):
    if author_info is None:
        return "Information not available"
    return author_info.get(key, {}).get('value', 'Information not available')

def compare_values(value1, value2, comparison_type):
    if value1 == "Information not available" or value2 == "Information not available":
        return "Information not available"

    try:
        value1 = float(value1)
        value2 = float(value2)
    except ValueError:
        return "Information not available"

    if comparison_type == "higher":
        return max(value1, value2)
    elif comparison_type == "lower":
        return min(value1, value2)
    else:
        return "Invalid comparison type"

# Fonction pour obtenir des informations d'institution à partir de SemOpenAlex
def get_institution_info_from_semopenalex(institution_uri):
    if not institution_uri:
        return None

    sparql_query = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX ns3: <https://semopenalex.org/ontology/>
    PREFIX ns4: <https://dbpedia.org/property/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?citedByCount ?worksCount ?homepage ?name ?countryCode ?rorType
    WHERE {{
        {institution_uri} a ns3:Institution ;
        ns3:citedByCount ?citedByCount ;
        ns3:worksCount ?worksCount ;
        foaf:homepage ?homepage ;
        foaf:name ?name ;
        ns4:countryCode ?countryCode ;
        ns4:acronym ?acronym ;
        ns3:rorType ?rorType .
    }}
    """
    endpoint = "https://semoa.skynet.coypu.org/sparql"
    response = requests.post(endpoint, data={"query": sparql_query}, headers={"Accept": "application/sparql-results+json"})
    if response.status_code == 200:
        results = response.json()
        if results['results']['bindings']:
            return results['results']['bindings'][0]
    return None

# Lire le fichier JSON complet (qui est une liste de dictionnaires)
with open('processed_sch_set2_test_questions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

predictions = []
predictionsa = []
null_count = 0  # Compteur pour les réponses nulles

# Parcourir chaque question dans le fichier
for test_data in data:
    try:
        question_id = test_data['id']
        question_text = test_data['question']
        context = test_data.get('context', "")
        author_dblp_uri = test_data.get('author_dblp_uri')

        print(f"Processing ID: {question_id}")

        # Étape 1: Essayer de répondre avec les données de SemOpenAlex ou DBLP
        if isinstance(author_dblp_uri, str):
            # Cas d'un seul auteur
            author_name = get_author_name_from_dblp(author_dblp_uri)
            author_info = get_author_info_from_semopenalex(author_name) if author_name else None
            institution_info = get_institution_info_from_semopenalex(extract_info(author_info, 'memberOf'))

            if "citedness" in question_text.lower():
                answer = extract_info(author_info, 'myc')
            elif "hindex" in question_text.lower():
                answer = extract_info(author_info, 'hindex')
            elif "i10index" in question_text.lower():
                answer = extract_info(author_info, 'i10Index')
            elif "cited by count" in question_text.lower() or "citedbycount" in question_text.lower() or "citedby count" in question_text.lower():
                answer = extract_info(author_info, 'citedByCount')
            elif "works count" in question_text.lower() or "workscount" in question_text.lower():
                answer = extract_info(author_info, 'worksCount')
            elif "cited by count" and "where" in question_text.lower():
              answer = extract_info(institution_info, 'citedByCount')
            elif "cited by count" and "institution" in question_text.lower():
              answer = extract_info(institution_info, 'citedByCount')
            elif "kind" and "institution" in question_text.lower():
              answer = extract_info(institution_info, 'rorType')
            elif "type" in question_text.lower():
              answer = extract_info(institution_info, 'rorType')
            elif "What is the number of publications" in question_text.lower() and "institution" in question_text.lower():
              answer = extract_info(institution_info, 'worksCount')
            elif "What is the number of publications" in question_text.lower() and "affiliation" in question_text.lower():
              answer = extract_info(institution_info, 'worksCount')
            elif "What is the number of publications" in question_text.lower() and "cited" in question_text.lower():
              answer = extract_info(institution_info, 'citedByCount')
            elif "How many publications" in question_text.lower() and "institution" in question_text.lower():
              answer = extract_info(institution_info, 'worksCount') 
            elif "How many publications" in question_text.lower() and "affiliation" in question_text.lower():
              answer = extract_info(institution_info, 'worksCount')
            elif "How many books has " in question_text.lower() in question_text.lower():
              answer = extract_info(author_info, 'worksCount') 
            elif "short name" in question_text.lower():
              answer = extract_info(institution_info, 'acronym')
            else:
                answer = "Information not available"

        elif isinstance(author_dblp_uri, list) and len(author_dblp_uri) == 2:
            # Cas comparatif pour deux auteurs
            author_name1 = get_author_name_from_dblp(author_dblp_uri[0])
            author_info1 = get_author_info_from_semopenalex(author_name1) if author_name1 else None

            author_name2 = get_author_name_from_dblp(author_dblp_uri[1])
            author_info2 = get_author_info_from_semopenalex(author_name2) if author_name2 else None

            if "higher" in question_text.lower() and "hindex" in question_text.lower():
                hindex1 = extract_info(author_info1, 'hindex')
                hindex2 = extract_info(author_info2, 'hindex')
                answer = compare_values(hindex1, hindex2, "higher")
            elif "higher" in question_text.lower() and "i10index" in question_text.lower():
                i10index1 = extract_info(author_info1, 'i10Index')
                i10index2 = extract_info(author_info2, 'i10Index')
                answer = compare_values(i10index1, i10index2, "higher")
            elif "higher" in question_text.lower() and "citedbycount" in question_text.lower():
                citedByCount1 = extract_info(author_info1, 'citedByCount')
                citedByCount2 = extract_info(author_info2, 'citedByCount')
                answer = compare_values(citedByCount1, citedByCount2, "higher")
            elif "higher" and  "works count" in question_text.lower():
                worksCount1 = extract_info(author_info1, 'worksCount')
                worksCount2 = extract_info(author_info2, 'worksCount')
                answer = compare_values(worksCount1, worksCount2, "higher")
            elif "higher" and "twoyearscitedness" in question_text.lower():
                myc1 = extract_info(author_info1, 'myc')
                myc2 = extract_info(author_info2, 'myc')
                answer = compare_values(myc1, myc2, "higher")
            else:
                answer = "Information not available"


        else:
            answer = "Information not available"

        # Étape 2: Si aucune réponse n'est trouvée via SPARQL, utiliser le modèle BERT pour prédire la réponse
        if answer == "Information not available":
            qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
            result = qa_pipeline(question=question_text, context=context)
            answer = result['answer']

        if not answer:  # Si l'answer est vide
            null_count += 1
            answer = "No answer found"

        # Ajouter les prédictions à la liste
        predictions.append(
            {
                "id": question_id,
                "answer": answer
                }
            )
        predictionsa.append({"id": question_id, "question": question_text, "answer": answer, "context": context})

    except Exception as e:
        print(f"An error occurred while processing ID {question_id}: {str(e)}")

# Sauvegarder les prédictions dans un fichier JSON
with open('answers2.txt', 'w', encoding='utf-8') as outfile:
    json.dump(predictions, outfile, ensure_ascii=False, indent=4)

with open('answers2context.txt', 'w', encoding='utf-8') as outfile:
    json.dump(predictionsa, outfile, ensure_ascii=False, indent=4)


print(f"Processing complete. Total null predictions: {null_count}")
















# import json
# import requests
# from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

# # Charger le modèle et le tokenizer pour le question-answering
# model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
# tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

# # Fonction pour obtenir le nom de l'auteur à partir de DBLP
# def get_author_name_from_dblp(author_dblp_uri):
#     sparql_query = f"""
#     PREFIX dblp: <https://dblp.org/rdf/schema#>
#     PREFIX foaf: <http://xmlns.com/foaf/0.1/>

#     SELECT ?name
#     WHERE {{
#         {author_dblp_uri} dblp:creatorName ?name .
#     }}
#     """
#     endpoint = "https://dblp-april24.skynet.coypu.org/sparql"
#     response = requests.post(endpoint, data={'query': sparql_query}, headers={'Accept': 'application/sparql-results+json'})

#     if response.status_code == 200:
#         results = response.json()
#         if results['results']['bindings']:
#             return results['results']['bindings'][0]['name']['value']
#     print(f"Errors or no results in DBLP query: {response.text}")
#     return None

# # Fonction pour obtenir les informations de l'auteur à partir de SemOpenAlex
# def get_author_info_from_semopenalex(author_name):
#     sparql_query = f"""
#     PREFIX dcterms: <http://purl.org/dc/terms/>
#     PREFIX foaf: <http://xmlns.com/foaf/0.1/>
#     PREFIX ns2: <https://semopenalex.org/ontology/>
#     PREFIX org: <http://www.w3.org/ns/org#>
#     PREFIX ns3: <http://purl.org/spar/bido/>

#     SELECT ?author ?name ?memberOf ?citedByCount ?worksCount ?hindex ?i10Index ?myc
#     WHERE {{
#         ?author foaf:name ?name .
#         ?author org:memberOf ?memberOf .
#         ?author ns2:citedByCount ?citedByCount .
#         ?author ns2:worksCount ?worksCount .
#         ?author ns3:h-index ?hindex .
#         ?author ns2:2YrMeanCitedness ?myc .
#         ?author ns2:i10Index ?i10Index .

#         FILTER(lcase(str(?name)) = lcase("{author_name}"))
#     }}
#     """
#     endpoint = "https://semoa.skynet.coypu.org/sparql"
#     response = requests.post(endpoint, data={'query': sparql_query}, headers={'Accept': 'application/sparql-results+json'})

#     if response.status_code == 200:
#         results = response.json()
#         if results['results']['bindings']:
#             return results['results']['bindings'][0]

#     return None

# def extract_info(author_info, key):
#     if author_info is None:
#         return "Information not available"
#     return author_info.get(key, {}).get('value', 'Information not available')

# def compare_values(value1, value2, comparison_type):
#     if value1 == "Information not available" or value2 == "Information not available":
#         return "Information not available"

#     try:
#         value1 = float(value1)
#         value2 = float(value2)
#     except ValueError:
#         return "Information not available"

#     if comparison_type == "higher":
#         return max(value1, value2)
#     elif comparison_type == "lower":
#         return min(value1, value2)
#     else:
#         return "Invalid comparison type"

# # Fonction pour obtenir des informations d'institution à partir de SemOpenAlex
# def get_institution_info_from_semopenalex(institution_uri):
#     if not institution_uri:
#         return None

#     sparql_query = f"""
#     PREFIX dcterms: <http://purl.org/dc/terms/>
#     PREFIX foaf: <http://xmlns.com/foaf/0.1/>
#     PREFIX ns3: <https://semopenalex.org/ontology/>
#     PREFIX ns4: <https://dbpedia.org/property/>
#     PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

#     SELECT ?citedByCount ?worksCount ?homepage ?name ?countryCode ?rorType
#     WHERE {{
#         {institution_uri} a ns3:Institution ;
#         ns3:citedByCount ?citedByCount ;
#         ns3:worksCount ?worksCount ;
#         foaf:homepage ?homepage ;
#         foaf:name ?name ;
#         ns4:countryCode ?countryCode ;
#         ns3:rorType ?rorType .
#     }}
#     """
#     endpoint = "https://semoa.skynet.coypu.org/sparql"
#     response = requests.post(endpoint, data={"query": sparql_query}, headers={"Accept": "application/sparql-results+json"})
#     if response.status_code == 200:
#         results = response.json()
#         if results['results']['bindings']:
#             return results['results']['bindings'][0]
#     return None

# # Lire le fichier JSON complet (qui est une liste de dictionnaires)
# with open('processed_sch_set2_test_questions.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# predictions = []
# null_count = 0  # Compteur pour les réponses nulles

# # Parcourir chaque question dans le fichier
# for test_data in data:
#     try:
#         question_id = test_data['id']
#         question_text = test_data['question']
#         context = test_data.get('context', "")
#         author_dblp_uri = test_data.get('author_dblp_uri')

#         print(f"Processing ID: {question_id}")

#         # Étape 1: Essayer de répondre avec les données de SemOpenAlex ou DBLP
#         if isinstance(author_dblp_uri, str):
#             # Cas d'un seul auteur
#             author_name = get_author_name_from_dblp(author_dblp_uri)
#             author_info = get_author_info_from_semopenalex(author_name) if author_name else None
#             institution_info = get_institution_info_from_semopenalex(extract_info(author_info, 'memberOf'))

#             if "citedness" in question_text.lower():
#                 answer = extract_info(author_info, 'myc')
#             elif "hindex" in question_text.lower():
#                 answer = extract_info(author_info, 'hindex')
#             elif "i10index" in question_text.lower():
#                 answer = extract_info(author_info, 'i10Index')
#             elif "cited by count" in question_text.lower() or "citedbycount" in question_text.lower() or "citedby count" in question_text.lower():
#                 answer = extract_info(author_info, 'citedByCount')
#             elif "works count" in question_text.lower() or "workscount" in question_text.lower():
#                 answer = extract_info(author_info, 'worksCount')
#             elif "cited by count" and "where" in question_text.lower():
#               answer = extract_info(institution_info, 'citedByCount')
#             elif "cited by count" and "institution" in question_text.lower():
#               answer = extract_info(institution_info, 'citedByCount')
#             elif "many papers" in question_text.lower() or "many publications" in question_text.lower():
#               answer = extract_info(institution_info, 'worksCount')
#             else:
#                 answer = "Information not available"

#         elif isinstance(author_dblp_uri, list) and len(author_dblp_uri) == 2:
#             # Cas comparatif pour deux auteurs
#             author_name1 = get_author_name_from_dblp(author_dblp_uri[0])
#             author_info1 = get_author_info_from_semopenalex(author_name1) if author_name1 else None

#             author_name2 = get_author_name_from_dblp(author_dblp_uri[1])
#             author_info2 = get_author_info_from_semopenalex(author_name2) if author_name2 else None

#             if "higher" in question_text.lower() and "hindex" in question_text.lower():
#                 hindex1 = extract_info(author_info1, 'hindex')
#                 hindex2 = extract_info(author_info2, 'hindex')
#                 answer = compare_values(hindex1, hindex2, "higher")
#             elif "higher" in question_text.lower() and "i10index" in question_text.lower():
#                 i10index1 = extract_info(author_info1, 'i10Index')
#                 i10index2 = extract_info(author_info2, 'i10Index')
#                 answer = compare_values(i10index1, i10index2, "higher")
#             elif "higher" in question_text.lower() and "citedbycount" in question_text.lower():
#                 citedByCount1 = extract_info(author_info1, 'citedByCount')
#                 citedByCount2 = extract_info(author_info2, 'citedByCount')
#                 answer = compare_values(citedByCount1, citedByCount2, "higher")
#             elif "higher" and  "works count" in question_text.lower():
#                 worksCount1 = extract_info(author_info1, 'worksCount')
#                 worksCount2 = extract_info(author_info2, 'worksCount')
#                 answer = compare_values(worksCount1, worksCount2, "higher")
#             elif "higher" and "twoyearscitedness" in question_text.lower():
#                 myc1 = extract_info(author_info1, 'myc')
#                 myc2 = extract_info(author_info2, 'myc')
#                 answer = compare_values(myc1, myc2, "higher")
#             else:
#                 answer = "Information not available"


#         else:
#             answer = "Information not available"

#         # Étape 2: Si aucune réponse n'est trouvée via SPARQL, utiliser le modèle BERT pour prédire la réponse
#         if answer == "Information not available":
#             qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
#             result = qa_pipeline(question=question_text, context=context)
#             answer = result['answer']

#         if not answer:  # Si l'answer est vide
#             null_count += 1
#             answer = "No answer found"

#         # Ajouter les prédictions à la liste
#         predictions.append({"id": question_id, "question": question_text, "prediction": answer})

#     except Exception as e:
#         print(f"An error occurred while processing ID {question_id}: {str(e)}")

# # Sauvegarder les prédictions dans un fichier JSON
# with open('answers2.txt', 'w', encoding='utf-8') as outfile:
#     json.dump(predictions, outfile, ensure_ascii=False, indent=4)

# print(f"Processing complete. Total null predictions: {null_count}")