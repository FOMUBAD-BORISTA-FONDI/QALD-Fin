import requests
import json
import re
import os
import wikipedia

def get_author_name_from_dblp(author_dblp_uri):
    sparql_query = f"""
    PREFIX dblp: <https://dblp.org/rdf/schema#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name
    WHERE {{
        {author_dblp_uri} dblp:primaryCreatorName ?name .
    }}
    """
    endpoint = "https://dblp-april24.skynet.coypu.org/sparql"
    response = requests.post(endpoint, data={'query': sparql_query}, headers={'Accept': 'application/sparql-results+json'})

    if response.status_code == 200:
        results = response.json()
        if results['results']['bindings']:
            name = results['results']['bindings'][0]['name']['value']
            print(f"Author name from DBLP: {name}")
            return name
        else:
            print(f"No name found for {author_dblp_uri}")
            return None
    else:
        print(f"Error in SPARQL query execution for DBLP: {response.status_code}")
        return None

def get_author_info_from_semopenalex(author_name):
    query = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX ns2: <https://semopenalex.org/ontology/>
    PREFIX org: <http://www.w3.org/ns/org#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX ns3: <http://purl.org/spar/bido/>

    SELECT ?author ?name ?memberOf ?citedByCount ?worksCount ?hindex ?i10Index ?2YrMeanCitedness
    WHERE {{
        {{
            ?author foaf:name ?name .
            ?author org:memberOf ?memberOf .
            ?author ns2:citedByCount ?citedByCount .
            ?author ns2:worksCount ?worksCount .
            ?author ns3:h-index ?hindex .
            ?author ns2:2YrMeanCitedness ?2YrMeanCitedness .
            ?author ns2:i10Index ?i10Index .

            FILTER(lcase(str(?name)) = lcase("{author_name}"))
        }}
        UNION
        {{
            ?author ns2:alternativeName ?altName .
            ?author foaf:name ?name .
            ?author org:memberOf ?memberOf .
            ?author ns2:citedByCount ?citedByCount .
            ?author ns2:worksCount ?worksCount .
            ?author ns3:h-index ?hindex .
            ?author ns2:2YrMeanCitedness ?2YrMeanCitedness .
            ?author ns2:i10Index ?i10Index .

            FILTER(lcase(str(?altName)) = lcase("{author_name}") && lcase(str(?altName)) != lcase(str(?name)))
        }}
    }}
    """

    endpoint = "https://semoa.skynet.coypu.org/sparql"
    response = requests.post(endpoint, data={"query": query}, headers={"Accept": "application/sparql-results+json"})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data from SemOpenAlex: {response.status_code}")
        return None

def get_institution_info_from_semopenalex(institution_uri):
    query = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX ns3: <https://semopenalex.org/ontology/>
    PREFIX ns4: <https://dbpedia.org/property/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?citedByCount ?worksCount (STR(?homepage) AS ?homepage) (STR(?name) AS ?name) (STR(?countryCode) AS ?countryCode) (STR(?rorType) AS ?rorType)
    WHERE {{
        <{institution_uri}> a ns3:Institution ;
        ns3:citedByCount ?citedByCount ;
        ns3:worksCount ?worksCount ;
        foaf:homepage ?homepage ;
        foaf:name ?name ;
        ns4:countryCode ?countryCode ;
        ns3:rorType ?rorType .
    }}
    """

    endpoint = "https://semoa.skynet.coypu.org/sparql"
    response = requests.post(endpoint, data={"query": query}, headers={"Accept": "application/sparql-results+json"})
    if response.status_code == 200:
        results = response.json()
        if results['results']['bindings']:
            return results
        else:
            return None
    else:
        print(f"Error in SPARQL query execution for SemOpenAlex: {response.status_code}")
        return None

def get_wikipedia_text(author_name):
    try:
        page = wikipedia.page(author_name)
        text = page.content
        text = re.sub(r'\\u[\dA-Fa-f]{4}', '', text)

        return text
    except:
        print(f"No Wikipedia page found for {author_name}")
        return None



def formulate_info(author_details, institution_details, wikipedia_text):
    if author_details and author_details['results']['bindings']:
        author_info = author_details['results']['bindings'][0]
        author_name = author_info.get('name', {}).get('value', "Unknown name")
        author_hindex = author_info.get('hindex', {}).get('value', "Unknown")
        author_i10index = author_info.get('i10Index', {}).get('value', "Unknown")
        author_cbc = author_info.get('citedByCount', {}).get('value', "Unknown")
        author_wc = author_info.get('worksCount', {}).get('value', "Unknown")
        author_myc = author_info.get('2YrMeanCitedness', {}).get('value', "Unknown")
    else:
        author_name = "Unknown"
        author_hindex = author_i10index = author_cbc = author_wc = author_myc = "Unknown"

    if institution_details and institution_details['results']['bindings']:
        inst_info = institution_details['results']['bindings'][0]
        institution_name = inst_info.get('name', {}).get('value', "Unknown institution")
        institution_type = inst_info.get('rorType', {}).get('value', "Unknown type")
        institution_country = inst_info.get('countryCode', {}).get('value', "Unknown country")
        institution_cbc = inst_info.get('citedByCount', {}).get('value', "Unknown")
        institution_wc = inst_info.get('worksCount', {}).get('value', "Unknown")
        institution_homepage = inst_info.get('homepage', {}).get('value', "Unknown")
    else:
        institution_name = institution_type = institution_country = institution_cbc = institution_wc = institution_homepage = "Unknown"

    author_info = (
        f"Author: {author_name}\n"
        f"Affiliation: {institution_name}\n"
        f"Numerical metrics:\n"
        f"- hIndex: {author_hindex}\n"
        f"- i10Index: {author_i10index}\n"
        f"- Cited by count: {author_cbc}\n"
        f"- Works count: {author_wc}\n"
        f"- Two-year mean citedness: {author_myc}\n"
    )

    institution_info = (
        f"Institution: {institution_name}\n"
        f"Type: {institution_type}\n"
        f"Location: {institution_country}\n"
        f"Numerical metrics:\n"
        f"- Cited by count: {institution_cbc}\n"
        f"- Works count: {institution_wc}\n"
        f"Homepage: {institution_homepage}\n"
    )

    wikipedia_summary = f"Additional information: {wikipedia_text[:500]}..." if wikipedia_text else ""

    return f"{author_info}\n{institution_info}\n{wikipedia_summary}"




def process_question(question):
    author_uris = question.get('author_dblp_uri', [])
    question_id = question.get('id')
    question_text = question.get('question')

    contexts = []
    all_author_uris = []

    if isinstance(author_uris, str):
        author_uris = [{'author_dblp_uri': author_uris}]

    for author_uri_dict in author_uris:
        for key, uri in author_uri_dict.items():
            if uri:
                all_author_uris.append(uri)
                author_name = get_author_name_from_dblp(uri)
                if author_name:
                    author_details = get_author_info_from_semopenalex(author_name)
                    if author_details and author_details['results']['bindings']:
                        author_info = author_details['results']['bindings'][0]
                        member_of_uri = author_info['memberOf']['value']
                        institution_details = get_institution_info_from_semopenalex(member_of_uri)
                        wikipedia_text = get_wikipedia_text(author_name)
                        context = formulate_info(author_details, institution_details, wikipedia_text)
                        contexts.append(context)

    combined_context = " ".join(contexts)

    if len(all_author_uris) == 1:
        author_uri_representation = all_author_uris[0]
    else:
        author_uri_representation = all_author_uris

    return {
        "id": question_id,
        "question": question_text,
        "author_dblp_uri": author_uri_representation,
        "context": combined_context.strip()
    }

def process_questions_from_file(input_file_path, output_file_path):
    processed_questions = []

    # Charger les questions existantes si le fichier existe
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            processed_questions = json.load(f)

    processed_ids = set(q['id'] for q in processed_questions)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    for question in questions:
        question_id = question.get('id')
        if question_id not in processed_ids:
            processed_question = process_question(question)
            if processed_question:
                processed_questions.append(processed_question)

    # Écrire la liste complète des questions traitées dans le fichier de sortie
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_questions, f, ensure_ascii=False, indent=2)

    print(f"Questions traitées écrites dans {output_file_path}")

if __name__ == "__main__":
    input_file_path = "sch_set2_test_questions.json"
    output_file_path = "processed_sch_set2_test_questions.json"

    process_questions_from_file(input_file_path, output_file_path)