import os
import csv
import networkx as nx
from collections import defaultdict
from transformers import AutoTokenizer
import requests
from fuzzywuzzy import process

# Set the Hugging Face token as an environment variable
os.environ['HUGGINGFACE_TOKEN'] = "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"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def extract_symptoms(response):
    lines = response.split('\n')
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("symptoms:"):
            return '\n'.join(lines[i:]).strip()
    return response.strip()

def analyze_symptoms(user_input):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    messages = [
        {"role": "system", "content": "You are a medical assistant. Analyze the given text for symptoms. Respond with a list of detected symptoms only, starting with 'Symptoms:'. Do not provide any other explanation or analysis."},
        {"role": "user", "content": user_input},
    ]
    try:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_text = tokenizer.decode(inputs[0])
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.3,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        response = query(payload)
        if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
            result = response[0]['generated_text']
            assistant_response = result.split("[/INST]")[-1].strip()
            symptoms = extract_symptoms(assistant_response)
            return [s.strip().lower() for s in symptoms.replace("Symptoms:", "").strip().split(",")]
        else:
            return []
    except Exception as e:
        print(f"Error: An unexpected error occurred. Details: {str(e)}")
        return []

def create_knowledge_graph(csv_file):
    G = nx.Graph()
    symptoms_dict = defaultdict(set)
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            disease = row['Disease'].strip().lower()
            symptom = row['Symptom'].strip().lower()

            G.add_node(disease, node_type='disease')
            G.add_node(symptom, node_type='symptom')
            G.add_edge(disease, symptom)
            symptoms_dict[symptom].add(disease)

    print(f"Knowledge graph created with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    print(f"Number of diseases: {len([n for n, d in G.nodes(data=True) if d['node_type'] == 'disease'])}")
    print(f"Number of symptoms: {len([n for n, d in G.nodes(data=True) if d['node_type'] == 'symptom'])}")
    return G, symptoms_dict

def get_potential_diseases(G, detected_symptoms):
    potential_diseases = defaultdict(set)
    all_symptoms = [n for n, d in G.nodes(data=True) if d['node_type'] == 'symptom']
    
    for symptom in detected_symptoms:
        matches = process.extract(symptom, all_symptoms, limit=3)
        for match, score in matches:
            if score > 80:  # Adjust threshold as needed
                for disease in G.neighbors(match):
                    potential_diseases[disease].add(match)
    
    return potential_diseases

def ask_additional_questions(G, potential_diseases, confirmed_symptoms):
    question_count = 0
    max_questions = 10  # Increased number of questions

    for disease, symptoms in potential_diseases.items():
        all_disease_symptoms = set(G.neighbors(disease))
        remaining_symptoms = all_disease_symptoms - symptoms - confirmed_symptoms
        if not remaining_symptoms:
            continue
        
        print(f"\nChecking symptoms for {disease.capitalize()}:")
        for symptom in remaining_symptoms:
            while True:
                response = input(f"Do you experience {symptom}? (yes/no/unsure) or type 'quit' to exit: ").lower()
                if response in ['yes', 'no', 'unsure', 'quit']:
                    break
                print("Invalid input. Please enter 'yes', 'no', 'unsure', or 'quit'.")
            
            if response == 'quit':
                return confirmed_symptoms, True  # User wants to quit
            elif response == 'yes':
                confirmed_symptoms.add(symptom)
            
            question_count += 1
            if question_count >= max_questions:
                return confirmed_symptoms, False

    return confirmed_symptoms, False

def diagnose_disease(G, potential_diseases, confirmed_symptoms, threshold=0.5):
    diagnosed_diseases = []
    for disease, initial_symptoms in potential_diseases.items():
        all_disease_symptoms = set(G.neighbors(disease))
        matched_symptoms = initial_symptoms.union(confirmed_symptoms)
        match_ratio = len(matched_symptoms) / len(all_disease_symptoms)
        confidence = min(match_ratio * 100, 100)  # Ensure confidence doesn't exceed 100%
        if match_ratio >= threshold:
            diagnosed_diseases.append((disease, confidence))
    return diagnosed_diseases

def main():
    print("Welcome to the Disease Diagnosis Chatbot.")
    print("DISCLAIMER: This chatbot is for informational purposes only and does not replace professional medical advice.")
    print("Please consult with a healthcare professional for accurate diagnosis and treatment.")
    print("Type 'quit' at any time to exit.")

    G, symptoms_dict = create_knowledge_graph('new_disease_symptom_relations.csv')
    all_symptoms = set()

    while True:
        user_input = input("\nDescribe your symptoms (or type 'add' to add more symptoms, 'list' to see current symptoms): ")
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'add':
            continue
        elif user_input.lower() == 'list':
            if all_symptoms:
                print("Current symptoms:", ", ".join(all_symptoms))
            else:
                print("No symptoms recorded yet.")
            continue

        detected_symptoms = set(analyze_symptoms(user_input))
        all_symptoms.update(detected_symptoms)

        print("\nDetected Symptoms:")
        print(", ".join(all_symptoms))

        potential_diseases = get_potential_diseases(G, all_symptoms)
        
        if not potential_diseases:
            print("No potential diseases found based on the provided symptoms.")
            print("Please try describing your symptoms in more detail or consult a healthcare professional.")
            continue

        print("\nPotential diseases based on symptoms:")
        for disease in potential_diseases:
            print(f"- {disease.capitalize()}")

        confirmed_symptoms = set()
        while True:
            print("\nAsking additional questions to refine the diagnosis...")
            new_confirmed_symptoms, quit_flag = ask_additional_questions(G, potential_diseases, confirmed_symptoms)

            if quit_flag:
                print("Exiting the program as per user request.")
                return

            confirmed_symptoms.update(new_confirmed_symptoms)
            all_symptoms.update(confirmed_symptoms)

            diagnosed_diseases = diagnose_disease(G, potential_diseases, confirmed_symptoms)

            if diagnosed_diseases:
                print("\nBased on the symptoms provided, you may have:")
                for disease, confidence in diagnosed_diseases:
                    print(f"- {disease.capitalize()} (Confidence: {confidence:.2f}%)")
                    matched_symptoms = potential_diseases[disease].union(confirmed_symptoms)
                    print(f"  Matched symptoms: {', '.join(matched_symptoms)}")
                print("\nPlease consult with a healthcare professional for an accurate diagnosis.")
            else:
                print("\nUnable to determine a specific disease based on the provided symptoms.")
                print("Please consult with a healthcare professional for proper diagnosis.")

            more = input("\nDo you want to continue the diagnosis process? (yes/no): ").lower()
            if more != 'yes':
                break

if __name__ == "__main__":
    main()