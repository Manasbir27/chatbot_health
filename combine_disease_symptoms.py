import csv

# Read the CSV file
diseases = []
all_symptoms = []

with open('merged_disease_symptoms.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        disease = row['Disease'].lower()
        diseases.append(disease)
        
        symptoms = [s.strip() for s in row['Combined_Symptoms'].split(',')]
        all_symptoms.extend(symptoms)

# Group diseases by symptoms
groups = {}
for symptom in set(all_symptoms):
    symptom = symptom.lower()
    groups[symptom] = set()

with open('merged_disease_symptoms.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        disease = row['Disease'].lower()
        symptoms = [s.strip().lower() for s in row['Combined_Symptoms'].split(',')]
        for symptom in symptoms:
            if symptom in groups:
                groups[symptom].add(disease)

# Combine diseases and their common symptoms
combined = {}
symptoms = list(groups.keys())

for i in range(len(symptoms)):
    set_ = groups[symptoms[i]]
    for j in range(i+1, len(symptoms)):
        set1 = groups[symptoms[j]]
        set_inter = set_.intersection(set1)
        if len(set_inter) > 1:
            dis = ' '.join(sorted(set_inter))
            if dis not in combined:
                combined[dis] = set([symptoms[i]])
            else:
                combined[dis].add(symptoms[i])

# Write the results to a new CSV file
with open('combined_disease_symptoms.csv', 'w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Diseases', 'Common_Symptoms'])
    
    for diseases, common_symptoms in combined.items():
        csv_writer.writerow([diseases, ', '.join(sorted(common_symptoms))])

print("Advanced combined disease symptoms have been written to 'combined_disease_symptoms.csv'")

# Print an example of the results
example_key = list(combined.keys())[min(100, len(combined.keys())-1)]
print(f"Example: {example_key}")
print(f"Common symptoms: {combined[example_key]}")
