import csv
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os

def clean_csv(input_csv, output_csv):
    """
    Cleans the CSV by removing lemmatization, stemming, and any preprocessing.
    Writes the cleaned data to a new CSV file.
    """
    diseases = []
    all_symptoms = []
    
    with open(input_csv, 'r', encoding='utf-8') as file_in, \
         open(output_csv, 'w', newline='', encoding='utf-8') as file_out:
        
        csv_reader = csv.DictReader(file_in)
        fieldnames = csv_reader.fieldnames
        csv_writer = csv.DictWriter(file_out, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        for row in csv_reader:
            # Clean Disease
            disease = row['Disease'].strip().lower()
            row['Disease'] = disease
            diseases.append(disease)
            
            # Clean Symptoms
            symptoms = [s.strip().lower() for s in row['Combined_Symptoms'].split(',')]
            row['Combined_Symptoms'] = ', '.join(symptoms)
            all_symptoms.extend(symptoms)
            
            csv_writer.writerow(row)
    
    print(f"Cleaned data has been written to '{output_csv}'.")

def create_knowledge_graph(csv_file):
    """
    Creates a knowledge graph from the given CSV file.
    Returns the graph and a dictionary of symptoms with their associated diseases.
    """
    # Create a new graph
    G = nx.Graph()

    # Create a dictionary to store unique symptoms
    symptoms_dict = defaultdict(set)

    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            disease = row['Disease']
            symptom_list = [s.strip() for s in row['Combined_Symptoms'].split(',')]

            # Add disease node
            G.add_node(disease, node_type='disease')

            # Add symptom nodes and edges
            for symptom in symptom_list:
                if symptom:  # Ensure symptom is not empty
                    G.add_node(symptom, node_type='symptom')
                    G.add_edge(disease, symptom)
                    symptoms_dict[symptom].add(disease)

    return G, symptoms_dict

def save_new_knowledge_graph_relations(G, output_csv):
    """
    Saves the new relations from the knowledge graph to a CSV file.
    This CSV can be used for further steps in chatbot retrieval.
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as file_out:
        csv_writer = csv.writer(file_out)
        csv_writer.writerow(['Disease', 'Symptom', 'Relation'])

        # Iterate over edges to save the relations
        for disease, symptom in G.edges():
            if G.nodes[disease]['node_type'] == 'disease' and G.nodes[symptom]['node_type'] == 'symptom':
                csv_writer.writerow([disease, symptom, 'has_symptom'])

    print(f"New knowledge graph relations saved to '{output_csv}'.")

def visualize_graph(G, output_file):
    """
    Visualizes the full knowledge graph and saves it as an image.
    """
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Draw disease nodes
    disease_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'disease']
    nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes, node_color='lightblue', node_size=500, alpha=0.8, label='Diseases')
    
    # Draw symptom nodes
    symptom_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'symptom']
    nx.draw_networkx_nodes(G, pos, nodelist=symptom_nodes, node_color='lightgreen', node_size=300, alpha=0.6, label='Symptoms')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Disease-Symptom Knowledge Graph", fontsize=20)
    plt.axis('off')
    plt.legend(scatterpoints=1)
    plt.tight_layout()
    plt.savefig(output_file, format="png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Full knowledge graph visualization saved as '{output_file}'.")

def visualize_single_disease(G, disease, output_file):
    """
    Visualizes a single disease and its associated symptoms.
    """
    if disease not in G:
        print(f"Disease '{disease}' not found in the graph.")
        return

    subgraph = nx.ego_graph(G, disease, radius=1)
    pos = nx.spring_layout(subgraph, k=0.9, iterations=50, seed=42)
    
    plt.figure(figsize=(15, 15))
    
    # Highlight the disease node
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[disease], node_color='#FF6B6B', node_size=3000, alpha=0.8, label='Disease')
    
    # Highlight symptom nodes
    symptom_nodes = [node for node in subgraph.nodes() if node != disease]
    nx.draw_networkx_nodes(subgraph, pos, nodelist=symptom_nodes, node_color='#4ECDC4', node_size=1500, alpha=0.6, label='Symptoms')
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='#45B7D1', width=1)
    
    # Add labels
    nx.draw_networkx_labels(subgraph, pos, {disease: disease}, font_size=14, font_weight='bold')
    nx.draw_networkx_labels(subgraph, pos, {node: node for node in symptom_nodes}, font_size=10)
    
    # Add edge labels
    edge_labels = {(disease, symptom): "has_symptom" for symptom in symptom_nodes}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"Disease: {disease} and its Symptoms", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.legend(scatterpoints=1)
    plt.tight_layout()
    plt.savefig(output_file, format="png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Single disease graph for '{disease}' saved as '{output_file}'.")

def print_statistics(G):
    """
    Prints statistics about the knowledge graph.
    """
    disease_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'disease']
    symptom_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'symptom']
    
    print(f"\nGraph Statistics:")
    print(f"-----------------")
    print(f"Number of disease nodes: {len(disease_nodes)}")
    print(f"Number of symptom nodes: {len(symptom_nodes)}")
    print(f"Number of edges (connections): {G.number_of_edges()}")
    
    # Top 10 most common symptoms
    symptom_disease_count = [(symptom, len(list(G.neighbors(symptom)))) for symptom in symptom_nodes]
    top_symptoms = sorted(symptom_disease_count, key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 Most Common Symptoms:")
    for idx, (symptom, count) in enumerate(top_symptoms, start=1):
        print(f"{idx}. {symptom}: {count} diseases")

def main():
    # File paths
    input_csv = 'merged_disease_symptoms.csv'
    cleaned_csv = 'cleaned_disease_symptoms.csv'
    output_csv_knowledge_graph = 'new_disease_symptom_relations.csv'
    full_graph_output = 'full_disease_symptom_knowledge_graph.png'
    single_disease_output = 'single_disease_graph.png'

    # Check if input CSV exists
    if not os.path.exists(input_csv):
        print(f"Input CSV file '{input_csv}' not found. Please ensure the file exists in the current directory.")
        return

    # Step 1: Clean the CSV
    clean_csv(input_csv, cleaned_csv)

    # Step 2: Create Knowledge Graph
    G, symptoms = create_knowledge_graph(cleaned_csv)

    # Step 3: Save New Knowledge Graph Relations for Chatbot
    save_new_knowledge_graph_relations(G, output_csv_knowledge_graph)

    # Step 4: Visualize Full Knowledge Graph
    visualize_graph(G, full_graph_output)

    # Step 5: Visualize a Single Disease
    disease_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'disease']
    if disease_nodes:
        random_disease = random.choice(disease_nodes)
        visualize_single_disease(G, random_disease, single_disease_output)
    else:
        print("No disease nodes found in the graph to visualize.")

    # Step 5: Print Statistics
    print_statistics(G)

if __name__ == "__main__":
    main()
