import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_kg_viz():
    # Create a directed graph
    G = nx.DiGraph()

    # Define some entity types and relations for a "rank-rank" knowledge graph
    entities = [
        ("rank-retrieve", "Crate"),
        ("rank-fusion", "Crate"),
        ("rank-rerank", "Crate"),
        ("rank-eval", "Crate"),
        ("rank-train", "Crate"),
        ("anno", "Project"),
        ("lattice", "Project"),
        ("hop", "Project"),
        ("BM25", "Algorithm"),
        ("HNSW", "Algorithm"),
        ("RRF", "Algorithm"),
        ("MaxSim", "Algorithm"),
        ("NDCG", "Metric"),
        ("LambdaRank", "Algorithm"),
    ]

    relations = [
        ("rank-rank", "rank-retrieve", "includes"),
        ("rank-rank", "rank-fusion", "includes"),
        ("rank-rank", "rank-rerank", "includes"),
        ("rank-rank", "rank-eval", "includes"),
        ("rank-rank", "rank-train", "includes"),
        ("rank-retrieve", "BM25", "implements"),
        ("rank-retrieve", "HNSW", "implements"),
        ("rank-fusion", "RRF", "implements"),
        ("rank-rerank", "MaxSim", "implements"),
        ("rank-eval", "NDCG", "implements"),
        ("rank-train", "LambdaRank", "implements"),
        ("lattice", "anno", "uses"),
        ("hop", "rank-fusion", "uses"),
        ("hop", "rank-rerank", "uses"),
        ("anno", "rank-retrieve", "uses"),
    ]

    # Add nodes with attributes
    for entity, type_ in entities:
        G.add_node(entity, type=type_)
    G.add_node("rank-rank", type="Ecosystem")

    # Add edges with labels
    for src, dst, label in relations:
        G.add_edge(src, dst, label=label)

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Draw
    plt.figure(figsize=(12, 8))
    
    # Color mapping
    colors = {
        "Ecosystem": "#ff7f0e",
        "Crate": "#1f77b4",
        "Project": "#2ca02c",
        "Algorithm": "#d62728",
        "Metric": "#9467bd"
    }
    node_colors = [colors.get(G.nodes[n].get("type", "Project"), "#333333") for n in G.nodes()]

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.9)
    
    # Edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="gray", arrowsize=20)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif", font_color="white")
    
    # Edge Labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("rank-rank Ecosystem Knowledge Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Default to a local relative path to keep this portable (and public-safe).
    output_path = "kg_structure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_kg_viz()
