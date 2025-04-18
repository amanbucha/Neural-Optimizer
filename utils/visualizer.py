import networkx as nx
import matplotlib.pyplot as plt

def visualize_fx(graph_module, title="Graph"):
    G = nx.DiGraph()
    for node in graph_module.graph.nodes:
        G.add_node(str(node), label=str(node.target))
        for arg in node.all_input_nodes:
            G.add_edge(str(arg), str(node))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, font_size=10)
    plt.title(title)
    plt.savefig(f"./images/{title}")