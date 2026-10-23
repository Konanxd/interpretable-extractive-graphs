import networkx as nx
import numpy as np


class GraphBuilder:
    def __init__(self, directed: bool = False):
        self.graph = nx.DiGraph() if directed else nx.Graph()

    def compute_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed_emb = embeddings / norms

        sim_matrix = np.dot(normed_emb, normed_emb.T)
        np.fill_diagonal(sim_matrix, 0.0)

        return sim_matrix

    def add_edge_similarity(self, sim_matrix: np.ndarray, threshold: float):
        n = sim_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if sim_matrix[i, j] >= threshold:
                    self.graph.add_edge(
                        i, j, weight=float(sim_matrix[i, j]), type="similarity"
                    )

        print(self.graph.edges(data=True))

    def add_dependency_edges(self, edges):
        for head, dep in edges:
            self.graph.add_edge(head, dep, weight=1.0, type="dependency")

    def page_rank(self, alpha: float = 0.85):
        pagerank_scores = nx.pagerank(self.graph, alpha=alpha)
        sorted_scores = sorted(
            pagerank_scores.items(), key=lambda item: item[1], reverse=True
        )

        print("Node Scores:\n")
        for node, score in sorted_scores:
            print(f"Node {node}: {score:.4f}")
