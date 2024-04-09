from typing import List
import networkx as nx
from networkx import Graph
import community
import numpy as np
from sklearn.cluster import SpectralClustering


# Function to get coarsened graphs
def get_coarsened_graphs(G: Graph, num_levels: int, **kwargs) -> List[Graph]:
    """
    Coarsens the input graph G0 to create a hierarchy of coarsened graphs.

    Parameters:
    - G: The input graph to be coarsened.
    - num_levels: The number of coarsened graphs to generate.
    - **kwargs: Additional keyword arguments.

    Returns:
    - abs_graph_ls: A list of coarsened graphs.

    """
    G2 = nx.from_numpy_matrix(nx.to_numpy_matrix(G, weight='weight'))

    if kwargs['coarsening_alg'].lower()[0:2] == 'sc':
        # Apply spectral clustering based coarsening
        abs_graph_ls = apply_coarsening_spectral(G2, num_levels=num_levels, **kwargs)
    else:
        raise ValueError("not implemented")

    return abs_graph_ls


def get_clusters(partition):
    clusters = {}
    for i, j in partition.items():
        clusters.setdefault(j, []).append(i)
    return clusters


def get_induced_graph(clusters, G: Graph) -> Graph:
    """
    Get the induced graph based on the given clusters and original graph.

    Parameters:
    clusters (list): List of clusters, where each cluster is represented as a list of node indices.
    G (networkx.Graph): Original graph.

    Returns:
    networkx.Graph: Induced graph based on the given clusters.
    """
    cluster_asign = np.zeros([len(clusters), len(G)])
    for j in range(len(clusters)):
        cluster_asign[j, clusters[j]] = 1

    adj = nx.to_numpy_matrix(G)
    adj_parent = np.matmul(np.matmul(cluster_asign, adj), np.transpose(cluster_asign))
    adj_parent -= np.diag(np.diag(adj_parent) / 2)
    return nx.from_numpy_matrix(adj_parent)


def best_spectral_partition(G: Graph, k_min, k_max):
    """
    Finds the best spectral partition of a graph searching within a given range of cluster sizes.

    Parameters:
        G (Graph): The input graph.
        k_min (int): The minimum number of clusters to consider.
        k_max (int): The maximum number of clusters to consider.

    Returns:
        dict: A dictionary representing the best partition found, where the keys are node indices and the values are cluster labels.

    """
    modularity_best = 0
    partition_best = None
    modularity_ls = []
    for K in range(k_min, k_max + 1):
        # Perform spectral clustering algorithm
        adj_mat = nx.to_numpy_matrix(G)
        sc = SpectralClustering(K, affinity='precomputed', n_init=1000)
        sc.fit(adj_mat)
        partition = dict([(i, label) for i, label in enumerate(sc.labels_)])
        modularity_ = community.modularity(partition, G)
        modularity_ls.append(modularity_)
        if modularity_ > modularity_best:
            modularity_best = modularity_
            K_best = K
            partition_best = partition

    print(f'modularity score in k_min={k_min} and k_max={k_max}:  {modularity_ls}')
    return partition_best


def apply_coarsening_spectral(G: Graph, num_levels, **kwargs):
    """
    Apply spectral coarsening to a graph.

    Args:
        G (networkx.Graph): The input graph.
        num_levels (int): The number of coarsening levels to apply.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of coarsened graphs.
    """
    coarsened_graphs = []

    if 'sc0' in kwargs['coarsening_alg'].lower():
        n_ = len(G)
        lo_lim, up_lim = (.7, 1.3)
        if 'planar' in kwargs.get('graph_name', 'none'):
            lo_lim, up_lim = (.5, 1.3)
        k_min = max(2, int(np.sqrt(n_) * lo_lim))
        k_max = min(n_, int(np.sqrt(n_) * up_lim))
        if kwargs['coarsening_alg'].lower() == 'sc02':
            k_min = k_max = 2
        # Get the best spectral partition
        partition = best_spectral_partition(G, k_min=k_min, k_max=k_max)

        nx.set_node_attributes(G, partition, "parent_cluster")
        clusters = dict([(i, None) for i in range(len(G.nodes))])
        nx.set_node_attributes(G, clusters, "child_nodes")
        coarsened_graphs.append(G)

        clusters = get_clusters(partition)
        # Get the induced graph based on the clusters
        #     G_parent = community.induced_graph(partition, graph=G)
        G_parent = get_induced_graph(clusters, G)
        partition_parent = dict([(i, None) for i in range(len(G_parent.nodes))])
        # Set the child_nodes attribute for each node in G_parent
        nx.set_node_attributes(G_parent, clusters, "child_nodes")
        # Set the parent_cluster attribute for each node in G_parent
        nx.set_node_attributes(G_parent, partition_parent, "parent_cluster")
        coarsened_graphs.append(G_parent)

    return coarsened_graphs
