###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################

import os
import pickle
import numpy as np
from scipy import sparse as sp
import networkx as nx


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def graph_loader_batch(
        data_dir,
        min_num_nodes=20,
        max_num_nodes=1000,
        name='ENZYMES',
        node_attributes=True,
        graph_labels=True
):
    """
    Load multiple graphs from a dataset.

    Args:
        data_dir (str): The directory path where the dataset is located.
        min_num_nodes (int, optional): The minimum number of nodes a graph should have. Defaults to 20.
        max_num_nodes (int, optional): The maximum number of nodes a graph should have. Defaults to 1000.
        name (str, optional): The name of the dataset. Defaults to 'ENZYMES'.
        node_attributes (bool, optional): Whether to load node attributes. Defaults to True.
        graph_labels (bool, optional): Whether to load graph labels. Defaults to True.

    Returns:
        list: A list of graphs.

    """
    print(f'Loading graph dataset:  {str(name)}')
    G = nx.Graph()
    # load data
    path = os.path.join(data_dir, name)
    data_adj = np.loadtxt(
        os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)

    # Load node attributes if specified
    if node_attributes:
        data_node_att = np.loadtxt(
            os.path.join(path, '{}_node_attributes.txt'.format(name)),
            delimiter=','
        )
    data_node_label = np.loadtxt(
        os.path.join(path, '{}_node_labels.txt'.format(name)),
        delimiter=','
    ).astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, '{}_graph_indicator.txt'.format(name)),
        delimiter=','
    ).astype(int)

    # Load graph labels if specified
    if graph_labels:
        data_graph_labels = np.loadtxt(
            os.path.join(path, '{}_graph_labels.txt'.format(name)),
            delimiter=','
        ).astype(int)

    data_tuple = list(map(tuple, data_adj))

    # Add edges to the graph
    G.add_edges_from(data_tuple)

    # Add node attributes and labels to the graph
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=data_node_label[i])

    # Remove isolated nodes from the graph
    G.remove_nodes_from(list(nx.isolates(G)))

    # Remove self-loops from the graph
    G.remove_edges_from(nx.selfloop_edges(G))

    # Split the graph into subgraphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # Find the nodes for each subgraph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        if min_num_nodes <= G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()

    print('Graphs are Loaded')
    return graphs


# load cora, citeseer and pubmed dataset
def graph_load(dataset='citeseer'):
    """
    Load a single graph dataset.

    :param dataset: The name of the dataset.
    :return: The adjacency matrix, features matrix, and graph object.
    """
    print('Loading graph dataset: ' + str(dataset))
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pickle.load(
            open("data/{}/ind.{}.{}".format(dataset, dataset, names[i]), 'rb'), encoding='latin1'
        )
        objects.append(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset: there are some isolated nodes in the graph.
        # Find isolated nodes and add them as zero-vectors into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
