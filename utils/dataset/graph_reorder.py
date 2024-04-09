
import numpy as np
import networkx as nx
import torch_geometric as pyg
from networkx import Graph
from easydict import EasyDict as edict
from utils.graph_helper import *


def sort_graph_nodes(G, G_parent, order='BFS'):
    """
    Reorder the node ordering of the partitions in graph G based on the order type.
    It returns the modified graph G and its parent graph G_parent.

    Parameters:
    - G (Graph): The graph to be reordered.
    - G_parent (Graph): The parent graph of G.
    - order (str): The order type. Can be one of 'BFS', 'DFS', 'BFSAC', 'BFSDC'.

    Returns:
    - G (Graph): The modified graph G with reordered nodes.
    - G_parent (Graph): The modified parent graph G_parent with reordered child nodes.
    """
    if G_parent is None:
        return G

    # First fix node ordering so that they partitioned are sorted
    G, G_parent = fix_node_orders_based_on_partitions(G, G_parent)

    if order in [0, '0', 'rand']:
        return G, G_parent

    # Create a list of networkx subgraphs
    part_ls = []
    for node_ind in range(G_parent.num_nodes):
        child_nodes = G_parent.child_nodes[node_ind]
        
        # Create a subgraph of G using the child nodes
        edge_index_, part_edges = pyg.utils.subgraph(
            subset=child_nodes,
            edge_index=G.edge_index,
            edge_attr=np.arange(G.num_edges),
            relabel_nodes=True
        )
        G_part_ = Data2(
            edge_index=edge_index_,
            edge_weight=G.edge_weight[part_edges],
            num_nodes=len(child_nodes)
        )
        
        # Convert the Data2 object to a networkx graph
        G_part_nx = nx.from_numpy_matrix(G_part_.adj(is_weighted=True).cpu().numpy())
        part_ls.append(G_part_nx)

    if order == 'BFS':
        # Perform BFS traversal on each subgraph and collect the node ordering
        node_list_bfs = []
        i = 0
        for ii in range(len(part_ls)):
            # Get the degree of each node in the subgraph and sort the nodes based on their degree in descending order
            node_degree_list = [(n, d) for n, d in part_ls[ii].degree()]
            degree_sequence = sorted(
                node_degree_list, key=lambda tt: tt[1], reverse=True)
            # Perform BFS traversal starting from the node with the highest degree
            bfs_tree = nx.bfs_tree(part_ls[ii], source=degree_sequence[0][0])
            # Collect the nodes visited during the BFS traversal of each subgraph
            node_list_bfs += list(np.array(bfs_tree.nodes()) + i)
            i += len(part_ls[ii])
        perm = node_list_bfs

    elif order == 'DFS':
        # Perform DFS traversal on each subgraph and collect the node ordering
        node_list_dfs = []
        i = 0
        for ii in range(len(part_ls)):
            # Get the degree of each node in the subgraph and sort the nodes based on their degree in descending order
            node_degree_list = [(n, d) for n, d in part_ls[ii].degree()]
            degree_sequence = sorted(
                node_degree_list, key=lambda tt: tt[1], reverse=True)
            # Perform DFS traversal starting from the node with the highest degree
            dfs_tree = nx.dfs_tree(part_ls[ii], source=degree_sequence[0][0])
            # Collect the nodes visited during the DFS traversal of each subgraph
            node_list_dfs += list(np.array(dfs_tree.nodes()) + i)
            i += len(part_ls[ii])
        perm = node_list_dfs

    elif order in ['BFSAC', 'BFSDC']:
        # Perform modified BFS traversal on each subgraph
        perm = bfs_modified(part_ls, sort_order=order )

    else:
        raise ValueError("Not a valid node order")

    perm = dict([(jj, ii) for ii, jj in enumerate(perm)])
    G.perm = perm
    G_parent.child_perm = perm

    i = 0
    for i_p, part in enumerate(G_parent.child_nodes):
        G_parent.child_nodes[i_p] = list(range(i, i + len(part)))
        i += len(part)

    # Fix the child node order of G
    child_nodes = [None] * G.num_nodes
    for i in range(G.num_nodes):
        child_nodes[perm[i]] = G.child_nodes[i]
    G.child_nodes = child_nodes

    # Fix the edge indices of G
    for jj in range(G.num_edges):
        G.edge_index[0, jj] = perm[G.edge_index[0, jj].item()]
        G.edge_index[1, jj] = perm[G.edge_index[1, jj].item()]

    return G, G_parent


def fix_node_orders_based_on_partitions(G, G_parent):
    """
    Performs node reordering on a graph represented by G
    based on the partitioning information provided by G_parent.
    For example, if partitions are [[1,3,5][2,4]], the nodes' ids
    are changed so that partitions become [[1,2,3][4,5]].
    These changes affect both G and G_parent.

    Parameters:
    - G (Graph): The graph to be reordered.
    - G_parent (Graph): The parent graph of G.

    Returns:
    - G (Graph): The modified graph G with reordered nodes.
    - G_parent (Graph): The modified parent graph G_parent with reordered child nodes.
    """
    permutation = []
    i = 0
    for i_p, part in enumerate(G_parent.child_nodes):
        child_nodes_ls = []
        for child_node_id in part:
            permutation.append(child_node_id)
            child_nodes_ls.append(i)
            i += 1
        # fix the node id of part i_p in G_parent
        G_parent.child_nodes[i_p] = child_nodes_ls
    permutation = dict([(jj, ii) for ii, jj in enumerate(permutation)])
    G.perm = permutation
    G_parent.child_perm = permutation

    # fix the child node order of G
    child_nodes = [None] * G.num_nodes
    for i in range(G.num_nodes):
        child_nodes[permutation[i]] = G.child_nodes[i]
    G.child_nodes = child_nodes

    # fix the edge indices of G
    for jj in range(G.num_edges):
        G.edge_index[0, jj] = permutation[G.edge_index[0, jj].item()]
        G.edge_index[1, jj] = permutation[G.edge_index[1, jj].item()]
    return G, G_parent

def bfs_modified(part_ls, sort_order='BFSAC'):
    """
    Traverse the nodes in BFS order, when adding a neighbor to the tree,
    In the queue, the one that has minimum cutset weight between the node and the nodes of the tree is selected first
    w: is the cutset weight
    """
    node_list_bfs = []
    i = 0
    # Iterate over each subgraph in part_ls
    for G_ in part_ls:
        # Create a list of tuples containing node id and node attributes
        node_selfedge_list = [
            (node, edict(
                id=node,
                w=G_.get_edge_data(node, node, default={"weight": 0})["weight"],
                w_self=G_.get_edge_data(node, node, default={"weight": 0})["weight"],
                deg0=G_.degree(node),
                deg=G_.degree(node, weight='weight'),
                visited=False,
                pred=None
                )
            ) for node in list(G_.nodes())
        ]
        node_dict = dict(node_selfedge_list)
        # Perform modified BFS traversal on the subgraph
        mbfs_list = bfs_modified_visit(G_, node_dict=node_dict, sort_order=sort_order)
        node_list_bfs += list(np.array(mbfs_list) + i)
        i += len(G_)

    return node_list_bfs


def bfs_modified_visit(G: Graph, node_dict, sort_order='BFSAC'):
    """
    Perform modified BFS traversal on a graph G based on the sort order.

    Parameters:
    - G (Graph): The graph to be traversed.
    - node_dict (dict): A dictionary containing node information.
    - sort_order (str): The sort order. Can be one of 'BFSAC', 'BFSDC'.

    Returns:
    - tree_list (list): The node ordering obtained from the traversal.
    """
    if sort_order == 'BFSAC':
        # BFS with Ascending cutset weight
        sorting_key_fn = lambda item: (item.w, item.w_self, -item.deg0) if not item.visited \
            else (np.inf, np.inf, np.inf)
    elif sort_order == 'BFSDC':
        # BFS with Descending cutset weight
        sorting_key_fn = lambda item: (-item.w, -item.w_self, item.deg0) if not item.visited \
            else (np.inf, np.inf, np.inf)

    tree_list = []
    while np.array([not node.visited for node in node_dict.values()]).any():
        source = min(node_dict.values(), key=sorting_key_fn)
        Q = [source]
        while np.array([not node.visited for node in Q]).any():
            u = min(Q, key=sorting_key_fn) # sort the queue based on the sort criteria
            tree_list.append(u.id)
            node_dict[u.id].visited = True
            for v_id in G.neighbors(u.id):
                if not node_dict[v_id].visited:
                    node_dict[v_id].w += G[u.id][v_id]["weight"]
                    Q.append(node_dict[v_id])

    return tree_list


