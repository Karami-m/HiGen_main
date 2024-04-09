###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os
import pickle
from typing import List
import networkx as nx
import torch
import random
from utils.graph_helper import *
from utils.dataset.simple_example_graph import load_simple_example_graph
from utils.dataset.graph_load import save_graph_list, graph_load, graph_loader_batch
from utils.dataset.make_hierarchical_graph import graph_to_hierarchical_graph
from utils.dataset.graph_reorder import bfs_modified

__all__ = ['save_graph_list', 'graph_loader_batch', 'get_multilevel_graph_dataset']

all_orders = ['def', 'DD', 'DA', 'BFS', 'DFS', 'BFSDC', 'BFSAC']

graph_type_tags = {
    'enzyme': 'enz',
    'enzyme_1level': 'enz',
    'SBM': 'sbm',
    'SBM_1level': 'sbm',
    'DD': 'dd',
    'DD2': 'dd',
    'Ego': 'eg',
    'Ego2': 'eg',
    'FIRSTMM': 'db',
    'FIRSTMM2': 'db',
}


def get_multilevel_graph_dataset(
        graph_type,
        node_orders,
        max_num_level: int,
        data_dir: str = 'data',
        is_overwrite_precompute: bool = True,
        **kwargs
) -> List[List[HG]]:
    """
    Get a multilevel graph dataset based on the specified graph type and parameters.

    Args:
        graph_type (str): The type of graph dataset to generate. Possible values are:
            - 'simple': Simple demo graphs
            - 'enzyme': Enzyme graphs with multiple levels
            - 'enzyme_1level': Enzyme graphs with a single level
            - 'sbm': SBM (Stochastic Block Model) graphs with multiple levels
            - 'DD2':  Protein graphs with 2 levels
            - 'DD':   Protein graphs with 3 levels
            - 'DD_1level': Protein graphs with a single level
            - 'FIRSTMM2': FIRSTMM graphs with 2 levels
            - 'FIRSTMM': FIRSTMM graphs with 3 levels
            - 'FIRSTMM_1level': FIRSTMM graphs with a single level
            - 'Ego2': Ego graphs with 2 levels
            - 'Ego': Ego graphs with 3 levels
            - 'Ego_1level': Ego graphs with a single level
        node_orders: The list of orders of the nodes to sort the graph and its
        subgraphs after partitioning. We normally used only one ordering  over all graphs
        rather than multiple ordering in our experimentation.
        max_num_level (int): The maximum number of levels in the hierarchical graph.
        data_dir (str): The directory where the graph data is stored.
        is_overwrite_precompute (bool): Whether to overwrite precomputed data.
        **kwargs: Additional keyword arguments.

    Returns:
        List[List[HG]]: A list of (list of (hierarchical graphs) for all orders).
    """
    kwargs.update({'graph_name': graph_type})

    if not isinstance(node_orders, list):
        node_orders = [node_orders]

    ## load datasets
    graphs_dataset = []
    # simple demo graphs
    if graph_type == 'simple':
        graphs_dataset = load_simple_example_graph()

    elif graph_type in ['enzyme', 'enzyme_1level']:
        input_path = 'data/ENZYMES.pkl'
        with open(input_path, 'rb') as f:
            graphs = pickle.load(f)

        if graph_type == 'enzyme_1level':
            graphs_dataset = []
            for i, G_nx in enumerate(graphs):
                kwargs.update({'g_id': f'enz{i}'})
                HG_ls = get_HG_of_orders(
                    G_nx,
                    node_orders=node_orders,
                    num_levels=1, **kwargs
                )
                graphs_dataset.append(HG_ls)

        # Enzyme 2 level
        else:
            kwargs.update(
                {
                    'splice_out_inds':
                        {2: [],
                         3: [1],
                         4: [1, 2],
                         5: [1, 2, 3]
                         }
                }
            )
            graphs_dataset = get_hierarchical_graph(
                graphs,
                graph_type=graph_type,
                max_num_level=max_num_level,
                orders=node_orders,
                **kwargs
            )

    # SBM 2 level
    elif graph_type.lower() in ['sbm']:
        if is_overwrite_precompute:
            input_path = 'data/sbm_200.pt'
            if os.path.isfile(input_path):
                adjs_ls, eigvals_, eigvecs_, n_nodes_, max_eigval_, min_eigval_, same_sample_, n_max_ = torch.load(
                    input_path)
            else:
                raise ValueError(f'file for {graph_type} is not available')
            graphs = []

            for adj_ in adjs_ls:
                G = nx.from_numpy_array(adj_.numpy())
                graphs.append(G)
        else:
            graphs = [None] * 200

        kwargs.update(
            {
                'splice_out_inds':
                    {2: [],
                     3: [1],
                     4: [1, 2],
                     5: [1, 2, 3]
                     }
            }
        )
        graphs_dataset = get_hierarchical_graph(
            graphs,
            graph_type=graph_type,
            max_num_level=max_num_level,
            orders=node_orders,
            **kwargs
        )

    # protein dataset
    elif graph_type in ['DD2', 'DD', 'DD_1level']:
        if is_overwrite_precompute:
            graphs = graph_loader_batch(
                data_dir,
                min_num_nodes=100,
                max_num_nodes=500,
                name='DD',
                node_attributes=False,
                graph_labels=True
            )
        else:
            graphs = [None] * 918

        if graph_type == 'DD_1level':
            graphs_dataset = []
            for ii, G_i in enumerate(graphs):
                kwargs.update({'g_id': f'dd{ii}'})
                hyper_graphs_ls = get_HG_of_orders(
                    G_i,
                    node_orders=node_orders,
                    num_levels=1,
                    **kwargs
                )
                graphs_dataset.append(hyper_graphs_ls)

        elif graph_type == 'DD':
            # Protein 3 level
            kwargs.update(
                {
                    'splice_out_inds':
                        {2: [],
                         3: [],
                         4: [1],
                         5: [1, 2]
                         }
                }
            )
            graphs_dataset = get_hierarchical_graph(
                graphs,
                graph_type=graph_type,
                max_num_level=max_num_level,
                orders=node_orders,
                **kwargs
            )


        elif graph_type == 'DD2':
            # Protein 2 level
            kwargs.update(
                {
                    'splice_out_inds':
                        {2: [],
                         3: [1],
                         4: [1, 2],
                         5: [1, 2, 3]
                         }
                }
            )
            graphs_dataset = get_hierarchical_graph(
                graphs,
                graph_type=graph_type,
                max_num_level=max_num_level,
                orders=node_orders,
                **kwargs
            )

    elif graph_type in ['FIRSTMM2', 'FIRSTMM', 'FIRSTMM_1level']:
        graphs = graph_loader_batch(
            data_dir,
            min_num_nodes=0,
            max_num_nodes=10001,
            name='FIRSTMM_DB',
            node_attributes=False,
            graph_labels=True
        )

        # FIRSTMM 3 level
        if graph_type == 'FIRSTMM':
            kwargs.update(
                {
                    'splice_out_inds':
                        {3: [],
                         4: [1],
                         5: [1, 3],
                         6: [1, 2, 4],
                         7: [1, 2, 4, 5]}
                }
            )
            graphs_dataset = get_hierarchical_graph(
                graphs,
                graph_type=graph_type,
                max_num_level=max_num_level,
                orders=node_orders,
                **kwargs
            )

        # FIRSTMM 2 level
        elif graph_type == 'FIRSTMM2':
            kwargs.update(
                {
                    'splice_out_inds':
                        {2: [],
                         3: [1],
                         4: [0, 2],
                         5: [0, 2, 3],
                         6: [0, 2, 3, 4],
                         7: [0, 2, 3, 4, 5]}
                }
            )
            graphs_dataset = get_hierarchical_graph(
                graphs,
                graph_type=graph_type,
                max_num_level=max_num_level,
                orders=node_orders,
                **kwargs
            )

        elif graph_type == 'FIRSTMM_1level':
            graphs_dataset = []
            for ii, G_nx in enumerate(graphs):
                kwargs.update({'g_id': f'db{ii}'})
                HG_ls = get_HG_of_orders(
                    G_nx,
                    num_levels=1,
                    node_orders=node_orders,
                    **kwargs
                )
                graphs_dataset.append(HG_ls)

    elif graph_type in ['Ego2', 'Ego', 'Ego_1level']:
        _, _, G = graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_i = nx.ego_graph(G, i, radius=3)
            if G_i.number_of_nodes() >= 50 and (G_i.number_of_nodes() <= 400):
                graphs.append(G_i)

        # Ego 3 level
        if graph_type == 'Ego':
            kwargs.update(
                {
                    'splice_out_inds':
                        {2: [],
                         3: [],
                         4: [1],
                         5: [1, 2]}
                }
            )
            graphs_dataset = get_hierarchical_graph(
                graphs,
                graph_type=graph_type,
                max_num_level=max_num_level,
                orders=node_orders,
                **kwargs
            )

        # Ego 2 level
        elif graph_type == 'Ego2':
            kwargs.update(
                {
                    'splice_out_inds':
                        {2: [],
                         3: [1],
                         4: [0, 2],
                         5: [0, 2, 3]}
                }
            )
            graphs_dataset = get_hierarchical_graph(
                graphs,
                graph_type=graph_type,
                max_num_level=max_num_level,
                orders=node_orders,
                **kwargs
            )

        if graph_type == 'Ego_1level':
            graphs_dataset = []
            for ii, G_i in enumerate(graphs):
                kwargs.update({'g_id': f'ego{ii}'})
                HG_ls = get_HG_of_orders(
                    G_i,
                    node_orders=node_orders,
                    num_levels=1,
                    **kwargs
                )
                graphs_dataset.append(HG_ls)

    return graphs_dataset


def get_hierarchical_graph(
        graphs,
        graph_type,
        max_num_level,
        orders,
        data_path='data/',
        **kwargs
) -> List[List[HG]]:
    """
    Get a list of graphs and create a coarsened graph in HG format. Save the coarsened graphs for future use.

    Args:
        graphs (List): A list of graphs.
        graph_type (str): The type of the graph.
        max_num_level (int): The maximum number of levels for coarsening.
        orders (List): A list of node orders.
        data_path (str, optional): The path to save the coarsened graphs. Defaults to 'data/'.
        **kwargs: Additional keyword arguments.

    Returns:
        List[List[HG]]: A list of hierarchical graphs.

    """
    save_dir = os.path.join(data_path, graph_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    g_type_tag = graph_type_tags[graph_type]
    hg_dataset = []
    for i, G_nx in enumerate(graphs):

        if 'all' in orders:
            orders = all_orders

        hg_all_orders = []
        # Iterate over each node order
        # It assumes that we might have several ordering. In experimentation, we only used one ordering such as BFS.
        for node_order in orders:
            file_name = os.path.join(
                save_dir,
                f"{graph_type}_{i}_{node_order}_multilevel_{kwargs['coarsening_alg']}.p"
            )

            if not os.path.exists(file_name):
                print(f'Making HG: {graph_type}, size: {G_nx.number_of_nodes()} ')
                kwargs.update({'g_id': f'{g_type_tag}{i}'})
                # Get the hierarchical graph for the current node order
                hg = get_HG_of_orders(
                    G_nx,
                    num_levels=max_num_level,
                    node_orders=[node_order],
                    **kwargs
                )[0]
                # Save the hierarchical graph to a file
                with open(file_name, "wb") as f:
                    pickle.dump(hg, f)

            else:
                print(f"Loading HGs dataset from: {file_name} ")
                # Load the hierarchical graph from the file
                with open(file_name, "rb") as f:
                    hg = pickle.load(f)

            hg_all_orders.append(hg)
        hg_dataset.append(hg_all_orders)
    return hg_dataset


def get_HG_of_orders(
        G,
        node_orders: list,
        num_levels: int,
        to_remove_minor_CG: bool = True,
        **kwargs
) -> List[HG]:
    """
    Given the input graph G and a list of order types, it returns a list of coarsened graph in HG format.

    Parameters:
    - G: The input graph.
    - node_orders: A list of order types.
    - num_levels: The number of levels in the hierarchical graph.
    - to_remove_minor_CG: A boolean indicating whether to remove minor connected components of the graph.
    - **kwargs: Additional keyword arguments.

    Returns:
    - HG_ords_list: A list of coarsened graphs in HG format.

    """
    G = nx.from_numpy_matrix(nx.to_numpy_matrix(G))

    # Get the connected components of the graph
    CG_ls = [G.subgraph(c) for c in nx.connected_components(G)]
    CG_ls = [nx.from_numpy_matrix(nx.to_numpy_matrix(CG_)) for CG_ in CG_ls]

    # Sort connected components (sub-graphs) from largest to smallest size
    CG_ls = sorted(CG_ls, key=lambda x: x.number_of_nodes(), reverse=True)

    # Remove minor connected components if specified
    if to_remove_minor_CG and len(CG_ls) > 1:
        print('Deleting all minor components of Graph: ',
              kwargs.get('g_id', f'graph_len={len(G)}'))
        G = CG_ls[0]
        CG_ls = [CG_ls[0]]

    node_degree_list = [(n, d) for n, d in G.degree()]

    # If 'all' is in node_orders, use all available orders
    if 'all' in node_orders:
        node_orders = all_orders

    # Create a list of hierarchical graphs over all orders
    # in the exp, we used only one order
    HG_ords_list = []

    for order_ in node_orders:
        # Create hierarchical graph using default order
        if order_ == 'def':
            G_ord = G
            HG_ord = graph_to_hierarchical_graph(
                G_ord,
                num_levels=num_levels,
                order='0',
                **kwargs
            )

        # Create hierarchical graph using Degree Descent or Degree Ascent order
        elif order_ in ['DD', 'DA']:
            # Degree Descent or Degree Ascent
            degree_seq = sorted(node_degree_list, key=lambda tt: tt[1],
                                reverse=True if order_=='DD' else False)
            G_ord = nx.from_numpy_matrix(
                nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_seq])
            )
            HG_ord = graph_to_hierarchical_graph(
                G_ord,
                num_levels=num_levels,
                order=order_,
                **kwargs
            )

        elif order_ in ['BFS', 'DFS']:
            # BFS & DFS and sotrted by largest-degree node
            node_list_sorted = []
            for ii in range(len(CG_ls)):
                node_degree_list = [(n, d) for n, d in CG_ls[ii].degree()]
                degree_seq = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
                if order_ == 'BFS':
                    sort_tree = nx.bfs_tree(CG_ls[ii], source=degree_seq[0][0])
                elif order_ == 'DFS':
                    sort_tree = nx.dfs_tree(CG_ls[ii], source=degree_seq[0][0])
                node_list_sorted += list(sort_tree.nodes())

            G_ord = nx.from_numpy_matrix(
                nx.to_numpy_matrix(G,nodelist=node_list_sorted)
            )
            HG_ord = graph_to_hierarchical_graph(
                G_ord,
                num_levels=num_levels,
                order=order_,
                **kwargs
            )

        elif order_ in ['BFSAC', 'BFSDC']:
            # Perform modified BFS traversal
            node_list_sorted = bfs_modified([G], sort_order=order_)
            G_ord = nx.from_numpy_matrix(
                nx.to_numpy_matrix(G, nodelist=node_list_sorted)
            )
            HG_ord = graph_to_hierarchical_graph(
                G_ord,
                num_levels=num_levels,
                order=order_,
                **kwargs
            )

        # Create hierarchical graph using random order
        elif order_ == 'rand':
            perm_ = list(range(G.number_of_nodes()))
            random.shuffle(perm_)
            G_ord = nx.from_numpy_matrix(
                nx.to_numpy_matrix(G, nodelist=perm_)
            )
            HG_ord = graph_to_hierarchical_graph(
                G_ord,
                num_levels=num_levels,
                order='rand',
                **kwargs
            )

        else:
            raise ValueError("Not a Valid Order")

        HG_ords_list.append(HG_ord)

    return HG_ords_list
