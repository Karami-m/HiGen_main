import numpy as np
import networkx as nx
import torch
import torch_geometric as pyg
from utils.graph_helper import *
from utils.graph_helper import bipartite_subgraph
from utils.dataset.graph_corsening import get_coarsened_graphs
from utils.dataset.graph_reorder import sort_graph_nodes


def graph_to_hierarchical_graph(
        G_nx,
        num_levels: int,
        order: str,
        **kwargs
) -> HG:
    """
    Given a graph, it returns a Hierarchical Graph (HG) using a coarsening algorithm.

    Args:
        G_nx: The input graph as a NetworkX graph object.
        num_levels (int): The number of levels in the hierarchical graph.
        order (str): The order type for sorting the graph nodes.
        **kwargs: Additional keyword arguments.

    Returns:
        HG: The resulting Hierarchical Graph.

    Notes:
        - If num_levels is 1, the function returns a single-level HG with no coarsening applied.
        - If num_levels is greater than 1, the function applies a coarsening algorithm to generate the HG.
    """
    i_g = kwargs.get('g_id')

    # HG is single level, i.e. no coarsening is applied
    if num_levels == 1:
        G_nx = G_nx.to_directed() if not nx.is_directed(G_nx) else G_nx
        edges = list(G_nx.edges(data='weight'))
        edge_index_weight = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()
        edge_index, edge_weight = edge_index_weight[0:2, :], edge_index_weight[2, :]
        num_nodes = G_nx.number_of_nodes()

        # HG consists of only one level and one partition
        G0_partition = Data2(
            is_part=True,
            ord=order,
            level=0,
            part_id=0,
            id=i_g,
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_feat=torch.zeros([num_nodes, 0]),
            child_part=[None] * num_nodes,  # no child_part at the leaf level
            child_bipart=[None] * edge_index.shape[1],  # no child_bipart at the leaf level
        )
        G0 = collate_partitions(
            level=0,
            part_ls=[G0_partition],
            bipart_ls=[],
            g_id=i_g
        )

        out = HG(
            g_levels_list=[G0],
            max_level=1,
            id=i_g,
            ord=order
        )
        return out

    else:
        abs_graphs_ls = get_coarsened_graphs(G_nx, num_levels=num_levels, **kwargs)
        num_nodes_ls = [
            G_.number_of_nodes()
            for G_ in abs_graphs_ls
        ]
        print(f'(Number of Nodes, Number of levels) :  ({num_nodes_ls[::-1]}, {len(abs_graphs_ls)})')

        if kwargs.get('return_abs', False):
            # only return abstract graph in this case
            return abs_graphs_ls[-1]

        abs_graphs_ls.reverse()
        HG0_ls = [
            nx2pyg_Data2(abs_graphs_ls[l])
            for l in range(0, len(abs_graphs_ls))
        ]
        if len(abs_graphs_ls)>2:
            raise NotImplementedError('depth of the HG > 2')

        # G_root is a singleton graph that only contain some graph stats,
        # its level=-1 so that not mistaken with coarsened graph at level=0
        G_root = Data2(
            num_nodes=1,
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_weight=[None],  # not used
            child_nodes=[list(range(HG0_ls[0].num_nodes))],
            child_part=[None],  # part of lower level that is child of nodes in this graph
            child_bipart=[None],
            id=i_g,
        )

        # resort graph nodes based on order type
        HG0_ls[0], G_root = sort_graph_nodes(
            G=HG0_ls[0],
            G_parent=G_root,
            order=order
        )
        for l in range(1, len(HG0_ls)):
            HG0_ls[l], HG0_ls[l - 1] = sort_graph_nodes(
                G=HG0_ls[l],
                G_parent=HG0_ls[l - 1],
                order=order
            )

        HG_ls = []
        for i in reversed(range(0, len(HG0_ls))):
            # if the number of levels are not similar in the dataset,
            # we can add None at the top or bottom of the tree
            level = i if kwargs['level_startFromRoot'] else i + num_levels - len(HG0_ls)
            pyg_G = nx2pyg_level(
                G=HG0_ls[i],
                G_parent=HG0_ls[i - 1] if i > 0 else G_root,
                order=order,
                level=level,
                g_id=i_g
            )
            HG_ls.append(pyg_G)

        HG_ls.reverse()
        return HG(HG_ls, max_level=num_levels, id=i_g, ord=order)


def nx2pyg_Data2(G) -> Data2:
    """
    Converts a graph in nx format into a customized PyG_Data2
    """
    G = G.to_directed() if not nx.is_directed(G) else G
    edges = list(G.edges(data='weight'))
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index, edge_weight = edges[0:2, :], edges[2, :]
    child_nodes = [
        G.nodes[node]['child_nodes']
        for node in range(len(G))
    ]

    return Data2(
        num_nodes=G.number_of_nodes(),
        edge_index=edge_index,
        edge_weight=edge_weight,
        child_nodes=child_nodes,
        child_part=[None] * G.number_of_nodes(),
        child_bipart=[None] * edge_index.shape[1],
    )


def nx2pyg_level(
        G, G_parent,
        order: str,
        level: int,
        g_id: int
) -> Data2:
    """
    Convert a networkx graph to a PyTorch Geometric (pyg) graph at a specific level of hierarchy.

    Args:
        G (networkx.Graph): The input networkx graph.
        G_parent (networkx.Graph): The parent graph at the previous level of hierarchy.
        order (str): The order of the graph.
        level (int): The level of hierarchy.
        g_id (int): The ID of the graph.
    """
    # First, convert partition subgraphs
    part_ls = []
    for node_ind in range(G_parent.num_nodes):
        child_nodes = G_parent.child_nodes[node_ind]
        edge_index, part_edges = pyg.utils.subgraph(
            subset=child_nodes,
            edge_index=G.edge_index,
            edge_attr=np.arange(G.num_edges),
            relabel_nodes=True
        )

        # Make subgraphs in pyg
        part_pyg = Data2(
            ord=order,
            level=level,
            id=g_id,
            num_nodes=len(child_nodes),
            edge_index=edge_index,
            edge_weight=G.edge_weight[part_edges],
            is_part=True,
            part_id=node_ind,
            edge_feat=torch.zeros([edge_index.shape[1], 0]),
            child_part=[G.child_part[ind_] for ind_ in child_nodes],
            child_bipart=[G.child_bipart[ind_] for ind_ in part_edges],
        )
        # Add subgraphs as an attr on parent node
        part_ls.append(part_pyg)
        G_parent.child_part[node_ind] = part_pyg

    # Second, make bipartite subgraphs in pyg
    for edge_ind in range(G_parent.num_edges):
        u, v = (G_parent.edge_index[0, edge_ind].item(), G_parent.edge_index[1, edge_ind].item())
        if (u == v):
            continue # self loops does not have a bipart

        else:
            child_nodes_left = G_parent.child_nodes[u]
            child_nodes_right = G_parent.child_nodes[v]
            edge_index, bipart_edges = bipartite_subgraph(
                subset=(child_nodes_left, child_nodes_right),
                edge_index=G.edge_index,
                edge_attr=np.arange(G.num_edges),
                relabel_nodes=True
            )

            bipart_ = Bipart(
                ord=order,
                level=level,
                id=g_id,
                num_nodes=len(child_nodes),
                edge_index=edge_index,
                edge_weight=G.edge_weight[bipart_edges],
                edge_feat=torch.zeros([edge_index.shape[1], 0]),
                child_bipart=[G.child_bipart[ind_] for ind_ in bipart_edges],
                part_left=G_parent.child_part[u],
                part_right=G_parent.child_part[v],
            )

        G_parent.child_bipart[edge_ind] = bipart_

    # Collate partition and bipartite subgraphs into a single graph at the current level
    G_level = collate_partitions(
        level=level,
        part_ls=part_ls,
        bipart_ls=G_parent.child_bipart,
        g_id=g_id
    )
    return G_level
