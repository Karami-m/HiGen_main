from typing import List
import torch
from utils.graph_helper import *
from utils.graph_helper import HG


def load_simple_example_graph() -> List[List[HG]]:
    """
    Returns a simple example hierarchical graph (lists of HG objects representing the graph dataset)
    """
    # make partition subgraphs
    ord_ = '0'
    G1_part0 = Data2(
        ord=ord_,
        level=1,
        num_nodes=4,
        is_part=True,
        part_id=0,
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 0, 2, 1, 3, 2, 3, 3, 0],
             [1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 0, 3]],
            dtype=torch.long
        ),
        edge_weight=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        edge_feat=torch.zeros([12, 0]),
        child_part=[None] * 4,  # childs of nodes are part of lower level
        child_bipart=[None] * 12,  # childs of edges are bipart of lower level
    )
    G1_part1 = Data2(
        ord=ord_,
        level=1,
        num_nodes=5,
        is_part=True,
        part_id=1,
        edge_index=torch.tensor(
            [[0, 1, 1, 3, 0, 2, 2, 3, 3, 4],
             [1, 0, 3, 1, 2, 0, 3, 2, 4, 3]],
            dtype=torch.long
        ),
        edge_weight=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        edge_feat=torch.zeros([10, 0]),
        child_part=[None] * 5,
        child_bipart=[None] * 10,
    )
    G1_part2 = Data2(
        ord=ord_,
        level=1,
        num_nodes=3,
        is_part=True,
        part_id=2,
        edge_index=torch.tensor(
            [[0, 1, 0, 1, 1, 2],
             [1, 0, 1, 0, 2, 1]],
            dtype=torch.long
        ),
        edge_weight=torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long),
        edge_feat=torch.zeros([6, 0]),
        child_part=[None] * 3,
        child_bipart=[None] * 6,
    )

    # make bipartite subgraphs
    bp_01 = Bipart(
        level=1,
        ord=ord_,
        edge_index=torch.tensor(
            [[0, 2],
             [3, 1]],
            dtype=torch.long
        ),
        edge_weight=torch.tensor([1, 1], dtype=torch.long),
        edge_feat=torch.zeros([2, 0]),
        child_bipart=[None] * 2,
        part_left=G1_part0,
        part_right=G1_part1, )

    bp_12 = Bipart(
        level=1,
        ord=ord_,
        edge_index=torch.tensor(
            [[3],
             [1]],
            dtype=torch.long
        ),
        edge_weight=torch.tensor([1], dtype=torch.long),
        edge_feat=torch.zeros([1, 0]),
        child_bipart=[None] * 1,
        part_left=G1_part1,
        part_right=G1_part2, )

    bp_10 = bp_01.t()
    bp_21 = bp_12.t()

    # make partition subgraphs
    G0_part0 = Data2(
        ord=ord_,
        level=0,
        num_nodes=3,
        edge_index=torch.tensor(
            [[0, 0, 1, 1, 1, 2, 2],
             [0, 1, 0, 1, 2, 1, 2]],
            dtype=torch.long
        ),
        is_part=True,
        part_id=0,
        edge_weight=torch.tensor([6, 2, 2, 5, 1, 1, 3], dtype=torch.long),
        edge_feat=torch.zeros([7, 0]),
        child_part=[G1_part0, G1_part1, G1_part2],
        child_bipart=[None, bp_01, bp_10, None, bp_12, bp_21, None],
    )

    G0_part1 = Data2(
        ord=ord_,
        level=0,
        num_nodes=2,
        edge_index=torch.tensor(
            [[0, 0, 1, 1],
             [0, 1, 0, 1]], dtype=torch.long),
        is_part=True,
        part_id=0,
        edge_weight=torch.tensor([6, 2, 2, 5], dtype=torch.long),
        edge_feat=torch.zeros([4, 0]),
        child_part=[G1_part0, G1_part1],
        child_bipart=[None, bp_01, bp_10, None],
    )

    G0_part2 = Data2(
        ord=ord_,
        level=0,
        num_nodes=2,
        edge_index=torch.tensor(
            [[0, 0, 1, 1],
             [0, 1, 0, 1]], dtype=torch.long),
        is_part=True,
        part_id=0,
        edge_weight=torch.tensor([5, 1, 1, 3], dtype=torch.long),
        edge_feat=torch.zeros([4, 0]),
        child_part=[G1_part1, G1_part2],
        child_bipart=[None, bp_12, bp_21, None],
    )

    # Create HGs (list of levels) based on parts and bipartites in each level
    G0 = collate_partitions(
        level=0,
        part_ls=[G0_part0], bipart_ls=[],
        g_id='sim0'
    )
    G1 = collate_partitions(
        level=1,
        part_ls=G0.child_part,
        bipart_ls=G0.child_bipart,
        g_id='sim0'
    )
    for key in G1.keys: print(f'{key}, =, {G1[key]}')
    hg0 = HG([G0, G1], max_level=2, ord=ord_)
    hg0.set_id('sim0')
    hg_ord_0 = [hg0]

    #
    G0 = collate_partitions(
        level=0,
        part_ls=[G0_part1],
        bipart_ls=[],
        g_id='sim1'
    )
    G1 = collate_partitions(
        level=1,
        part_ls=G0.child_part,
        bipart_ls=G0.child_bipart,
        g_id='sim1'
    )
    for key in G1.keys: print(f'{key}, =, {G1[key]}')
    hg1 = HG([G0, G1], max_level=2, ord=ord_)
    hg1.set_id('sim1')
    hg_ord_1 = [hg1]

    G0 = collate_partitions(
        level=0,
        part_ls=[G0_part2],
        bipart_ls=[], g_id='sim2'
    )
    G1 = collate_partitions(
        level=1,
        part_ls=G0.child_part,
        bipart_ls=G0.child_bipart,
        g_id='sim2'
    )
    for key in G1.keys: print(f'{key} = {G1[key]}')
    hg2 = HG([G0, G1], max_level=2, ord=ord_)
    hg2.set_id('sim2')
    hg_ord_2 = [hg1]

    dataset = [hg_ord_0, hg_ord_1, hg_ord_2]
    return dataset
