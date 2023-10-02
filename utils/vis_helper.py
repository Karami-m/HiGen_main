import copy, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch_geometric as pyg


def draw_HG_list_separate(HG_list,
                          fname='exp/gen_graph',
                          layout='spring',
                          is_single=False,
                          k=1,
                          alpha=1,
                          better_vis=True):
    draw_graph_list_separate(
        [better_vis_fn(HG_[-1], better_vis) for HG_ in HG_list],
        fname=fname,
        layout=layout,
        is_single=is_single,
        k=k,
    )

    for i, HG_nx in enumerate(HG_list):
        draw_HG(HG_nx, title=fname+'_{:03d}'.format(i), better_vis=better_vis)


def aggeregate_pos(pos, communities):
    pos_1 = {}
    for j in communities.keys():
        pos_agg = [pos[v] for v in communities[j]]
        pos_agg = np.average(np.stack(pos_agg, 0), 0)
        pos_1[j] = pos_agg

    return pos_1


def aggeregate_color(node_color, communities):
    node_color_1 = {}
    for j in communities.keys():
        col_agg = [node_color[v] for v in communities[j]]
        col_agg = np.average(np.array(col_agg))
        node_color_1[j] = col_agg
    return node_color_1



def draw_HG(HG_nx, title='', pos=None, layout='spring', figsize=(5, 5), better_vis=True, scale_logarithmic = True):
    L = len(HG_nx)
    G = HG_nx[-1]
    partition = nx.get_node_attributes(G, "part_id_of_node")
    # cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    cmap = cm.get_cmap('hsv', max(partition.values()) + 1)
    node_color = list(partition.values())

    if pos is None:
        if layout == 'spring':
            pos = nx.drawing.layout.spring_layout(
                G, k=1 / np.sqrt(G.number_of_nodes()), iterations=100)
        elif layout == 'spectral':
            pos = nx.drawing.layout.spectral_layout(G)

    plt.switch_backend('agg')
    plt.subplot(L, 1, L)
    plt.gcf().set_size_inches(figsize[0], figsize[1] * len(HG_nx))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.axis("off")
    draw_graph(G, partition, title='', pos=pos, node_color=node_color, cmap=cmap)
    G0 = copy.deepcopy(G)
    pos0= copy.deepcopy(pos)
    node_color0 = copy.deepcopy(node_color)
    partition0=copy.deepcopy(partition)

    for l in reversed(range(L - 1)):
        G = HG_nx[l]
        if G is None:
            continue

        # remove isolated nodes for better visulization
        G = better_vis_fn(G, better_vis=better_vis)

        clusters = nx.get_node_attributes(G, "child_nodes")
        pos = aggeregate_pos(pos, clusters)
        node_color = aggeregate_color(node_color, clusters)
        node_color = list(node_color.values())
        partition = nx.get_node_attributes(G, "part_id_of_node")

        plt.subplot(L, 1, l + 1)
        plt.gcf().set_size_inches(figsize[0], figsize[1] * len(HG_nx))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis("off")
        draw_graph(G, partition, title='', pos=pos, node_color=node_color, cmap=cmap, scale_logarithmic=scale_logarithmic)
        nx.draw_networkx_nodes(G0, pos0, alpha=.0) # this will scale the graphs so that clusterc centres are adjusted

    plt.draw()
    plt.tight_layout()
    try:
        plt.savefig(title + '.png', dpi=300)
    except:
        fname_ls = title.split(os.sep)
        fname_ls[-1] = fname_ls[-1].replace('_', '')
        fname = os.path.join(*fname_ls)
        plt.savefig(fname + '.png', dpi=300)
    plt.close()

def draw_graph(G,
               partition,
               title='',
               layout='spring',
               pos=None,
               node_color=None,
               cmap=None,
               scale_logarithmic=True,
               alpha=0.8):

    if pos is None:
        if layout == 'spring':
            pos = nx.drawing.layout.spring_layout(
                G, k=1 / np.sqrt(G.number_of_nodes()), iterations=100)
        elif layout == 'spectral':
            pos = nx.drawing.layout.spectral_layout(G)

    labels = nx.get_edge_attributes(G, 'edge_weight')
    labels = dict([(key_, int(val_)) for (key_, val_) in labels.items()])
    if scale_logarithmic == True:
        node_size = [(1 + np.log2(labels[(i, i)] + 1e-3))/np.log2(1.3) * 30 if (i, i) in labels else 20 for i in range(len(G))]
        edge_width = list((1 + np.log2(np.array(list(labels.values())) + 1e-3)/np.log2(1.3))* 1.3)  # list(labels.values())
    else:
        node_size = [labels[(i, i)] * 30 if (i, i) in labels else 20 for i in range(len(G))]
        edge_width = list(labels.values())
    if node_color == None:
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        node_color = list(partition.values())

    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=node_size,
                           cmap=cmap, node_color=node_color, alpha=alpha, )

    nx.draw_networkx_edges(G, pos, alpha=alpha/2., width=edge_width)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "parent_cluster"), font_size=1, font_color="red")


def draw_graph_list_separate(G_list,
                             fname='exp/gen_graph',
                             layout='spring',
                             is_single=False,
                             k=1,
                             node_size=55,
                             alpha=1,
                             width=1.3):

    for i, G in enumerate(G_list):
        plt.switch_backend('agg')

        plt.axis("off")

        if layout == 'spring':
            pos = nx.drawing.layout.spring_layout(G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
        elif layout == 'spectral':
            pos = nx.drawing.layout.spectral_layout(G)

        if is_single:
            nx.drawing.nx_pylab.draw_networkx_nodes(
                G,
                pos,
                node_size=node_size,
                node_color='#336699',
                alpha=1,
                linewidths=0,
                font_size=0)
            nx.drawing.nx_pylab.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.drawing.nx_pylab.draw_networkx_nodes(
                G,
                pos,
                node_size=1.5,
                node_color='#336699',
                alpha=1,
                linewidths=0.2,
                font_size=1.5)
            nx.drawing.nx_pylab.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

        plt.draw()
        plt.tight_layout()
        try:
            plt.savefig(fname+'_{:03d}_leaf.png'.format(i), dpi=300)
        except:
            fname_ls = fname.split(os.sep)
            fname_ls[-1] = fname_ls[-1].replace('_', '')
            fname = os.path.join(*fname_ls)
            plt.savefig(fname+'_%d_leaf.png'%i, dpi=300)
        plt.close()

def better_vis_fn(G_nx, better_vis=True):
    if better_vis == False:
        return G_nx
    # To remove isolated nodes for better visulization
    G_nx.remove_nodes_from(list(nx.isolates(G_nx)))

    # To display the largest connected component for better visualization
    CGs = [G_nx.subgraph(c) for c in nx.connected_components(G_nx)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    G_nx = CGs[0]
    return G_nx

