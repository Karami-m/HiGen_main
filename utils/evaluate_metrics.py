import time, os, sys
import argparse
import lzma
import pickle
from glob import glob
import numpy as np
import networkx as nx
from easydict import EasyDict as edict
from utils.GGM_metrics.gin_evaluator import nn_based_eval
from utils.eval_helper import *


def evaluate(graph_gt, graph_pred, stats, kernel="gaussian_tv", to_correct_edges=False):
    if to_correct_edges:
        # convert all edge weights to binary
        graph_pred_ = []
        for G_ in graph_pred:
            if not G_ is None:
                adj_ = np.minimum(nx.to_numpy_matrix(G_), 1)
                if not np.sum(adj_ - adj_.transpose()) == 0:
                    print("------ Generated graph is not symmetric ..... ")
                graph_pred_.append(nx.from_numpy_matrix(adj_))
        graph_pred = graph_pred_

    out = edict()
    if 'mmd_degree' in stats:
        out.mmd_degree = float("Inf")
        out.mmd_degree = degree_stats(graph_gt, graph_pred, kernel="gaussian_tv")
    if 'mmd_clustering' in stats:
        out.mmd_clustering = float("Inf")
        out.mmd_clustering = clustering_stats(graph_gt, graph_pred, kernel="gaussian_tv")
    if 'mmd_spectral' in stats:
        out.mmd_spectral = float("Inf")
        out.mmd_spectral = spectral_stats(graph_gt, graph_pred, kernel="gaussian_tv")
    if 'mmd_4orbits' in stats:
        out.mmd_4orbits = float("Inf")
        out.mmd_4orbits = orbit_stats_all(graph_gt, graph_pred, kernel="gaussian_tv")

    if 'emd_degree' in stats:
        out.emd_degree = float("Inf")
        out.emd_degree = degree_stats(graph_gt, graph_pred, kernel="gaussian_emd")
    if 'emd_clustering' in stats:
        out.emd_clustering = float("Inf")
        out.emd_clustering = clustering_stats(graph_gt, graph_pred, kernel="gaussian_emd")
    if 'emd_spectral' in stats:
        out.emd_spectral = float("Inf")
        out.emd_spectral = spectral_stats(graph_gt, graph_pred, kernel="gaussian_emd")
    if 'emd_4orbits' in stats:
        out.emd_4orbits = float("Inf")
        out.emd_4orbits = orbit_stats_all(graph_gt, graph_pred, kernel="gaussian_emd")

    # GGM metric based
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if 'ggm_mmd' in stats or 'ggm_pr' in stats or 'ggm_dc' in stats:

        res_ = nn_based_eval(graph_gt, graph_pred, N_gin=10)
        out.ggm_mmd, out.ggm_mmd_std = res_['MMD_RBF']
        out.ggm_pr, out.ggm_pr_std = -res_['F1_PR'][0], res_['F1_PR'][1]
        out.ggm_dc, out.ggm_dc_std = -res_['F1_DC'][0], res_['F1_DC'][1]

    return out


def compute_emmd(graphs_gt, graphs_gen, out, to_correct_edges=False, kernel="gaussian_tv", stats=['mmd_degree', 'mmd_clustering', 'mmd_spectral']):
    """
    :param graphs_gt:
    :param graphs_gen:
    :param to_correct_edges:
    :param kernel:
    :param stats: can be from ['degree', 'clustering', 'spectral', '4orbits']
    :return:
    """

    out_ = evaluate(
        graphs_gt,
        graphs_gen,
        stats=stats,
        kernel=kernel,
        to_correct_edges=False
    )
    out.update(out_)
    return out

metric_map={
    'mmd_degree': 1,
    'mmd_clustering': 3,
    'mmd_spectral': 5,
    'mmd_4orbits': 7,
    'ggm_mmd':9,
    'ggm_pr':11,
    'ggm_dc':13,
    'emd_degree': 15,
    'emd_clustering': 17,
    'emd_spectral': 19,
    'emd_4orbits': 21,
}
def compute_metric(dataset_name, exp_dir, stats, log_file_name, to_correct_edges,
                   start_epoch, end_epoch, step, gen_file_type, dev_graphs, test_graphs, tag=''):
    metrics_log_file= os.path.join(exp_dir, log_file_name+tag+'.txt')
    out_fname = os.path.join(exp_dir, log_file_name + tag + '.pickle')
    if os.path.exists(out_fname):
        with open(out_fname, "rb") as f:
            _out = pickle.load(f)
            out_dict = _out['out_dict']
            out_array = _out['out_array']
            dev_best = _out.get('dev_best', edict())
            test_best = _out.get('test_best', edict())
            epoch_best = _out.get('epoch_best', edict())
    else:
        out_dict = {'out_dev': {}, 'out_test': {}}
        out_array = np.zeros([0, 40]) + np.inf
        dev_best = edict()
        test_best = edict()
        epoch_best = edict()

    str_out = "epoch  ," + ",".join([" {:24s}"]*len(stats)).format( *[f"{_stat}_dev, {_stat}_ts" for _stat in stats]) + "\n"
    with open(metrics_log_file, 'a+') as f:
        str_out = "\n {}\n Dataset : {} \n exp_dir : {:20} \n TAG : {:20} \n\n".format('-'*10 ,dataset_name, exp_dir, tag) + str_out
        f.write(str_out)
    print(str_out)

    for epoch in range(start_epoch, end_epoch+1, step):
        out_dev = out_dict['out_dev'].get(epoch, edict())
        out_test = out_dict['out_test'].get(epoch, edict())

        stats_dev, stats_test = [], []
        for stat_ in stats:
            if not hasattr(out_dev, stat_):
                stats_dev.append(stat_)
            if not hasattr(out_test, stat_):
                stats_test.append(stat_)
        if len(stats_dev)+len(stats_test)==0:
            continue

        gen_file = os.path.join(exp_dir, "gen_graphs_epoch_{:07}.pickle".format(epoch))
        if not glob(gen_file+'*'):
            continue

        try:
            if gen_file_type=='xz':
                with lzma.open(gen_file+".xz", "rb") as f:
                    gen_HGs = pickle.load(f)
            else:
                with open(gen_file, "rb") as f:
                    gen_HGs = pickle.load(f)
        except:
            print("couldn't load : ", gen_file)
            continue

        level_ = -1
        gen_graphs = [HG_nx_[level_] for HG_nx_ in gen_HGs if not HG_nx_[level_] is None]

        out_dev_ = evaluate(dev_graphs, gen_graphs, stats=stats_dev, to_correct_edges=False) # kernel=kernel,
        out_test_ = evaluate(test_graphs, gen_graphs, stats=stats_test, to_correct_edges=False) # kernel=kernel,
        out_dev.update(out_dev_)
        out_test.update(out_test_)
        out_dev.epoch = epoch
        out_test.epoch = epoch

        str_epoch = "{:7}".format(epoch)
        str_out = " , ".join(["{:12.2e}, {:12.2e}".format(out_dev[stat_], out_test[stat_]) for stat_ in stats ])
        str_out = str_epoch+str_out + "\n"
        print(str_out)
        with open(metrics_log_file, 'a+') as f:
            f.write(str_out)

        out_dict['out_dev'].update({epoch: out_dev})
        out_dict['out_test'].update({epoch: out_test})

        out_array_ = np.zeros([1, out_array.shape[1]]) + np.inf
        out_array_[0, 0] = epoch
        for stat_ in stats:
            out_array_[0, metric_map[stat_]] = out_dev[stat_]
            out_array_[0, metric_map[stat_]+1] = out_test[stat_]
        out_array = np.concatenate([out_array, out_array_] , 0)

        #update the bests
        for stat_ in stats:
            if out_dev[stat_] < dev_best.get(stat_, np.inf):
                dev_best[stat_] = out_dev[stat_]
                test_best[stat_] = out_test[stat_]
                epoch_best[stat_] = epoch

        with open(out_fname, "wb") as f:
            pickle.dump(
                {'out_dict':out_dict,
                 'out_array': out_array,
                 'dev_best' : dev_best,
                 'test_best': test_best,
                 'epoch_best': epoch_best,
                 'exp_dir': exp_dir,
                 'metric_map': metric_map},
                f)

    # to compute the best again

    min_ind = np.argmin(out_array, axis=0)
    out_best_ = np.zeros([1, out_array.shape[1]])+ np.inf
    for stat_ in stats:
        out_best_[0, metric_map[stat_]] = out_array[min_ind[metric_map[stat_]], metric_map[stat_]]
        out_best_[0, metric_map[stat_]+1] = out_array[min_ind[metric_map[stat_]], metric_map[stat_]+1]
        if dev_best.get(stat_, np.inf) == np.inf: # this is
            dev_best[stat_] = out_best_[0, metric_map[stat_]]
            test_best[stat_] = out_best_[0, metric_map[stat_]+1]
            epoch_best[stat_] = min_ind[metric_map[stat_]]
    out_array = np.concatenate([out_array, out_best_], 0)

    out_best_ = np.zeros([1, out_array.shape[1]])+ np.inf
    for stat_ in stats:
        out_best_[0, metric_map[stat_]] = out_array[min_ind[metric_map[stat_]+1], metric_map[stat_]]
        out_best_[0, metric_map[stat_]+1] = out_array[min_ind[metric_map[stat_]+1], metric_map[stat_]+1]
    out_array = np.concatenate([out_array, out_best_], 0)

    str_out = " , ".join(["{:12.2e}, {:12.2e}".format(dev_best.get(stat_, np.inf), test_best.get(stat_, np.inf)) for stat_ in stats])
    str_out = "BEST" + str_out + "\n"
    print(str_out)
    with open(metrics_log_file, 'a+') as f:
        f.write(str_out)

    with open(out_fname, "wb") as f:
        pickle.dump({'out_dict': out_dict,
                     'out_array': out_array,
                     'dev_best': dev_best,
                     'test_best': test_best,
                     'epoch_best': epoch_best,
                     'exp_dir': exp_dir,
                     'metric_map': metric_map},
                    f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Running Evaluation of Generated graphs")
    parser.add_argument('-d', '--dataset_name', type=str, default="")
    parser.add_argument('-e', '--exp_dir', type=str, default="")
    parser.add_argument('--exp_subdir', type=str, default="")
    parser.add_argument('--gen_file_type', type=str, default="xz")
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--start_epoch', type=int, default=10)
    parser.add_argument('--end_epoch', type=int, default=100)
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--num_gen', type=int, default=-1)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    exp_dir = args.exp_dir  # "dist=mix_multinomial+Bern,NLL_avg=adv,context_parent=GATgran_indv,gnn_aug_part=GATgran,gnn_aug_bipart=GATgran,to_joint_aug_bp=none,LP_model=b13_m12_s1,node_order=BFSDC,link_cntx_type=cat62,hidden_dim=64,embedding_dim=64-lr=1e-4"
    exp_subdir_ls = [args.exp_subdir]  # ["gen_completion=NumEdgesNumNodes2-", "-"]
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    step = args.step
    gen_file_type = args.gen_file_type
    tag = args.tag
    num_gen = None if args.num_gen == -1 else args.num_gen

    base_dir = os.path.join(os.getcwd(), "exp/{}/".format(dataset_name))
    dev_test_dir = os.path.join(os.getcwd(), "data/DEV_TEST/{}/".format(dataset_name))
    print(base_dir)

    stats = ['degree', 'clustering', 'spectral']  # '4orbits'
    log_file_name = 'emmd_log'
    to_correct_edges = False

    dev_file = os.path.join(dev_test_dir, "DEV_graphs_{}.pickle".format(dataset_name))
    test_file = os.path.join(dev_test_dir, "TEST_graphs_{}.pickle".format(dataset_name))

    with open(dev_file, "rb") as f:
        dev_graphs = pickle.load(f)
    with open(test_file, "rb") as f:
        test_graphs = pickle.load(f)

    if num_gen:
        dev_graphs = dev_graphs[:num_gen]
        test_graphs = test_graphs[:num_gen]

    for exp_subdir_ in exp_subdir_ls:
        exp_dir = os.path.join(base_dir, exp_dir, exp_subdir_)

        compute_metric(dataset_name=dataset_name, exp_dir=exp_dir, stats=stats, log_file_name=log_file_name,
                       to_correct_edges=to_correct_edges,
                       start_epoch=start_epoch, end_epoch=end_epoch, step=step, gen_file_type=gen_file_type, tag=tag)
    sys.exit(0)