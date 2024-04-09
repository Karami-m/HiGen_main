
from __future__ import (division, print_function)
import os, sys, time
import pickle, lzma
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.utils.data
import torch.optim as optim
from tensorboardX import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

from utils.higen_data import *
from utils.dataset.dataset_preprocessing import *
from utils.logger import get_logger
from utils.train_helper import snapshot, load_model
from utils.vis_helper import draw_HG_list_separate
from utils.graph_helper import *

try:
    ###
    # for solving the issue of multi-worker https://github.com/pytorch/pytorch/issues/973
    import resource
    rsclimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rsclimit[1]))
    ###
except:
    pass

logger = get_logger('exp_logger')

np.random.RandomState(seed=1234)

class HiGeN_Runner(object):

    def __init__(self, config):
        config.logger = logger
        self.config = config
        config.model.temperature = 1
        self.use_amp = config.get('use_amp', False)
        self.seed = config.seed
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.device = config.device
        self.writer = SummaryWriter(config.save_dir)
        self.is_vis = config.test.is_vis
        self.better_vis = config.test.better_vis
        self.num_vis = config.test.num_vis
        self.vis_num_row = config.test.vis_num_row
        self.is_single_plot = config.test.is_single_plot
        self.num_gpus = len(self.gpus)
        self.is_shuffle = False
        self.verbose = config.get('verbose', 1)

        self.train_ratio = config.dataset.train_ratio
        self.dev_ratio = config.dataset.dev_ratio
        self.num_levels = config.dataset.num_levels
        self.model_conf.num_levels = config.dataset.num_levels

        if self.train_conf.is_resume:
            self.config.save_dir = self.train_conf.resume_dir

        ## to load and save graphs
        if self.dataset_conf.is_overwrite_precompute:
            _graphs0 = get_multilevel_graph_dataset(
                config.dataset.name,
                node_orders=self.dataset_conf.node_order,
                data_dir=config.dataset.data_path,
                max_num_level=self.num_levels,
                is_overwrite_precompute=self.dataset_conf.is_overwrite_precompute,
                level_startFromRoot=config.dataset.level_startFromRoot,
                load_afew=config.dataset.get('load_afew', np.inf),
                coarsening_alg=config.dataset.get('coarsening_alg', 'Louvain'),
            )

            _graphs = []
            for i_g, _graph in enumerate(_graphs0):
                if _graph[0][0] is None:
                    print("removing graph : ", _graph[0][-1].get('graph_name', i_g))
                    continue
                _graphs.append(_graph)

            self.num_graphs = len(_graphs)
            self.num_train = int(float(self.num_graphs) * self.train_ratio)
            self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
            self.num_test_gt = self.num_graphs - self.num_train
            self.num_test_gen = config.test.num_test_gen

            ### to shuffle all graphs
            if self.is_shuffle:
                self.npr = np.random.RandomState(self.seed)
                self.npr.shuffle(_graphs)

            self.graphs_train = _graphs[:self.num_train]
            self.graphs_dev = _graphs[:self.num_dev]
            self.graphs_test = _graphs[self.num_train:]

            self.file_names_train, _ = save_load_graphs(self.graphs_train, 'train', config=config)
            self.file_names_dev, _ = save_load_graphs(self.graphs_dev, 'dev', config=config)
            self.file_names_test, _ = save_load_graphs(self.graphs_test, 'test', config=config)

        else:
            self.file_names_train, self.graphs_train = save_load_graphs(None, 'train', config=config, to_load=True)
            self.file_names_dev, self.graphs_dev    = save_load_graphs(None, 'dev', config=config, to_load=True)
            self.file_names_test, self.graphs_test  = save_load_graphs(None, 'test', config=config, to_load=True)

            self.num_train = len(self.graphs_train)
            self.num_dev = len(self.graphs_dev)
            self.num_test_gt = len(self.graphs_test)
            self.num_graphs = self.num_test_gt + self.num_train
            self.num_test_gen = config.test.num_test_gen

        logger.info(f'Train/val/test = {self.num_train}/{self.num_dev}/{self.num_test_gt}')

        def compute_edge_ratio(G_list):
            num_edges_max, num_edges = .0, .0
            for gg in G_list:
                num_nodes = gg.num_nodes
                num_edges += gg.num_edges
                num_edges_max += num_nodes ** 2
            ratio = (num_edges_max - num_edges) / num_edges if num_edges > 0 else 0.
            return ratio
        
        self.config.dataset.sparse_ratio = [
            compute_edge_ratio([self.graphs_train[i_t][i_ord][l] for i_t in range(self.num_train)
                                for i_ord in range(self.model_conf.num_canonical_order) if
                                not self.graphs_train[i_t][i_ord][l] is None
                                ]) \
            for l in range(self.num_levels)
        ]
        for l in range(self.num_levels):
            logger.info(f'LEVEL {l}, No Edges vs. Edges in training set = {self.config.dataset.sparse_ratio[l]}')

        self.stat_HG_pmf = PMF_stat_HG(self.graphs_train)
        self.config.model.max_num_nodes = self.stat_HG_pmf.max_num_nodes
        self.config.model.stat_HG_pmf = self.stat_HG_pmf


    def train(self):
        # to free up unused RAM
        del (self.graphs_train)
        del (self.graphs_dev)
        del (self.graphs_test)

        # to create data loader
        train_dataset = HigenData(self.config, file_names=self.file_names_train, num_graphs=self.num_train, tag='train')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_conf.batch_size,
            shuffle=self.train_conf.shuffle,
            num_workers=self.train_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=False,
            drop_last=False)

        # create models
        if self.model_conf.name == 'model_higen':
            from model_higen import Graph_Gen_MultiLevels
        else:
            raise NotImplementedError()

        model = Graph_Gen_MultiLevels(self.config).to(self.device)

        # to create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=self.train_conf.lr, momentum=self.train_conf.momentum,
                                  weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        # AUTOMATIC MIXED PRECISION
        if self.config.use_gpu:
            amp = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp)
        else:
            amp = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=self.use_amp)
        scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp,
            init_scale=8192.0, growth_factor=1.0005, backoff_factor=.9999, growth_interval=1000,
        ) if self.use_amp else None

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_epoch,
            gamma=self.train_conf.lr_decay)

        # To reset gradient
        optimizer.zero_grad()

        # To resume training
        resume_epoch = 0
        if self.train_conf.is_resume:
            model_file = os.path.join(self.train_conf.resume_dir,
                                      self.train_conf.resume_model)
            load_model(model, model_file,self.device, optimizer=optimizer, scheduler=lr_scheduler, scaler=scaler)
            resume_epoch = self.train_conf.resume_epoch

        # Training Loop
        iter_count = 0
        results = defaultdict(list)
        for epoch in range(resume_epoch, self.train_conf.max_epoch):
            model.train()
            train_iterator = train_loader.__iter__()

            for inner_iter in range(len(train_loader) // self.num_gpus):
                optimizer.zero_grad()

                time0_ = time.time()
                data_batch = next(train_iterator) # train_iterator.next()
                data_batch = train_dataset.to_device(data_batch, device=self.device)
                if self.verbose > 2:
                    print(f"++++ Time to load a batch: {time.time() - time0_}")
                iter_count += 1

                time0_ = time.time()
                with amp:
                    train_loss = model(data_batch)
                if self.use_amp:
                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < 128.:
                        scaler.set_backoff_factor(1.)
                    if self.verbose > 2:
                        logger.info(f"grad-scale: {scaler.get_scale()}")
                else:
                    train_loss.backward()
                    optimizer.step()
                train_loss_ = train_loss.data.cpu().numpy()

                if self.verbose > 2:
                    print(f"++++ Time to do one iteration: {time.time() - time0_}")

                self.writer.add_scalar('train_loss', train_loss_, iter_count)
                results['train_loss'].append(train_loss_)
                results['train_step'].append(iter_count)

                if (iter_count % self.train_conf.display_iter == 0) or (iter_count == 1):
                    train_loss_avg_ = np.array(
                        results['train_loss'][ -1 * min(self.train_conf.display_iter, len(results['train_loss'])) : ]
                    ).mean()
                    logger.info(
                        "NLL Loss @ EPOCH {:04d} iteration {:07d} = {:04f}".format(epoch + 1, iter_count, train_loss_avg_))

                torch.cuda.empty_cache()  # todo: to remove
            lr_scheduler.step()

            # to snapshot the model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("SNAPSHOT @ epoch {:04d}".format(epoch + 1))
                snapshot(model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler, scaler=scaler)

            train_loss_avg_ = np.array(
                results['train_loss'][-1 * min(len(train_loader) , len(results['train_loss'])) : ]
            ).mean()
            logger.info("{} NLL Loss @ epoch {:04d}= {:05f}".format('*'*20, epoch+1, train_loss_avg_))

        with open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb') as f_train_stat:
            pickle.dump(results, f_train_stat)
        self.writer.close()
        return

    def test(self, compute_metrics=None, save_graphs=True, make_compact=True):
        visualize_level = 'all'  # 'leaf'

        ### Compute Erdos-Renyi baseline
        if self.config.test.is_test_ER:
            p_ER = sum([aa.number_of_edges() for aa in self.graphs_train]) / sum(
                [aa.number_of_nodes() ** 2 for aa in self.graphs_train])
            HG_gen_nx_ls = [nx.fast_gnp_random_graph(self.max_num_nodes, p_ER, seed=ii) for ii in range(self.num_test_gen)]

        else:
            self.stat_HG_pmf.compute_stat_of_degree(self.graphs_train)

            save_name = os.path.join(self.config.save_dir, 'train_graphs.png')
            if self.config.get("draw_training", False):
                draw_HG_list_separate(
                    [self.graphs_train[i][0] for i in range(min(self.num_vis, len(self.graphs_train)) )],
                    fname=save_name[:-4], is_single=True, layout='spring')
            del(self.graphs_train)

            ## to load model
            from model_higen import Graph_Gen_MultiLevels
            model = Graph_Gen_MultiLevels(self.config).to(self.device)

            model_file = os.path.join(self.config.test.test_model_dir, self.test_conf.test_model_name)
            logger.info(f" *** *** Model snapshot name: {self.test_conf.test_model_name}")
            load_model(model, model_file, self.device)

            ## To Generate Graphs
            model.eval()
            HG_gen_ls, HG_gen_nx_ls = [], []
            num_nodes_pred = []
            num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))
            stat_HG_gen = self.stat_HG_pmf.sample2(n_sample=self.num_test_gen, leaf_edge_weight=1)
            g_root_gen = [
                HG.make_root_graph(
                    n_levels=n_levels,
                    num_leafnodes=n_node_,
                    sum_leafedges=sum_edge_,
                    num_leafedges=num_edge_,
                    hg_name='HG%d' % i_sample,
                ).to(self.device)
                for i_sample, (n_levels, n_node_, sum_edge_, num_edge_) in enumerate(stat_HG_gen)]

            gen_runtime = []
            for ii in tqdm(range(num_test_batch)):
                batch_indexes = (ii * self.test_conf.batch_size,
                                 min((ii + 1) * self.test_conf.batch_size, self.num_test_gen))
                with torch.no_grad():
                    start_time = time.time()
                    HG_gen_ = model.generate(g_root_gen[batch_indexes[0]: batch_indexes[1]])
                    gen_runtime += [time.time() - start_time]
                    if make_compact:
                        HG_gen_nx_ls += [
                            HG_.to_networkx(level=visualize_level, add_child_cluster=True, add_node_parts=False)
                            for HG_ in HG_gen_]
                    else:
                        HG_gen_ls += [HG_ for HG_ in HG_gen_]
                    num_nodes_pred += [HG_[-1].num_nodes for HG_ in HG_gen_]
                    del HG_gen_
            logger.info(f'Average test time per mini-batch = {np.mean(gen_runtime)}')

            if not make_compact:
                HG_gen_nx_ls = [HG_.to_networkx(level=visualize_level, add_child_cluster=True, add_node_parts=False) for HG_ in HG_gen_ls]

        test_epoch = self.test_conf.test_model_name
        test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]

        ## to store Generated Graphs
        if save_graphs:
            fname = os.path.join(
                self.config.save_dir,
                f'gen_graphs_epoch_{test_epoch}.pickle')
            with lzma.open(fname + ".xz", "wb") as f:
                pickle.dump(HG_gen_nx_ls, f)

        ## to Visualize Generated Graphs
        if self.is_vis:
            save_name = os.path.join(self.config.save_dir,
                                     f'{self.config.test.test_model_name[:-4]}_gen_graphs_epoch_{test_epoch}.png')
            draw_HG_list_separate(HG_gen_nx_ls[:self.num_vis], fname=save_name[:-4], is_single=True, layout='spring',
                                  better_vis=self.better_vis)

        model.train()

        model_total_params = sum(0 if isinstance(p_, torch.nn.parameter.UninitializedParameter) else
                                 p_.numel() for p_ in model.parameters() if p_.requires_grad)
        logger.info("TOTAL NUMBER of parameters of model = %6e" % model_total_params)
        with open(os.path.join(self.config.save_dir, 'model_summary.sum'), 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print(model)
            sys.stdout = original_stdout

        return

    def store_test_dev(self):
        visualize_level = 'all'  # 'leaf'

        self.stat_HG_pmf.compute_stat_of_degree(self.graphs_train)

        save_folder = os.path.join(self.config.dataset.data_path, f"DEV_TEST/{self.config.dataset.name}")
        save_name = os.path.join(save_folder, 'train_graphs.png')

        HG_tr_nx = [HG_[0].to_networkx(level=visualize_level, add_node_parts=False, add_child_cluster=True) for HG_ in self.graphs_dev]
        draw_HG_list_separate(HG_tr_nx[: min(self.num_vis, len(self.graphs_train))],
                              fname=save_name[:-4], is_single=True, layout='spring')

        HG_dev_nx = [HG_[0].to_networkx(level=visualize_level, add_node_parts=True, add_child_cluster=True) for HG_ in self.graphs_dev]
        HG_test_nx = [HG_[0].to_networkx(level=visualize_level, add_node_parts=True, add_child_cluster=True) for HG_ in self.graphs_test]
        graph_dev_nx = [HG_[0].to_networkx(level='leaf', add_node_parts=True, add_child_cluster=True) for HG_ in self.graphs_dev]
        graph_test_nx = [HG_[0].to_networkx(level='leaf', add_node_parts=True, add_child_cluster=True) for HG_ in self.graphs_test]

        save_folder = os.path.join(self.config.dataset.data_path, f"DEV_TEST/{self.config.dataset.name}")
        fname = os.path.join(
            save_folder,
            f'DEV_HG_{self.config.dataset.name}.pickle')
        with open(fname, "wb") as f:
            pickle.dump(HG_dev_nx, f)

        fname = os.path.join(
            save_folder,
            f'TEST_HG_{self.config.dataset.name}.pickle')
        with open(fname, "wb") as f:
            pickle.dump(HG_test_nx, f)

        fname = os.path.join(
            save_folder,
            f'DEV_graphs_{self.config.dataset.name}.pickle')
        with open(fname, "wb") as f:
            pickle.dump(graph_dev_nx, f)

        fname = os.path.join(
            save_folder,
            f'TEST_graphs_{self.config.dataset.name}.pickle')
        with open(fname, "wb") as f:
            pickle.dump(graph_test_nx, f)

