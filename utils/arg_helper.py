import os, time, copy, yaml
from sys import platform
import argparse
from easydict import EasyDict as edict
import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_act

class struct(object):
    def __init__(self):
        self = None

class HParams(object):
    """Returns the dictionary of hyperparameters."""

    def __init__(self, **kwargs):
        self.dict_ = kwargs
        self.__dict__.update(self.dict_)

    def update_config(self, in_string):
        """To update the dictionary with a comma separated list."""
        if in_string is None or in_string == '':
            return self
        pairs = in_string.split(",")
        pairs = [pair.split("=") for pair in pairs]
        for key, val in pairs:
            if val.lower() == 'false':
                val = False
            elif val.lower() == 'true':
                val = True
            self.dict_[key] = type(self.dict_[key])(val)
        self.__dict__.update(self.dict_)
        return self

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, val):
        self.dict_[key] = val
        self.__dict__.update(self.dict_)
def parse_arguments(command_str=None):
    parser = argparse.ArgumentParser(
        description="Running Experiments of Deep Prediction")
    parser.add_argument('-c', '--config_file', type=str, required=True, help="The path of the config file")
    parser.add_argument('-l', '--log_level', type=str, default='INFO',
                        help="Logging Level from: [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
    parser.add_argument('--comment', help="Experiment comment")
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--subdir', type=str, default="")
    parser.add_argument('--hp', type=str, default="")
    parser.add_argument('-t', '--test', help="Test model", action='store_true')
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--draw_training', help="draw training data at test time", action='store_true')
    parser.add_argument('--amp', help="To train/test with automatic mixed precision", action='store_true')

    if command_str:
        args = parser.parse_args(command_str.split(' '))  # To be for debug ONLY
    else:
        args = parser.parse_args()
    return args


register_act('relu', nn.ReLU)

pe_encoder_setting_dict = {
    'LapPE': 'posenc_LapPE',
    'RWSE': 'posenc_RWSE',
    'HKdiagSE': 'posenc_HKdiagSE',
    'ElstaticSE': 'posenc_ElstaticSE',
    'SignNet': 'posenc_SignNet',
    'EquivStableLapPE': 'posenc_EquivStableLapPE',
    'Cycles': 'posenc_Cycles',
    'StLapPE': 'posenc_StableLapPE',
    # 'GraphormerBias': ''
}
def update_config_with_defaults(config):
    if type(config.model.node_feat_types) == str:
        config.model.node_feat_types = config.model.node_feat_types.split('+')

    for _node_feat_type in config.model.node_feat_types:
        _pe_encoder_setting = pe_encoder_setting_dict[_node_feat_type]
        if _pe_encoder_setting in ['posenc_LapPE', 'posenc_SignNet',
                                   'posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE', 'posenc_Cycles', 'posenc_StableLapPE']:
            pecfg = copy.deepcopy(config.get(_pe_encoder_setting, edict()))
            pecfg.enable = False    # Use extended positional encodings
            pecfg.model = 'none'    # Neural-net model type within the PE encoder 'DeepSet', 'Transformer', 'Linear', 'none', ...
            pecfg.dim_pe = 16       # Size of Positional Encoding embedding
            pecfg.layers = 3        # Number of layers in PE encoder model
            pecfg.n_heads = 4       # Number of attention heads in PE encoder when model == 'Transformer'
            pecfg.post_layers = 0   # Number of layers to apply in LapPE encoder post its pooling stage
            pecfg.raw_norm_type = 'none' # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
            # In addition to appending PE to the node features, pass them also as
            # a separate variable in the PyG graph batch object.
            pecfg.pass_as_var = False

            if _pe_encoder_setting in ['posenc_LapPE', 'posenc_SignNet', 'posenc_EquivStableLapPE', 'posenc_StableLapPE']:
                pecfg.eigen = pecfg.get('eigen', edict())
                pecfg.eigen.laplacian_norm = 'sym'      # The normalization scheme for the graph Laplacian: 'none', 'sym', or 'rw'
                pecfg.eigen.eigvec_norm = 'L2'          # The normalization scheme for the eigen vectors of the Laplacian
                pecfg.eigen.max_freqs = 10              # Maximum number of top smallest frequencies & eigenvectors to use

            if _pe_encoder_setting == 'posenc_SignNet':
                # Config for SignNet-specific options.
                pecfg.phi_out_dim = 4
                pecfg.phi_hidden_dim = 64

            if _pe_encoder_setting == 'posenc_EquivStableLapPE':
                pecfg.enable = False
                pecfg.raw_norm_type = 'none'

            if _pe_encoder_setting in ['posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE']:
                pecfg.model = 'Linear'  # Neural-net model type within the PE encoder 'DeepSet', 'Transformer', 'Linear', 'none', ...
                pecfg.layers = 1  # Number of layers in PE encoder model
                pecfg.kernel = pecfg.get('kernel', edict())
                # List of times to compute the heat kernel for (the time is equivalent to
                # the variance of the kernel) / the number of steps for random walk kernel
                # Can be overridden by `posenc.kernel.times_func`
                pecfg.kernel.times = []
                if not hasattr(pecfg.kernel, 'times') or len(pecfg.kernel.times) == 0:
                    if hasattr(pecfg.kernel, 'times_func') and pecfg.kernel.times_func != '':
                        pecfg.kernel.times = list(eval(pecfg.kernel.times_func))

            if _pe_encoder_setting == 'posenc_Cycles':
                pecfg.enable = True
                pecfg.raw_norm_type = 'none'

            if _pe_encoder_setting == 'posenc_StableLapPE':
                pecfg.enable = True
                pecfg.raw_norm_type = 'none'


        if hasattr(config, _pe_encoder_setting):
            pecfg.update(config[_pe_encoder_setting])
        config[_pe_encoder_setting] = pecfg

    cfg = copy.deepcopy(config.get('gt', edict()))
    cfg.layer_type = 'CustomGatedGCN+Transformer'    # Type of Graph Transformer layer to use
    cfg.layers = 3                                   # Number of Transformer layers in the model
    cfg.n_heads = 4                                  # Number of attention heads in the Graph Transformer
    cfg.dim_hidden = 64                              # Size of the hidden node and edge representation
    cfg.full_graph = True                            # Full attention SAN transformer including all possible pairwise edges
    cfg.gamma = 1e-5                                 # SAN real vs fake edge attention weighting coefficient
    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.pna_degrees = []
    cfg.dropout = 0.0                                # Dropout in feed-forward module.
    cfg.attn_dropout = 0.0                           # Dropout in self-attention.
    cfg.layer_norm = False                           # Dropout in self-attention.
    cfg.batch_norm = True                            # Dropout in self-attention.
    cfg.residual = True
    if 'bigbird' in cfg.layer_type.lower():
        # BigBird model/GPS-BigBird layer.
        cfg.bigbird = edict()
        cfg.bigbird.attention_type = "block_sparse"
        cfg.bigbird.chunk_size_feed_forward = 0
        cfg.bigbird.is_decoder = False
        cfg.bigbird.add_cross_attention = False
        cfg.bigbird.hidden_act = "relu"
        cfg.bigbird.max_position_embeddings = 128
        cfg.bigbird.use_bias = False
        cfg.bigbird.num_random_blocks = 3
        cfg.bigbird.block_size = 3
        cfg.bigbird.layer_norm_eps = 1e-6

    if hasattr(config, 'gt'):
        cfg.update(config.gt)
        cfg.layers = config.model.get('layers_gps', cfg.layers)
        cfg.dim_hidden = config.model.get('hidden_dim_gps', cfg.dim_hidden)
        cfg.dim_inner = config.model.get('hidden_dim_gps', cfg.dim_inner)
        cfg.layer_type = config.model.get('layer_type_gps', cfg.layer_type)
        cfg.dropout = config.model.get('dropout_gps', cfg.dropout)
    config.gt = cfg

    cfg = copy.deepcopy(config.get('pgnn', edict()))
    cfg.layers = 8
    cfg.dim_hidden = 64
    cfg.dim_inner= 16
    cfg.dim_out= 16
    cfg.norm= 'instance'
    cfg.dropout = 0.0
    cfg.act= 'gelu'
    if hasattr(config, 'pgnn'):
        cfg.update(config.pgnn)
        cfg.layers = config.model.get('layers_gps', cfg.layers)
        cfg.dim_hidden = config.model.get('hidden_dim_gps', cfg.dim_hidden)
        cfg.dim_inner = config.model.get('hidden_dim_gps', cfg.dim_inner)
        cfg.dim_out = config.model.get('hidden_dim_gps', cfg.dim_out)
        cfg.dropout = config.model.get('dropout_gps', cfg.dropout)
    config.pgnn = cfg

    return config


def get_config(config_file, exp_dir=None, is_test=False,
               tag=None, gpu=None, hp_txt='', subdir=None,
               test_model_name=None, draw_training=False):
    """ To construct hyperparameters
    """
    config = edict(yaml.load(open(config_file, 'r')))
    hp = HParams(**config.model).update_config(hp_txt)
    config.model.update(hp.dict_)

    # to overwrite the node_order from hp_txt
    node_order = config.model.get('node_order', None)
    if not ((node_order is None) and node_order == 'none'):
        config.dataset.node_order = node_order

    # to create hyperparameters
    config.run_id = str(os.getpid())
    config.exp_name = '_'.join([time.strftime('%Y-%b-%d-%H-%M-%S'), config.run_id]) if (tag is None) or tag == '' else tag

    if exp_dir is not None:
        config.exp_dir = exp_dir
    else:
        config.exp_dir = os.path.join(config.exp_dir, config.dataset.name, subdir)

    config = update_config_with_defaults(config)

    if config.train.is_resume and not is_test:
        if config.train.is_resume and not is_test:
            if config.train.resume_dir is None:
                config.train.resume_dir = config.save_dir
        config.save_dir = config.train.resume_dir
        config.config_save_name = os.path.join(config.save_dir, f'config_resume_{config.run_id}.yaml')

    elif is_test:
        config.save_dir = os.path.join(config.test.test_model_dir, tag)
        config.config_save_name = os.path.join(config.save_dir, f'config_test_{config.run_id}.yaml' if not platform == "win32" else f'conf_{config.run_id}.yaml')
    else:
        config.save_dir = os.path.join(config.exp_dir, config.exp_name)
        config.config_save_name = os.path.join(config.save_dir, f'config_{config.run_id}.yaml')

    # to snapshot hyperparameters
    make_dir(config.exp_dir)
    make_dir(config.save_dir)

    if (not gpu is None) and 'cuda' in config.device:
        config.device = 'cuda:{}'.format(gpu)
        config.use_gpu = config.use_gpu and torch.cuda.is_available()
    if (not gpu is None) and 'mps' in config.device:
        if torch.backends.mps.is_available(): # For Apple silicon
            config.device = torch.device("mps")
        else:
            config.use_gpu = False
    if (not gpu is None) and (gpu<0):
        config.use_gpu = False
    print("*** *** CUDA is AVAILABLE *** ***" if config.use_gpu else "*** *** Running on CPU *** ***")
    if not config.use_gpu:
        config.device = 'cpu'

    config.draw_training= draw_training

    if test_model_name:
        config.test.test_model_name = "model_snapshot_%07d.pth"%(int(test_model_name))

    with open(config.config_save_name, 'w') as f:
        yaml.dump(edict2dict(config), f, default_flow_style=False)

    return config


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals
    return dict_obj


def make_dir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
