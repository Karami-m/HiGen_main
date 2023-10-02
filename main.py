import os
import sys
import traceback
from pprint import pprint
import numpy as np
import torch

from higen_runner import HiGeN_Runner
from utils.arg_helper import parse_arguments, get_config
from utils.logger import setup_logging

torch.set_printoptions(profile='full')

command_str = None
################################################################
## TEMP DEBUG
# command_str = "-c configs/higen_simple2.yaml --hp dist=mix_multinomial+Bern"
# command_str = "-c configs/higen_Enz.yaml --hp dist=mix_multinomial+Bern"
# command_str = "-c configs/higen_Ego2.yaml --hp dist=mix_multinomial+Bern"

################################################################

def make_config(command_str=command_str):
    args = parse_arguments(command_str)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = get_config(args.config_file, is_test=args.test,
                        tag='{}-{}'.format(args.hp, args.tag) if not args.tag is None else args.hp,
                        subdir=args.subdir,
                        hp_txt=args.hp,
                        gpu=args.gpu,
                        test_model_name=args.model_name,
                        draw_training=args.draw_training)
    config.use_amp = args.amp
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config._verbose = 3

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # to log some info
    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
    logger = setup_logging(args.log_level, log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance g_id = {}".format(config.run_id))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print("<>" * 30)
    pprint(config)
    print("<>" * 30)
    print('device', config.device)
    logger.info(f"Use AUTOMATIC MIXED PRECISION (AMP): {config.use_amp}")
    return config, args, logger


#####################################

def run_main(config, args, logger):

    runner = HiGeN_Runner(config)

    if args.draw_training:
        runner.store_test_dev()

    elif not args.test:
        runner.train()
    else:
        config.train.is_resume = False
        print("save dir : ", config.save_dir)
        runner.test(compute_metrics=None, save_graphs=True)


if __name__ == '__main__':
    config, args, logger = make_config(command_str=command_str)
    run_main(config, args, logger)
    sys.exit(0)