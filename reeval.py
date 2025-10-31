### reeval.py
# Evaluate Leaf Accuracy and HCA for a given checkpoint.
###
import argparse
import os
import pandas as pd
import yaml
from pathlib import Path
import sys
sys.path.append('loader')
import torch
from loader import get_dataloader
from models import get_model
from utils import get_logger
from engine import eval_one_epoch_detailed
from pprint import pprint

def main():

    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    os.system('echo $CUDA_VISIBLE_DEVICES')

    # setup random seed
    torch.manual_seed(cfg.get('seed', 1))
    torch.cuda.manual_seed(cfg.get('seed', 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # setup data loader
    splits = ['test']
    data_loader = get_dataloader(cfg['data'], splits, cfg['data']['batch_size'])
    # hierarchical information
    param_names = data_loader['param_names']
    leaf_nodes = data_loader['leaf_nodes']
    intnl_nodes = data_loader['intnl_nodes']
    sublabels = data_loader['sublabels'].cuda()
    hlabels = data_loader['hlabels'].cuda()
    nodes_per_depth = data_loader['nodes_per_depth']
    init_classname = None
    # change the initial label space
    if cfg.get("init_label_set", "leaf") == "leaf":
        init_classname = [param_names[i] for i in leaf_nodes]
    #used only by promptsrc
    df = pd.DataFrame(param_names)
    all_nodes_info = list(df.iloc[:][0])

    model = get_model(cfg['model'], init_classname, all_nodes_info=all_nodes_info).cuda()
    # load the best model for evaluation
    checkpoint = os.path.join(args.folder, "ckpt", args.eval_ckpt + ".pth")
    if os.path.isfile(checkpoint):
        model.load_custom_ckpt(checkpoint)
    else:
        print(f'checkpoint ({checkpoint}) not found!')
    model.eval()
    # setup model
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    eval_meters_detailed = eval_one_epoch_detailed(model, data_loader['test'], param_names, device, -1, cfg, args, leaf_nodes, intnl_nodes, sublabels, hlabels=hlabels, nodes_per_depth=nodes_per_depth)
    result_detailed = {x for x in eval_meters_detailed.items() if x[0][0] != 'n' and 'depth_' not in x[0]}
    print("Final result_detailed")
    pprint(result_detailed)
    logger.info(f"Final result_detailed: {result_detailed}" )
    logger.info(f"======")

if __name__ == '__main__':
    global cfg, args, logger, logdir
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--folder',
        type=str,
        help='Path to the folder',
    )
    parser.add_argument(
        '--subsample',
        type=str,
        default='all',
        choices=['base', 'novel', 'all'],
        help='Which classes to subsample',
    )
    parser.add_argument(
        '--eval_dataset',
        type=str,
        default='',
        help='Dataset for evaluation',
    )
    parser.add_argument(
        '--bz',
        type=int,
        default=1024,
        help='batch size',
    )
    parser.add_argument(
        '--debug',
        default=False,
        action="store_true",
        help='debug mode',
    )
    parser.add_argument(
        '--extra_zs_clip_ks',
        type=str,
        default=None,
        help='todo'
    )
    parser.add_argument(
        '--extra_pt_clip_ks',
        type=str,
        default=None,
        help='todo'
    )
    parser.add_argument(
        '--pretrained_pt_clip_dir',
        type=str,
        default=None,
        help='todo'
    )
    
    args = parser.parse_args()
    args.eval_ckpt = ''

    # check if there is any yml files in the folder
    yml_file = None
    for f in os.listdir(args.folder):
        if f[-3:] == "yml":
            assert yml_file is None, "Error, multi config file detected!"     
            yml_file = args.folder + "/" + f
        elif f == "ckpt":
            assert os.path.exists(os.path.join(args.folder, f, "last.pth"))
            if os.path.exists(os.path.join(args.folder, f, "best.pth")):
                args.eval_ckpt = "best"
            else:
                args.eval_ckpt = "last"

    #assert args.eval_ckpt is not None
    assert yml_file is not None

    with open(yml_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    cfg['data']['eval_batch_size'] = args.bz
    if args.eval_dataset != '':
        cfg['data']['data_root'] = os.path.join("./prepro/raw", args.eval_dataset)
        cfg['data']['test'] = os.path.join("./prepro/data", args.eval_dataset, "gt_test.txt")
        cfg['data']['hierarchy'] = os.path.join("./prepro/data", args.eval_dataset, "tree.npy")
        logdir = Path(args.folder) / args.eval_dataset
    else:
        logdir = Path(args.folder)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("RUNDIR: {}".format(logdir))

    logger = get_logger(logdir)
    logger.info("Start logging")


    ###
    cfg['data']['subsample'] = args.subsample
    cfg['extra_zs_clip_ks'] = args.extra_zs_clip_ks
    cfg['extra_pt_clip_ks'] = args.extra_pt_clip_ks
    cfg['pretrained_pt_clip_dir'] = args.pretrained_pt_clip_dir
    main()

