### train.py
# Main script for training.
###
import argparse
import os
import yaml
import random
import shutil
from tqdm import tqdm
import sys
sys.path.append('loader')
import torch
from loader import get_dataloader
from models import get_model
from utils import get_logger, print_args, dotted_set, smart_convert
from optim.optimizer import build_optimizer
from optim.lr_scheduler import build_lr_scheduler
from tensorboardX import SummaryWriter
from engine import train_one_epoch, eval_one_epoch, eval_one_epoch_detailed
from pprint import pprint
import random
import pandas as pd
import ast

def main():
    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')
    os.system('echo $CUDA_VISIBLE_DEVICES')

    # print args
    print_args(args, cfg)
    if "eval_freq" not in cfg:
        cfg["eval_freq"] = -1

    random.seed(cfg.get('seed', 1))
    os.environ['PYTHONHASHSEED'] = str(cfg.get('seed', 1))
    torch.cuda.manual_seed_all(cfg.get('seed', 1))
    
    # setup random seed
    torch.manual_seed(cfg.get('seed', 1))
    torch.cuda.manual_seed(cfg.get('seed', 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ['train', 'test']
    data_loader = get_dataloader(cfg['data'], splits, cfg['data']['batch_size'])
    # hierarchical information
    param_names = data_loader['param_names']
    sublabels = data_loader['sublabels'].cuda()
    leaf_nodes = data_loader['leaf_nodes']
    intnl_nodes = data_loader['intnl_nodes']
    depth = data_loader['depth']
    hlabels = data_loader['hlabels'].cuda()
    nodes_per_depth = data_loader['nodes_per_depth']
    intnl_depths = data_loader['intnl_depths']

    # setup model
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # change the initial label space
    init_classname = None
    if cfg["init_label_set"] == "leaf":
        init_classname = [param_names[i] for i in leaf_nodes]

    treecut_generator = None
    if cfg.get('treecut_generator', None) is not None:
        model_dict = cfg['treecut_generator']
        model_dict.update(data_loader['tree_info'])
        treecut_generator = get_model(model_dict).cuda()
    
    #used only by promptsrc
    df = pd.DataFrame(param_names)
    all_nodes_info = list(df.iloc[:][0])

    model = get_model(cfg['model'], init_classname, all_nodes_info=all_nodes_info).cuda()

    # setup optimizer
    optim_param = []
    optim_name = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            optim_param.append(param)
            optim_name.add(name)

    print(f"Parameters to be updated: {optim_name}")
    optim = build_optimizer(optim_param, cfg['optim'])
    sched = build_lr_scheduler(optim, cfg['optim'])

    # training epochs
    best_result = 0
    for epoch in tqdm(range(cfg["optim"]["max_epoch"])):
        model.train()
        if treecut_generator is not None:
            train_meters = train_one_epoch(model, optim, sched, data_loader['train'], param_names, device, epoch, cfg, args, treecut_generator, leaf_nodes, intnl_nodes, sublabels, max_depth=depth, intnl_depths=intnl_depths)
        else:
            train_meters = train_one_epoch(model, optim, sched, data_loader['train'], param_names, device, epoch, cfg, args, leaf_nodes=leaf_nodes)

        # logging for training
        curr_lr = optim.param_groups[0]['lr']
        writer.add_scalar('train/lr', curr_lr, epoch + 1)
        logger.info(f"=== Epoch {epoch} done !! ===")
        for k in train_meters:
            writer.add_scalar(f"train/{k}", train_meters[k], epoch + 1)
            if k[0] == 'n':
                continue
            print(f"Average result of {k} over epoch: " + str(train_meters[k]))
            logger.info(f"Average result of {k} over epoch: " + str(train_meters[k]))
        logger.info(f"======")

        if cfg["eval_freq"] != -1 and (epoch + 1) % cfg["eval_freq"] == 0:
            model.eval()
            eval_meters = eval_one_epoch(model, data_loader['test'], param_names, device, epoch, cfg, args, leaf_nodes, intnl_nodes, sublabels)
            # logging for training
            logger.info(f"=== Evaluating {epoch}  !! ===")
            for k in eval_meters:
                writer.add_scalar(f"val/{k}", eval_meters[k], epoch + 1)
                if k[0] == 'n':
                    continue
                print(f"Average result of {k} on the test set: " + str(eval_meters[k]))
                logger.info(f"Average result of {k} on the test set: " + str(eval_meters[k]))

            result = (eval_meters["acc"] + eval_meters["consistency"]) / 2

            print(f"Best result : {best_result} ; Epoch result : {result}" )
            logger.info(f"Best result : {best_result} ; Epoch result : {result}" )

            if result >= best_result:
                best_result = result
                print("Saving the best model")
                # save the best model
                torch.save(model.state_dict(), os.path.join(logdir, "ckpt", "best.pth"))
        torch.save(model.state_dict(), os.path.join(logdir, "ckpt", "last.pth"))
        if args.save_frequency is not None:
            if epoch % args.save_frequency == args.save_frequency - 1:
                torch.save(model.state_dict(), os.path.join(logdir, "ckpt", f"epoch{epoch}.pth"))

        logger.info(f"======")

        # debug mode
        if args.debug:
            break

    # load the best model for evaluation
    if os.path.exists(os.path.join(logdir, "ckpt", "best.pth")):
        logger.info(f"=== Evaluating Best model  !! ===")
        model.load_state_dict(torch.load(os.path.join(logdir, "ckpt", "best.pth")))
    elif os.path.exists(os.path.join(logdir, "ckpt", "last.pth")):
        logger.info(f"=== Evaluating Last model  !! ===")
        model.load_state_dict(torch.load(os.path.join(logdir, "ckpt", "last.pth")))
    else:
        logger.info(f"=== Evaluating without training  !! ===")
        pass
    model.eval()
    eval_meters = eval_one_epoch(model, data_loader['test'], param_names, device, -1, cfg, args, leaf_nodes, intnl_nodes, sublabels)
    result = {x for x in eval_meters.items() if x[0][0] != 'n'}

    # print(f"Final result : {result}" )
    print("Final result")
    pprint(result)
    logger.info(f"Final result : {result}" )
    logger.info(f"======")

    eval_meters_detailed = eval_one_epoch_detailed(model, data_loader['test'], param_names, device, -1, cfg, args, leaf_nodes, intnl_nodes, sublabels, hlabels=hlabels, nodes_per_depth=nodes_per_depth)
    result_detailed = {x for x in eval_meters_detailed.items() if x[0][0] != 'n' and 'depth_' not in x[0]}
    print("Final result_detailed")
    pprint(result_detailed)
    logger.info(f"Final result_detailed: {result_detailed}" )
    logger.info(f"======")


def merge_args_to_config():
    assert len(args.opts) % 2 == 0
    for k,v in zip(args.opts[::2], args.opts[1::2]):
        v=smart_convert(v)
        original = dotted_set(cfg, k, v)
        print(f'[attr {k}]:{original}=>{v}  ' + ',warning' if original is None else '')
    
    if isinstance(cfg['data']['sampler']['nshot'], float):
        cfg['data']['sampler']['nshot'] = int(cfg['data']['sampler']['nshot'])

def modify_config():
    cfg['model']['max_epoch'] = cfg['optim']['max_epoch']

    if cfg['model'].get('selected_layer_indices', None) is not None:
        if isinstance(cfg['model']['selected_layer_indices'], str):
            print("eval raw str <selected_layer_indices>")
            cfg['model']['selected_layer_indices'] = ast.literal_eval(
                cfg['model']['selected_layer_indices']
            )
        assert isinstance(cfg['model']['selected_layer_indices'], (list, tuple))
        

if __name__ == '__main__':
    global cfg, args, writer, logger, logdir
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        help='Configuration file to use',
    )
    parser.add_argument(
        '--debug',
        action = "store_true",
        default=False,
        help='Run in debug mode',
    )
    parser.add_argument(
        '--trial',
        type=str,
        help='different trial of the exp',
    )
    parser.add_argument(
        '--save_frequency',
        type=int,
        default=None,
        help='save after the specified number of epochs. save no intermediate epochs by default.',
    )
    parser.add_argument(
        '--show_crossattn_weights',
        action='store_true',
        help='whether to show cross attention weights during training',
    )

    # add the argument to dynamically add parameters from cmd instructions. (inspired by CoOp's repo)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )


    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if args.debug:
        outdir = 'runs/debug'
    else:
        outdir = 'runs'

    merge_args_to_config()

    modify_config()

    logdir = os.path.join(outdir, cfg['data']['name'], cfg['model']['arch'], cfg['exp'], "trial_" + args.trial)

    if not os.path.exists(os.path.join(logdir, "ckpt")):
        os.makedirs(os.path.join(logdir, "ckpt"))

    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))

    # write config file to experiment directory
    config_basename = os.path.basename(args.config)
    backup_path = os.path.join(logdir, f"{config_basename}.rawcopied")
    shutil.copy(args.config, backup_path)
    filename, ext = os.path.splitext(config_basename)
    modified_path = os.path.join(logdir, f"{filename}-merged{ext}")
    with open(modified_path, 'w') as f:
        yaml.dump(cfg, f, indent=4, default_flow_style=False, sort_keys=False)
    
    logger = get_logger(logdir)
    logger.info("Start logging")

    main()
