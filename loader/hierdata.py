import os
import logging
import numpy as np
from collections import Counter
import torch.utils.data as data
import torch

from .utils import prepro_node_name
from .img_flist import ImageFilelist
from .sampler import get_sampler
from .collate import get_collate_fn
from .transforms import get_transform

# Subsample a dataset.
# A relabeler should be provided, which is under the same director with cfg['hierachy'], with the name 'relabelr_{subsample}.npy'
# This Function will modify the dataset in-place, drop samples that are not in the relabeler, and relabel the targets according to the relabeler.
def subsample_dataset(dataset, cfg):
    subsample=cfg.get('subsample', 'all')
    subsample_valid_options = ['all', 'base', 'novel']
    assert subsample in subsample_valid_options, f'subsample should be in {subsample_valid_options}, got {subsample}'
    if subsample == 'all':
        print('No subsampling, using all classes.')
        return
    
    hierachy_path = cfg['hierarchy']
    # subhierachy_path = os.path.join(os.path.dirname(hierachy_path), os.path.basename(hierachy_path).replace('.npy', f'_{subsample}.npy'))
    submapping_path = os.path.join(os.path.dirname(hierachy_path), f'relabeler_{subsample}.npy')
    relabeler = np.load(submapping_path, allow_pickle=True).item()

    print(f"SUBSAMPLE {subsample.upper()} CLASSES!")

    new_data, new_target = [], []
    for impath, target, index in zip(dataset.data, dataset.target, dataset.indices):
        if target not in relabeler:
            continue
        new_data.append(impath)
        new_target.append(relabeler[target])

    dataset.data = new_data
    dataset.target = new_target
    dataset.indices = list(range(len(new_data)))
    
    return dataset

logger = logging.getLogger('mylogger')

def HierDataLoader(cfg, splits, batch_size):
    """Function to build data loader(s) for the specified splits given the parameters.
    """
    data_root = cfg.get('data_root', '/path/to/dataset')
    if not os.path.isdir(data_root):
        raise Exception('{} does not exist'.format(data_root))

    num_workers = cfg.get('n_workers', 4)

    smplr_dict = cfg.get('sampler', {'name': 'random'})

    data_loader = dict()
    for split in splits:

        bz = batch_size if 'train' in split else cfg.get('eval_batch_size', 128)
        aug = True if 'train' in split else False
        trans = get_transform(cfg.get('transform', 'imagenet'), aug)

        data_list = cfg.get(split, None)
        if not os.path.isfile(data_list):
            raise Exception('{} not available'.format(data_list))

        dataset = ImageFilelist(root_dir=data_root, flist=data_list, transform=trans)
        subsample_dataset(dataset, cfg)
        if 'train' in split:
            cls_num_list = []
            counter = Counter(dataset.target)
            n_class = len(counter)
            for i in range(n_class):
                cls_num_list.append(counter.get(i, 1e-7))
            data_loader['cls_num_list'] = np.asarray(cls_num_list)

        shuffle = True if 'train' in split else False
        drop_last = cfg.get('drop_last', False) if 'train' in split else False
        rot = cfg.get('rot', False) if 'train' in split else False
        collate_fn = get_collate_fn(rot)
        if ('train' in split) and (smplr_dict['name'] != 'random'):
            sampler = get_sampler(dataset, smplr_dict)
            data_loader[split] = data.DataLoader(
                dataset, batch_size=bz, sampler=sampler, shuffle=False, collate_fn=collate_fn,
                drop_last=drop_last, pin_memory=True, num_workers=num_workers
            )
        else:
            data_loader[split] = data.DataLoader(
                dataset, batch_size=bz,  sampler=None, shuffle=shuffle, collate_fn=collate_fn,
                drop_last=drop_last, pin_memory=True, num_workers=num_workers
            )

        logger.info("{split}: {size}".format(split=split, size=len(dataset)))
    hierachy_path = cfg['hierarchy']
    subsample=cfg.get('subsample', 'all')
    if subsample != 'all':
        hierachy_path = os.path.join(os.path.dirname(hierachy_path), os.path.basename(hierachy_path).replace('.npy', f'_{subsample}.npy'))
    
    print('loading tree from {}'.format(hierachy_path))

    # retrieve semantic information from the class hierarchy
    tree = np.load(hierachy_path, allow_pickle=True).tolist()
    
    # re-arange intnl-id to make sure that shallower intnl_nodes will have smaller intnl-id
    # (because in the training process, the intnl_nodes' childset is used by an ascending order of its intnl-id, we want to make sure that shallower intl_nodes's childset will never be accessed later than deepeer ones)
    sorted_intnl_nodes = sorted([tree.nodes.get(i) for i in tree.intnl_nodes.values()], key=lambda x:x.depth)
    tree.intnl_nodes = {i:node.name for i, node in enumerate(sorted_intnl_nodes)}
    tree._gen_sublabels()

    # get the parameter list of each internal node
    tree.gen_param_lists()
    n_intnl_node = len(tree.intnl_nodes)
    intnl_nodes = [tree.nodes.get(tree.intnl_nodes[i]).param_list for i in range(n_intnl_node)]
    intnl_depths = [tree.nodes.get(tree.intnl_nodes[i]).depth for i in range(n_intnl_node)]

    # get the parameter list of leaf nodes
    n_leaf_node = len(tree.leaf_nodes)
    #leaf_nodes是一个list，每个元素是一个node_id
    leaf_nodes = np.asarray([tree.get_nodeId(tree.leaf_nodes[i]) - 1 for i in range(n_leaf_node)])

    # get semantic names in the hierarchy
    nodes = sorted([v for v in tree.nodes.values()], key=lambda x: x.node_id)[1:]
    param_names = [prepro_node_name(x.name) for x in nodes]
    # import pdb; pdb.set_trace()
    # generate codewords for all the nodes
    tree.gen_codewords('class')    # for treecut
    codewords = np.asarray([x.codeword for x in nodes])

    # generate the dependence matrix of the internal nodes
    tree.gen_dependence()

    # generate parameter masks for internal nodes (这里的mask对应的似乎是文中的B^T)
    masks = -np.ones((n_intnl_node, len(param_names)))
    intnl_params = np.asarray([tree.get_nodeId(tree.intnl_nodes[i]) - 1 for i in range(n_intnl_node)])
    for i in range(n_intnl_node):
        # passed parameters
        leaf_idx = (tree.sublabels[:, i] >= 0)
        masks[i, leaf_nodes[leaf_idx]] = 1
        intnl_idx = (tree.dependence[:, i] == 1)
        masks[i, intnl_params[intnl_idx]] = 1
        # blocked parameters
        intnl_idx = (tree.dependence[i, 1:] == 1)
        masks[i, intnl_params[1:][intnl_idx]] = 0
    
    tree._gen_extra()

    data_loader['tree_info'] = {
        'dependence': torch.from_numpy(tree.dependence).float(),
        'masks': torch.from_numpy(masks).int(),
        'codewords': torch.from_numpy(codewords).int()
    }
    data_loader['sublabels'] = torch.from_numpy(tree.sublabels).float()
    data_loader['intnl_nodes'] = intnl_nodes
    data_loader['leaf_nodes'] = leaf_nodes
    data_loader['param_names'] = param_names
    data_loader['hlabels'] = torch.from_numpy(tree.hlabels).float()
    data_loader['nodes_per_depth'] = tree.depth_nodes
    data_loader['leafnodes2depth'] = torch.from_numpy(tree.leafnodes2depth).long()
    data_loader['leafnames'] = tree.leafnames
    data_loader['depth'] = tree.depth
    data_loader['intnl_depths'] = intnl_depths

    logger.info("Building data loader with {} workers".format(num_workers))

    # import pdb; pdb.set_trace()
    return data_loader


# unit-test
if __name__ == '__main__':
    # import pdb
    cfg = {
        'data_root': '/raw/cifar100',
        'train': '/data/cifar100/gt_train.txt',
        'val': '/data/cifar100/gt_valid.txt',
        'hierarchy': '/data/cifar100/tree.npy',
        'n_workers': 4,
        'rot': False,
        'sampler': {'name': 'class_balanced'},
        'transform': 'clip'
    }
    splits = ['train', 'val']
    data_loader = HierDataLoader(cfg, splits, batch_size=4)
    for (step, value) in enumerate(data_loader['train']):
        if len(value) > 3:
            img, label, index, rot_label = value
        else:
            img, label, index = value
        # pdb.set_trace()

