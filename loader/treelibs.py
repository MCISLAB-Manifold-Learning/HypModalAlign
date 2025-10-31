### treelibs.py
# Classes for building/analyzing tree hierarchy and hierarchical parameterization.
###

import numpy as np
import os
from termcolor import colored
from datasets import load_dataset
from collections import defaultdict, deque
from tqdm import tqdm
import re


class TreeNode():

    def __init__(self, name, path, depth, node_id, child_idx=-1, parent=None):
        self.name = name                # node name
        self.path = path                # dataset path to this node
        self.depth = depth              # depth of this node
        self.node_id = node_id          # node index in the tree
        self.children = {}              # list of children names
        self.child_idx = child_idx      # child index of its parent
        self.param_list = []            # list of parameter indices for classification at this node
        self.parent = parent            # name of the parent node
        self.codeword = None            # codeword for hierarchical parameterization

    def add_child(self, child):
        self.children[len(self.children)] = child

    def init_codeword(self, cw_size):
        self.codeword = np.zeros([cw_size])

    def set_codeword(self, idx):
        self.codeword[idx] = 1

    def __str__(self):
        attr = 'name={}, node_id={}, depth={}, children={}'.format(
                    self.name, self.node_id, self.depth,
                    ','.join([chd for chd in self.children.values()])
                )
        return  attr

    def copy(self):
        new_node = TreeNode(self.name, self.path, self.depth, self.node_id, self.child_idx, self.parent)
        new_node.children = self.children.copy()
        new_node.codeword = self.codeword
        if self.param_list:
            new_node.param_list = self.param_list.copy()
        return new_node


class Tree():

    def __init__(self, data_root):
        """ Build a tree based on the dataset folder hierarachy.
            data_root: the root directory of the hierarchical dataset
        """
        self.root = TreeNode('root', data_root, 0, 0)
        self.depth = 0                          # the maximum depth of the tree (initialized as 0)
        self.nodes = {'root': self.root}        # list of all the nodes in the tree
        self.nid2name = {0: 'root'}             # a mapping from node index to node name

        # build tree
        self._buildTree(self.root)

        # generate dictionary of internal (non-leaf) nodes (including the root node)
        intnl_nodes = sorted([v for v in self.nodes.values() if len(v.children) > 0], key=lambda x: x.node_id)
        self.intnl_nodes = {i: x.name for i, x in enumerate(intnl_nodes)} # a mapping from intnl_node_id to node name

        # generate dictionary of leaf nodes
        leaf_nodes = sorted([v for v in self.nodes.values() if len(v.children) == 0], key=lambda x: x.node_id)
        self.leaf_nodes = {i: x.name for i, x in enumerate(leaf_nodes)}   # a mapping from leaf_node_id to node name

        # a node is either an internal node or a leaf node
        assert len(self.leaf_nodes) + len(self.intnl_nodes) == len(self.nodes)
        # a mapping from leaf node labels to sublabels for each internal node
        self._gen_sublabels()
        # self._gen_hieids()
        self._gen_extra()

    def _buildTree(self, root, depth=0):
        """ Traverse the root directory to build the tree (starting with depth=0).
            root: node to be used as the root
        """

        for chd in os.listdir(root.path):
            chd_path = os.path.join(root.path, chd)

            # if this child is a node (internal/leaf), then add it to the tree
            if os.path.isdir(chd_path):
                assert chd not in self.nodes
                child_idx = len(root.children)
                root.add_child(chd)
                node_id = len(self.nodes)
                child = TreeNode(chd, chd_path, depth + 1, node_id, child_idx, parent=root.name)
                self.nodes[chd] = child
                self.nid2name[node_id] = chd

                # keep traverse its children
                self._buildTree(child, depth + 1)

        self.depth = max(self.depth, depth)

    def _gen_sublabels(self):
        """ Generate sublabels for each internal nodes.
        """
        self.sublabels = -np.ones([len(self.leaf_nodes), len(self.intnl_nodes)])
        name2inid = {v: k for k, v in self.intnl_nodes.items()}
        # generate sublabels for each leaf node class
        for leaf_id, name in self.leaf_nodes.items():
            node = self.nodes.get(name)
            parent = node.parent
            while parent:
                parent_inid = name2inid.get(parent)
                self.sublabels[leaf_id, parent_inid] = node.child_idx
                node = self.nodes.get(parent)
                parent = node.parent
    
    def _gen_extra(self):
        """ hieids实际上就是从根节点到每一个节点的路径,不包括根节点，因为根节点无意义。
        """
        self.hieids = -np.ones([len(self.leaf_nodes), self.depth + 1])
        # hieids[i, j]表示根节点到第i个叶子结点的路径上第j层的节点id (root开始为第0层)   
        self.hlabels = -np.ones([len(self.leaf_nodes), self.depth + 1])
        # hlabels[i, j]表示根节点到第i个叶子结点的路径上第j层的节点在该层的节点列表-self_depth_nodes[j]中的索引
        self.depth_nodes = [[] for _ in range(self.depth + 1)]
        #self.depth_nodes[i, :]表示第i层的节点id列表 （层数从root开始为第0层）

        for node in self.nodes.values():
            self.depth_nodes[node.depth].append(node.node_id)

        self.depth_nodes = [sorted(i) for i in self.depth_nodes]#每一个depth的nodes列表升序排列一下(其实顺序无所谓，只要保证hlabel对应上就行)

        for leaf_id, name in self.leaf_nodes.items():
            stack = []
            #对于每一个节点，获取其到根节点的路径
            tmpnode = self.nodes.get(name)
            while tmpnode.node_id != 0:
                stack.append(tmpnode)
                tmpnode = self.nodes.get(tmpnode.parent)
            stack.append(tmpnode) #将根节点也加入路径
            depth = 0
            while len(stack) > 0:
                tmpnode = stack.pop()
                #记录leaf结点的路径与标签
                self.hieids[leaf_id, depth] = tmpnode.node_id
                try:
                    self.hlabels[leaf_id, depth] = self.depth_nodes[depth].index(tmpnode.node_id)
                except:
                    print(f'error: name-{name},leafid-{leaf_id}, curid-{tmpnode.node_id}, depth-{depth}, lst-{self.depth_nodes[depth]}')
                    exit(0)
                depth = depth + 1

        self.leafnodes2depth = np.array([self.nodes.get(leaf_name).depth for leaf_name in self.leaf_nodes.values()])
        self.leafnames = [self.leaf_nodes.get(i) for i in range(len(self.leaf_nodes))]

        self.intnlid2depth = np.array([self.nodes.get(self.intnl_nodes[i]).depth for i in range(len(self.intnl_nodes))] )

    def show(self, node_name='root', root_depth=-1, max_depth=np.Inf, cls_alias=None):
        """ Display the sub-tree architecture under the specified root within the max_depth.
            node_name: the name of the root node to display
            root_depth: Just for recursive calls; no need to specify root_depth when calling
                this function externally.
            max_depth: the maximum depth to display
            cls_alias: a mapping from the class (leaf node) name to the alias
        """

        root = self.nodes.get(node_name, None)
        if not root:
            raise ValueError('{} is not in the tree'.format(node_name))

        if root_depth == -1:
            if cls_alias is not None and len(root.children) == 0: # leaf node
                print(colored('{} ({})'.format(root.name, cls_alias[root.name]), 'red'))
            else:
                print(root.name)
            root_depth = root.depth
            max_depth = min(self.depth, max_depth)

        if root.depth - root_depth < max_depth:
            for chd in root.children.values():
                child = self.nodes[chd]
                print('--' * (child.depth - root_depth), end='')
                if cls_alias is not None and len(child.children) == 0: # leaf node
                    print(colord('{} ({})'.format(child.name, cls_alias[child.name]), 'red'))
                else:
                    print(child.name)
                self.show(chd, root_depth, max_depth)
    #codewords: Each node has a codeword vector. The length of this vector equals the number of leaf nodes. A position marked '1' in the vector indicates that the corresponding leaf node is a descendant of the current node.
    def gen_codewords(self, cw_type='class'):
        """ Generate codewords for all the nodes.
            cw_type: choice = ['class', 'param-td', 'param-bu']
                class: Codewords encoding the classes.
                param-td: Top-down codewords encoding the parameters.
                param-bu: Bottom-up codewords encoding the parameters.
        """
        if cw_type == 'class':
            # leaf nodes
            n_leaf_node = len(self.leaf_nodes)
            for leaf_id, name in self.leaf_nodes.items():
                node = self.nodes.get(name)
                node.init_codeword(n_leaf_node)
                node.set_codeword(leaf_id)

            # internal nodes
            for inid, name in self.intnl_nodes.items():
                node = self.nodes.get(name)
                node.codeword = (self.sublabels[:, inid] >= 0).astype(int)

        elif cw_type == 'param-td' or cw_type == 'param-bu':
            num_nodes = len(self.nodes)
            for node_id, name in self.nid2name.items():
                node = self.nodes.get(name)
                node.init_codeword(num_nodes - 1)
                if node_id > 0:
                    node.set_codeword(node_id - 1)
                    if cw_type=='param-td':
                        # inherent the codeword from the parent node
                        parent_cw = self.nodes.get(node.parent).codeword
                        node.codeword += parent_cw

    def gen_param_lists(self):
        """ Generate parameter list for each internal nodes.
        """
        # self.param_list = []
        for name in self.intnl_nodes.values():
            node = self.nodes.get(name)
            for chd in node.children.values():
                param_idx = self.nodes.get(chd).node_id - 1
                node.param_list.append(param_idx)

    def gen_dependence(self):
        """ Generate dependence matrix for internal nodes.
        """
        self.dependence = np.eye(len(self.intnl_nodes))
        name2inid = {v: k for k, v in self.intnl_nodes.items()}
        for inid, name in self.intnl_nodes.items():
            node = self.nodes.get(name)
            parent = node.parent
            while parent:
                parent_inid = name2inid.get(parent)
                self.dependence[inid, parent_inid] = 1
                node = self.nodes.get(parent)
                parent = node.parent

    def get_class(self, node_name='root', verbose=False):
        root = self.nodes.get(node_name, None)
        if not root:
            raise ValueError('{} is not in the tree'.format(node_name))

        def traverse(root):
            for chd in root.children.values():
                child = self.nodes[chd]
                if len(child.children) == 0:
                    class_list.append(child.name)
                traverse(child)

        class_list = []
        if len(root.children) == 0:
            class_list.append(root.name)
        else:
            traverse(root)
        if verbose:
            print(node_name, class_list)
        return class_list

    def get_nodeId(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        return node.node_id

    def get_parent(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        return node.parent

class TreeForRareSpecies(Tree):
    """ A tree class for the rare species dataset.
    """
    def _buildTree(self, root):
        self.path_to_display_name = {}
        dataset_path = root.path
        keys = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        
        # 1. load dataset
        dataset = load_dataset(dataset_path)
        self.name_to_children = defaultdict(set)
        
        # construct path
        for data in tqdm(dataset['train'], desc="construct path"):
            parent = 'root'
            current_path = []
            
            for key in keys:
                name = data[key].lower()
                current_path.append(name)
                path_str = '_'.join(current_path)
                
                self.name_to_children[parent].add(path_str)
                parent = path_str
        
        # build tree recursively
        self._buildTreeRecur(self.root)

    def _buildTreeRecur(self, root, depth=0):
        print(f'buildnode:{root.name}')
        for chd in self.name_to_children[root.name]:
            child_idx = len(root.children)
            root.add_child(chd)
            node_id = len(self.nodes)
            child = TreeNode(chd, None, depth + 1, node_id, child_idx, parent=root.name)
            self.nodes[chd] = child
            self.nid2name[node_id] = chd

            # keep traverse its children
            self._buildTreeRecur(child, depth + 1)

        self.depth = max(self.depth, depth)


if __name__ == '__main__':
    tree_sun = np.load('../prepro/data/sun/tree.npy', allow_pickle=True).tolist()
    tree_imagenet = np.load('../prepro/data/imagenet/tree.npy', allow_pickle=True).tolist()
    tree_cifar100 = np.load('../prepro/data/cifar100/tree.npy', allow_pickle=True).tolist()
    tree_sun._gen_extra()
    tree_imagenet._gen_extra()
    tree_cifar100._gen_extra()
    
    from utils import prepro_node_name
    import pandas as pd
    from collections.abc import Iterable as Iterable
    def show_name(id_lists, tree):
        nodes = sorted([v for v in tree.nodes.values()], key=lambda x: x.node_id)[1:]
        param_names = [prepro_node_name(x.name) for x in nodes]
        df = pd.DataFrame(["root"] + param_names)
        if not isinstance(id_lists[0], (Iterable)):
            id_lists = [i for i in id_lists if i >= 0]
            return df.iloc[id_lists][0].tolist()
        else:
            return [show_name(i, tree) for i in id_lists]
    import pdb; pdb.set_trace()
    print('depth of trees: sun={}, imagenet={}, cifar100={}'.format(tree_sun.depth, tree_imagenet.depth, tree_cifar100.depth))
    print('------------------------------------------')
    
    # print('leaf_nodes的id刚好和label是对应的:')
    # for idx in tree_sun.leaf_nodes:
    #     print(idx, tree_sun.leaf_nodes[idx])
    
    # #统计深度信息
    # depth_list_dict = {}
    # for mode in ['leaf-only', 'all']:
    #     for name, tree in {'sun':tree_sun, 'imagenet':tree_imagenet, 'cifar100':tree_cifar100}.items():
    #         depth_list = []
    #         if mode == 'leaf-only':
    #             for idx in tree.leaf_nodes:
    #                 depth = tree.nodes.get(tree.leaf_nodes[idx]).depth
    #                 depth_list.append(depth)
    #         elif mode == 'all':
    #             for node in tree.nodes.values():
    #                 depth_list.append(node.depth)
    #         else:
    #             raise ValueError('error')
    #         depth_list_dict[f'{name}-{mode}'] = depth_list
    # print(depth_list_dict.keys())
    # #绘制直方图
    # import math
    # import matplotlib.pyplot as plt
    # n = len(depth_list_dict)
    # cols = math.ceil(math.sqrt(n))  # 列数为平方根向上取整
    # rows = math.ceil(n / cols)     # 行数为总数量除以列数向上取整
    # fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
    # if n == 1:
    #     axs = [axs]
    # for i, ((name, lst), ax) in enumerate(zip(depth_list_dict.items(), axs.flatten())):
    #     # 绘制直方图
    #     ax.hist(lst, bins=range(min(lst), max(lst) + 2), edgecolor='black', align='left')
    #     # 设置子图标题和标签
    #     ax.set_title(f'Histogram of {name}')
    #     ax.set_xlabel('Depth')
    #     ax.set_ylabel('Frequency')
    # # 如果有空余的子图，隐藏它们
    # for j in range(i + 1, len(axs.flatten())):
    #     axs.flatten()[j].axis('off')
    # # 调整子图之间的间距
    # plt.tight_layout()
    # # 显示图表
    # plt.savefig('./depth_statistics.png')
    
    #查看leafnodes2depth
    # tree_sun._gen_hieids()
    # print(tree_sun.leafnodes2depth)

    # leafnodes的text和df里是不一样的
    # from utils import prepro_node_name
    # import pandas as pd
    # nodes = sorted([v for v in tree_sun.nodes.values()], key=lambda x: x.node_id)[1:]
    # param_names = [prepro_node_name(x.name) for x in nodes]
    # df = pd.DataFrame(param_names)
    # print(tree_sun.leaf_nodes)
    # print(df)

    # tree = tree_cifar100
    # print(tree.nodes.get(tree.root.children[0]).depth)
    # from utils import prepro_node_name
    # import pandas as pd
    # nodes = sorted([v for v in tree.nodes.values()], key=lambda x: x.node_id)[1:]
    # param_names = [prepro_node_name(x.name) for x in nodes]
    # df = pd.DataFrame(param_names)
    # print('df', df)
    # tree._gen_hieids()
    # import pdb; pdb.set_trace()

    # print('nodes_id of every depth:')
    # print('\n'.join([str(i) for i in tree.depth_nodes]))

    # print('nodes_name of every depth:')
    # # depth_node_names = [[tree.nid2name[j + 1] for j in i] for i in tree.depth_nodes]
    # depth_node_names = [list(df.iloc[i][0]) for i in tree.depth_nodes]
    # print('\n'.join([str(i) for i in depth_node_names]))

    # print('num of nodes at every depth:', [len(i) for i in tree.depth_nodes])

    # print('------------------------------------------')
    # num_to_show = 10

    # print('showing 10 hieids(paths from root to nodes):\n', tree.hieids[:10])
    # print('text version:', [list(df.iloc[i][0]) for i in tree.hieids[:10]])
    # print(tree.hlabels)

