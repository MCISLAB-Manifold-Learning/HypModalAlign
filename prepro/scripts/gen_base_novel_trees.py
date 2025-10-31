from argparse import ArgumentParser
import os
import numpy as np
import sys
import math
import copy
import random
from collections import defaultdict

#add {project_root}/loader to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'loader'))
from treelibs import Tree, TreeNode

def balanced_leaf_sampling(tree):
    #debug
    leaf_nids = []
    parent_nids = []
    
    # 1. construct a mapping from father to children
    parent_leaf_map = defaultdict(list)
    for leaf_id, leaf_name in tree.leaf_nodes.items():
        leaf_node = tree.nodes[leaf_name]
        parent_name = leaf_node.parent
        parent_leaf_map[parent_name].append(leaf_name)
        #debug
        leaf_nid = leaf_node.node_id
        parent_nid = tree.nodes.get(parent_name).node_id
        leaf_nids.append(leaf_nid)
        parent_nids.append(parent_nid)

    # 2. calcluate the quota
    n = len(tree.leaf_nodes)
    base_quota = (n + 1) // 2  # ceil(n/2)
    
    # 3. Allocate initial quota to parent nodes (based on the ratio of leaf children)
    quotas = {}
    total_quota = 0
    for parent, leaves in parent_leaf_map.items():
        quotas[parent] = round(base_quota * len(leaves) / n)
        total_quota += quotas[parent]
    
     # 4. Dynamically adjust to strict equality
    diff = base_quota - total_quota
    if diff != 0:
        # Sort by the number of leaf children, prioritize adjusting large sets
        sorted_parents = sorted(parent_leaf_map, key=lambda p: len(parent_leaf_map[p]), reverse=True)
        for parent in sorted_parents:
            if diff > 0 and quotas[parent] < len(parent_leaf_map[parent]):
                quotas[parent] += 1
                diff -= 1
            elif diff < 0:
                quotas[parent] -= 1
                diff += 1
            if diff == 0:
                break
    
    # 5. sample based on the quota
    base_leaf_names = []
    for parent, leaves in parent_leaf_map.items():
        k = quotas[parent]
        selected = random.sample(leaves, k)
        base_leaf_names.extend(selected)
    base_leaf_names = sorted(base_leaf_names, key=lambda x:tree.nodes.get(x).node_id)
    return base_leaf_names


def show_tree_info(tree):
    print('----------------------------------Tree Information----------------------------------')
    print(f"Number of nodes: {len(tree.nodes)}")
    print(f"Number of internal nodes: {len(tree.intnl_nodes)}")
    print(f"Number of leaf nodes: {len(tree.leaf_nodes)}")
    print(f"Root node: {tree.root}")
    print("Internal nodes:")
    for nid, name in tree.intnl_nodes.items():
        print(f"  Node ID: {nid}, Name: {name}, Depth: {tree.nodes[name].depth}")
    print("Leaf nodes:")
    for nid, name in tree.leaf_nodes.items():
        print(f"  Node ID: {nid}, Name: {name}")
    print('----------------------------------------------------------------------------------')

def extract_sub_tree(tree, leaf_names):
    """
    Extract a sub-tree from the given tree, containing only the specified leaf nodes.
    """
    # identify the nodes to keep
    subtree = copy.deepcopy(tree)
    nodes_to_save = {}
    for leaf in leaf_names:
        node = tree.nodes.get(leaf)
        if node:
            nodes_to_save[node.name] = node
            while node.parent is not None:
                node = tree.nodes.get(node.parent)
                if node and node not in nodes_to_save:
                    nodes_to_save[node.name] = node
    
    # create a mapping from orignal node ids to new subtree node ids
    new_nid2name = {nid: node.name for nid, node in enumerate(sorted(nodes_to_save.values(), key=lambda x: x.node_id))}
    new_name2nid = {name: nid for nid, name in new_nid2name.items()}
    nid_mapping = {node.node_id: new_name2nid[node.name] for node in nodes_to_save.values()}

    # clear nodes
    for name in list(subtree.nodes.keys()):
        if name not in nodes_to_save:
            del subtree.nodes[name]

    assert len(subtree.nodes) == len(nodes_to_save), "Subtree nodes count mismatch"

    #clear nid2name dictionary
    subtree.nid2name = {}

    # clear children and nodes' attributes
    for node in subtree.nodes.values():
        new_children = {}
        for child in node.children.values():
            if child in nodes_to_save:
                new_children[len(new_children)] = child
                subtree.nodes[child].parent = node.name
                subtree.nodes[child].child_idx = len(new_children) - 1
        node.children = new_children
        node.node_id = nid_mapping[node.node_id] # reassign node id
        subtree.nid2name[node.node_id] = node.name
    
    intnl_nodes = sorted([name for name in subtree.intnl_nodes.values() if name in nodes_to_save], key=lambda x: subtree.nodes.get(x).depth)
    subtree.intnl_nodes = {i: name for i, name in enumerate(intnl_nodes)}

    subtree.leaf_nodes = {i:name for i, name in enumerate(leaf_names)}

    subtree._gen_sublabels()
    # subtree._gen_extra()
    origin_leafname2id = {v: k for k, v in tree.leaf_nodes.items()}
    leafid_mapping = {origin_leafname2id[leafname]:subid for subid, leafname in subtree.leaf_nodes.items()}
    return subtree, leafid_mapping



def parse_args():
    parser = ArgumentParser(description="Generate base novel trees")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of data")
    parser.add_argument("--seed", type=int, default=42, help="Directory of data")
    return parser.parse_args()

def main(args):
    random.seed(args.seed)
    tree_path = os.path.join(args.data_dir, "tree.npy")
    
    tree = np.load(tree_path, allow_pickle=True).tolist()

    base_leaf_names = balanced_leaf_sampling(tree)
    novel_leaf_names = [leaf for leaf in tree.leaf_nodes.values() if leaf not in base_leaf_names]

    print('Original Tree Information:')
    show_tree_info(tree)

    base_tree, base_mapping = extract_sub_tree(tree, base_leaf_names)
    print("Base Tree Information:")
    show_tree_info(base_tree)
    novel_tree, novel_mapping = extract_sub_tree(tree, novel_leaf_names)
    print("Novel Tree Information:")
    show_tree_info(novel_tree)
    np.save(os.path.join(args.data_dir, "tree_base.npy"), base_tree)
    np.save(os.path.join(args.data_dir, "relabeler_base.npy"), base_mapping)
    np.save(os.path.join(args.data_dir, "tree_novel.npy"), novel_tree)
    np.save(os.path.join(args.data_dir, "relabeler_novel.npy"), novel_mapping)






if __name__ == "__main__":
    args = parse_args()
    main(args)