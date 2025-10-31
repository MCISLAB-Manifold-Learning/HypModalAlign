### engine.py
# Functions for train/test an epoch.
###
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import averageMeter, multilabel_accuracy, accuracy, hierConsistency
import manifolds.lorentz as L
import yaml

from models.my_zsclip import ZeroShotCLIP
import copy
from utils import load_extra_pt_model, compute_modified_logits



# train one single epoch
def train_one_epoch(model, optimizer, sched, data_loader, param_names, device, epoch, cfg, args, treecut_generator=None, \
        leaf_nodes=None, intnl_nodes=None, sublabels=None ,*, max_depth=None, dim=512, intnl_depths=None):
    random_cut = True if treecut_generator is not None else False

    need_align = cfg['model']['hierachical'] and cfg['loss'].get('align_weight', 0) > 0
    if need_align:
        # check params needed for tree aligning loss
        assert all([i is not None for i in [max_depth, dim]])

    # setup average meters
    meter = {}
    meter["loss"] = averageMeter()
    meter["acc"] = averageMeter()
    if intnl_nodes is not None:
        meter["loss_node"] = averageMeter()
        meter.update({f'n{i}_acc': averageMeter() for i in range(len(intnl_nodes))})
        if cfg['data']['sampler']['name'] == 'nshot':
            meter['consistency'] = hierConsistency(data_loader.sampler.sampled_idx, len(data_loader.dataset.indices))
        else:
            meter['consistency'] = hierConsistency(data_loader.dataset.indices)

    print_freq = cfg["print_freq"]
    if cfg['model']['name'] in ['RN50', 'RN101']:
        model.eval()
    else:
        model.train()
    if random_cut:
        treecut_generator.train()

    df = pd.DataFrame(param_names)
    leaf_node_all_names = list(df.iloc[leaf_nodes][0])
    curv_i, curv_t, curv_m = None, None, None # print
    # train 1 epoch
    for (step, value) in tqdm(enumerate(data_loader)):
        image = value[0].cuda()
        target = value[1].cuda()
        index = value[2].cuda()
        bz = target.shape[0]
        model.model.clamp_curv()

        # classification at each internal node
        loss_node = 0
        loss_align = 0
        if intnl_nodes is not None:
            loss_node = []
            if need_align:
                text_tree_nodes = torch.zeros(bz, max_depth, dim, device=image.device, dtype=torch.half)
                image_tree_nodes = torch.zeros(bz, max_depth, dim, device=image.device, dtype=torch.half)
                nums = torch.zeros(bz, dtype=torch.long, device=image.device)

            for i, x in enumerate(intnl_nodes):
                tgt_mapping = sublabels[:, i]
                ntarget = tgt_mapping[target].long()
                idx = (ntarget >= 0)
                if sum(idx) > 0:
                    # forward
                    ntarget, nindex = ntarget[idx], index[idx]
                    param_idx = torch.tensor(x).cuda()
                    text = list(df.iloc[param_idx.cpu()][0])
                    ret = model(image[idx], text)

                    if need_align:
                        nlogits, image_features, text_features = [ret[k] for k in ['logits_per_image', 'image_features', 'text_features']]
                        text_tree_nodes[idx, nums[idx]] = text_features[ntarget]
                        image_tree_nodes[idx, nums[idx]] = image_features[torch.arange(len(ntarget)), ntarget]
                        nums[idx] += 1
                    else:
                        nlogits = ret['logits_per_image']
                    
                    # compute loss and accuracy
                    nloss = F.cross_entropy(nlogits, ntarget)
                    loss_node.append(nloss)
                    nconf, npred = torch.softmax(nlogits, dim=-1).max(dim=-1)
                    niscorrect = (npred == ntarget)
                    nacc = niscorrect.float().mean() * 100.0
                    meter[f'n{i}_acc'].update(nacc.item(), sum(idx))
                    meter['consistency'].update(nindex, niscorrect)
            loss_node = torch.stack(loss_node).mean()
            meter['loss_node'].update(loss_node.item(), bz)
            if need_align:
                curv_i, curv_t = model.model.curv_i.exp(), model.model.curv_t.exp()
                #calculate curvature of the image-text manifold if needed
                curv_version = cfg['model'].get('curv_version', 'learn_common')
                r_image, r_text = None, None
                if curv_version in ['learn2derive1', 'fixed2derive1']:                       
                    image_mean_euc = image_tree_nodes.detach().mean(dim=(0,1)).unsqueeze(0)
                    text_mean_euc = text_tree_nodes.detach().mean(dim=(0,1)).unsqueeze(0)
                    r_image = torch.norm(image_mean_euc)
                    r_text = torch.norm(text_mean_euc)
        
                curv_m = model.model.get_curv_m(r_image=r_image, r_text=r_text)
                scaling_factor_i = cfg["loss"].get("scaling_factor_i", 1.)
                scaling_factor_t = cfg["loss"].get("scaling_factor_t", 1.)
                loss_align = L.tree_aligning_loss(text_tree_nodes, image_tree_nodes, curv_i=curv_i, curv_t=curv_t, curv_m = curv_m, scaling_factor_i=scaling_factor_i, scaling_factor_t=scaling_factor_t)
            
        if random_cut:
            # classification at leaf nodes with random cut
            label_set, tgt_mapping = treecut_generator.get_randomcut()
            param_idx = torch.where(label_set)[0]
            ltarget = tgt_mapping[target]
            text = list(df.iloc[param_idx.cpu()][0])
            extra = {'text_ids':param_idx} if cfg['model']['arch'] == 'promptsrc' else {}
            output = model(image, text, **extra)
        else:
            output = model(image)
            ltarget = target

        leaf_logits = output['logits_per_image']
        n_classes = leaf_logits.shape[1]
        if intnl_nodes is not None:
            conf, pred = torch.softmax(leaf_logits, dim=-1).max(dim=-1)
            iscorrect = (pred == ltarget)
            meter['consistency'].update(index, iscorrect)

        assert cfg["loss"]["name"] == "ce", f"{cfg['loss']['name']} not implemented."
        loss = F.cross_entropy(leaf_logits, ltarget)
        acc = accuracy(leaf_logits, ltarget)[0].item()

        if "loss_scl" in output:
            loss += output["loss_scl"]

        loss *= cfg['loss'].get('mta_weight', 1)

        loss += cfg['loss'].get('lambda', 0) * loss_node

        loss += cfg['loss'].get('align_weight', 0) * loss_align


        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # lr update
        if cfg["optim"].get("use_scheduler", True) == True:
            sched.step()

        # update meters
        meter["loss"].update(loss.item(), bz)
        meter["acc"].update(acc, bz)

        # print information
        if (step+1) % print_freq == 0 or args.debug:
            print(f"[Train] Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Acc: {acc}")

    if cfg['model']['arch'] == 'promptsrc':
        model.gauss_fusion()

    meter = {k:meter[k].avg for k in meter.keys()}
    return meter



# eval one single epoch
@torch.no_grad()
def eval_one_epoch(model, data_loader, param_names, device, epoch, cfg, args, leaf_nodes=None, intnl_nodes=None, sublabels=None):
    # setup average meters
    meter = {}
    meter["loss"] = averageMeter()
    meter["acc"] = averageMeter()
    if intnl_nodes is not None:
        meter["loss_node"] = averageMeter()
        meter.update({f'n{i}_acc': averageMeter() for i in range(len(intnl_nodes))})
        meter['consistency'] = hierConsistency(data_loader.dataset.indices)

    print_freq = cfg["print_freq"]
    model.eval()
    df = pd.DataFrame(param_names)

    for (step, value) in tqdm(enumerate(data_loader)):
        image = value[0].cuda()
        target = value[1].cuda()
        index = value[2].cuda()
        bz = target.shape[0]

        # classification at each internal node
        loss_node = 0
        if intnl_nodes is not None:
            loss_node = []
            for i, x in enumerate(intnl_nodes):
                tgt_mapping = sublabels[:, i]
                ntarget = tgt_mapping[target].long()
                idx = (ntarget >= 0)
                if sum(idx) > 0:
                    # forward
                    ntarget, nindex = ntarget[idx], index[idx]
                    param_idx = torch.tensor(x).cuda()
                    text = list(df.iloc[param_idx.cpu()][0])
                    nlogits = model(image[idx], text)['logits_per_image']

                    # compute loss and accuracy
                    nloss = F.cross_entropy(nlogits, ntarget)
                    loss_node.append(nloss)
                    nconf, npred = torch.softmax(nlogits, dim=-1).max(dim=-1)
                    niscorrect = (npred == ntarget)
                    nacc = niscorrect.float().mean() * 100.0
                    meter[f'n{i}_acc'].update(nacc.item(), sum(idx))
                    meter['consistency'].update(nindex, niscorrect)
            loss_node = torch.stack(loss_node).mean()
            meter['loss_node'].update(loss_node.item(), bz)

        # classification at leaf nodes
        text = list(df.iloc[leaf_nodes][0])
        output = model(image, text)

        leaf_logits = output['logits_per_image']
        n_classes = leaf_logits.shape[1]
        if intnl_nodes is not None:
            conf, pred = torch.softmax(leaf_logits, dim=-1).max(dim=-1)
            iscorrect = (pred == target)
            meter['consistency'].update(index, iscorrect)

        if cfg["loss"]["name"] == "ce":
            loss = F.cross_entropy(leaf_logits, target)
            acc = accuracy(leaf_logits, target)[0].item() # assuming top 1 acc
        elif cfg["loss"]["name"] == "bce":
            # need to find all the root path of the node
            y = torch.zeros(bz, n_classes).to(device)
            y[range(bz), target] = 1
            assert y.shape == leaf_logits.shape # both the shape should be (bz, num label set)
            loss = F.binary_cross_entropy_with_logits(leaf_logits, y)

            # computer accuracy
            prob = torch.sigmoid(leaf_logits)
            prob = 1.0 * (prob > 0.5)
            # compute the metric
            acc = multilabel_accuracy(prob.cpu(), y.cpu())

        loss += cfg['loss'].get('lambda', 0) * loss_node

        # update meters
        meter["loss"].update(loss.item(), bz)
        meter["acc"].update(acc, bz)
        # print information
        if (step+1) % print_freq == 0 or args.debug:
            print(f"[Eval] Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Acc: {acc}")

        # debug mode
        if args.debug:
            break

    meter = {k:meter[k].avg for k in meter.keys()}
    return meter


# eval one single epoch
# Compared to the original eval_one_epoch, a hierarchical loss measured depth-wise, HCA-depthwise, has been additionally implemented. Attempts were made to support the use of external models (such as the original CLIP model for auxiliary prediction), although these are not reported in the paper.
@torch.no_grad()
def eval_one_epoch_detailed(model, data_loader, param_names, device, epoch, cfg, args, leaf_nodes=None, intnl_nodes=None, sublabels=None, *, hlabels=None, nodes_per_depth=None):
    # setup average meters
    meter = {}
    meter["loss"] = averageMeter()
    meter["acc"] = averageMeter()
    if intnl_nodes is not None:
        meter["loss_node"] = averageMeter()
        meter.update({f'n{i}_acc': averageMeter() for i in range(len(intnl_nodes))})
        meter['consistency'] = hierConsistency(data_loader.dataset.indices)

    if nodes_per_depth is not None:
        meter.update({f'depth_{i}_acc': averageMeter() for i in range(len(nodes_per_depth))})
        meter['consistency-depthwise'] = hierConsistency(data_loader.dataset.indices)

    # Whether to use external knowledge, default is None.
    extra_zs_clip_ks = cfg.get('extra_zs_clip_ks', None)
    extra_pt_clip_ks = cfg.get('extra_pt_clip_ks', None)
    pretrained_pt_clip_dir = cfg.get('pretrained_pt_clip_dir', None)
    assert extra_pt_clip_ks is None or extra_zs_clip_ks is None
    extra_type, extra_ks = ('pt',extra_pt_clip_ks) if extra_pt_clip_ks is not None else ('zs', extra_zs_clip_ks)
    extra_model = None

    if extra_ks is not None:
        if isinstance(extra_ks, str):
            extra_ks = eval(extra_ks)
        if isinstance(extra_ks, (int, float)):
            extra_ks = [int(extra_ks)]
        elif isinstance(extra_ks, (list, tuple)):
            extra_ks = [int(i) for i in extra_ks]
        else:
            raise ValueError(f"extra_{extra_type}_clip_ks must be int, float, list or tuple, got {type(extra_ks)}")
        
        if extra_type == 'zs':
            # load pure zs clip model
            from models.promptsrc import load_clip_to_cpu as load_clip_to_cpu
            print('load an extra zs_clip to evaluate for generalization')
            clip_model = load_clip_to_cpu(cfg['model'], True).cuda()
            extra_model = ZeroShotCLIP(clip_model)
        elif extra_type == 'pt':
            extra_model = load_extra_pt_model(pretrained_pt_clip_dir, param_names, model, leaf_nodes)
        else:
            raise RuntimeError()
        for key in list(meter.keys()):
            if 'loss' not in key:
                for k in extra_ks:
                    meter[f'{key}_{extra_type}modified_k{k}'] = copy.deepcopy(meter[key])

    print_freq = cfg["print_freq"]
    model.eval()

    df = pd.DataFrame(param_names)

    # used by nodes_per_depth, where the root node is included
    df_all = pd.DataFrame(["root"] + param_names)

    for (step, value) in tqdm(enumerate(data_loader)):
        image = value[0].cuda()
        target = value[1].cuda()
        index = value[2].cuda()
        bz = target.shape[0]

        # classification at each internal node
        loss_node = 0
        if intnl_nodes is not None:
            loss_node = []
            for i, x in enumerate(intnl_nodes):
                tgt_mapping = sublabels[:, i]
                ntarget = tgt_mapping[target].long()
                idx = (ntarget >= 0)
                if sum(idx) > 0:
                    # forward
                    ntarget, nindex = ntarget[idx], index[idx]
                    param_idx = torch.tensor(x).cuda()
                    text = list(df.iloc[param_idx.cpu()][0])
                    nlogits = model(image[idx], text)['logits_per_image']

                    # compute loss and accuracy
                    nloss = F.cross_entropy(nlogits, ntarget)
                    loss_node.append(nloss)
                    nconf, npred = torch.softmax(nlogits, dim=-1).max(dim=-1)
                    niscorrect = (npred == ntarget)
                    nacc = niscorrect.float().mean() * 100.0
                    meter[f'n{i}_acc'].update(nacc.item(), sum(idx))
                    meter['consistency'].update(nindex, niscorrect)

                    # If needed, inject extra knowledge for children set of nodes.
                    if extra_model is not None:
                        for k in extra_ks:
                            # get top-k logits
                            topk_values, topk_indices = torch.topk(nlogits, k=min(k, nlogits.shape[1]), dim=1)
                            extra_scores = compute_modified_logits(extra_model, image[idx], text, topk_indices, extra_type)
                            
                            extra_best_idx = extra_scores.argmax(dim=1)  # [batch_size]
                            extra_pred = topk_indices.gather(1, extra_best_idx.unsqueeze(1)).squeeze(1)
                            
                            extra_iscorrect = (extra_pred == ntarget)
                            extra_acc = extra_iscorrect.float().mean() * 100.0
                            meter[f'n{i}_acc_{extra_type}modified_k{k}'].update(extra_acc.item(), sum(idx))
                            meter[f'consistency_{extra_type}modified_k{k}'].update(nindex, extra_iscorrect)

            loss_node = torch.stack(loss_node).mean()
            meter['loss_node'].update(loss_node.item(), bz)

        # classification at leaf nodes
        text = list(df.iloc[leaf_nodes][0])
        output = model(image, text)

        leaf_logits = output['logits_per_image']
        n_classes = leaf_logits.shape[1]
        if intnl_nodes is not None:
            conf, pred = torch.softmax(leaf_logits, dim=-1).max(dim=-1)
            iscorrect = (pred == target)
            meter['consistency'].update(index, iscorrect)
            if nodes_per_depth is not None:
                meter['consistency-depthwise'].update(index, iscorrect)

        assert cfg["loss"]["name"] == "ce", f"{cfg['loss']['name']} not implemented."
        loss = F.cross_entropy(leaf_logits, target)
        acc = accuracy(leaf_logits, target)[0].item() # assuming top 1 acc
        
        # If needed, inject extra knowledge for leaf nodes.
        if extra_model is not None:
            for k in extra_ks:
                topk_values, topk_indices = torch.topk(leaf_logits, k=min(k, leaf_logits.shape[1]), dim=1)
                extra_scores = compute_modified_logits(extra_model, image, text, topk_indices, extra_type)
                extra_best_idx = extra_scores.argmax(dim=1)  # [batch_size]
                extra_pred = topk_indices.gather(1, extra_best_idx.unsqueeze(1)).squeeze(1)
                
                extra_acc = (extra_pred == target).float().mean() * 100.0
                meter[f'acc_{extra_type}modified_k{k}'].update(extra_acc.item(), bz)
                
                if intnl_nodes is not None:
                    extra_iscorrect = (extra_pred == target)
                    meter[f'consistency_{extra_type}modified_k{k}'].update(index, extra_iscorrect)
                    if nodes_per_depth is not None:
                        meter[f'consistency-depthwise_{extra_type}modified_k{k}'].update(index, extra_iscorrect)
                    
        loss += cfg['loss'].get('lambda', 0) * loss_node

        # update meters
        meter["loss"].update(loss.item(), bz)
        meter["acc"].update(acc, bz)

        # depth-wise accuracy
        if nodes_per_depth is not None:
            # index=0 means root node, which is trival to classify
            for depth, x in enumerate(nodes_per_depth[1:], start=1):
                tgt_mapping_depth = hlabels[:, depth]
                dtarget = tgt_mapping_depth[target].long()
                idx = (dtarget >= 0)
                if sum(idx) > 0:
                    # forward
                    dtarget, dindex = dtarget[idx], index[idx]
                    text = list(df_all.iloc[x][0])
                    dlogits = model(image[idx], text)['logits_per_image']

                    # compute loss and accuracy
                    dloss = F.cross_entropy(dlogits, dtarget)
                    dconf, dpred = torch.softmax(dlogits, dim=-1).max(dim=-1)
                    discorrect = (dpred == dtarget)
                    dacc = discorrect.float().mean() * 100.0
                    meter[f'depth_{depth}_acc'].update(dacc.item(), sum(idx))
                    meter['consistency-depthwise'].update(dindex, discorrect)
                    # If needed, inject extra knowledge for depthwise set of nodes.
                    if extra_model is not None:
                        for k in extra_ks:
                            topk_values, topk_indices = torch.topk(dlogits, k=min(k, dlogits.shape[1]), dim=1)
                            extra_scores = compute_modified_logits(extra_model, image[idx], text, topk_indices, extra_type)
                            extra_best_idx = extra_scores.argmax(dim=1)  # [batch_size]
                            extra_pred = topk_indices.gather(1, extra_best_idx.unsqueeze(1)).squeeze(1)
                            extra_iscorrect = (extra_pred == dtarget)
                            extra_acc = extra_iscorrect.float().mean() * 100.0
                            meter[f'depth_{depth}_acc_{extra_type}modified_k{k}'].update(extra_acc.item(), sum(idx))
                            meter[f'consistency-depthwise_{extra_type}modified_k{k}'].update(dindex, extra_iscorrect)
                
        # print information
        if (step+1) % print_freq == 0 or args.debug:
            print(f"[Eval] Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Acc: {acc}")

        # debug mode
        if args.debug:
            break

    meter = {k:meter[k].avg for k in meter.keys()}
    return meter