import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import math
from models import promptsrc_clip as clip
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

from .imagenet_templates import IMAGENET_TEMPLATES

from .adaptive_weight_generator import AdaptiveWeightGenerator
import pandas as pd
import manifolds.lorentz as L
_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg["name"]
    pretrained_version = cfg.get("pretrained_version", None)
    if pretrained_version is None:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
    elif pretrained_version.upper() == "BIOCLIP":
        print('loading bioclip backbone')
        url = clip._MODELS_BIOCLIP[backbone_name]
        state_dict = torch.load(url)  
        extra_dict = clip._EXTRA_BIOCLIP[backbone_name]
        state_dict.update(extra_dict)
    else:
        raise NotImplementedError()


    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg['prompt_depth_vision'],
                          "language_depth": cfg['prompt_depth_text'],
                          "vision_ctx": cfg['n_ctx_vision'],
                          "language_ctx": cfg['n_ctx_text']}
        print(design_details)
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        print(design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = {'x':x}
        outputs = self.transformer(combined)
        x = outputs['x']
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, all_nodes_info):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg['prompt_depth_text'] >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg['n_ctx_text']
        ctx_init = cfg['ctx_init']
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            raise RuntimeError()
            # 由于我们需要对任意的candidate text set生成embedding，而非仅仅局限于叶节点； 因此ctx_vectors必须是2维张量（参见encode_text方法中对self.ctx进行expand操作）； 当然，这里如果直接将ctx_vectors随机初始化为2维张量也不会引发运行时错误，但效果应该会很差，因此这里增加了报错。
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg['n_ctx_vision']}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            all_teacher_features_all = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

                x_all = [single_template.replace("{}", name) for name in all_nodes_info]
                x_tokenized_all = torch.cat([clip.tokenize(p) for p in x_all])
                text_features_all = clip_model_temp.encode_text(x_tokenized_all.cuda())
                all_teacher_features_all.append(text_features_all.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        self.fixed_embeddings_all = torch.cat(all_teacher_features_all, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts
    
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

    
    # encode specific text that is different from self.classnames
    def encode_text(self, text: list, token_embedding):
        n_cls = len(text)
        text = [name.replace("_", " ") for name in text]
        name_lens = [len(_tokenizer.encode(name)) for name in text]
        prompts = [self.prompt_prefix + " " + name + "." for name in text]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            dtype = self.dtype
            embedding = token_embedding(tokenized_prompts).type(dtype)
            embedding = embedding.to(tokenized_prompts.device)
 
        ctx = self.ctx

        ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = embedding[:, :1, :]  # SOS
        suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts, tokenized_prompts



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, all_nodes_info):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, all_nodes_info)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg["max_epoch"]
        self.n_cls = len(classnames)
        dim = clip_model.ln_final.weight.shape[0]

        self.cross_attn = AdaptiveWeightGenerator(dim, dtype=clip_model.dtype).to(dtype=clip_model.dtype)
        curv_init = cfg.get('curv_init', 0.5)
        self.curv_i_init = cfg.get('curv_i_init', curv_init)
        self.curv_t_init = cfg.get('curv_t_init', curv_init)
        self.curv_m_init = cfg.get('curv_m_init', curv_init)
        curv_version = self.cfg.get('curv_version', 'learn_common')
        self.curv_version = curv_version
        assert curv_version in ['learn_common', 'learn3', 'learn2derive1', 'fixed3', 'fixed2derive1']

        if curv_version == 'learn_common':
            #learn one common curvature parameter
            self.curv = nn.Parameter(
                torch.tensor(curv_init).log(), requires_grad=True
            )
            self.curv_t, self.curv_i, self.curv_m = self.curv, self.curv, self.curv
        elif 'fixed' in curv_version :
            if curv_version == 'fixed3':
                suffixs = ['t', 'i', 'm']
            else:
                suffixs = ['t', 'i']
            for suffix in suffixs:
                # 动态创建带后缀的参数变量 (self.curv_t, self.curv_i, self.curv_m)
                setattr(
                    self, 
                    f'curv_{suffix}',  # 动态生成参数名称
                    nn.Parameter(
                        torch.tensor(getattr(self, f'curv_{suffix}_init')).log(),  # 初始化张量并取对数
                        requires_grad=False  # 启用梯度计算
                    )
                )            
        else:
            if curv_version == 'learn3':
                suffixs = ['t', 'i', 'm']
            else:
                suffixs = ['t', 'i']
            for suffix in suffixs:
                # 动态创建带后缀的参数变量 (self.curv_t, self.curv_i, self.curv_m)
                setattr(
                    self, 
                    f'curv_{suffix}',  # 动态生成参数名称
                    nn.Parameter(
                        torch.tensor(getattr(self, f'curv_{suffix}_init')).log(),  # 初始化张量并取对数
                        requires_grad=True  # 启用梯度计算
                    )
                )
        
        self.last_curv_m = None

        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        print('curv_init', curv_init)   

    # note that self.curv_i, self.curv_t, self.curv_m are all log curvature. while this function should return original curvature
    def get_curv_m(self, r_image=None, r_text=None):
        if self.curv_version in ['learn_common', 'learn3', 'fixed3']:
            return self.curv_m.exp()
        elif self.curv_version in ['learn2derive1', 'fixed2derive1']:
            # we find that too small r will lead to optimal curv_m at boundary point.
            # suffciently large r can guarentee a internal point (suppose c > 0.01 any for c_m, c_t, c_i, r > 10 can guarentee a internal point)
            # and we experimentally find that the value of optimal c_m as well as its gradient is insensitive to the value of r as long as c_m is an internal point.
            # For example:
            # For:
            #   c1:0.44617098569869995, c2:0.45430800318717957, r1:2.0, r2:2.0 
            # we have:
            #   c3_optimal: 0.450436 c1.grad: 0.501852 c2.grad: 0.498446
            # For:
            #   c1:0.44617098569869995, c2:0.45430800318717957, r1:5.0, r2:5.0 
            # we have:
            #   c3_optimal: 0.450214 c1.grad: 0.504300 c2.grad: 0.495705
            # For:
            #   c1:0.44617098569869995, c2:0.45430800318717957, r1:10.0, r2:10.0
            # we have:
            #   c3_optimal: 0.450206 c1.grad: 0.504450 c2.grad: 0.495522
            # For:
            #   c1:0.44617098569869995, c2:0.45430800318717957, r1:2, r2:10
            # we have:
            #   c3_optimal: 0.454229 c1.grad: 0.018185 c2.grad: 0.981723
            # For:
            #   c1:0.44617098569869995, c2:0.45430800318717957, r1:10, r2:50
            # we have:
            #   c3_optimal: 0.454004 c1.grad: 0.038396 c2.grad: 0.961622


            # heuristically, c_m should be the internal point
            # so, although we can pass the calculated r_image and r_text to this function,
            # we rescale them to be efficienly large
            curv_i, curv_t = self.curv_i.exp(), self.curv_t.exp()
            with torch.no_grad():
                it_ratio = r_image / r_text
                min_ri = 1 / torch.sqrt(curv_i)
                min_rt = 1 / torch.sqrt(curv_t)
                ri = max(min_ri, it_ratio * min_rt)
                rt = ri / it_ratio
            
            curv_m =  L.OptimalC3Function.apply(curv_i, curv_t, ri, rt)
            return curv_m
        else:
            raise ValueError(f"self.curv_version:{self.curv_version} is illegal")
    
    def compute_logits_hierachical(self, img_multi_features, text_features, return_weights=False):
        # text_fea.shape = (n, dim)
        # img_multi_features = (bsz, decoupled_size, dim) or (bsz, num_multi_features, dim)
        # 使用文本特征作为查询(query)
        ret_dict = {}
        query = text_features.unsqueeze(0).expand(img_multi_features.shape[0], -1, -1)  # (bsz, n, dim)

        key = value = img_multi_features #(bsz, decoupled_size, dim)
        # 应用交叉注意力
        ret = self.cross_attn(
            query=query.transpose(0, 1),
            key=key.transpose(0, 1),
            value=value.transpose(0,1),
            need_weights = return_weights
        )  # (num_i, bsz, dim)
  
        fused_image_features = ret[0]
        if return_weights:
            attn_weights = ret[1]
            # for visualization
            ret_dict["attn_weights"] = attn_weights 
        else:
            ret_dict["attn_weights"] = None
        fused_image_features = fused_image_features.transpose(0,1) #(bsz, num_i, dim)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # different ways to align text and fused image features
        # by default, we use 'many_to_many' as a straightforward way to align text and fused image features
        # to alleviate overfitting, we also tried 'ony_to_many' stragety.
        if self.cfg.get('align_version', 'many-to-many') == 'many-to-many':
            # version 1: directly compute (many to many)
            fused_image_features = fused_image_features / fused_image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = (logit_scale * fused_image_features * text_features.unsqueeze(0)).sum(dim=-1)
            ret_dict['logits'] = logits
            ret_dict['image_features'] = fused_image_features
            ret_dict['text_features'] = text_features
        
        # elif self.cfg['align_version'] == 'one-to-many':
        #     # version 2: pooling (one to many)
        #     pooled_image_features = fused_image_features.mean(dim=1) #(bsz, dim)
        #     pooled_image_features = pooled_image_features / pooled_image_features.norm(dim=-1, keepdim=True)
        #     logit_scale = self.logit_scale.exp()
        #     logits = logit_scale * pooled_image_features @ text_features.t()

        else:
            raise NotImplementedError(f"Unknown align-version: {self.cfg['align_version']}")

        return ret_dict
    
    #given unnormalized features
    def forward_with_text_features(self, image, text_features, hierachical, selected_layer_indices, return_weights, text_ids=None):
        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image.type(self.dtype), hierachical=hierachical, selected_layer_indices=selected_layer_indices)
        # Compute the prompted image and text features and the prompted logits
        if hierachical:
            ret = self.compute_logits_hierachical(image_features, text_features, return_weights=return_weights)
            fused_image_features = ret['image_features']
            logits = ret['logits']
            #get the last layers' representation
            image_features = image_features[:, -1, :]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()
            fused_image_features = None
        
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            if text_ids is None:
                fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            else:
                fixed_embeddings = self.prompt_learner.fixed_embeddings_all[text_ids]
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
            return {
                "text_features": text_features,
                "fixed_embeddings": fixed_embeddings,
                "zero_shot_features": zero_shot_features,
                "image_features": image_features,
                "zero_shot_logits": zero_shot_logits,
                "logits": logits,
                "fused_image_features": fused_image_features
            }
        else:
            return {"logits":logits}

    def forward(self, image, hierachical=True, selected_layer_indices=None, return_weights=False):
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        return self.forward_with_text_features(image=image, text_features=text_features, hierachical=hierachical, selected_layer_indices=selected_layer_indices, return_weights=return_weights)
        


    def forward_custom_label_set(self, image, text: list, token_embedding, hierachical=False, selected_layer_indices=None, return_weights=False, text_ids=None):
        logit_scale = self.logit_scale.exp()
        prompts, tokenized_prompts = self.prompt_learner.encode_text(text, token_embedding)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return self.forward_with_text_features(text_features=text_features, image=image, hierachical=hierachical, selected_layer_indices=selected_layer_indices,return_weights=return_weights, text_ids=text_ids)
        

    def compute_logits_hierachical(self, img_multi_features, text_features, return_weights=False):
        # text_fea.shape = (n, dim)
        # img_multi_features = (bsz, decoupled_size, dim) or (bsz, num_multi_features, dim)
        # 使用文本特征作为查询(query)
        ret_dict = {}
        query = text_features.unsqueeze(0).expand(img_multi_features.shape[0], -1, -1)  # (bsz, n, dim)

        key = value = img_multi_features #(bsz, decoupled_size, dim)
        # 应用交叉注意力
        ret = self.cross_attn(
            query=query.transpose(0, 1),
            key=key.transpose(0, 1),
            value=value.transpose(0,1),
            need_weights = return_weights
        )  # (num_i, bsz, dim)
  
        fused_image_features = ret[0]
        if return_weights:
            attn_weights = ret[1]
            # for visualization
            ret_dict['attn_weights'] = attn_weights 
        fused_image_features = fused_image_features.transpose(0,1) #(bsz, num_i, dim)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # different ways to align text and fused image features
        # by default, we use 'many_to_many' as a straightforward way to align text and fused image features
        # to alleviate overfitting, we also tried 'ony_to_many' stragety.
        if self.cfg.get('align_version', 'many-to-many') == 'many-to-many':
            # version 1: directly compute (many to many)
            fused_image_features = fused_image_features / fused_image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = (logit_scale * fused_image_features * text_features.unsqueeze(0)).sum(dim=-1)
            ret_dict['logits'] = logits
            ret_dict['image_features'] = fused_image_features
            ret_dict['text_features'] = text_features
        
        # elif self.cfg['align_version'] == 'one-to-many':
        #     # version 2: pooling (one to many)
        #     pooled_image_features = fused_image_features.mean(dim=1) #(bsz, dim)
        #     pooled_image_features = pooled_image_features / pooled_image_features.norm(dim=-1, keepdim=True)
        #     logit_scale = self.logit_scale.exp()
        #     logits = logit_scale * pooled_image_features @ text_features.t()

        else:
            raise NotImplementedError(f"Unknown align-version: {self.cfg['align_version']}")

        return ret_dict
    
    #not used in this file
    def encode_text(self, text: list, token_embedding, normalize=True):
        prompts, tokenized_prompts = self.prompt_learner.encode_text(text, token_embedding)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    #not used in this file
    def encode_image(self, image, normalize=True, hierachical=False, selected_layer_indices=None, attn_weights_list=None):
        image_features = self.image_encoder(image.type(self.dtype), hierachical=hierachical, selected_layer_indices=selected_layer_indices)
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    # def encode_text(self, text: list, token_embedding, normalize=True):

    #     prompts, tokenized_prompts = self.prompt_learner.encode_text(text, token_embedding)
    #     text_features = self.text_encoder(prompts, tokenized_prompts)

    #     if normalize:
    #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #     return text_features

    # def encode_image(self, image, normalize=True, hierachical=None, attn_weights_list=None):
    #     assert hierachical in [True, False]
    #     if hierachical:
    #         selected_layer_indices = self.cfg['selected_layer_indices']
    #         ret = self.image_encoder(image.type(self.dtype), hierachical=True, selected_layer_indices=selected_layer_indices, attn_weights_list=attn_weights_list)
    #         if attn_weights_list is not None:
    #             image_features, attn_weights_list = ret
    #         else:
    #             image_features = ret
    #     else:
    #         ret = self.image_encoder(image.type(self.dtype), hierachical=False, attn_weights_list=attn_weights_list)
    #         if attn_weights_list is not None:
    #             image_features, attn_weights_list = ret
    #         else:
    #             image_features = ret

    #     if normalize:
    #         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     if attn_weights_list is not None:
    #         return image_features, attn_weights_list
    #     else:
    #         return image_features

    def clamp_curv(self):
        self.curv_i.data = torch.clamp(self.curv_i.data, **self._curv_minmax)
        self.curv_t.data = torch.clamp(self.curv_t.data, **self._curv_minmax)
        if hasattr(self, 'curv_m'):
            self.curv_m.data = torch.clamp(self.curv_m.data, **self._curv_minmax)




class PromptSRC(nn.Module):
    def __init__(self, cfg, classnames, all_nodes_info):
        super().__init__()
        self.cfg = cfg
        self.check_cfg(self.cfg)
        self.classnames = classnames
        self.all_nodes_info = all_nodes_info
        self.build_model()

    def check_cfg(self, cfg):
        assert cfg["prec"] in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        print("Loading CLIP (backbone:" + cfg["name"] + ") into PromptSRC")
        clip_model = load_clip_to_cpu(cfg)

        if cfg["prec"] == "fp32" or cfg["prec"] == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.classnames, clip_model, self.all_nodes_info)
        self.token_embedding = clip_model.token_embedding

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = ["cross_attn", "prompt_learner", "curv"]

        for name, param in self.model.named_parameters():
            if all([i not in name for i in name_to_update]):
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        # if cfg.get("init_weights", None) is not None:
        #     load_pretrained_weights(self.model, cfg["init_weights"])

        
        # NOTE: only give prompt_learner to the optimizer
        # self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg['max_epoch']
        self.step_counter = 1
        N = cfg['max_epoch']
        mean = cfg['gpa_mean']
        stdev = cfg['gpa_std']
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        # self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        # Keep model with GPA
        self.previous_model_gpa = None
    
    def forward(self, image, labelset=None, show_weights=False, text_ids=None):
        output = {}
        hierachical = self.cfg.get('hierachical', True)
        if hierachical:
            selected_layer_indices = self.cfg.get('selected_layer_indices', [4, 7, 11])
            return_weights=self.cfg.get("show_crossattn_weights", False)
        else:
            selected_layer_indices = self.cfg.get('selected_layer_indices', None)  
            return_weights = None

        if labelset is None:
            ret = self.model(image, hierachical=hierachical, selected_layer_indices=selected_layer_indices)
            
        else:
            ret = self.model.forward_custom_label_set(image, labelset, token_embedding=self.token_embedding, hierachical=hierachical, selected_layer_indices=selected_layer_indices, return_weights=return_weights, text_ids=text_ids)

        if "fixed_embeddings" in ret:
            normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, zero_shot_logits, logits, fused_image_features = [ ret[key] for key in ["text_features", "fixed_embeddings", "zero_shot_features", "image_features",  "zero_shot_logits", "logits", "fused_image_features"]]
            output["image_features"] = fused_image_features
            output["text_features"] = normalized_text_features
            output["logits_per_image"] = logits
            # print(f"normalized_text_features.shape:{normalized_text_features.shape}")
            # print(f"zs_clip_text_embeddings.shape:{zs_clip_text_embeddings.shape}")
            # print(f'image_ft.shape:{image_ft.shape}')
            # print(f'zs_image_embedd.shape:{zs_image_embedd.shape}')
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                    reduction='mean') * self.cfg["image_loss_weight"]
            # Calculate the L_SCL_text loss and L_SCL_logits
            if normalized_text_features.shape == zs_clip_text_embeddings.shape:
                loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                        reduction='mean') * self.cfg["text_loss_weight"]
                L_SCL_logits = F.kl_div(
                    F.log_softmax(logits / 1, dim=1),
                    F.log_softmax(zero_shot_logits / 1, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (1 * 1) / logits.numel()
                
            else:
                loss_scl_text = 0
                L_SCL_logits = 0
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            
            output["loss_scl"] = L_SCL
        else:
            output["logits_per_image"] = ret["logits"]
            for k in ["image_features", "text_features", "attn_weights"]:
                if k in ret:
                    output[k] = ret[k]
        return output

    # load the pretrained_ckpt from author
    def load_author_pretrained_ckpt(self, model_path):   
        map_location = None if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=map_location)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        #print(state_dict.keys())
        # Ignore fixed token vectors
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]

        #print(state_dict["prompt_learner.ctx"]) 
        self.model.load_state_dict(state_dict, strict=False)

        # double check the state dict is loaded
        assert torch.equal(self.model.prompt_learner.ctx, state_dict["prompt_learner.ctx"])

    # load the pretrained_ckpt from author
    # def load_custom_ckpt(self, model_path):
    #     map_location = None if torch.cuda.is_available() else "cpu"
    #     state_dict = torch.load(model_path, map_location=map_location)

    #     # Ignore fixed token vectors
    #     if "token_prefix" in state_dict:
    #        del state_dict["token_prefix"]

    #     if "token_suffix" in state_dict:
    #        del state_dict["token_suffix"]

    #     #print(state_dict.keys())

    #     name_to_copy = ["prompt_learner.proj.weight", "prompt_learner.compound_prompts_text.0", "prompt_learner.compound_prompt_projections.0.bias", "prompt_learner.proj.bias", "prompt_learner.compound_prompt_projections.0.weight", "prompt_learner.compound_prompt_projections.1.weight", "prompt_learner.ctx", "prompt_learner.compound_prompt_projections.1.bias", "prompt_learner.compound_prompts_text.1"] + ['cross_attn.in_proj_weight', 'cross_attn.in_proj_bias', 'cross_attn.out_proj.weight', 'cross_attn.out_proj.bias'] + ['curv']
    #     #print(self.model.state_dict().keys())

    #     for n in name_to_copy:
    #         # if "cross_attn" in n:
    #         #     import pdb; pdb.set_trace()
    #         print(f'copying {n}')
    #         self.model.state_dict()[n].copy_(state_dict["model." + n])
        
    #         assert torch.equal(self.model.state_dict()[n], state_dict["model." + n])
    
    def load_custom_ckpt(self, model_path):
        map_location = None if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(model_path, map_location=map_location)

        # Ignore fixed token vectors
        for key in list(state_dict.keys()):
            if "token_prefix" in key:
               del state_dict[key]
               print(f'ignoring {key}')
            elif "token_suffix" in key:
                del state_dict[key]
                print(f'ignoring {key}')

        self.load_state_dict(state_dict, strict=False)

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage
        
    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2
        
    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def gauss_fusion(self):
        print('gauss_fusion')
        model = self.model
        self.step_counter = self.step_counter + 1
        current_epoch_weight = self.gauss[self.step_counter - 2]
        current_model_weights = copy.deepcopy(model.state_dict())
        weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
        if self.previous_model_gpa is None:
            self.previous_model_gpa = weighted_state_dict
        else:
            self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)
        print(self.step_counter, self.model.total_epochs)
        if self.step_counter == self.model.total_epochs + 1:
            
            print("Using GPA model for final inference...")
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)

    def encode_image(self, image, normalize=True):
        image_features = self.model.encode_image(image, normalize=normalize)
        return image_features

    def encode_text(self, text : list, normalize=True):
        text_features = self.model.encode_text(text, self.token_embedding, normalize=normalize)
        return text_features

def load(cfg, classnames: list, all_nodes_info:list):
    model = PromptSRC(cfg, classnames, all_nodes_info)
    return model
