import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from models import maple_clip as clip
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .adaptive_weight_generator import AdaptiveWeightGenerator
import manifolds.lorentz as L
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    #backbone model指的是视觉骨干网络的架构
    backbone_name = cfg["name"] 
    pretrained_version = cfg.get("pretrained_version", None)
    if pretrained_version is None:
        url = clip._MODELS[backbone_name]
        #缓存模型，并返回缓存的路径
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
    elif pretrained_version.upper() == "HYCOCLIP":
        print('loading bioclip backbone')
        url = clip._MODELS_HYCOCLIP[backbone_name]
        state_dict = torch.load(url)  
        extra_dict = clip._EXTRA_HYCOCLIP[backbone_name]
        state_dict.update(extra_dict)
    else:
        raise NotImplementedError()
    

    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg["n_ctx"],
                    #   "hierachical": cfg["hierarchical"],
                      "use_invite": cfg.get("use_invite", True),
                      }
    print('design_details', design_details)
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        # combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        combined = {'x':x, 'compound_prompts_deeper':compound_prompts_deeper_text, 'counter':0}  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        # x = outputs[0]  # extract the x back from here
        x = outputs['x']  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg["n_ctx"]
        ctx_init = cfg["ctx_init"] if len(cfg["ctx_init"]) > 0 else None
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        # Default is 1, which is compound shallow prompting
        assert cfg["prompt_depth"] >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg["prompt_depth"]  # max=12, but will create 11 such shared prompts

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # 这里用register_buffer是因为：我们只学prompt，而SOS, CLS, EOS这些是不会改变的
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

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


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

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = embedding[:, :1, :]  # SOS
        suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, tokenized_prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


    def get_visual_prompt(self,):
        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return self.proj(self.ctx), visual_deep_prompts   # pass here original, as for visual 768 is required




class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        #相当于对原生的clip_model做了一下包装
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
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

    def forward_original(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        #这里需要修改image_encode，获取多层次的image_features,我的想法就是返回的features张量多加一个维度，表示不同层次
        #查找发现，需要修改maple_model.py下的CLIP类
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return {'logits':logits}

    def forward(self, image, hierachical=True, selected_layer_indices=None, return_weights=False):
        if not hierachical:
            return self.forward_original(image)
        tokenized_prompts = self.tokenized_prompts
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        img_multi_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, hierachical=True, selected_layer_indices=selected_layer_indices)

        return self.compute_logits_hierachical(img_multi_features, text_features, return_weights=return_weights)


    def forward_custom_label_set_original(self, image, text: list, token_embedding):
        text_features = self.encode_text(text=text, token_embedding=token_embedding, normalize=True)
        image_features = self.encode_image(image, normalize=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    def forward_custom_label_set(self, image, text: list, token_embedding, hierachical=True, selected_layer_indices=None, return_weights=False):
        if not hierachical:
            return {"logits":self.forward_custom_label_set_original(image, text, token_embedding)}

        text_features = self.encode_text(text=text, token_embedding=token_embedding, normalize=False)
        img_multi_features = self.encode_image(image, normalize=False, hierachical=True, selected_layer_indices=selected_layer_indices)

        # step1: feature decoupling (for now, nothing is done)
        decoupled_size = img_multi_features.shape[1]
        img_multi_features_decoupled = img_multi_features

        # step2: attention fusion and compute logits
        return self.compute_logits_hierachical(img_multi_features_decoupled, text_features, return_weights=return_weights)


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

    def encode_text(self, text: list, token_embedding, normalize=True):

        prompts, tokenized_prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner.encode_text(text, token_embedding)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)

        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image, normalize=True, hierachical=False, selected_layer_indices=None, attn_weights_list=None):
        shared_ctx, deep_compound_prompts_vision = self.prompt_learner.get_visual_prompt()
        if hierachical:
            # selected_layer_indices = self.cfg['selected_layer_indices']
            ret = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, hierachical=True, selected_layer_indices=selected_layer_indices, attn_weights_list=attn_weights_list)
            if attn_weights_list is not None:
                image_features, attn_weights_list = ret
            else:
                image_features = ret
        else:
            ret = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, hierachical=False, attn_weights_list=attn_weights_list)
            if attn_weights_list is not None:
                image_features, attn_weights_list = ret
            else:
                image_features = ret

        #todo: conduct feature decoupling
        #目前测试阶段，先在sun数据集上（sun只有三层，且比较规整），暂时不做decoupling

        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if attn_weights_list is not None:
            return image_features, attn_weights_list
        else:
            return image_features

    def clamp_curv(self):
        self.curv_i.data = torch.clamp(self.curv_i.data, **self._curv_minmax)
        self.curv_t.data = torch.clamp(self.curv_t.data, **self._curv_minmax)
        if hasattr(self, 'curv_m'):
            self.curv_m.data = torch.clamp(self.curv_m.data, **self._curv_minmax)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MaPLe(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.check_cfg(self.cfg)
        self.classnames = classnames
        self.build_model()


    def check_cfg(self, cfg):
        assert cfg["prec"] in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        print("Loading CLIP (backbone:" + cfg["name"] + ") into Maple")
        clip_model = load_clip_to_cpu(cfg)

        if cfg["prec"] == "fp32" or cfg["prec"] == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.classnames, clip_model)
        self.token_embedding = clip_model.token_embedding
        #self.logit_scale = self.model.logit_scale        

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = ["cross_attn", "prompt_learner", "curv"]

        for name, param in self.model.named_parameters():
            if all([i not in name for i in name_to_update]):
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def forward(self, image, labelset=None, treecut_node=None, leaf_nodes=False, show_weights=False):
        output = {}
        hierachical = self.cfg.get('hierachical', True)
        if hierachical:
            select_layer_indices = self.cfg.get('selected_layer_indices', [4, 7, 11])
            return_weights=self.cfg.get("show_crossattn_weights", False)
        else:
            select_layer_indices = self.cfg.get('selected_layer_indices', None)  
            return_weights = None
        # use the init classname for classification
        if labelset is None:
            ret = self.model(image, hierachical=hierachical, selected_layer_indices=select_layer_indices,  return_weights=return_weights)
        else:
            ret = self.model.forward_custom_label_set(image, labelset, self.token_embedding, hierachical=hierachical, selected_layer_indices=select_layer_indices, return_weights=return_weights)
        output["logits_per_image"] = ret["logits"]
        for k in ["image_features", "text_features", "attn_weights"]:
            if k in ret:
                output[k] = ret[k]
        return output

    # given a list of classes, return the text feature of those classes
    # return output: (len(text), feat_size)
    def encode_text(self, text : list, normalize=True):
        text_features = self.model.encode_text(text, self.token_embedding, normalize=normalize)
        return text_features

    # encode image and return image feature
    # return output: (bz, feat_size)
    def encode_image(self, image, normalize=True):
        image_features = self.model.encode_image(image, normalize=normalize)
        return image_features


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
    def load_custom_ckpt(self, model_path):
        map_location = None if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(model_path, map_location=map_location)

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
           del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
           del state_dict["token_suffix"]

        #print(state_dict.keys())

        name_to_copy = ["prompt_learner.proj.weight", "prompt_learner.compound_prompts_text.0", "prompt_learner.compound_prompt_projections.0.bias", "prompt_learner.proj.bias", "prompt_learner.compound_prompt_projections.0.weight", "prompt_learner.compound_prompt_projections.1.weight", "prompt_learner.ctx", "prompt_learner.compound_prompt_projections.1.bias", "prompt_learner.compound_prompts_text.1"] + ['cross_attn.in_proj_weight', 'cross_attn.in_proj_bias', 'cross_attn.out_proj.weight', 'cross_attn.out_proj.bias'] + ['curv_i', 'curv_t', 'curv_m', 'curv']
        #print(self.model.state_dict().keys())

        for n in name_to_copy:
            # if "cross_attn" in n:
            #     import pdb; pdb.set_trace()
            if "model." + n in state_dict.keys():
                print(f'copying {n}')
                self.model.state_dict()[n].copy_(state_dict["model." + n])
            
                assert torch.equal(self.model.state_dict()[n], state_dict["model." + n])
            else:
                print(f'warning! model.{n} not exists, skipping')


def load(cfg, classnames: list):
    model = MaPLe(cfg, classnames)
    return model


