import os
import numpy as np
import logging
import datetime
from collections import OrderedDict
import yaml
from models import get_model

def get_logger(logdir):
    logger = logging.getLogger('mylogger')
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def cvt2normal_state(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def cvt2module_state(state_dict):
    """Converts a state dict from a normal state to a dataParallel module
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # add `module.`
        new_state_dict['module.' + k] = v
    return new_state_dict

def dotted_set(d, path, value):
    """Set value in a nested dictionary using a dot-separated path (automatically creates intermediate dictionaries)
    :param d: Target dictionary (can be nested)
    :param path: Dot-separated path (e.g., "user.profile.age")
    :param value: Value to set
    """
    keys = path.split('.')
    current = d
    # Traverse each level, dynamically creating intermediate dictionaries
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}  # Create missing intermediate dictionary
        current = current[key]
    to_ret = current.get(keys[-1], None)
    # Set the value at the final level
    current[keys[-1]] = value
    return to_ret

def smart_convert(s: str):
    """Smart type conversion (priority: bool → int → float → original string)
    Args:
        s: Input string
    Returns:
        Converted value (bool/int/float/str)
    """
    # Step 1: Try converting to boolean (supports common case variations)
    bool_lower = s.lower()
    # if bool_lower in ('true', 'yes', 'y', '1', 'on'):
    if bool_lower in ('true'):
        return True
    # elif bool_lower in ('false', 'no', 'n', '0', 'off', 'none'):
    elif bool_lower in ('false'):
        return False
    
    if bool_lower in ('none'):
        return None
    
    # Step 2: Try converting to integer (but exclude cases with decimal points or scientific notation)
    # Check if the string does not contain '.', 'e', or 'E' (scientific notation)
    if '.' not in s and 'e' not in s.lower():
        try:
            return int(s)
        except ValueError:
            pass
    
    # Step 3: Try converting to float (supports scientific notation)
    try:
        return float(s)  # Supports formats like "3.14", "1e-5", "-inf", "1.", "1.0", etc.
    except ValueError:
        pass
    
    # Step 4: Return original string if conversion fails
    return s


def compute_modified_logits(model, images, texts, topk_indices, model_type):
    model.eval()
    if model_type == 'zs':
        logits = model.compute_similarity_for_topk(
                                images, texts, topk_indices
                            )
        return logits
    elif model_type == 'pt':
        batch_size, k = topk_indices.shape
        image_features = model.encode_image(images, normalize=True)
        flat_indices = topk_indices.flatten()  # [batch_size * k]
        selected_texts = [texts[idx.item()] for idx in flat_indices]
        text_features = model.encode_text(selected_texts, normalize=True)  # [batch_size * k, embed_dim]
        text_features = text_features.view(batch_size, k, -1)
        image_features = image_features.unsqueeze(1)
        
        # similarity matrix：[batch_size, k]
        scores = (image_features * text_features).sum(dim=-1)
        
        return scores
    else:
        raise ValueError(f"Unsupported model type {model_type}")

def load_extra_pt_model(folder, param_names, model, leaf_nodes):
    yml_file = None
    for f in os.listdir(folder):
        if f[-3:] == "yml":
            assert yml_file is None, "Error, multi config file detected!"     
            yml_file = folder + "/" + f
        elif f == "ckpt":
            assert os.path.exists(os.path.join(folder, f, "last.pth"))
            eval_ckpt = "last"
    assert yml_file is not None
    with open(yml_file) as fp:
        cfg_extra = yaml.load(fp, Loader=yaml.SafeLoader)
    
    init_classname = None
    # change the initial label space
    if cfg_extra.get("init_label_set", "leaf") == "leaf":
        init_classname = [param_names[i] for i in leaf_nodes]

    all_nodes_info = None
    if hasattr(model, 'all_nodes_info'):
        all_nodes_info = model.all_nodes_info
    extra_model = get_model(cfg_extra['model'], init_classname, all_nodes_info=all_nodes_info).cuda()
    checkpoint = os.path.join(folder, "ckpt", eval_ckpt + ".pth")
    if os.path.isfile(checkpoint):
        extra_model.load_custom_ckpt(checkpoint)
    else:
        print(f'checkpoint ({checkpoint}) not found!')
    extra_model.eval()
    return extra_model