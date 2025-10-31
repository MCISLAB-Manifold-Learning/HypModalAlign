### __init__.py
# Get model instance with designated parameters.
###

import copy
import torch.nn as nn
import logging

from .clip import load as clip_model
from .coop import load as coop_model
from .cocoop import load as cocoop_model
from .maple import load as maple_model
from .promptsrc import load as promptsrc_model
from .treecut_generator import treecut_generator

logger = logging.getLogger('mylogger')


def get_model(model_dict, init_classname=None, verbose=False, all_nodes_info=None):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if name == 'clip':
        model, _ = model(**param_dict)

    elif name in ['coop', 'cocoop', 'maple', 'promptsrc']:
        if name == 'promptsrc':
            model = model(param_dict, init_classname, all_nodes_info=all_nodes_info)
        else:
            model = model(param_dict, init_classname)

    elif name == 'treecut_generator':
        model = model(**param_dict)

    else:
        model = model(**param_dict)

    if verbose:
        logger.info(model)

    return model


def _get_model_instance(name):
    try:
        return {
            'clip': clip_model,
            'coop': coop_model,
            'cocoop': cocoop_model,
            'maple': maple_model,
            'promptsrc': promptsrc_model,
            'treecut_generator': treecut_generator,
        }[name]
    except:
        raise BaseException('Model {} not available'.format(name))


