import os
import sys
import json
import shutil
import torch
from datetime import datetime

import numpy as np

def persistence(args, module_file, main_file):

    output_dir, model_path = args.output, args.model_path
    # Initial checkpoint
    checkpoint = {
        "loss": 0,
        "epoch": 0,
        "model_state_dict": None,
        "optimizer_state_dict": None
    }

    # Create log dir if does not exist
    checkpoints_path = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(output_dir):
        os.makedirs(checkpoints_path)
        shutil.copy(os.path.abspath(main_file), output_dir)
        shutil.copy(os.path.abspath(module_file), output_dir)
    else:
        if model_path is None:
            ans = input("Folder already exists! Overwrite? [Y/n]: ")
            if not ans.lower() in ['y', 'yes', '']:
                raise FileExistsError("Folder already exists: %s"%(output_dir))
            shutil.copy(os.path.abspath(__file__), output_dir)
            shutil.copy(os.path.abspath(module_file), output_dir)

        else:
            checkpoint = torch.load(model_path)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    return checkpoint

def save_checkpoint(output_dir, container):
    """Save given information (eg. model, optimizer, epoch number etc.) into log_idr

    Parameters
    ----------
    output_dir : str
        Path to save
    container : dict
        Information to be saved
    """
    path = os.path.join(output_dir, "checkpoints", "checkpoint_%s.pt"%datetime.now().strftime("%Y%m%d_%H%M%S"))
    torch.save(container, path)


def get_lr(optimizer):
    """Return current learning rate of the given optimizer

    Parameters
    ----------
    optimizer : PyTorch Optimizer
        Optimizer

    Returns
    -------
    float
        learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def join_path(*args, **kwargs):
    return os.path.join(*args, **kwargs)

def seed(x):
    np.random.seed(x)
    torch.manual_seed(x)
