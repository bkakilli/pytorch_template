import os
import sys
import importlib

from torch.utils.data import DataLoader

def get_loaders(args):

    print("Loading dataset.")
    dataset = importlib.import_module(".dataset", package="datasets."+args.dataset)

    dataloaders = []
    for split in ["train", "val", "test"]:
        d = dataset.Dataset(root=args.root, split=split, crossval_id=args.crossval_id, augmentation=(not args.no_augmentation))
        dl = DataLoader(d, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=dataset.custom_collate_fn)
        dataloaders.append(dl)

    return dataloaders