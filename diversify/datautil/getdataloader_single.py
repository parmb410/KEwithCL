# getdataloader_single.py

import numpy as np
from torch.utils.data import DataLoader

import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset

import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}

def get_dataloader(args, tr, val, tar):
    train_loader = DataLoader(dataset=tr, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    valid_loader = DataLoader(dataset=val, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    target_loader = DataLoader(dataset=tar, batch_size=args.batch_size,
                               num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader

def get_act_dataloader(args):
    source_datasetlist = []
    target_datalist = []
    pcross_act = task_act[args.task]

    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)
    for i, item in enumerate(tmpp):
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=actutil.act_train())
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)

    # Combines datasets as in your original pipeline
    tr = combindataset(args, source_datasetlist)
    val = subdataset(args, source_datasetlist, type='val')
    tar = combindataset(args, target_datalist)
    return get_dataloader(args, tr, val, tar) + (tr, val, tar)

def inject_domain_labels(dataset, domain_labels):
    """
    Safely injects domain labels into a dataset.
    Handles common PyTorch dataset structures.
    """
    # Try to use the setter if available
    if hasattr(dataset, 'set_domain_labels') and callable(getattr(dataset, 'set_domain_labels')):
        dataset.set_domain_labels(domain_labels)
    # Otherwise, set the dlabels attribute directly if it exists
    elif hasattr(dataset, 'dlabels'):
        dataset.dlabels = np.array(domain_labels, dtype=int)
    else:
        # Recursively set on constituent datasets if this is a ConcatDataset or similar
        if hasattr(dataset, 'datasets'):  # e.g., torch.utils.data.ConcatDataset
            idx = 0
            for subds in dataset.datasets:
                n = len(subds)
                inject_domain_labels(subds, domain_labels[idx:idx + n])
                idx += n
        else:
            raise RuntimeError(
                "Dataset does not support domain label injection. Please add set_domain_labels method or dlabels attribute."
            )
