import torch
import data.tta_datasets as tta_datasets
from torch.utils.data import DataLoader

def load_tta_dataset(args):
    """
    Load specific Test-Time Adaptation dataset based on args.
    """
    root = args.myroot

    if 'modelnet' in args.dataset_name.lower():
        if args.corruption == 'clean':
            # inference_dataset = tta_datasets.ModelNet_h5(args, root)
            inference_dataset = tta_datasets.ModelNet40C(args, root)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)

    elif 'scanobject' in args.dataset_name.lower():
        inference_dataset = tta_datasets.ScanObjectNN_C(args, root)

    elif 'shapenet' in args.dataset_name.lower():
        inference_dataset = tta_datasets.ShapeNetC2(args, root)

    else:
        raise NotImplementedError(f'Dataset {args.dataset_name} is not implemented')

    return inference_dataset