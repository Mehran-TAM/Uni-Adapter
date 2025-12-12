import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# =============================================================================
# Helper Functions
# =============================================================================

def load_data(data_path, corruption, severity):
    """
    Helper to load .npy files for corrupted datasets (ModelNet-C, ScanObjectNN-C, ShapeNet-C).
    """
    if corruption == 'clean':
        data_file = os.path.join(data_path, 'data_original.npy')
    else:
        # Standard naming convention for corrupted datasets
        data_file = os.path.join(data_path, f'data_{corruption}_{severity}.npy')
    
    label_file = os.path.join(data_path, 'label.npy')

    # Mixed corruptions edge case (if used in your dataset generation)
    if 'mixed_corruptions' in corruption:
        data_file = os.path.join(data_path, f'{corruption}.npy')
        label_file = os.path.join(data_path, 'mixed_corruptions_labels.npy')

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")

    all_data = np.load(data_file, allow_pickle=True)
    all_label = np.load(label_file, allow_pickle=True)
    
    return all_data, all_label

def load_h5(h5_name):
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data, label

# =============================================================================
# Dataset Classes
# =============================================================================

class ModelNet_h5(Dataset):
    """
    Loader for clean ModelNet40 from H5 file.
    """
    def __init__(self, args, root):
        # Try standard filenames
        possible_names = ['modelnet40_test.h5', 'clean.h5', f'{args.corruption}.h5']
        h5_path = None
        
        for name in possible_names:
            temp_path = os.path.join(root, name)
            if os.path.exists(temp_path):
                h5_path = temp_path
                break
        
        if h5_path is None:
             raise FileNotFoundError(f"Could not find H5 file in {root}. Checked: {possible_names}")

        self.data, self.label = load_h5(h5_path)
        
        # Adjust 1-based indexing if necessary (common in some ModelNet versions)
        if np.min(self.label) == 1:
            self.label = self.label - 1

        self.class_name = [
            "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", 
            "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot", 
            "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", 
            "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink", 
            "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", 
            "wardrobe", "xbox"
        ]

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        
        if isinstance(label, np.ndarray):
            label = label.item()
            
        class_name = self.class_name[label]
        rgb = torch.ones_like(torch.asarray(pointcloud)).float()
        
        return pointcloud, label, class_name, rgb

    def __len__(self):
        return self.data.shape[0]


class ModelNet40C(Dataset):
    """
    Loader for ModelNet40-C (Corrupted) from .npy files.
    """
    def __init__(self, args, root):
        self.data, self.label = load_data(root, args.corruption, args.severity)
        
        # Optional debug mode
        if getattr(args, 'debug', False):
            self.data = self.data[:5]
            self.label = self.label[:5]

        self.class_name = [
            "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", 
            "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot", 
            "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", 
            "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink", 
            "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", 
            "wardrobe", "xbox"
        ]

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        
        if isinstance(label, np.ndarray):
            label = label.item()
            
        class_name = self.class_name[label]
        rgb = torch.ones_like(torch.asarray(pointcloud)).float()
        
        return pointcloud, label, class_name, rgb

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNN_C(Dataset):
    """
    Loader for ScanObjectNN-C (Corrupted).
    """
    def __init__(self, args, root):
        self.data, self.label = load_data(root, args.corruption, args.severity)
        
        if getattr(args, 'debug', False):
            self.data = self.data[:5]
            self.label = self.label[:, :5] if self.label.ndim > 1 else self.label[:5]

        self.class_name = [
            "bag", "bin", "box", "cabinet", "chair", "desk", "display", 
            "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"
        ]

    def __getitem__(self, item):
        pointcloud = self.data[item]
        
        # ScanObjectNN labels sometimes stored as [1, N] or [N, 1]
        try:
            label = self.label[0][item] 
        except:
            label = self.label[item]

        if isinstance(label, np.ndarray):
             label = label.item()
             
        class_name = self.class_name[int(label)]
        rgb = torch.ones_like(torch.asarray(pointcloud)).float()
        
        return pointcloud, int(label), class_name, rgb

    def __len__(self):
        return self.data.shape[0]


def load_data_partseg(root, args):
    import glob
    all_data = []
    all_label = []
    all_seg = []

    file = glob.glob(os.path.join(root, args.corruption + '_4.h5'))

    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        # f = h5py.File(file, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')  # part seg label
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg

class ShapeNetC(Dataset):
    def __init__(self, args, root, npoints=2048, class_choice=None, sub=None):
        # if args.corruption == 'clean':
        self.data, self.label, self.seg = load_data_partseg(root, args)

        self.cat2id = {
            'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
            'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
            'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15
        }
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]  # number of parts for each category
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        # self.partition = partition
        self.class_choice = class_choice
        self.npoints = npoints

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        seg = self.seg[item]  # part seg label

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        pointcloud = pointcloud[choice, :]
        seg = seg[choice]

        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class ShapeNetC2(Dataset):
    """
    Loader for ShapeNetCore-C (Corrupted).
    """
    def __init__(self, args, root):
        self.data, self.label = load_data(root, args.corruption, args.severity)
        
        if getattr(args, 'debug', False):
            self.data = self.data[:5]
            self.label = self.label[:5]

        self.class_name = [
            "airplane", "bag", "basket", "bathtub", "bed", "bench", "bottle", "bowl", "bus", 
            "cabinet", "can", "camera", "cap", "car", "chair", "clock", "dishwasher", 
            "monitor", "table", "telephone", "tin_can", "tower", "train", "keyboard", 
            "earphone", "faucet", "file", "guitar", "helmet", "jar", "knife", "lamp", 
            "laptop", "speaker", "mailbox", "microphone", "microwave", "motorcycle", "mug", 
            "piano", "pillow", "pistol", "pot", "printer", "remote_control", "rifle", 
            "rocket", "skateboard", "sofa", "stove", "vessel", "washer", "cellphone", 
            "birdhouse", "bookshelf"
        ]

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        
        if isinstance(label, np.ndarray):
             label = label.item()
             
        class_name = self.class_name[int(label)]
        rgb = torch.ones_like(torch.asarray(pointcloud)).float()
        
        return pointcloud, int(label), class_name, rgb

    def __len__(self):
        return self.data.shape[0]