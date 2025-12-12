import sys
import os
import torch
import logging
import numpy as np
import random
from datetime import datetime

from utils.params import parse_args
from utils.hyperparams import get_hyperparams
from utils.load_models import load_vlm_model
from Uni_Adapter import test_zeroshot_3d_core
from data.data_utils import load_tta_dataset
from utils.tokenizer import SimpleTokenizer
from utils.distributed import init_distributed_device
from utils.logger import setup_logging

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    
    # 1. Setup Environment
    if args.name is None:
        args.name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
    log_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(os.path.join(log_dir, 'out.log'), logging.INFO)
    init_distributed_device(args)
    random_seed(args.seed, args.rank if hasattr(args, 'rank') else 0)

    logging.info(f"Running Experiment: {args.name}")
    logging.info(f"Args: {args}")
    

    # 2. Get Hyperparameters (Paper Settings)
    hp = get_hyperparams(args.dataset_name)
    logging.info(f"Hyperparameters: {hp}")
    
    # if args.batch_size > 1:
    #     logging.warning(f"Batch size {args.batch_size} is more than recommended minimum {1} for dataset {args.dataset_name}.")
    #     args.use_new_approximation = False

    # 3. Load Models
    clip_model, model = load_vlm_model(args)
    tokenizer = SimpleTokenizer()

    # 4. Define Corruptions
    corruptions = [
        'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
        'distortion_rbf', 'distortion_rbf_inv', 'density', 'density_inc',
        'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
    ]
    if args.corruption != 'all':
        corruptions = [args.corruption]

    # 5. Main Loop
    results_summary = {}
    GREEN = "\033[92m"
    RESET = "\033[0m"

    for corr in corruptions:
        args.corruption = corr 
        logging.info(f"\n{'='*20} Processing Corruption: {corr} {'='*20}")
        
        # green print:
        print(f"{GREEN}")
        print(f"Loading data for corruption: {corr}")
        print(f"{RESET}")
        # Load Data
        dataset = load_tta_dataset(args)
        test_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.workers, 
            pin_memory=True, 
            drop_last=False
        )

        # Run Core Logic
        result = test_zeroshot_3d_core(
            test_loader=test_loader,
            validate_dataset_name=args.validate_dataset_name,
            model=model,
            clip_model=clip_model,
            tokenizer=tokenizer,
            args=args,
            hp=hp
        )
        
        results_summary[corr] = result['acc1']
    
    logging.info(f"Summary of Results: {results_summary}")
    logging.info(f"Average Top-1: {np.mean(list(results_summary.values())):.3f}")

if __name__ == '__main__':
    main()