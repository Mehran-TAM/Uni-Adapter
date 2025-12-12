import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Uni-Adapter: Adapt-As-You-Walk Through the Clouds")

    # ========================= System & Paths =========================
    parser.add_argument('--name', type=str, default=None, help="Experiment name")
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Directory to save logs/results')
    

    # Config & Metadata Paths (NO HARDCODING IN CODE)
    parser.add_argument('--templates-path', type=str, default='./data/templates.json', help='Path to templates.json')
    parser.add_argument('--labels-path', type=str, default='./data/labels.json', help='Path to labels.json')
    

    

    

    # ========================= Model Config =========================
    parser.add_argument('--vlm3d', type=str, default='uni3d', choices=['uni3d', 'ulip', 'openshape'])
    parser.add_argument('--model', type=str, default='create_uni3d', help='Model creation function name')
    parser.add_argument("--patch-dropout", type=float, default=0., help="flip patch dropout.")
    parser.add_argument('--use-new-approximation', default=True, type=bool, help='Whether to use the new approximation method for cache logits')
    parser.add_argument('--drop-path-rate', default=0.0, type=float)

    # Uni3D Specifics
    parser.add_argument('--precomputed-text-features', type=str, default=r"C:\Users\reza_moradi\Desktop\point\AAAI\code\precomputed_text_features\Uni3D\text_features_large.pt", help="Path to Uni3D checkpoint")

    parser.add_argument('--clip-uni3d-model', type=str, default="EVA02-E-14-plus", help="CLIP backbone name")
    parser.add_argument('--clip-uvi3d-path', type=str, default=r"C:\Users\reza_moradi\Desktop\point\AAAI\code\pretrained\open_clip_pytorch_model.bin", help="CLIP backbone name")
    parser.add_argument('--pc-model-uni3d', type=str, default="eva02_large_patch14_448", help="Point cloud backbone") # for Uni3D
    parser.add_argument('--pretrained-pc-uni3d', type=str, default=r"C:\Users\reza_moradi\Desktop\point\AAAI\code\pretrained\uni3d_L_ensembled_model.pt", help="Path to Uni3D checkpoint")
    
    
    parser.add_argument('--pc-feat-dim-uni3d', type=int, default=1024, help="Point cloud feature dimension") # for Uni3D
    parser.add_argument('--embed-dim-uni3d', type=int, default=1024, help="Embedding dimension") # for Uni3D
    parser.add_argument('--num-group-uni3d', type=int, default=512, help="Embedding dimension") # for Uni3D
    parser.add_argument('--group_size_uni3d', type=int, default=64, help="Embedding dimension") # for Uni3D
    parser.add_argument('--pc_encoder_dim_uni3d', type=int, default=512, help="Embedding dimension") # for Uni3D


    # OpenShape Specifics
    parser.add_argument("--oshape-version", type=str, choices=["vitg14", "vitl14"], default="vitg14")
    parser.add_argument('--pc-model-oshape', type=str, default="eva02_large_patch14_448", help="Point cloud backbone") # for OpenShape
    parser.add_argument('--pretrained-pc-oshape', type=str, default=r"C:\Users\reza_moradi\Desktop\point\AAAI\code\pretrained\uni3d_g_ensembled_model.pt", help="Path to Uni3D checkpoint")
    
    parser.add_argument('--pc-feat-dim-oshape', type=int, default=1024, help="Point cloud feature dimension") # for OpenShape
    parser.add_argument('--embed-dim-oshape', type=int, default=1024, help="Embedding dimension") # for OpenShape
    parser.add_argument('--num-group-oshape', type=int, default=1024, help="Embedding dimension") # for OpenShape
    parser.add_argument('--group_size_oshape', type=int, default=1024, help="Embedding dimension") # for OpenShape
    parser.add_argument('--pc_encoder_dim_oshape', type=int, default=1024, help="Embedding dimension") # for OpenShape



    # ULIP Specifics
    parser.add_argument('--slip-ckpt-path', type=str, default=None, help="Path to SLIP weights")
    parser.add_argument('--ulip-version', type=str, default="ulip2", help="ulip version")
    parser.add_argument('--pc-model-ulip', type=str, default="eva02_large_patch14_448", help="Point cloud backbone") # for ULIP
    parser.add_argument('--pretrained-pc-ulip', type=str, default=r"C:\Users\reza_moradi\Desktop\point\AAAI\code\pretrained\uni3d_g_ensembled_model.pt", help="Path to Uni3D checkpoint")
    
    parser.add_argument('--pc-feat-dim-ulip', type=int, default=1024, help="Point cloud feature dimension") # for ULIP
    parser.add_argument('--embed-dim-ulip', type=int, default=1024, help="Embedding dimension") # for ULIP
    parser.add_argument('--num-group-ulip', type=int, default=1024, help="Embedding dimension") # for ULIP
    parser.add_argument('--group_size_ulip', type=int, default=1024, help="Embedding dimension") # for ULIP
    parser.add_argument('--pc_encoder_dim_ulip', type=int, default=1024, help="Embedding dimension") # for ULIP





    # ========================= Data Config =========================
    # Dataset Root
    parser.add_argument('--myroot', type=str, default="C:\\Users\\reza_moradi\\Desktop\\point\\AAAI\\code\\datasets\\modelnet40_c", help='Root path to specific dataset point clouds')
    
    parser.add_argument('--dataset_name', type=str, default='modelnet', help='modelnet, scanobject, shapenetcore')
    parser.add_argument('--validate_dataset_name', default='modelnet40_openshape', type=str, help="Key in labels.json")
    parser.add_argument('--template_key', default='modelnet40_64', type=str, help="Key in templates.json")
    
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--npoints', default=1024, type=int)
    
    # Corruptions
    parser.add_argument('--corruption', type=str, default='all', help='clean, all, or specific corruption name')
    parser.add_argument('--severity', type=int, default=5)

    # ========================= Runtime =========================
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--distributed', action='store_true')

    args = parser.parse_args()
    return args