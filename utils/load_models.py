import torch
import logging
from collections import OrderedDict
import open_clip
import models.uni3d as uni3d_models
from models import openshape, ulip

def load_vlm_model(args):
    logging.info(f"=> Loading Model: {args.vlm3d}")
    
    if args.vlm3d == 'uni3d':

        args.pc_feat_dim = args.pc_feat_dim_uni3d
        args.embed_dim = args.embed_dim_uni3d
        args.num_group = args.num_group_uni3d
        args.group_size = args.group_size_uni3d
        args.pc_encoder_dim = args.pc_encoder_dim_uni3d
        return _load_uni3d(args)
    
    elif args.vlm3d == 'ulip':

        args.pc_feat_dim = args.pc_feat_dim_ulip
        args.embed_dim = args.embed_dim_ulip
        args.num_group = args.num_group_ulip
        args.group_size = args.group_size_ulip
        args.pc_encoder_dim = args.pc_encoder_dim_ulip
        return _load_ulip(args)
    
    elif args.vlm3d == 'openshape':

        args.pc_feat_dim = args.pc_feat_dim_oshape
        args.embed_dim = args.embed_dim_oshape
        args.num_group = args.num_group_oshape
        args.group_size = args.group_size_oshape
        args.pc_encoder_dim = args.pc_encoder_dim_oshape
        return _load_openshape(args)
    
    else:
        raise ValueError(f"Unknown model type: {args.vlm3d}")

def _load_uni3d(args):
    # Load CLIP
    # clip_model, _, _ = open_clip.create_model_and_transforms(
    #     model_name=args.clip_uni3d_model, 
    #     pretrained=args.clip_uvi3d_path
    # )
    # clip_model.to(args.device)
    clip_model = []

    # Load Uni3D
    model = getattr(uni3d_models, args.model)(args=args)
    
    if args.pretrained_pc_uni3d:
        logging.info(f"Loading Uni3D checkpoint from {args.pretrained_pc_uni3d}")
        checkpoint = torch.load(args.pretrained_pc_uni3d, map_location='cpu')
        sd = checkpoint.get('module', checkpoint)
        # Strip 'module.' prefix if present
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
        
    model.to(args.device)
    return clip_model, model

def _load_ulip(args):
    if not args.slip_ckpt or not args.pointbert_ckpt:
        raise ValueError("For ULIP, --slip-ckpt and --pointbert-ckpt must be provided")

    # Load CLIP Text Encoder
    clip_model = ulip.create_clip_text_encoder(args)
    slip_sd = torch.load(args.slip_ckpt, map_location='cpu')['state_dict']
    
    
    slip_sd = {k.replace('module.', ''): v for k, v in slip_sd.items()}
    slip_sd = {k:v for k,v in slip_sd.items()
                        if k.startswith('positional_embedding') or 
                           k.startswith('text_projection') or 
                           k.startswith('logit_scale') or 
                           k.startswith('transformer') or 
                           k.startswith('token_embedding') or 
                           k.startswith('ln_final')}
    
    clip_model_dict = OrderedDict(slip_sd)
    clip_model.load_state_dict(clip_model_dict)

    clip_model.half().to(args.device)
    clip_model.eval()

    # Load Point Encoder
    lm3d_model = ulip.create_ulip(args)
    print('len(lm3d_model.state_dict()):', len(lm3d_model.state_dict()))

    point_sd = torch.load(args.pointbert_ckpt, map_location='cpu')['state_dict']
    point_sd = {k.replace('module.', ''): v for k, v in point_sd.items()}
    point_sd = {k:v for k, v in point_sd.items() 
                         if k.startswith('pc_projection') or k.startswith('point_encoder')}
    print('len(pretrain_point_sd):', len(point_sd), '\n')
    
    lm3d_model_dict = OrderedDict(point_sd)
    lm3d_model.load_state_dict(lm3d_model_dict)

    lm3d_model.half().to(args.device)
    lm3d_model.eval()

    
    return clip_model, lm3d_model


def _load_openshape(args):
    # Placeholder for OpenShape loading logic
    # Ensure paths are taken from args, not hardcoded strings
    if args.oshape_version == 'vitg14':
        clip_name = 'ViT-bigG-14'
        pretrained_path = args.clip_ckpt
    else:
        clip_name = 'ViT-L-14'
        pretrained_path = args.clip_ckpt

    open_clip_model, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained=pretrained_path)
    open_clip_model.half().to(args.device)
    
    # Config loading (simplified)
    # config = load_config(args.openshape_config) 
    # lm3d_model = openshape.create_openshape(config)
    # ... Load checkpoint from args.ckpt_path ...
    
    lm3d_model = None # Stub
    
    return open_clip_model, lm3d_model