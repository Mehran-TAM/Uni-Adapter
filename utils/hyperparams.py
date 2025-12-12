def get_hyperparams(dataset_name):
    """
    Returns TTA hyperparameters.
    Reference: Uni-Adapter paper. 
    Lambda ~ 0.1 balances smoothing with label integrity.
    Cluster Capacity ~ 30 balances diversity with noise.
    """
    
    # Defaults (aligns with ModelNet)
    params = {
        "shot_capacity": 30,    # As per paper finding
        "beta": 150,            # Confidence decay
        "threshold": 0.5,       # Graph adjacency threshold
        "lambda_reg": 0.11,     # Approx 0.1
    }

    if "modelnet" in dataset_name.lower():
        params["lambda_reg"] = 0.07 # 0.11
        params["threshold"] = 0.5

    elif "scanobject" in dataset_name.lower():
        # ScanObjectNN usually requires slightly stronger regularization
        params["lambda_reg"] = 0.20 
        params["threshold"] = 0.5
        
    elif "shapenet" in dataset_name.lower():
        params["lambda_reg"] = 0.07 
        params["threshold"] = 0.45 
        # Example: ShapeNet might perform better with the exact inverse or older approx
        params["use_new_approximation"] = False 

    return params