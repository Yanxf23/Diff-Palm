import torch as th

def dev():
    return th.device("cuda" if th.cuda.is_available() else "cpu")

def setup_dist():
    # Skip distributed setup
    pass

def get_rank():
    return 0  # Single process only

def get_world_size():
    return 1  # Pretend there is only one process

def synchronize():
    pass

def sync_params(params):
    pass  # No syncing needed

def load_state_dict(path, map_location=None):
    import torch
    return torch.load(path, map_location=map_location)

def master_only_print(*args, **kwargs):
    print(*args, **kwargs)