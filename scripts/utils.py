from vit_pytorch import ViT, MAE, Dino
from vit_pytorch.mpp import MPP
from vit_pytorch.simmim import SimMIM
from torch import nn, optim
import torch
import os
import random
import numpy as np
from typing import Tuple


def create_model(type: str, device: str) -> Tuple[nn.Module, optim.Optimizer]:
    
    print("=" * 50)
    print(f"\n          Creating {type.upper()} model ...\n")
    print("=" * 50)
    
    # create encoder
    encoder = create_encoder()
    
    ######################## MAE ########################
    if type == "mae":        
        model = MAE(
            encoder = encoder,
            masking_ratio = 0.75,   # the paper recommended 75% masked patches
            decoder_dim = 512,      # paper showed good results with just 512
            decoder_depth = 6       # anywhere from 1 to 8
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.05
        )

    ######################## DINO ########################
    elif type == "dino":        
        model = Dino(
            net = encoder,
            image_size = 256,
            hidden_layer = 'to_latent',          # hidden layer name or index, from which to extract the embedding
            projection_hidden_size = 512,        # projector network hidden dimension
            projection_layers = 3,               # number of layers in projection network
            num_classes_K = 65336,               # output logits dimensions (referenced as K in paper)
            student_temp = 0.1,                  # student temperature
            teacher_temp = 0.04,                 # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale = 0.4,        # upper bound for local crop - 0.4 was recommended in the paper 
            global_lower_crop_scale = 0.4,       # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay = 0.996,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay = 0.99,  # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = 3e-4,
            weight_decay = 0.04
        )
    
    ######################## MPP ########################
    elif type == "mpp":
        pass
    
    ######################## simMIM ########################
    elif type == "simmim":
        model = SimMIM(
            encoder = encoder,
            masking_ratio = 0.5  # they found 50% to yield the best results
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    ######################## ERROR ########################
    else:
        raise ValueError(f"Model type {type} not recognized. Choose from 'mae', 'dino', 'mpp', 'simmim'.")
    
    return model, optimizer


def create_encoder() -> ViT:
    encoder = ViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 1000,   # this gets overwritten below so doesn't matter
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 1536
    )
    # remove classifier head
    encoder.mlp_head = torch.nn.Identity()
    return encoder


def decoder_skip(exc, sample=None, key=None, url=None):
    # return True to silently skip the sample
    return True


def seed_everything(seed) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True