from vit_pytorch import ViT, MAE, Dino
from torch import nn, optim
import torch
import os
import random
import numpy as np


def create_model(type: str, device: str) -> nn.Module:
    
    print("=" * 50)
    print("=" * 50)
    print("=" * 50)
    print(f"\n          Creating {type.upper()} model ...\n")
    print("=" * 50)
    print("=" * 50)
    print("=" * 50)
    
    # MAE
    if type == "mae":
        encoder = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
        )
        model = MAE(
            encoder = encoder,
            masking_ratio = 0.75,   # the paper recommended 75% masked patches
            decoder_dim = 512,      # paper showed good results with just 512
            decoder_depth = 6       # anywhere from 1 to 8
        ).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
    # DINO
    elif type == "dino":
        encoder = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
        )
        model = Dino(
            encoder,
            image_size = 256,
            hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
            projection_hidden_size = 256,      # projector network hidden dimension
            projection_layers = 4,             # number of layers in projection network
            num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
            student_temp = 0.9,                # student temperature
            teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
            global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
        ).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=3e-4)
        optimizer = optim.Adam(model.parameters(), lr=3e-3)
    
    # MP3
    elif type == "mp3":
        pass  # Placeholder for MP3 model creation
    
    # simMIM
    elif type == "simmim":
        pass  # Placeholder for simMIM model creation
    
    else:
        raise ValueError(f"Model type {type} not recognized. Choose from 'mae', 'dino', 'mp3', 'simmim'.")
    
    return encoder, model, optimizer


def decoder_skip(exc, sample=None, key=None, url=None):
    # return True to silently skip the sample
    return True


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True