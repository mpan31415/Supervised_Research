from vit_pytorch import Dino
from torch import nn, optim
import torch
import torch.nn.functional as F
import os
import random
import numpy as np
from typing import Tuple
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from models import MAE, ViT


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
    
    ######################## ERROR ########################
    else:
        raise ValueError(f"Model type {type} not recognized. Choose from 'mae', 'dino'.")
    
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
    
    
def find_corner_indices(Z, k=3):
    """
    Returns indices of k points closest to each of the 4 corners.
    """
    corners = np.array([
        [Z[:, 0].min(), Z[:, 1].min()],  # bottom-left
        [Z[:, 0].min(), Z[:, 1].max()],  # top-left
        [Z[:, 0].max(), Z[:, 1].min()],  # bottom-right
        [Z[:, 0].max(), Z[:, 1].max()],  # top-right
    ])

    indices = set()
    for c in corners:
        dists = np.linalg.norm(Z - c, axis=1)
        # Get indices of k smallest distances
        nearest_k = np.argsort(dists)[:k]
        for idx in nearest_k:
            indices.add(idx)

    return list(indices)


def create_2x2_grid(tensors, border_width=4):
    """Combines 4 tensors with a white border between them."""
    # Add padding to the right and bottom of each tensor to create borders
    # F.pad format: (left, right, top, bottom)
    img0 = F.pad(tensors[0], (0, border_width, 0, border_width), value=1.0)
    img1 = F.pad(tensors[1], (0, 0, 0, border_width), value=1.0)
    img2 = F.pad(tensors[2], (0, border_width, 0, 0), value=1.0)
    img3 = F.pad(tensors[3], (0, 0, 0, 0), value=1.0)
    
    top = torch.cat((img0, img1), dim=2)
    bottom = torch.cat((img2, img3), dim=2)
    grid = torch.cat((top, bottom), dim=1)
    
    return grid.permute(1, 2, 0).cpu().numpy()


def overlay_corner_grids(ax, Z, image_tensors, zoom=0.12):
    """Places 2x2 grids in the 4 corners of the plot with connecting lines."""
    
    # 1. Find data extremes
    x_min, x_max = Z[:, 0].min(), Z[:, 0].max()
    y_min, y_max = Z[:, 1].min(), Z[:, 1].max()
    
    # Define theoretical corners in data space
    corners = [
        [x_min, y_min], # Bottom-Left
        [x_min, y_max], # Top-Left
        [x_max, y_min], # Bottom-Right
        [x_max, y_max]  # Top-Right
    ]
    
    # Fixed positions for the grids (in 'axes fraction', 0 to 1)
    grid_positions = [(0.12, 0.12), (0.12, 0.88), (0.88, 0.12), (0.88, 0.88)]

    for corner_coords, grid_pos in zip(corners, grid_positions):
        # 2. Find the 4 nearest points to this corner
        dists = np.linalg.norm(Z - corner_coords, axis=1)
        nearest_4_idx = np.argsort(dists)[:4]
        
        # 3. Create the grid image
        grid_img = create_2x2_grid([image_tensors[i] for i in nearest_4_idx])
        imagebox = OffsetImage(grid_img, zoom=zoom)
        
        # 4. Place grid anchored to the corner (axes fraction)
        ab = AnnotationBbox(
            imagebox, 
            grid_pos,
            xybox=grid_pos,
            xycoords='axes fraction',
            boxcoords='axes fraction',
            frameon=True,
            pad=0.2
        )
        ax.add_artist(ab)
        
        # 5. Draw lines from each of the 4 data points to the grid
        # for idx in nearest_4_idx:
        #     point_data = Z[idx]
            
        #     # Draw line from data point to the fixed grid position
        #     ax.annotate(
        #         '', xy=point_data, xycoords='data',
        #         xytext=grid_pos, textcoords='axes fraction',
        #         arrowprops=dict(arrowstyle="-", color='black', alpha=0.15, linewidth=0.7)
        #     )
