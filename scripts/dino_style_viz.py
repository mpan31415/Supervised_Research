import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import create_model

current_dir = Path.cwd()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    print(f"Added project root to sys.path: {project_root}")
from dataset.utils import load_images_as_tensors


################## 0. CONFIG ##################
# images
SAMPLE_IMAGES_DIR = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/sample_images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# model checkpoint
ckpt_dir = "/cluster/home/jiapan/Supervised_Research/checkpoints"
model_type = "mae"
ckpt_name = "epoch_30.pth"

# plot save dir
plot_save_dir = str(project_root) + "/scripts/eval_plots/"


################## 1. LOAD DATA ##################
# Load all images and convert them to tensors
image_tensors_list = load_images_as_tensors(SAMPLE_IMAGES_DIR, IMAGE_EXTS)
num_loaded = len(image_tensors_list)
print(f"\n--- Summary ---")
print(f"Successfully loaded and converted **{num_loaded}** images to PyTorch tensors.")
if num_loaded > 0:
    # Check the type and shape of the first tensor
    first_tensor = image_tensors_list[0]
    print(f"Type of first element: **{type(first_tensor)}**")
    print(f"Shape of first tensor (C, H, W): **{first_tensor.shape}**")
    print(f"Data type (dtype): **{first_tensor.dtype}**")
    print(f"Min/Max values: **{first_tensor.min():.4f}** / **{first_tensor.max():.4f}**")
print("\nAll image tensors are now available in the `image_tensors_list` list.")


################## 2. CREATE ENCODER & LOAD CHECKPOINT ##################
# new model object
model, _ = create_model(type=model_type, device=DEVICE)

# load checkpoint
ckpt_path = os.path.join(ckpt_dir, model_type, ckpt_name)
state_dict = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Successfully loaded MAE model weights from: {ckpt_path}")


###### 1. Extract patch embeddings from images ######
NUM_IMAGES = 20   # DINOv2 uses 3; this is perfect
PATCH_H, PATCH_W = 16, 16  # adapt to your model
D = 768

image_batch = torch.stack(image_tensors_list[:NUM_IMAGES]).to(DEVICE)

with torch.no_grad():
    tokens = model.get_patch_embeddings(image_batch)   # (B, 1+N, D)
    print("=" * 100)
    print("Extracted patch embeddings shape:", tokens.shape)
    print("=" * 100)
    patch_tokens = tokens[:, 1:, :]                    # (B, N, D)

patch_tokens = patch_tokens.cpu().numpy()


###### 2. Perform PCA per image (foreground selection) ######
foreground_patches = []
foreground_masks = []

for i in range(NUM_IMAGES):
    patches = patch_tokens[i]  # (N, D)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(patches).squeeze()  # (N,)
    
    # pc1 distribution information
    print(f"Image {i+1} - PC1 min: {pc1.min():.4f}, max: {pc1.max():.4f}, mean: {pc1.mean():.4f}, std: {pc1.std():.4f}")

    mask = pc1 > 0   # foreground mask
    foreground_masks.append(mask)

    foreground_patches.append(patches[mask])


###### 3. Second PCA across images (semantic alignment) ######
all_fg_patches = np.concatenate(foreground_patches, axis=0)

pca_rgb = PCA(n_components=3)
rgb_feats = pca_rgb.fit_transform(all_fg_patches)

# Normalize for visualization
rgb_feats -= rgb_feats.min(axis=0)
rgb_feats /= rgb_feats.max(axis=0)


###### 4. Assign RGB colors back to patches ######
colored_patches = []
idx = 0

for i in range(NUM_IMAGES):
    mask = foreground_masks[i]
    num_fg = mask.sum()

    colors = np.zeros((PATCH_H * PATCH_W, 3))
    colors[mask] = rgb_feats[idx:idx + num_fg]
    idx += num_fg

    colored_patches.append(colors.reshape(PATCH_H, PATCH_W, 3))


###### 5. Visualization ######
fig, axes = plt.subplots(2, NUM_IMAGES, figsize=(4 * NUM_IMAGES, 8))

for i in range(NUM_IMAGES):
    # Original image
    img = image_batch[i].permute(1, 2, 0).cpu().numpy()
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Image {i+1}")
    axes[0, i].axis("off")

    # PCA-colored patches
    axes[1, i].imshow(colored_patches[i])
    axes[1, i].set_title("PCA-colored patches")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig(plot_save_dir + "dino_style_pca_patches.png", dpi=300)
plt.close()

print(f"✅ Visualization saved to: {plot_save_dir}dino_style_pca_patches.png")
