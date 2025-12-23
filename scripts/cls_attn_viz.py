import sys
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import scienceplots
from einops import repeat

from utils import create_model

current_dir = Path.cwd()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    print(f"Added project root to sys.path: {project_root}")
from dataset.utils import load_images_as_tensors

# set plotting style
plt.style.use(['science', 'ieee'])
# Disable the requirement for a system LaTeX installation
plt.rcParams.update({
    "text.usetex": False,
})


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
ckpt_name = "epoch_100.pth"

# plot save dir
plot_save_dir = str(project_root) + "/plots/" + model_type + "/"


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
print(f"✅ Successfully loaded model weights from: {ckpt_path}")


###### 1. Extract patch embeddings from images ######
START_IDX = 5
NUM_IMAGES = 16

sample_tensors = image_tensors_list[START_IDX:START_IDX+NUM_IMAGES]
image_batch = torch.stack(sample_tensors, dim=0).to(DEVICE)

PATCH_H, PATCH_W = 16, 16
D = 768

# extract encoder
encoder = model.encoder

###### 2. Get the attention maps of the CLS token ######
with torch.no_grad():
    
    # patchify
    patch_embeddings = encoder.to_patch_embedding(image_batch)
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    
    # add cls token and pos embeddings
    b, n, _ = patch_embeddings.shape
    cls_tokens = repeat(encoder.cls_token, '1 1 d -> b 1 d', b = b)
    patch_embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)
    patch_embeddings += encoder.pos_embedding[:, :(n + 1)]
    print(f"Patch embeddings with cls token shape: {patch_embeddings.shape}")
    
    # transformer
    latent_embeddings = encoder.transformer(patch_embeddings)
    print(f"Latent embeddings shape: {latent_embeddings.shape}")  # (NUM_IMAGES, num_patches + 1, D)
    
    # extract last layer attention
    last_attn_module = encoder.transformer.layers[-1][0]
    attn = last_attn_module.attn
    print(f"Attention weights shape from last block: {attn.shape}") # (b, num_heads, n, n)
    
    # extract CLS token attention (with all heads) and reshape to patch grid
    # CLS token is query index 0
    cls_attn = attn[:, :, 0, 1:]
    print(f"CLS token attention shape: {cls_attn.shape}")  # (b, num_heads, n-1)
    num_heads = cls_attn.shape[1]
    # cls_attn_mean = cls_attn.mean(dim=1)  # (B, 256)
    # print(f"CLS token mean attention shape: {cls_attn_mean.shape}")  # (b, n-1)
    cls_attn_map = cls_attn.reshape(b, num_heads, 16, 16)
    print(f"Reshaped CLS attention map shape: {cls_attn_map.shape}")
    
    
# generate subplots
# each row is one of the 20 images
# first column is input image
# each of the next 12 columns is one head's attention heatmap

fig, axes = plt.subplots(NUM_IMAGES, num_heads + 1, figsize=(3 * (num_heads + 1), 3 * NUM_IMAGES))
for img_idx in range(NUM_IMAGES):
    # plot input image
    axes[img_idx, 0].imshow(image_batch[img_idx].permute(1, 2, 0).cpu().numpy())
    axes[img_idx, 0].axis('off')
    # if img_idx == 0:
    #     axes[img_idx, 0].set_title("Input Image")
    
    # plot attention heatmaps for each head
    for head_idx in range(num_heads):
        # get attention map for this image and head
        attn_map = cls_attn_map[img_idx, head_idx].detach().cpu().numpy()
        # normalize attention map for better visualization
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        # plot
        axes[img_idx, head_idx + 1].imshow(attn_map, cmap='viridis')
        axes[img_idx, head_idx + 1].axis('off')
        # if img_idx == 0:
        #     axes[img_idx, head_idx + 1].set_title(f"Head {head_idx + 1}")

plt.tight_layout()
plt.savefig(plot_save_dir + "/" + model_type + "_cls_attn_viz.png", dpi=300)
plt.close()

print(f"✅ Saved CLS token attention visualization to: {plot_save_dir + '/' + model_type + '_cls_attn_viz.png'}")
