import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import create_model, overlay_corner_grids

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
model_type = "dino"
ckpt_name = "epoch_100.pth"

# plot save dir
plot_save_dir = str(project_root) + "/plots/" + model_type + "/"


################## 1. LOAD DATA ##################
# Load all images and convert them to tensors
image_tensors_list = load_images_as_tensors(SAMPLE_IMAGES_DIR, IMAGE_EXTS)

# --- Verification and Summary ---

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
# NOTE: to bypass deepcopy bug in DINO implementation, use create_model with type="mae" for both MAE and DINO
model, _ = create_model(type="mae", device=DEVICE)

# load checkpoint
ckpt_path = os.path.join(ckpt_dir, model_type, ckpt_name)
state_dict = torch.load(ckpt_path, map_location=DEVICE)

if model_type == "mae":
    model.load_state_dict(state_dict)
    encoder = model.encoder
    encoder.eval()
else:
    encoder = model.encoder
    encoder.load_state_dict(state_dict)
    encoder.eval()
print(f"✅ Successfully loaded model weights from: {ckpt_path}")


###### HELPER FUNCTION ######
def encode_in_batches(encoder, images, batch_size=32):
    latents = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(DEVICE)
            z = encoder(batch)
            latents.append(z.cpu())
            del batch, z
            torch.cuda.empty_cache()
    return torch.cat(latents, dim=0)


################## 3. PCA and t-SNE ON ALL IMAGES ##################
all_img_batch = torch.stack(image_tensors_list, dim=0)
latent_embeddings = encode_in_batches(
    encoder,
    all_img_batch,
    batch_size=100
)

print("\n--- Model Output ---")
print(f"Shape of latent embeddings (Encoder Output Size): **{latent_embeddings.shape}**")
latent_embeddings = latent_embeddings.cpu().numpy()   # convert to numpy for PCA

# PCA-2
Z_pca = PCA(n_components=2).fit_transform(latent_embeddings)

plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.scatter(Z_pca[:, 0], Z_pca[:, 1], s=15, alpha=0.4, c='gray', edgecolors='none')
overlay_corner_grids(ax, Z_pca, image_tensors_list, zoom=0.2)
# ax.set_title("PCA of Latent Embeddings (with extreme images)", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel("Principal Component 1", fontsize=18)
ax.set_ylabel("Principal Component 2", fontsize=18)
ax.grid(True)
plt.savefig(plot_save_dir + "/" + model_type + "_pca.png", dpi=300)
plt.close()
print("PCA plot saved.")

# PCA-50 then TSNE-2
Z_50 = PCA(n_components=50).fit_transform(latent_embeddings)
Z_tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    random_state=42
).fit_transform(Z_50)

plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=15, alpha=0.4, c='gray', edgecolors='none')
overlay_corner_grids(ax, Z_tsne, image_tensors_list, zoom=0.2)
# ax.set_title("t-SNE of Latent Embeddings (with extreme images)", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel("t-SNE Dimension 1", fontsize=18)
ax.set_ylabel("t-SNE Dimension 2", fontsize=18)
ax.grid(True)
plt.savefig(plot_save_dir + "/" + model_type + "_tsne.png", dpi=300)
plt.close()
print("t-SNE plot saved.")


################## 4. PCA EXPLAIN INFORMATION ##################
pca = PCA().fit(latent_embeddings)
indices = np.arange(0, 31)
expl_var_ratios = [0] + list(np.cumsum(pca.explained_variance_ratio_[:30]))

plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.plot(indices, expl_var_ratios, marker='o')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('Number of Components', fontsize=18)
ax.set_ylabel('Cumulative Explained Variance', fontsize=18)
# ax.set_title('PCA Explained Variance', fontsize=16)
ax.grid(True)
plt.savefig(plot_save_dir + "/" + model_type + "_expl_var.png", dpi=300)
plt.close()
print("Explained variance plot saved.")


##########################################
print("\n✅ Evaluation complete. Plots saved to:", plot_save_dir)
print("\n\n\n\n")
