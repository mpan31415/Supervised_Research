import matplotlib.pyplot as plt
from utils import load_images_as_tensors

# --- CONFIG ---
SAMPLE_IMAGES_DIR = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/sample_images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SAVE_PATH = "/cluster/home/jiapan/Supervised_Research/plots/sample_images.png"
ROWS, COLS = 4, 5

# 1. load data
image_tensors = load_images_as_tensors(SAMPLE_IMAGES_DIR, IMAGE_EXTS)

# 2. select the first 20 images
if len(image_tensors) < 20:
    print(f"Warning: Only found {len(image_tensors)} images. Adjusting grid.")
    samples = image_tensors
else:
    samples = image_tensors[:20]

# 3. create the 4x5 figure
fig, axes = plt.subplots(nrows=ROWS, ncols=COLS, figsize=(15, 12))
# fig.suptitle(f"Sample Grid: First {len(samples)} Images", fontsize=20)

for i, ax in enumerate(axes.flat):
    if i < len(samples):
        # convert (C, H, W) to (H, W, C) for matplotlib
        img_np = samples[i].permute(1, 2, 0).cpu().numpy()
        
        ax.imshow(img_np)
        # ax.set_title(f"Idx: {i}", fontsize=10)
    
    ax.axis('off')

# 4. save the result
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(SAVE_PATH, dpi=300)
plt.close()

print(f"✅ Successfully saved 4x5 grid to: {SAVE_PATH}")
