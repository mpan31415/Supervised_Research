import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import create_model, seed_everything

# ---------------- CONFIG ----------------
DATA_ROOT = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/labeled_shards"
SHARDS = "labeled_shard_{000000..000009}.tar"

CKPT_DIR = "/cluster/home/jiapan/Supervised_Research/checkpoints"
MODEL_TYPE = "mae"
CKPT_NAME = "epoch_100.pth"

TARGET_LABEL = "deathyear"

NUM_TUNE_LAYERS = 1     # <-- K: final transformer blocks to finetune
HEAD_TYPE = "linear"    # OPTIONS: "linear", "nonlinear"
LR_ENCODER = 1e-5
LR_HEAD = 1e-3

CONF_MAT_SAVE_NAME = f"{MODEL_TYPE}_partial{NUM_TUNE_LAYERS}_deathyear.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_EPOCHS = 100
SEED = 42
TRAIN_SPLIT = 0.8
# ----------------------------------------

seed_everything(SEED)

# ---------------- DATA LOADING ----------------
transform = transforms.Compose([transforms.ToTensor()])

def make_sample(sample):
    if "jpg" not in sample or "json" not in sample:
        return None
    labels = sample["json"]
    if labels.get(TARGET_LABEL) is None:
        return None
    image = transform(sample["jpg"])
    target = float(labels[TARGET_LABEL])
    return image, target

dataset = (
    wds.WebDataset(os.path.join(DATA_ROOT, SHARDS))
    .decode("pil")
    .map(make_sample)
)
all_samples = [s for s in dataset if s is not None]

# Random Split
indices = list(range(len(all_samples)))
random.shuffle(indices)
split_idx = int(len(all_samples) * TRAIN_SPLIT)
train_indices, val_indices = indices[:split_idx], indices[split_idx:]

train_base = [all_samples[i] for i in train_indices]
val_base = [all_samples[i] for i in val_indices]

# NOTE: "True" label if first is LARGER (newer)
class PairwiseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_a, year_a = self.samples[idx]
        img_b, year_b = self.samples[random.randint(0, len(self.samples) - 1)]
        label = torch.tensor(1.0 if year_a > year_b else 0.0, dtype=torch.float32)
        return img_a, img_b, label

train_loader = DataLoader(PairwiseDataset(train_base), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(PairwiseDataset(val_base), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- MODEL & PARTIAL UNFREEZING ----------------
model, _ = create_model(type=MODEL_TYPE, device=DEVICE)
model.load_state_dict(torch.load(os.path.join(CKPT_DIR, MODEL_TYPE, CKPT_NAME), map_location=DEVICE))

encoder = model.encoder
encoder.train()

# Freeze all, then unfreeze last K blocks + final norm
for p in encoder.parameters():
    p.requires_grad = False
for block in encoder.transformer.layers[-NUM_TUNE_LAYERS:]:
    for p in block.parameters():
        p.requires_grad = True
for p in encoder.transformer.norm.parameters():
    p.requires_grad = True

# Infer dim
with torch.no_grad():
    emb_dim = encoder(torch.zeros(1, 3, 256, 256).to(DEVICE)).shape[1]

# Pairwise Head
if HEAD_TYPE == "linear":
    classifier = nn.Linear(emb_dim * 2, 1).to(DEVICE)
else:
    classifier = nn.Sequential(
        nn.Linear(emb_dim * 2, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW([
    {"params": classifier.parameters(), "lr": LR_HEAD},
    {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": LR_ENCODER},
], weight_decay=1e-4)

# ---------------- TRAIN/EVAL ----------------
def run_epoch(loader, is_train=True):
    classifier.train() if is_train else classifier.eval()
    encoder.train() if is_train else encoder.eval()
    
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    
    with context:
        for img_a, img_b, y in loader:
            img_a, img_b, y = img_a.to(DEVICE), img_b.to(DEVICE), y.to(DEVICE)

            # Pass both through encoder (gradients will flow if is_train)
            z_a = encoder(img_a)
            z_b = encoder(img_b)
            z_combined = torch.cat([z_a, z_b], dim=1)
            
            logits = classifier(z_combined).squeeze(1)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            total_loss += loss.item()

    return total_loss / len(loader), correct / total

# ---------------- MAIN ----------------
print(f"Finetuning {NUM_TUNE_LAYERS} layers for Pairwise Deathyear...")

for epoch in range(NUM_EPOCHS):
    tr_loss, tr_acc = run_epoch(train_loader, is_train=True)
    val_loss, val_acc = run_epoch(val_loader, is_train=False)
    print(f"[Epoch {epoch+1:02d}] Train Acc: {tr_acc:.3f} | Val Acc: {val_acc:.3f}")

# ---------------- CONFUSION MATRIX ----------------
classifier.eval()
encoder.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for img_a, img_b, y in val_loader:
        z = torch.cat([encoder(img_a.to(DEVICE)), encoder(img_b.to(DEVICE))], dim=1)
        preds = (torch.sigmoid(classifier(z).squeeze(1)) > 0.5).float()
        y_true.append(y.cpu())
        y_pred.append(preds.cpu())

y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
cm = confusion_matrix(y_true, y_pred)
cm = cm[[1, 0], :][:, [1, 0]] # Reorder for [A-Larger, B-Larger]

plt.figure(figsize=(4, 4))
sns.heatmap(cm.astype(float)/cm.sum()*100, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["A Larger", "B Larger"], yticklabels=["A Larger", "B Larger"], cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()

save_path = os.path.join("/cluster/home/jiapan/Supervised_Research/plots/", MODEL_TYPE, CONF_MAT_SAVE_NAME)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Confusion matrix saved to {save_path}")
