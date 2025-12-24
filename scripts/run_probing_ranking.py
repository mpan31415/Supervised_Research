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
PROBE_TYPE = "linear"       # OPTIONS: "linear" or "nonlinear"

CONF_MAT_SAVE_NAME = f"{MODEL_TYPE}_{PROBE_TYPE}_deathyear.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 1e-3
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

# Load raw samples into memory
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
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

train_base_samples = [all_samples[i] for i in train_indices]
val_base_samples = [all_samples[i] for i in val_indices]

# ---------------- PAIRWISE DATASET ----------------
class PairwiseDataset(Dataset):
    """
    Returns pairs of images and a binary label:
    1.0 if Year(A) > Year(B)
    0.0 if Year(A) <= Year(B)
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_a, year_a = self.samples[idx]
        
        # Pick a random second image
        idx_b = random.randint(0, len(self.samples) - 1)
        img_b, year_b = self.samples[idx_b]
        
        # Label: Is A newer than B?
        label = torch.tensor(1.0 if year_a > year_b else 0.0, dtype=torch.float32)
        
        return img_a, img_b, label

train_dataset = PairwiseDataset(train_base_samples)
val_dataset = PairwiseDataset(val_base_samples)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Total base samples: {len(all_samples)}")
print(f"Train pairs per epoch: {len(train_dataset)}")

# ---------------- MODEL ----------------
model, _ = create_model(type=MODEL_TYPE, device=DEVICE)
ckpt_path = os.path.join(CKPT_DIR, MODEL_TYPE, CKPT_NAME)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()

# Freeze encoder
for p in model.parameters():
    p.requires_grad = False

# Infer embedding dim
with torch.no_grad():
    dummy = torch.zeros(1, 3, 256, 256).to(DEVICE)
    emb_dim = model.encoder(dummy).shape[1]

# Pairwise Probe: Input is concatenation of two embeddings (dim*2)
if PROBE_TYPE == "linear":
    classifier = nn.Linear(emb_dim * 2, 1).to(DEVICE)
else:
    classifier = nn.Sequential(
        nn.Linear(emb_dim * 2, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LR)

# ---------------- TRAIN/EVAL FUNCTIONS ----------------
def run_epoch(loader, is_train=True):
    if is_train:
        classifier.train()
    else:
        classifier.eval()
        
    total_loss = 0.0
    correct = 0
    total = 0

    # Gradient context
    context = torch.enable_grad() if is_train else torch.no_grad()
    
    with context:
        for img_a, img_b, y in loader:
            img_a, img_b, y = img_a.to(DEVICE), img_b.to(DEVICE), y.to(DEVICE)

            # Get embeddings (frozen)
            with torch.no_grad():
                z_a = model.encoder(img_a)
                z_b = model.encoder(img_b)
            
            # Concatenate features
            z_combined = torch.cat([z_a, z_b], dim=1)
            
            logits = classifier(z_combined).squeeze(1)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Stats
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            total_loss += loss.item()

    return total_loss / len(loader), correct / total

# ---------------- MAIN LOOP ----------------
print(f"\nStarting {PROBE_TYPE} pairwise deathyear comparison...")

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, is_train=True)
    val_loss, val_acc = run_epoch(val_loader, is_train=False)

    print(f"[Epoch {epoch+1:02d}] Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.4f}")

# ---------------- CONFUSION MATRIX ----------------
classifier.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for img_a, img_b, y in val_loader:
        img_a, img_b = img_a.to(DEVICE), img_b.to(DEVICE)
        z_comb = torch.cat([model.encoder(img_a), model.encoder(img_b)], dim=1)
        logits = classifier(z_comb).squeeze(1)
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        y_true.append(y.cpu())
        y_pred.append(preds.cpu())

y_true = torch.cat(y_true).numpy()
y_pred = torch.cat(y_pred).numpy()

cm = confusion_matrix(y_true, y_pred)
# Reorder to [A is Larger, B is Larger]
cm = cm[[1, 0], :][:, [1, 0]]

# Plotting
plt.figure(figsize=(4, 4))
sns.heatmap(cm.astype(float)/cm.sum()*100, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["A Larger", "B Larger"], yticklabels=["A Larger", "B Larger"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()

save_path = f"/cluster/home/jiapan/Supervised_Research/plots/{MODEL_TYPE}/{CONF_MAT_SAVE_NAME}"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"\n✅ Finished. Confusion matrix saved to {save_path}")
