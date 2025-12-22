import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import webdataset as wds
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from utils import create_model, seed_everything

# ---------------- CONFIG ----------------
DATA_ROOT = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/labeled_shards"
SHARDS = "labeled_shard_{000000..000009}.tar"

CKPT_DIR = "/cluster/home/jiapan/Supervised_Research/checkpoints"
MODEL_TYPE = "mae"
CKPT_NAME = "epoch_20.pth"

TARGET_LABEL = "is_military"    # OPTIONS: "is_military", "has_cross", or "deathyear"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
NUM_EPOCHS = 20
LR = 1e-3
SEED = 42

TRAIN_SPLIT = 0.8  # 80% train, 20% val
# ----------------------------------------

seed_everything(SEED)

# ---------------- DATA ----------------
transform = transforms.Compose([transforms.ToTensor()])

def make_sample(sample):
    if "jpg" not in sample or "json" not in sample:
        return None

    image = transform(sample["jpg"])
    labels = sample["json"]

    if labels[TARGET_LABEL] is None:
        return None

    if TARGET_LABEL in ["is_military", "has_cross"]:
        target = torch.tensor(float(labels[TARGET_LABEL]), dtype=torch.float32)
    else:  # deathyear
        target = torch.tensor(labels[TARGET_LABEL], dtype=torch.float32)
    
    # NOTE: these values are calculated using /dataset/compute_label_stats.py
    DEATHYEAR_MEAN = 1991.709
    DEATHYEAR_STD = 20.417

    if TARGET_LABEL == "deathyear":
        target = (target - DEATHYEAR_MEAN) / DEATHYEAR_STD

    return image, target

# load dataset
dataset = (
    wds.WebDataset(os.path.join(DATA_ROOT, SHARDS))
    .shuffle(1000)
    .decode("pil")
    .map(make_sample)
)

samples = [s for s in dataset if s is not None]

# ---------------- STRATIFIED SPLIT ----------------
# For boolean labels, stratify. For deathyear (continuous), do random split
if TARGET_LABEL in ["is_military", "has_cross"]:
    _, targets = zip(*samples)
    train_idx, val_idx = train_test_split(
        range(len(samples)),
        train_size=TRAIN_SPLIT,
        stratify=targets,
        random_state=SEED
    )
else:  # TARGET_LABEL == "deathyear"
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

train_samples = [samples[i] for i in train_idx]
val_samples = [samples[i] for i in val_idx]

train_loader = DataLoader(
    train_samples,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_samples,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"Train samples: {len(train_samples)}")
print(f"Val samples:   {len(val_samples)}")

# ---------------- CLASS RATIOS / DISTRIBUTION CHECK ----------------
def print_class_ratios(samples, label_name):
    labels = np.array([s[1] for s in samples])
    if label_name in ["is_military", "has_cross"]:
        num_pos = np.sum(labels == 1.0)
        num_neg = np.sum(labels == 0.0)
        total = len(labels)
        print(f"Label '{label_name}': {num_pos}/{total} positive ({num_pos/total:.3f}), "
              f"{num_neg}/{total} negative ({num_neg/total:.3f})")
    else:  # regression
        print(f"Label '{label_name}': min={labels.min():.3f}, max={labels.max():.3f}, "
              f"mean={labels.mean():.3f}, std={labels.std():.3f}")

print("\n--- Label statistics ---")
print_class_ratios(train_samples, TARGET_LABEL)
print_class_ratios(val_samples, TARGET_LABEL)

# ---------------- MODEL ----------------
model, _ = create_model(type=MODEL_TYPE, device=DEVICE)
ckpt_path = os.path.join(CKPT_DIR, MODEL_TYPE, CKPT_NAME)
state_dict = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# freeze encoder
for p in model.parameters():
    p.requires_grad = False

# infer embedding dim
with torch.no_grad():
    dummy = torch.zeros(1, 3, 256, 256).to(DEVICE)
    emb_dim = model.encoder(dummy).shape[1]

print("Encoder embedding dimension:", emb_dim)

# linear probe
classifier = nn.Linear(emb_dim, 1).to(DEVICE)

# choose loss based on task type
if TARGET_LABEL in ["is_military", "has_cross"]:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.MSELoss()

optimizer = optim.Adam(classifier.parameters(), lr=LR)

# ---------------- TRAIN ----------------
def evaluate(loader):
    classifier.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            z = model.encoder(x)
            out = classifier(z).squeeze(1)

            if TARGET_LABEL in ["is_military", "has_cross"]:
                logits = out
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.numel()
                loss = criterion(logits, y)
                total_loss += loss.item()
            else:  # regression
                loss = criterion(out, y)
                total_loss += loss.item()

    if TARGET_LABEL in ["is_military", "has_cross"]:
        acc = correct / total
        return total_loss / len(loader), acc
    else:
        return total_loss / len(loader), None

print("\nStarting linear probing...\n")

for epoch in range(NUM_EPOCHS):
    classifier.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            z = model.encoder(x)

        out = classifier(z).squeeze(1)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(val_loader)

    if TARGET_LABEL in ["is_military", "has_cross"]:
        print(
            f"[Epoch {epoch+1:02d}] "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
        )
    else:
        print(
            f"[Epoch {epoch+1:02d}] "
            f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

print("\n✅ Linear probing finished.")
