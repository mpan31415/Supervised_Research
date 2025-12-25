import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import webdataset as wds
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import create_model, seed_everything

# ---------------- CONFIG ----------------
DATA_ROOT = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/labeled_shards"
SHARDS = "labeled_shard_{000000..000009}.tar"
CKPT_DIR = "/cluster/home/jiapan/Supervised_Research/checkpoints"

MODEL_TYPE = "dino"
CKPT_NAME = "epoch_100.pth"

TARGET_LABEL = "is_military"    # OPTIONS: "is_military", "has_cross"

PROBE_TYPE = "linear"     # OPTIONS: "linear", "nonlinear"

CONF_MAT_SAVE_NAME = f"{MODEL_TYPE}_{PROBE_TYPE}_{TARGET_LABEL}.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 1e-3
SEED = 42

POS_CLASS_WEIGHT = 2.0

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

    target = torch.tensor(float(labels[TARGET_LABEL]), dtype=torch.float32)

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
_, targets = zip(*samples)
train_idx, val_idx = train_test_split(
    range(len(samples)),
    train_size=TRAIN_SPLIT,
    stratify=targets,
    random_state=SEED
)

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
    num_pos = np.sum(labels == 1.0)
    num_neg = np.sum(labels == 0.0)
    total = len(labels)
    print(f"Label '{label_name}': {num_pos}/{total} positive ({num_pos/total:.3f}), "
            f"{num_neg}/{total} negative ({num_neg/total:.3f})")

print("\n--- Label statistics ---")
print_class_ratios(train_samples, TARGET_LABEL)
print_class_ratios(val_samples, TARGET_LABEL)

# ---------------- MODEL ----------------
# NOTE: to bypass deepcopy bug in DINO implementation, use create_model with type="mae" for both MAE and DINO
model, _ = create_model(type="mae", device=DEVICE)

ckpt_path = os.path.join(CKPT_DIR, MODEL_TYPE, CKPT_NAME)
state_dict = torch.load(ckpt_path, map_location=DEVICE)

if MODEL_TYPE == "mae":
    model.load_state_dict(state_dict)
    encoder = model.encoder
    encoder.eval()
else:
    encoder = model.encoder
    encoder.load_state_dict(state_dict)
    encoder.eval()
print(f"✅ Successfully loaded model weights from: {ckpt_path}")

# freeze encoder
for p in encoder.parameters():
    p.requires_grad = False

# infer embedding dim
with torch.no_grad():
    dummy = torch.zeros(1, 3, 256, 256).to(DEVICE)
    emb_dim = encoder(dummy).shape[1]

print("Encoder embedding dimension:", emb_dim)

# create classifier
if PROBE_TYPE == "linear":
    # linear probe
    classifier = nn.Linear(emb_dim, 1).to(DEVICE)
elif PROBE_TYPE == "nonlinear":
    # nonlinear probe
    classifier = nn.Sequential(
        nn.Linear(emb_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(DEVICE)
else:
    raise ValueError(f"Invalid PROBE_TYPE: {PROBE_TYPE}")

# criterion and optimizer
pos_weight = torch.tensor([POS_CLASS_WEIGHT]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

            z = encoder(x)
            out = classifier(z).squeeze(1)

            logits = out
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            loss = criterion(logits, y)
            total_loss += loss.item()

    acc = correct / total
    return total_loss / len(loader), acc

print(f"\nStarting {PROBE_TYPE} probing...\n")

for epoch in range(NUM_EPOCHS):
    classifier.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            z = encoder(x)

        out = classifier(z).squeeze(1)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(val_loader)

    print(
        f"[Epoch {epoch+1:02d}] "
        f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
    )

print("\n✅ Linear probing finished.")


# -------- CONFUSION MATRIX --------
classifier.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        z = encoder(x)
        logits = classifier(z).squeeze(1)
        preds = (torch.sigmoid(logits) > 0.5).float()

        y_true.append(y.cpu())
        y_pred.append(preds.cpu())

y_true = torch.cat(y_true).numpy()
y_pred = torch.cat(y_pred).numpy()

# original cm: rows=true [neg, pos], cols=pred [neg, pos]
cm = confusion_matrix(y_true, y_pred)
# reorder to [Positive, Negative]
cm = cm[[1, 0], :][:, [1, 0]]

# -------- PLOT --------
plt.figure(figsize=(4, 4))
cm_percent = cm.astype(np.float32) / cm.sum() * 100

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=["True", "False"],
    yticklabels=["True", "False"],
    cbar=False,
    annot_kws={"size": 40}
)

plt.xlabel("Predicted", fontsize=40)
plt.ylabel("Actual", fontsize=40)

# Move x-axis labels to top
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().tick_params(axis='both', which='major', labelsize=40)

save_path = "/cluster/home/jiapan/Supervised_Research/plots/" + MODEL_TYPE + "/" + CONF_MAT_SAVE_NAME

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Confusion matrix saved to {save_path}")
