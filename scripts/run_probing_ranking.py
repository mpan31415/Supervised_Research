import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from utils import create_model, seed_everything

# ---------------- CONFIG ----------------
DATA_ROOT = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/labeled_shards"
SHARDS = "labeled_shard_{000000..000009}.tar"
CKPT_DIR = "/cluster/home/jiapan/Supervised_Research/checkpoints"

MODEL_TYPE = "dino"
CKPT_NAME = "epoch_100.pth"

TARGET_LABEL = "deathyear"
PROBE_TYPE = "linear"       # OPTIONS: "linear", "nonlinear"

BATCH_SIZE = 64            # Larger batch size possible for probing
NUM_EPOCHS = 100
SEED = 42
TRAIN_SPLIT = 0.8
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

seed_everything(SEED)

# ---------------- DATA PREP ----------------
# Standard Probing usually uses minimal augmentation to evaluate the raw features
transform = transforms.Compose([transforms.ToTensor()])

def make_sample(sample):
    if "jpg" not in sample or "json" not in sample: return None
    labels = sample["json"]
    if labels.get(TARGET_LABEL) is None: return None
    return sample["jpg"], float(labels[TARGET_LABEL])

dataset = wds.WebDataset(os.path.join(DATA_ROOT, SHARDS)).decode("pil").map(make_sample)
all_samples = [s for s in dataset if s is not None]

random.shuffle(all_samples)
split_idx = int(len(all_samples) * TRAIN_SPLIT)
train_base = all_samples[:split_idx]
val_base = all_samples[split_idx:]

class BalancedPairwiseDataset(Dataset):
    def __init__(self, samples, transform=None, pairs_per_epoch=None):
        self.samples = samples
        self.transform = transform
        self.len = pairs_per_epoch if pairs_per_epoch else len(samples)
        self.bins = [(0, 5), (5, 20), (20, 50), (50, 1000)]

    def __len__(self): return self.len

    def __getitem__(self, idx):
        img_a_pil, year_a = self.samples[idx % len(self.samples)]
        target_min, target_max = random.choice(self.bins)
        
        img_b_pil, year_b = None, None
        for _ in range(50):
            cand_img, cand_year = random.choice(self.samples)
            if target_min <= abs(year_a - cand_year) < target_max:
                img_b_pil, year_b = cand_img, cand_year
                break
        
        if img_b_pil is None: img_b_pil, year_b = random.choice(self.samples)

        img_a = self.transform(img_a_pil) if self.transform else img_a_pil
        img_b = self.transform(img_b_pil) if self.transform else img_b_pil
        label = torch.tensor(1.0 if year_a > year_b else 0.0, dtype=torch.float32)
        if year_a == year_b: label = torch.tensor(0.5)
            
        return img_a, img_b, label, year_a, year_b

train_loader = DataLoader(BalancedPairwiseDataset(train_base, transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BalancedPairwiseDataset(val_base, transform), batch_size=BATCH_SIZE, shuffle=False)

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

# Freeze encoder entirely
for p in encoder.parameters(): p.requires_grad = False

with torch.no_grad():
    emb_dim = encoder(torch.zeros(1, 3, 256, 256).to(DEVICE)).shape[1]

if PROBE_TYPE == "linear":
    classifier = nn.Linear(emb_dim * 2, 1).to(DEVICE)
else:
    classifier = nn.Sequential(
        nn.Linear(emb_dim * 2, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(DEVICE)

optimizer = optim.Adam(classifier.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# ---------------- EVAL ----------------
def evaluate(loader):
    classifier.eval()
    bins = {"0-5": [0,0], "5-20": [0,0], "20-50": [0,0], "50+": [0,0]}
    
    with torch.no_grad():
        for img_a, img_b, y, y_a, y_b in loader:
            mask = (y != 0.5)
            if not mask.any(): continue
            
            img_a, img_b, y = img_a.to(DEVICE), img_b.to(DEVICE), y.to(DEVICE)
            # Encoder is frozen, so we use it as a feature extractor
            z = torch.cat([encoder(img_a), encoder(img_b)], dim=1)
            logits = classifier(z).squeeze(1)
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == y).cpu().numpy()
            diffs = torch.abs(y_a - y_b).numpy()
            
            for i in range(len(diffs)):
                if y[i] == 0.5: continue
                d = diffs[i]
                b = "0-5" if d <= 5 else "5-20" if d <= 20 else "20-50" if d <= 50 else "50+"
                bins[b][0] += int(correct[i])
                bins[b][1] += 1
    return bins

# ---------------- TRAIN ----------------
print(f"Starting {PROBE_TYPE} probing (Frozen Encoder)...")
for epoch in range(NUM_EPOCHS):
    classifier.train()
    for img_a, img_b, y, _, _ in train_loader:
        img_a, img_b, y = img_a.to(DEVICE), img_b.to(DEVICE), y.to(DEVICE)
        
        with torch.no_grad():
            z = torch.cat([encoder(img_a), encoder(img_b)], dim=1)
        
        logits = classifier(z).squeeze(1)
        loss = criterion(logits, y)
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    val_bins = evaluate(val_loader)
    overall_acc = sum(b[0] for b in val_bins.values()) / sum(b[1] for b in val_bins.values())
    print(f"Epoch {epoch+1:02d} | Acc: {overall_acc:.3f} | " + 
          " ".join([f"{k}:{v[0]/v[1]:.2f}" for k,v in val_bins.items() if v[1]>0]))

print("Finetuning complete.")
