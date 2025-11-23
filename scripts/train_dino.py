from torchvision import transforms
import torch
from torch import nn, optim
from vit_pytorch import ViT, Dino

import webdataset as wds
import time
import os
import numpy as np
import random

os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

print("=" * 50)
print("=" * 50)
print("=" * 50)
print("\n          Starting DINO training ...\n")
print("=" * 50)
print("=" * 50)
print("=" * 50)

SEED = 42

################## TRAIN PARAMS ##################
BATCH_SIZE = 256     # 4096 in original paper
NUM_EPOCHS = 100
BATCHES_PER_EPOCH = int(222060 / BATCH_SIZE)

CHECKPOINT_EVERY = 10


################# 0. HELPER FUNCTIONS #################
tic = time.time()

def skip(exc, sample=None, key=None, url=None):
    # return True to silently skip the sample
    return True

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# seed everything
seed_everything(SEED)

print("Finished step 0 in", time.time() - tic, "seconds")


################# 1. DATA LOADING #################
tic = time.time()

ALL_DATA_DIR = "/cluster/work/lawecon_repo/gravestones/shards/images/transfer_2025-05-18_084428/"
dataset_shards = "transfer_2025-05-18_084428_image_shard-{000000..000103}.tar"

# create webdataset object
dataset = wds.WebDataset(ALL_DATA_DIR + dataset_shards, resampled=True, shardshuffle=True).shuffle(1000).decode("pil", handler=skip)
assert isinstance(dataset, torch.utils.data.IterableDataset)

print("Finished step 1 in", time.time() - tic, "seconds")


################# 2. DEFINE TRANSFORMS #################
tic = time.time()

transform_train = transforms.Compose(
    [   
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def make_sample(sample):
    if "jpg" in sample:
        image = sample["jpg"]
    elif "png" in sample:
        image = sample["png"]
    else:
        return None
    return (transform_train(image),)

print("Finished step 2 in", time.time() - tic, "seconds")


################# 3. CREATE DATALOADER #################
tic = time.time()

train_set = dataset.map(make_sample).batched(BATCH_SIZE)
train_loader = wds.WebLoader(train_set, batch_size=None, num_workers=4)

# Unbatch, shuffle between workers, then rebatch.
train_loader = train_loader.unbatched().shuffle(10).batched(BATCH_SIZE)

# Since we are using resampling, the dataset is infinite; set an artificial epoch size (number of batches per epoch).
train_loader = train_loader.with_epoch(BATCHES_PER_EPOCH)

print("Finished step 3 in", time.time() - tic, "seconds")


################# 4. CREATE MODEL AND OPTIMIZER #################
tic = time.time()

# ViT
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

# DINO with ViT backbone
dino = Dino(
    v,
    image_size = 256,
    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
).to(device)

# optimizer
optimizer = torch.optim.Adam(dino.parameters(), lr = 3e-4)

print("Finished step 4 in", time.time() - tic, "seconds")


################# 5. TRAIN #################
tic = time.time()

# Train the model
for epoch in range(NUM_EPOCHS):
    
    # NOTE: Each epoch should go through entire dataset once
    
    for i, data in enumerate(train_loader):
        
        # skip empty data
        if not data:
            continue
        
        # get image batch and move to device
        inputs = data[0].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = dino(inputs)
        loss.backward()
        optimizer.step()
        
        # update moving average of teacher encoder and teacher centers
        dino.update_moving_average()

        print(" [%3d,%3d] loss: %.5f" % (epoch + 1, i + 1, loss.item()))
        
    # save encoder checkpoint
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        ckpt_path = "/cluster/home/jiapan/Supervised_Research/checkpoints/dino/"
        torch.save(v.state_dict(), os.path.join(ckpt_path, f"encoder_epoch_{epoch+1}.pth"))
        print(f"Saved checkpoint for epoch {epoch+1}")

print("Finished Training in", time.time() - tic, "seconds")
