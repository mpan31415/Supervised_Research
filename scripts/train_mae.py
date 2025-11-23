from torchvision import transforms
import torch
from torch import nn, optim
from vit_pytorch import ViT, MAE

import webdataset as wds
import time
import os
import numpy as np

os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


################## TRAIN PARAMS ##################
BATCH_SIZE = 32
NUM_EPOCHS = 30
BATCHES_PER_EPOCH = 70


################# 0. HELPER FUNCTIONS #################
def get_tar_files(dataset_dir):
    tar_files = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".tar"):
            tar_files.append(os.path.join(dataset_dir, file))
    return tar_files


################# 1. DATA LOADING #################
tic = time.time()

ALL_DATA_DIR = "/cluster/work/lawecon_repo/gravestones/shards/images"
dataset_dir = "transfer_2025-05-18_084428"

# get all tar files from the dataset directory
tar_files = get_tar_files(os.path.join(ALL_DATA_DIR, dataset_dir))
print(f"Found {len(tar_files)} tar files.")

# use the first tar file for testing
tar_file = tar_files[0]

# create webdataset object
dataset = wds.WebDataset(tar_file, resampled=True, shardshuffle=True).shuffle(1000).decode("pil")
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
    image = sample["jpg"] if "jpg" in sample else sample["png"]
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

# MAE with ViT encoder
mae = MAE(
    encoder = v,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
).to(device)

# optimizer
optimizer = optim.SGD(mae.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

print("Finished step 4 in", time.time() - tic, "seconds")


################# 5. TRAIN #################
tic = time.time()

# Train the model
for epoch in range(NUM_EPOCHS):
    
    # NOTE: Each epoch should go through entire dataset once
    
    for i, data in enumerate(train_loader):
        inputs = data[0].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = mae(inputs)
        loss.backward()
        optimizer.step()

        print(" [%3d,%3d] loss: %.5f" % (epoch + 1, i + 1, loss.item()))
        
    # save encoder checkpoint after each epoch
    ckpt_path = "/cluster/home/jiapan/Supervised_Research/checkpoints/"
    torch.save(v.state_dict(), os.path.join(ckpt_path, f"mae_encoder_epoch_{epoch+1}.pth"))
    print(f"Saved checkpoint for epoch {epoch+1}")

print("Finished Training in", time.time() - tic, "seconds")
