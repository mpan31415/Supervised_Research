from torchvision import transforms
import torch
import os
os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"

import webdataset as wds
import time

from utils import get_tar_files


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
train_set = dataset.map(make_sample).batched(64)
train_loader = wds.WebLoader(train_set, batch_size=None, num_workers=4)

# Unbatch, shuffle between workers, then rebatch.
train_loader = train_loader.unbatched().shuffle(10).batched(64)

# Since we are using resampling, the dataset is infinite; set an artificial epoch size (number of batches per epoch).
train_loader = train_loader.with_epoch(10)
print("Finished step 3 in", time.time() - tic, "seconds")


################# 4. ITERATE ONCE #################
tic = time.time()
images, = next(iter(train_loader))
print(type(images))
print(images.shape)
print("Finished step 4 in", time.time() - tic, "seconds")
