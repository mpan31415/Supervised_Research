import webdataset as wds
import time
import os
from tqdm import tqdm

os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"



def skip(exc, sample=None, key=None, url=None):
    # Return True to silently skip the sample
    return True


ALL_DATA_DIR = "/cluster/work/lawecon_repo/gravestones/shards/images/transfer_2025-05-18_084428/"
dataset_shards = [
    f"transfer_2025-05-18_084428_image_shard-{n:06d}.tar"
    for n in range(0, 104)
]

num_samples = []

for i in range(len(dataset_shards)):
    
    shard_name = dataset_shards[i]
    
    # create webdataset object
    dataset = wds.WebDataset(ALL_DATA_DIR + shard_name).decode("pil", handler=skip)

    # Count images with tqdm progress bar
    count = 0
    for _ in dataset:
        count += 1

    num_samples.append(count)
    print(f"Shard {shard_name} has {count} samples.")

print("Number of samples in each shard:", num_samples)