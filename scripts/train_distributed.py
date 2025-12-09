import os
import sys
import time
import csv
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import webdataset as wds
from torch.utils.tensorboard import SummaryWriter

from utils import create_model, decoder_skip, seed_everything

os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    ################## READ CONFIG ##################
    config_path = sys.argv[1]
    cfg = load_config(config_path)

    MODEL_TYPE = cfg["model"]["type"].lower()

    ################## INIT DDP ##################
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        print("World size:", world_size)
        print("Loaded config:", config_path)

    ################## SEED EVERYTHING ##################
    SEED = cfg["training"]["seed"] + rank
    seed_everything(SEED)

    ################## TRAIN PARAMS ##################
    BATCH_SIZE = cfg["training"]["batch_size_per_gpu"]
    NUM_EPOCHS = cfg["training"]["num_epochs"]

    TOTAL_SAMPLES = cfg["training"]["total_samples_per_epoch"]
    BATCHES_PER_EPOCH = int(TOTAL_SAMPLES / (BATCH_SIZE * world_size))

    PRINT_EVERY_STEPS = cfg["training"]["print_every_steps"]
    CHECKPOINT_EVERY_EPOCHS = cfg["training"]["checkpoint_every_epochs"]

    ################# LOGGING (rank 0 only) #################
    if rank == 0:
        log_dir = os.path.join(cfg["logging"]["log_root"], MODEL_TYPE)
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)

        csv_path = os.path.join(log_dir, "train_log.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["global_step", "epoch", "loss"])

    ################# 1. DATA #################
    data_root = cfg["data"]["data_root"]
    shards = cfg["data"]["shards"]

    dataset = (
        wds.WebDataset(
            os.path.join(data_root, shards),
            resampled=True,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            shardshuffle=0
        )
        .shuffle(cfg["data"]["shuffle_buffer"])
        .decode("pil", handler=decoder_skip)
    )

    ################# 2. TRANSFORMS #################
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    def make_sample(sample):
        if "jpg" in sample:
            image = sample["jpg"]
        elif "png" in sample:
            image = sample["png"]
        else:
            return None
        return (transform_train(image),)

    ################# 3. DATALOADER #################
    train_set = dataset.map(make_sample).batched(BATCH_SIZE)

    train_loader = wds.WebLoader(
        train_set,
        batch_size=None,
        num_workers=cfg["data"]["num_workers"],
        persistent_workers=True,
    )

    train_loader = (
        train_loader
        .unbatched()
        .shuffle(10)
        .batched(BATCH_SIZE)
        .with_epoch(BATCHES_PER_EPOCH)
    )

    ################# 4. MODEL + OPTIMIZER #################
    model, optimizer = create_model(
        type=MODEL_TYPE,
        device=DEVICE,
    )

    model = model.to(DEVICE)
    
    if MODEL_TYPE == "mae":
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=True,
        )
    elif MODEL_TYPE == "dino":
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    ################# 5. TRAIN #################
    tic = time.time()
    global_step = 0
    
    print("Starting Training...")

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader):

            if not data:
                continue

            inputs = data[0].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            loss = model(inputs)
            loss.backward()
            optimizer.step()

            if MODEL_TYPE == "dino":
                model.module.update_moving_average()

            ################# GLOBAL LOSS #################
            with torch.no_grad():
                loss_tensor = loss.detach().clone()
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                global_loss = loss_tensor / world_size

            global_step += 1

            ################# LOGGING #################
            if rank == 0 and global_step % PRINT_EVERY_STEPS == 0:
                loss_val = global_loss.item()

                print(
                    f"[Epoch {epoch+1:2d} | Step {global_step:7d}] "
                    f"Global Loss: {loss_val:.5f}"
                )

                writer.add_scalar("train/global_loss", loss_val, global_step)
                csv_writer.writerow([global_step, epoch + 1, loss_val])
                csv_file.flush()
                
        ################# CHECKPOINT (PER EPOCH) #################
        if rank == 0 and (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:

            ckpt_path = os.path.join(
                cfg["logging"]["checkpoint_root"], MODEL_TYPE
            )
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(
                model.module.state_dict(),
                os.path.join(ckpt_path, f"epoch_{epoch+1}.pth")
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

    ################# FINAL FULL CHECKPOINT #################    
    if rank == 0:
        torch.save({
            "epoch": NUM_EPOCHS,
            "global_step": global_step,
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, os.path.join(ckpt_path, "final_checkpoint.pth")
        )
        print("Saved final checkpoint")

    ################# CLEANUP #################
    if rank == 0:
        writer.close()
        csv_file.close()
        print("Finished Training in", time.time() - tic, "seconds")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
