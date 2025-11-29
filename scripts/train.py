from torchvision import transforms
import torch

import webdataset as wds
import time
import os
import sys

from utils import create_model, decoder_skip, seed_everything

os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"



if __name__ == "__main__":
    
    
    ################## READ ARGS ##################
    args = sys.argv[1:]
    MODEL_TYPE = args[0].lower()
    

    ################## GET DEVICE ##################
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", DEVICE)


    ################## SEED EVERYTHING ##################
    SEED = 42
    seed_everything(SEED)


    ################## TRAIN PARAMS ##################
    BATCH_SIZE = 128     # 4096 in original paper
    NUM_EPOCHS = 10
    BATCHES_PER_EPOCH = int(200_000 / BATCH_SIZE)
    CHECKPOINT_EVERY = 1


    ################# 1. DATA LOADING #################
    ALL_DATA_DIR = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/images/"
    dataset_shards = "gravestones_shard_{000000..000099}.tar"
    # create webdataset object
    dataset = wds.WebDataset(ALL_DATA_DIR + dataset_shards, resampled=True, shardshuffle=True).shuffle(1000).decode("pil", handler=decoder_skip)


    ################# 2. DEFINE TRANSFORMS #################
    transform_train = transforms.Compose(
        [   
            # transforms.Resize((256, 256)),
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


    ################# 3. CREATE DATALOADER #################
    train_set = dataset.map(make_sample).batched(BATCH_SIZE)
    train_loader = wds.WebLoader(train_set, batch_size=None, num_workers=4)
    # Unbatch, shuffle between workers, then rebatch.
    train_loader = train_loader.unbatched().shuffle(10).batched(BATCH_SIZE)
    # Since we are using resampling, the dataset is infinite; set an artificial epoch size (number of batches per epoch).
    train_loader = train_loader.with_epoch(BATCHES_PER_EPOCH)


    ################# 4. CREATE MODEL AND OPTIMIZER #################
    encoder, model, optimizer = create_model(type=MODEL_TYPE, device=DEVICE)


    ################# 5. TRAIN #################
    tic = time.time()

    # Train the model
    for epoch in range(NUM_EPOCHS):

        # NOTE: Each epoch should go through entire dataset once
        for i, data in enumerate(train_loader):
            
            if not data:
                continue
            
            inputs = data[0].to(DEVICE)
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model(inputs)
            loss.backward()
            optimizer.step()
            
            # DINO moving average update
            if MODEL_TYPE == "dino":
                model.update_moving_average()

            print(" [%3d,%3d] loss: %.5f" % (epoch + 1, i + 1, loss.item()))
            
        # save encoder checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = f"/cluster/home/jiapan/Supervised_Research/checkpoints/{MODEL_TYPE}/"
            torch.save(encoder.state_dict(), os.path.join(ckpt_path, f"encoder_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoint for epoch {epoch+1}")

    print("Finished Training in", time.time() - tic, "seconds")
