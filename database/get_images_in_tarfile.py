import tarfile
import os
from pathlib import Path

# ---- CONFIG ----
# TAR_PATH = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/shards/gravestones_shard_000000.tar"
TAR_PATH = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/labeled_shards/labeled_shard_000010.tar"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# ----------------

def list_images_in_tar(tar_path):
    tar_path = Path(tar_path)

    if not tar_path.exists():
        print(f"Error: {tar_path} does not exist")
        return

    if not tarfile.is_tarfile(tar_path):
        print(f"Error: {tar_path} is not a valid tar file")
        return

    with tarfile.open(tar_path, "r") as tar_in:
        # gather eligible members
        members = [m for m in tar_in.getmembers() if m.isfile() and os.path.splitext(m.name)[1].lower() in IMAGE_EXTS]
        base_names = [os.path.basename(m.name) for m in members]
    
    num_members = len(members)
    num_unique_names = len(set(base_names))
    print(f"Total image files in tar: {num_members}")
    print(f"Unique image file names in tar: {num_unique_names}")

if __name__ == "__main__":
    
    from pathlib import Path
    dir_name = Path(TAR_PATH).parent.name
    print(f"Directory name of the tar file: {dir_name}")

    list_images_in_tar(TAR_PATH)
