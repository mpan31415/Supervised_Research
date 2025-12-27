import tarfile
import os
from pathlib import Path

# ---- CONFIG ----
TAR_PATH = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/shards/gravestones_shard_000000.tar"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SAMPLE_IMAGES_DIR = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/sample_images"
NUM_IMAGES_TO_SAVE = 1000
# ----------------

####################################################################################
def list_images_in_tar(tar_path):
    """Counts and prints the number of image files in a tar archive."""
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
    
    return members


####################################################################################
def extract_sample_images(tar_path, output_dir, num_to_save, image_exts):
    """
    Extracts a specified number of image files from a tar archive 
    and saves them to the output directory.
    """
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)

    if not tar_path.exists():
        print(f"Extraction Error: {tar_path} does not exist")
        return

    # create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sample images to: {output_dir}")

    count = 0
    
    try:
        with tarfile.open(tar_path, "r") as tar_in:
            for member in tar_in.getmembers():
                # check if the member is a file and has an eligible image extension
                if member.isfile() and os.path.splitext(member.name)[1].lower() in image_exts:
                    
                    # define the destination path
                    file_name = os.path.basename(member.name)
                    dest_path = output_dir / file_name

                    # extract the file
                    print(f"Extracting: {file_name}")
                    file_handle = tar_in.extractfile(member)
                    if file_handle:
                        with open(dest_path, "wb") as outfile:
                            outfile.write(file_handle.read())
                        count += 1
                    
                    # stop once the target number of images has been saved
                    if count >= num_to_save:
                        print(f"Successfully saved {count} sample images.")
                        break
            else:
                print(f"Finished archive. Saved a total of {count} sample images.")

    except tarfile.TarError as e:
        print(f"An error occurred while reading the tar file: {e}")
        
# --------------------

if __name__ == "__main__":
    
    from pathlib import Path
    dir_name = Path(TAR_PATH).parent.name
    print(f"Directory name of the tar file: {dir_name}")

    list_images_in_tar(TAR_PATH) 

    extract_sample_images(TAR_PATH, SAMPLE_IMAGES_DIR, NUM_IMAGES_TO_SAVE, IMAGE_EXTS)
