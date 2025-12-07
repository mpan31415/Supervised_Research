import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
import sys
import torch
from torchvision import transforms


def get_tar_files(dataset_dir):
    tar_files = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".tar"):
            tar_files.append(os.path.join(dataset_dir, file))
    return tar_files


def show_img_for(img, seconds=1):
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(seconds)
    plt.close()
    
    
def load_images_as_tensors(sample_dir, image_exts):
    """
    Loads all image files from a directory, converts them to PyTorch tensors, 
    and returns a list of tensors.
    """
    sample_path = Path(sample_dir)

    if not sample_path.exists():
        print(f"Error: Directory not found: {sample_path}")
        # Terminate gracefully if the source directory doesn't exist
        sys.exit(1)

    # 1. Define the transformation pipeline
    # ToTensor() converts a PIL Image (H x W x C, range 0-255) 
    # to a FloatTensor (C x H x W, range 0.0-1.0)
    transform = transforms.ToTensor()

    image_tensors = []
    
    # 2. Iterate over files in the directory
    print(f"Searching for images in: {sample_dir}")
    for file_path in sample_path.iterdir():
        # Check if the file has an eligible image extension
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            
            try:
                # 3. Load the image using PIL
                img = Image.open(file_path)
                
                # 4. Apply the transformation to get a PyTorch Tensor
                tensor = transform(img)
                
                image_tensors.append(tensor)
                # print(f"Loaded and converted: {file_path.name} (Tensor shape: {tensor.shape})")

            except Exception as e:
                print(f"Could not process {file_path.name}: {e}")

    return image_tensors


