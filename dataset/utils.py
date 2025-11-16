import os
import matplotlib.pyplot as plt


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