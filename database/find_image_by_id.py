import os
import tarfile
import numpy as np
from PIL import Image
from io import BytesIO

# -------- USER SETTINGS --------
tar_dir = "/cluster/work/lawecon_repo/gravestones/shards/images/transfer_2024-12-17_155646"   # directory with 170 .tar shards
html_fagid = "71308989"        # image filename without extension
extensions = (".png", ".jpg", ".jpeg")
# --------------------------------


def find_image_in_tars(tar_dir, html_fagid):
    target_names = [html_fagid + ext for ext in extensions]

    for tar_name in sorted(os.listdir(tar_dir)):
        if not tar_name.endswith(".tar"):
            continue

        tar_path = os.path.join(tar_dir, tar_name)
        print(f"Searching in {tar_name}...")

        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                base = os.path.basename(member.name)
                print(base)
                if base in target_names:
                    print(f"\n✅ Found {base} in {tar_name}")

                    f = tar.extractfile(member)
                    img_bytes = f.read()

                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    img_np = np.array(img)

                    return img_np, tar_name, base

    return None, None, None


if __name__ == "__main__":
    img_np, shard, fname = find_image_in_tars(tar_dir, html_fagid)

    if img_np is None:
        print("\n❌ Image not found in any shard.")
    else:
        print(f"\n✅ Image loaded from:")
        print(f"   Shard: {shard}")
        print(f"   File:  {fname}")
        print(f"   Shape: {img_np.shape}")
        print(f"   Dtype: {img_np.dtype}")

        # Print full NumPy array (WARNING: very large for big images)
        print("\nNumPy array:")
        print(img_np)
