import os
import json
import math
import webdataset as wds
import numpy as np
from collections import defaultdict

# ---------------- CONFIG ----------------
DATA_ROOT = "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/labeled_shards"
SHARDS = "labeled_shard_{000000..000009}.tar"

BOOLEAN_VARS = [
    "is_military",
    "is_veteran",
    "is_female",
    "has_cross",
]

NUMERIC_VARS = [
    "is_military_prob",
    "deathyear",
]
# ---------------------------------------


def main():

    # containers
    bool_counts = {
        k: {"true": 0, "false": 0, "missing": 0}
        for k in BOOLEAN_VARS
    }

    numeric_values = {
        k: []
        for k in NUMERIC_VARS
    }

    dataset = (
        wds.WebDataset(os.path.join(DATA_ROOT, SHARDS))
        .decode()
        .to_tuple("json")
    )

    num_samples = 0

    for (labels,) in dataset:
        num_samples += 1

        # booleans
        for k in BOOLEAN_VARS:
            v = labels.get(k, None)
            if v is None:
                bool_counts[k]["missing"] += 1
            elif bool(v):
                bool_counts[k]["true"] += 1
            else:
                bool_counts[k]["false"] += 1

        # numeric
        for k in NUMERIC_VARS:
            v = labels.get(k, None)
            if v is not None:
                numeric_values[k].append(v)

    print("\n================ DATASET STATISTICS ================\n")
    print(f"Total samples processed: {num_samples}\n")

    print("---- Boolean Variables ----")
    for k, c in bool_counts.items():
        total = c["true"] + c["false"]
        print(
            f"{k:18s} | "
            f"True: {c['true']:5d} | "
            f"False: {c['false']:5d} | "
            f"Missing: {c['missing']:5d} | "
            f"True ratio: {c['true'] / max(total, 1):.3f}"
        )

    print("\n---- Numeric Variables ----")
    for k, vals in numeric_values.items():
        if len(vals) == 0:
            print(f"{k:18s} | No valid values")
            continue

        arr = np.array(vals, dtype=np.float64)
        print(
            f"{k:18s} | "
            f"count: {len(arr):5d} | "
            f"min: {arr.min():.3f} | "
            f"max: {arr.max():.3f} | "
            f"mean: {arr.mean():.3f} | "
            f"std: {arr.std():.3f}"
        )

    print("\n====================================================\n")


if __name__ == "__main__":
    main()
