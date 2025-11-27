#!/usr/bin/env python3
"""
Create a new dataset of gravestone images by streaming images from many .tar shards.
- Walks root_dir recursively to find .tar files
- Shuffles shard order and member order inside each shard
- For each image found, extracts image_id (filename w/o extension)
- Queries DB_A for whether it's a gravestone (boolean)
- If True and not already added, writes the raw image bytes into output tar shards
- Stops when target_count images have been collected

Requirements:
  pip install psycopg pillow numpy
"""

import os
import tarfile
import random
import logging
import time
from io import BytesIO
from typing import List, Tuple, Optional, Set

import psycopg
from PIL import Image  # optional: to validate image bytes if desired

# ---------------- CONFIG ----------------
CONFIG = {
    # file system
    "root_dir": "/cluster/work/lawecon_repo/gravestones/shards/images",   # root containing many subdirs with .tar shards
    "output_dir": "/cluster/home/jiapan/new_dataset",      # where to write new shards
    "output_dir": "/cluster/home/jiapan/new_dataset",      # where to write new shards
    "output_prefix": "gravestones_shard",             # e.g. gravestones_shard_00001.tar
    # "images_per_output_shard": 2000,                  # adjust to taste; 2k is reasonable
    "images_per_output_shard": 50,                  # adjust to taste; 2k is reasonable
    "allowed_extensions": (".jpg", ".jpeg", ".png"),
    "shuffle_seed": 42,                               # for reproducibility; set None for non-deterministic

    # DB (is_gravestone filtering)
    "db": {
        "host": "id-hdb-psgr-cp7.ethz.ch",
        "dbname": "led",
        "user": "jiapan",
    },

    # Dataset target
    # "target_count": 2_000_000,  # goal number of gravestone images
    # "target_count": 10_000,  # goal number of gravestone images
    "target_count": 250,  # goal number of gravestone images

    # Safety / logging
    # "log_every": 1000,          # log progress every N accepted images
    "log_every": 50,          # log progress every N accepted images
    "validate_images": False,   # if True, will try to open images with PIL to ensure validity (slower)
}
# ----------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("collector")


def gather_tar_paths(root_dir: str) -> List[str]:
    """Recursively find all .tar files under root_dir."""
    tar_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".tar"):
                tar_paths.append(os.path.join(dirpath, fn))
    return tar_paths


def connect_db(db_cfg: dict):
    """Return a psycopg.Connection. Caller should close it."""
    # Build connect args - allow optional password / port keys
    kwargs = {k: v for k, v in db_cfg.items() if v is not None}
    conn = psycopg.connect(**kwargs)
    conn.autocommit = True
    return conn


def is_gravestone(conn: psycopg.Connection, image_id: str, table: str = "gravestones.tarfiles_map") -> bool:
    """
    Query DB A to determine if image_id is a gravestone.
    Modify SQL to match your schema: assumes a boolean column 'is_gravestone' OR lookups available.
    Example SQLs (adjust to your actual DB schema):
      - If the DB stores a boolean: SELECT is_gravestone FROM images_table WHERE image_id = %s
      - If your schema is different, change this function accordingly.
    """
    sql = "SELECT is_gravestone FROM gravestones.memorials WHERE fag_id = %s LIMIT 1"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (image_id,))
            row = cur.fetchone()
            if row and row[0] is not None:
                return bool(row[0])
            else:
                return False
    except Exception as e:
        # If DB query fails, log and treat as not a gravestone to avoid stopping entire pipeline.
        logger.exception("DB query failed for image_id=%s: %s", image_id, e)
        return False


def resize_image_bytes(img_bytes: bytes, size=(256, 256), output_format="JPEG", quality=95) -> bytes:
    """
    Decode image bytes, convert to RGB, resize to (256,256),
    and re-encode to JPEG (or PNG).
    """
    with Image.open(BytesIO(img_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.BILINEAR)

        out_buf = BytesIO()
        if output_format.upper() == "JPEG":
            img.save(out_buf, format="JPEG", quality=quality, subsampling=0)
        else:
            img.save(out_buf, format="PNG")

        return out_buf.getvalue()


def add_image_to_shard(tar_writer: tarfile.TarFile, img_bytes: bytes, name_in_tar: str):
    """
    Add an image represented by img_bytes into the open tar_writer.
    name_in_tar should be like '<image_id>.jpg'
    """
    bio = BytesIO(img_bytes)
    tinfo = tarfile.TarInfo(name=name_in_tar)
    bio.seek(0, os.SEEK_END)
    tinfo.size = bio.tell()
    bio.seek(0)
    tinfo.mtime = int(time.time())
    # set a default filemode readable
    tinfo.mode = 0o644
    tar_writer.addfile(tinfo, bio)


def open_new_output_shard(output_dir: str, prefix: str, idx: int) -> Tuple[tarfile.TarFile, str]:
    """Open a new tar file for writing and return (TarFile, path)."""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{prefix}_{idx:06d}.tar"
    path = os.path.join(output_dir, fname)
    tar = tarfile.open(path, mode="w")
    logger.info("Opened new output shard: %s", path)
    return tar, path


def process_shards(
    tar_paths: List[str],
    conn: psycopg.Connection,
    cfg: dict,
):
    """
    Main streaming loop:
      - iterate tar_paths (shuffled)
      - for each tar, open and list members (filter by extension), shuffle member order
      - for each member, extract raw bytes, derive image_id, query DB, and if positive add to output shards
      - stop when target_count reached
    """
    allowed_ext = tuple(e.lower() for e in cfg["allowed_extensions"])
    target = cfg["target_count"]
    images_per_shard = cfg["images_per_output_shard"]
    output_dir = cfg["output_dir"]
    prefix = cfg["output_prefix"]
    validate_images = cfg["validate_images"]

    # bookkeeping
    accepted_count = 0
    seen_ids: Set[str] = set()
    out_shard_idx = 0
    out_shard_writer, out_shard_path = open_new_output_shard(output_dir, prefix, out_shard_idx)
    current_out_count = 0
    shard_start_time = time.time()

    total_shards = len(tar_paths)
    shard_counter = 0

    for tar_path in tar_paths:
        shard_counter += 1
        logger.info("Opening shard %d/%d: %s", shard_counter, total_shards, tar_path)
        try:
            with tarfile.open(tar_path, "r") as tar_in:
                # gather eligible members
                members = [m for m in tar_in.getmembers() if m.isfile() and os.path.splitext(m.name)[1].lower() in allowed_ext]
                if not members:
                    continue

                # shuffle members
                random.shuffle(members)

                for member in members:
                    # stop early if we've reached our target
                    if accepted_count >= target:
                        break

                    base = os.path.basename(member.name)
                    if not base:
                        continue

                    name_no_ext, ext = os.path.splitext(base)

                    # Strip trailing "_0" if present
                    if name_no_ext.endswith("_0"):
                        image_id = name_no_ext[:-2]
                    else:
                        image_id = name_no_ext

                    # skip duplicates we've already added
                    if image_id in seen_ids:
                        continue

                    # read raw bytes
                    try:
                        fobj = tar_in.extractfile(member)
                        if fobj is None:
                            continue
                        img_bytes = fobj.read()
                    except Exception:
                        logger.exception("Failed to read member %s in %s", member.name, tar_path)
                        continue

                    # optional validation (slower)
                    if validate_images:
                        try:
                            _ = Image.open(BytesIO(img_bytes)).convert("RGB")
                        except Exception:
                            logger.debug("Invalid image bytes for %s (skipping)", member.name)
                            continue

                    # query DB whether gravestone
                    if is_gravestone(conn, image_id):
                        # add to current output shard
                        try:
                            # Resize before storing (3x256x256)
                            try:
                                resized_bytes = resize_image_bytes(
                                    img_bytes,
                                    size=(256, 256),
                                    output_format="JPEG",   # or "PNG"
                                    quality=95
                                )
                            except Exception:
                                logger.exception("Resize failed for %s", base)
                                continue
                            
                            # add downsized image
                            add_image_to_shard(out_shard_writer, resized_bytes, base)
                            
                        except Exception:
                            logger.exception("Failed to write image %s to output shard", image_id)
                            continue

                        seen_ids.add(image_id)
                        accepted_count += 1
                        current_out_count += 1

                        # logging
                        if accepted_count % cfg["log_every"] == 0:
                            logger.info("Accepted %d images so far (target %d)", accepted_count, target)

                        # rotate output shard if full
                        if current_out_count >= images_per_shard:
                            out_shard_writer.close()
                            
                            shard_elapsed = time.time() - shard_start_time
                            imgs_per_sec = current_out_count / max(shard_elapsed, 1e-6)
                            
                            logger.info(
                                "Closed output shard %s | images=%d | time=%.2f s | throughput=%.2f img/s",
                                out_shard_path,
                                current_out_count,
                                shard_elapsed,
                                imgs_per_sec,
                            )
                            
                            out_shard_idx += 1
                            out_shard_writer, out_shard_path = open_new_output_shard(output_dir, prefix, out_shard_idx)
                            current_out_count = 0
                            shard_start_time = time.time()   # reset timer for next shard

                # check after finishing members of this input shard
                if accepted_count >= target:
                    break
        except Exception:
            logger.exception("Failed to open/process tar: %s", tar_path)
            continue

        if accepted_count >= target:
            break

    # finalize: close current writer and if empty remove it
    try:
        final_elapsed = time.time() - shard_start_time
        final_imgs_per_sec = current_out_count / max(final_elapsed, 1e-6)

        out_shard_writer.close()

        logger.info(
            "Final output shard %s | images=%d | time=%.2f s | throughput=%.2f img/s",
            out_shard_path,
            current_out_count,
            final_elapsed,
            final_imgs_per_sec,
        )
        # if last shard is empty remove it
        if current_out_count == 0:
            try:
                os.remove(out_shard_path)
                logger.info("Removed empty final shard: %s", out_shard_path)
            except OSError:
                pass
    except Exception:
        logger.exception("Failed to finalize output shard")

    logger.info("Done. Collected %d images (target was %d).", accepted_count, target)
    return accepted_count


def main():
    cfg = CONFIG

    if cfg["shuffle_seed"] is not None:
        random.seed(cfg["shuffle_seed"])

    logger.info("Gathering tar paths under %s ...", cfg["root_dir"])
    tar_paths = gather_tar_paths(cfg["root_dir"])
    if not tar_paths:
        logger.error("No .tar shards found under %s", cfg["root_dir"])
        return

    logger.info("Found %d input shards.", len(tar_paths))
    # shuffle the shard order for randomness
    random.shuffle(tar_paths)

    # connect to DB
    try:
        conn = connect_db(cfg["db"])
    except Exception as e:
        logger.exception("Failed to connect to DB: %s", e)
        return

    try:
        start = time.time()
        accepted = process_shards(tar_paths, conn, cfg)
        elapsed = time.time() - start
        logger.info("Finished. Collected %d images in %.2f seconds (%.2f imgs/s)", accepted, elapsed, accepted / max(elapsed, 1e-6))
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
