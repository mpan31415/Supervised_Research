#!/usr/bin/env python3
"""
Create a new dataset of gravestone images by streaming images from many .tar shards.
"""

import os
import tarfile
import random
import logging
import time
from io import BytesIO
from typing import List, Tuple
from pathlib import Path

import psycopg
from PIL import Image

# ---------------- CONFIG ----------------
CONFIG = {
    "root_dir": "/cluster/work/lawecon_repo/gravestones/shards/images",
    "output_dir": "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/shards",
    "output_prefix": "gravestones_shard",
    
    "target_count": 2_000_000,
    "images_per_output_shard": 2000,
    "allowed_extensions": (".jpg", ".jpeg", ".png"),
    "shuffle_seed": 42,

    "db": {
        "host": "id-hdb-psgr-cp7.ethz.ch",
        "dbname": "led",
        "user": "jiapan",
    },
}
# ----------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("collector")


# ---------------- DB WITH RETRY ----------------

def connect_db_with_retry(db_cfg: dict, retry_delay=5):
    kwargs = {k: v for k, v in db_cfg.items() if v is not None}
    while True:
        try:
            conn = psycopg.connect(**kwargs)
            conn.autocommit = True
            logger.info("Connected to database.")
            return conn
        except Exception as e:
            logger.error("DB connection failed: %s. Retrying in %ds...", e, retry_delay)
            time.sleep(retry_delay)


def safe_db_query(conn: psycopg.Connection, sql: str, params: tuple):
    while True:
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()
        except Exception as e:
            logger.error("DB query failed (%s). Reconnecting...", e)
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(2)
            conn = connect_db_with_retry(CONFIG["db"])


def is_gravestone(conn: psycopg.Connection, image_id: str) -> bool:
    sql = "SELECT is_gravestone FROM gravestones.memorials WHERE fag_id = %s LIMIT 1"
    row = safe_db_query(conn, sql, (image_id,))
    return bool(row and row[0])


def get_first_tarfile_dir_name(conn: psycopg.Connection, image_id: str):
    sql = "SELECT tar_name FROM gravestones.tarfiles_map WHERE html_fagid = %s LIMIT 1"
    row = safe_db_query(conn, sql, (image_id,))
    return row[0] if row else None


# ---------------- IMAGE UTILS ----------------

def resize_image_bytes(img_bytes: bytes, size=(256, 256), output_format="JPEG", quality=95) -> bytes:
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
    bio = BytesIO(img_bytes)
    tinfo = tarfile.TarInfo(name=name_in_tar)
    bio.seek(0, os.SEEK_END)
    tinfo.size = bio.tell()
    bio.seek(0)
    tinfo.mtime = int(time.time())
    tinfo.mode = 0o644
    tar_writer.addfile(tinfo, bio)


# ---------------- SHARD UTILS ----------------

def gather_tar_paths(root_dir: str) -> List[str]:
    tar_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".tar"):
                tar_paths.append(os.path.join(dirpath, fn))
    return tar_paths


def open_new_output_shard(output_dir: str, prefix: str, idx: int) -> Tuple[tarfile.TarFile, str]:
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{prefix}_{idx:06d}.tar"
    path = os.path.join(output_dir, fname)
    tar = tarfile.open(path, mode="w")
    logger.info("Opened new output shard: %s", path)
    return tar, path


# ---------------- MAIN PIPELINE ----------------

def process_shards(tar_paths: List[str], conn: psycopg.Connection, cfg: dict):

    allowed_ext = tuple(e.lower() for e in cfg["allowed_extensions"])
    target = cfg["target_count"]
    images_per_shard = cfg["images_per_output_shard"]
    output_dir = cfg["output_dir"]
    prefix = cfg["output_prefix"]

    accepted_count = 0
    out_shard_idx = 0
    out_shard_writer, out_shard_path = open_new_output_shard(output_dir, prefix, out_shard_idx)
    current_out_count = 0
    shard_start_time = time.time()

    for tar_path in tar_paths:
        dir_name = Path(tar_path).parent.name
        try:
            with tarfile.open(tar_path, "r") as tar_in:
                logger.info("Processing input shard: %s", tar_path)
                members = [
                    m for m in tar_in.getmembers()
                    if m.isfile() and os.path.splitext(m.name)[1].lower() in allowed_ext
                ]
                random.shuffle(members)

                for member in members:
                    if accepted_count >= target:
                        break

                    base = os.path.basename(member.name)
                    name_no_ext, _ = os.path.splitext(base)

                    image_id = name_no_ext[:-2] if name_no_ext.endswith("_0") else name_no_ext

                    try:
                        fobj = tar_in.extractfile(member)
                        img_bytes = fobj.read()
                    except Exception:
                        continue

                    if is_gravestone(conn, image_id) and (
                        get_first_tarfile_dir_name(conn, image_id) == dir_name
                    ):
                        try:
                            resized_bytes = resize_image_bytes(img_bytes)
                            add_image_to_shard(out_shard_writer, resized_bytes, base)
                        except Exception:
                            continue

                        accepted_count += 1
                        current_out_count += 1

                        if current_out_count >= images_per_shard:
                            logger.info(
                                f"Collected {accepted_count} images so far. "
                                f"This shard took {time.time() - shard_start_time:.2f}s."
                            )
                            out_shard_writer.close()
                            out_shard_idx += 1
                            out_shard_writer, out_shard_path = open_new_output_shard(
                                output_dir, prefix, out_shard_idx
                            )
                            current_out_count = 0
                            shard_start_time = time.time()

                if accepted_count >= target:
                    break

        except Exception:
            logger.exception("Failed to process %s", tar_path)

    out_shard_writer.close()
    logger.info("Done. Collected %d images.", accepted_count)
    return accepted_count


# ---------------- ENTRY ----------------

def main():
    cfg = CONFIG
    if cfg["shuffle_seed"] is not None:
        random.seed(cfg["shuffle_seed"])

    tar_paths = gather_tar_paths(cfg["root_dir"])
    random.shuffle(tar_paths)

    conn = connect_db_with_retry(cfg["db"])

    try:
        start = time.time()
        accepted = process_shards(tar_paths, conn, cfg)
        elapsed = time.time() - start
        logger.info(
            "Finished. Collected %d images in %.2fs (%.2f img/s)",
            accepted, elapsed, accepted / max(elapsed, 1e-6)
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
