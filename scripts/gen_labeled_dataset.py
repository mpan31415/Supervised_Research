#!/usr/bin/env python3
"""
Read preprocessed 256x256 gravestone images from existing shards,
query labels from DB, and re-save into WebDataset shards
(image + JSON labels).
"""

import os
import tarfile
import random
import logging
import time
import json
from io import BytesIO
from typing import List

import psycopg

# ---------------- CONFIG ----------------
CONFIG = {
    "input_shards_dir": "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/shards",
    "output_dir": "/cluster/work/lawecon_repo/gravestones/rep_learning_dataset/labeled_shards",
    "output_prefix": "labeled_shard",

    "target_count": 10_000,
    "samples_per_output_shard": 1000,

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
logger = logging.getLogger("label_collector")


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


def get_labels(conn: psycopg.Connection, image_id: str):
    sql = """
        SELECT
            is_military,
            is_veteran,
            is_female,
            has_cross,
            is_military_prob,
            deathyear
        FROM gravestones.memorials
        WHERE fag_id = %s
        LIMIT 1
    """
    row = safe_db_query(conn, sql, (image_id,))
    if row is None:
        return None

    return {
        "is_military": bool(row[0]) if row[0] is not None else None,
        "is_veteran": bool(row[1]) if row[1] is not None else None,
        "is_female": bool(row[2]) if row[2] is not None else None,
        "has_cross": bool(row[3]) if row[3] is not None else None,
        "is_military_prob": float(row[4]) if row[4] is not None else None,
        "deathyear": int(row[5]) if row[5] is not None else None,
    }


# ---------------- TAR UTILS ----------------

def gather_input_shards(shards_dir: str) -> List[str]:
    return sorted(
        os.path.join(shards_dir, f)
        for f in os.listdir(shards_dir)
        if f.lower().endswith(".tar")
    )


def open_new_output_shard(output_dir: str, prefix: str, idx: int):
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{prefix}_{idx:06d}.tar"
    path = os.path.join(output_dir, fname)
    tar = tarfile.open(path, mode="w")
    logger.info("Opened new output shard: %s", path)
    return tar


def add_bytes_to_tar(tar_writer: tarfile.TarFile, data: bytes, name: str):
    bio = BytesIO(data)
    tinfo = tarfile.TarInfo(name=name)
    tinfo.size = len(data)
    tinfo.mtime = int(time.time())
    tinfo.mode = 0o644
    tar_writer.addfile(tinfo, bio)


# ---------------- MAIN PIPELINE ----------------

def process_shards(shard_paths: List[str], conn: psycopg.Connection, cfg: dict):

    allowed_ext = tuple(e.lower() for e in cfg["allowed_extensions"])
    target = cfg["target_count"]
    per_shard = cfg["samples_per_output_shard"]

    accepted = 0
    out_idx = 0
    out_count = 0

    out_tar = open_new_output_shard(cfg["output_dir"], cfg["output_prefix"], out_idx)

    for shard_path in shard_paths:
        logger.info("Processing input shard: %s", shard_path)

        try:
            with tarfile.open(shard_path, "r") as tar_in:
                members = [
                    m for m in tar_in.getmembers()
                    if m.isfile() and os.path.splitext(m.name)[1].lower() in allowed_ext
                ]
                random.shuffle(members)

                for m in members:
                    if accepted >= target:
                        break

                    base = os.path.basename(m.name)
                    stem, _ = os.path.splitext(base)

                    # match original logic
                    image_id = stem[:-2] if stem.endswith("_0") else stem

                    try:
                        img_bytes = tar_in.extractfile(m).read()
                    except Exception:
                        continue

                    labels = get_labels(conn, image_id)
                    if labels is None:
                        continue

                    try:
                        label_bytes = json.dumps(labels).encode("utf-8")
                    except Exception:
                        continue

                    add_bytes_to_tar(out_tar, img_bytes, f"{image_id}.jpg")
                    add_bytes_to_tar(out_tar, label_bytes, f"{image_id}.json")

                    accepted += 1
                    out_count += 1

                    if out_count >= per_shard:
                        out_tar.close()
                        out_idx += 1
                        out_tar = open_new_output_shard(
                            cfg["output_dir"], cfg["output_prefix"], out_idx
                        )
                        out_count = 0

        except Exception:
            logger.exception("Failed processing %s", shard_path)

        if accepted >= target:
            break

    out_tar.close()
    logger.info("Done. Collected %d samples.", accepted)
    return accepted


# ---------------- ENTRY ----------------

def main():
    cfg = CONFIG

    if cfg["shuffle_seed"] is not None:
        random.seed(cfg["shuffle_seed"])

    shard_paths = gather_input_shards(cfg["input_shards_dir"])
    random.shuffle(shard_paths)

    conn = connect_db_with_retry(cfg["db"])

    try:
        start = time.time()
        n = process_shards(shard_paths, conn, cfg)
        elapsed = time.time() - start
        logger.info(
            "Finished. Collected %d samples in %.2fs (%.2f samples/s)",
            n, elapsed, n / max(elapsed, 1e-6)
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
