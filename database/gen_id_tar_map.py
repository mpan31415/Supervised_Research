import psycopg
import pickle
import logging
import time
import os

logging.basicConfig(level=logging.INFO)

# ---------- Load ID list ----------
id_path = "/cluster/home/jiapan/gravestone_fag_ids.pkl"
with open(id_path, "rb") as f:
    fag_ids = pickle.load(f)

logging.info("Loaded %d fag_ids", len(fag_ids))

# ---------- DB Connection ----------
con = psycopg.connect(
    host="id-hdb-psgr-cp7.ethz.ch",
    dbname="led",
    user="jiapan",
)
cur = con.cursor()

sql = """
    SELECT tar_name
    FROM gravestones.tarfiles_map
    WHERE html_fagid = %s
    LIMIT 1
"""

mapping = {}

logging.info("Building {fag_id: tar_name} mapping...")
t0 = time.time()

for i, fag_id in enumerate(fag_ids):
    cur.execute(sql, (fag_id,))
    row = cur.fetchone()
    
    if row is not None:
        mapping[fag_id] = row[0]

    if i % 100 == 0 and i > 0:
        logging.info("Processed %d / %d", i, len(fag_ids))

logging.info("Built mapping for %d entries in %.2fs",
             len(mapping), time.time() - t0)

# ---------- Save Mapping ----------
out_path = "/cluster/home/jiapan/gravestone_map.pkl"
with open(out_path, "wb") as f:
    pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

logging.info("Saved mapping to %s (%.2f MB)",
             out_path,
             os.path.getsize(out_path) / 1024 / 1024)

cur.close()
con.close()
