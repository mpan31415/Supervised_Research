import psycopg
import pickle
import logging
import time
import os

logging.basicConfig(level=logging.INFO)

con = psycopg.connect(
    host="id-hdb-psgr-cp7.ethz.ch",
    dbname="led",
    user="jiapan",
)
cur = con.cursor()

sql = """
    SELECT m.fag_id, t.tar_name
    FROM gravestones.memorials m
    JOIN gravestones.tarfiles_map t
        ON m.fag_id = t.html_fagid
    WHERE m.is_gravestone = TRUE
    LIMIT 10000
"""

logging.info("Querying gravestone mapping from DB...")
t0 = time.time()

cur.execute(sql)
rows = cur.fetchall()

mapping = {}
for fag_id, tar_name in rows:
    mapping[str(fag_id)] = tar_name

logging.info("Fetched %d rows in %.2fs", len(mapping), time.time() - t0)

out_path = "/cluster/home/jiapan/gravestone_map.pkl"
with open(out_path, "wb") as f:
    pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

logging.info("Saved mapping to %s (%.2f MB)",
             out_path, 
             out_path and (os.path.getsize(out_path) / 1024 / 1024))

cur.close()
con.close()
