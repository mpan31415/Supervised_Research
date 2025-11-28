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
    SELECT fag_id
    FROM gravestones.memorials
    WHERE is_gravestone = TRUE
    LIMIT 1000000
"""

logging.info("Querying gravestone fag_ids from DB...")
t0 = time.time()

cur.execute(sql)
rows = cur.fetchall()

fag_ids = [str(row[0]) for row in rows]

logging.info("Fetched %d fag_ids in %.2fs", len(fag_ids), time.time() - t0)

out_path = "/cluster/home/jiapan/gravestone_fag_ids.pkl"
with open(out_path, "wb") as f:
    pickle.dump(fag_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

logging.info("Saved fag_id list to %s (%.2f MB)",
             out_path,
             os.path.getsize(out_path) / 1024 / 1024)

cur.close()
con.close()
