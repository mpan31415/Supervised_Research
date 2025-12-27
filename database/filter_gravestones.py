import psycopg
import logging

logging.basicConfig(level=logging.INFO)

IMAGE_ID = 6745123


con = psycopg.connect(
    host="id-hdb-psgr-cp7.ethz.ch",
    dbname="led",
    user="jiapan",
)

cur = con.cursor()

cur.execute("SET search_path TO gravestones;")

sql = "SELECT is_gravestone FROM gravestones.memorials WHERE fag_id = %s LIMIT 1"

res = False

with con.cursor() as cur:
    cur.execute(sql, (IMAGE_ID,))
    row = cur.fetchone()
    print(row)
    if row and row[0] is not None:
        res =  bool(row[0])

print("is_gravestone =", res)

con.close()
