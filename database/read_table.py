import psycopg
import logging

logging.basicConfig(level=logging.INFO)

con = psycopg.connect(
    host="id-hdb-psgr-cp7.ethz.ch",
    dbname="led",
    user="jiapan",
)

cur = con.cursor()

# Option 1: set search path (so table names need not be schema-qualified)
cur.execute("SET search_path TO gravestones;")

sql = """
SELECT *
FROM gravestones.graveyards
LIMIT 20;
"""

logging.info("executing")
cur.execute(sql)

rows = cur.fetchall()

for row in rows:
    print(row)

con.close()
