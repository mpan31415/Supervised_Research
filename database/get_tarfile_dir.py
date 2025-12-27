import psycopg
import logging

logging.basicConfig(level=logging.INFO)

html_fagid_query = "1"

con = psycopg.connect(
    host="id-hdb-psgr-cp7.ethz.ch",
    dbname="led",
    user="jiapan",
)

cur = con.cursor()

cur.execute("SET search_path TO gravestones;")

sql = """
SELECT tar_name
FROM gravestones.tarfiles_map
WHERE html_fagid = %s
LIMIT 1
"""

logging.info("executing query for html_fagid=%s", html_fagid_query)
cur.execute(sql, (html_fagid_query,))

# rows = cur.fetchall()
row = cur.fetchone()
dir_name = row[0]

print("tarfile dir name:", dir_name)

cur.close()
con.close()
