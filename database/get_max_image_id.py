import psycopg
import logging

logging.basicConfig(level=logging.INFO)

con = psycopg.connect(
    host="id-hdb-psgr-cp7.ethz.ch",
    dbname="led",
    user="jiapan",
)

cur = con.cursor()

# Set search path (optional since you're schema-qualifying anyway)
cur.execute("SET search_path TO gravestones;")

# sql = """
# SELECT MAX(html_fagid)
# FROM gravestones.tarfiles_map
# """

sql = """
SELECT MAX(fag_id)
FROM gravestones.memorials
"""

logging.info("executing query to get maximum html_fagid")
cur.execute(sql)

row = cur.fetchone()
max_html_fagid = row[0]

print("maximum html_fagid:", max_html_fagid)

cur.close()
con.close()
