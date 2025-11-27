import psycopg
import logging

logging.basicConfig(level=logging.INFO)

# Your query value (set this to the html_fagid you are looking for)
html_fagid_query = "71308989"

con = psycopg.connect(
    host="id-hdb-psgr-cp7.ethz.ch",
    dbname="led",
    user="jiapan",
)

cur = con.cursor()

# Set search path (optional since you're schema-qualifying anyway)
cur.execute("SET search_path TO gravestones;")

sql = """
SELECT tar_name
FROM gravestones.tarfiles_map
WHERE html_fagid = %s;
"""

logging.info("executing query for html_fagid=%s", html_fagid_query)
cur.execute(sql, (html_fagid_query,))

rows = cur.fetchall()

if rows:
    for row in rows:
        print("tar_name:", row[0])
else:
    print("No matching tar_name found.")

cur.close()
con.close()
