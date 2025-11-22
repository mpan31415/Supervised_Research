import psycopg

con = psycopg.connect(
    host="id-hdb-psgr-cp7.ethz.ch",
    dbname="led",
    user="jiapan",
)
cur = con.cursor()

# List all objects in the gravestones schema
cur.execute("""
    SELECT 
        c.relname AS object_name,
        c.relkind AS object_type
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'gravestones'
    ORDER BY c.relkind, c.relname;
""")

rows = cur.fetchall()

if not rows:
    print("Schema 'gravestones' has no objects at all.")
else:
    for name, kind in rows:
        print(f"{name:40} {kind}")
