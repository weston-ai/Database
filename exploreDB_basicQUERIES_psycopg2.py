import pandas as pd
pd.set_option('display.max_columns', None)
import psycopg2
import psycopg2.sql as sql
import os

os.getcwd()

# 1) Connect dynamically to database, using os environment variables
conn = psycopg2.connect(
    dbname=os.environ.get("PG_DBNAME"),
    user=os.environ.get("PG_USER"),
    password=os.environ.get("PG_PASSWORD"),
    host=os.environ.get("PG_HOST"),
    port=os.environ.get("PG_PORT")
)

cur= conn.cursor() # create the "cursur" object "cur"

# 2) List all databases (requires superuser access)
cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
for db in cur.fetchall():
    print("Database:", db[0])

# 3) List table names within the database
cur.execute("""
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
AND table_schema NOT IN ('pg_catalog', 'information_schema');
""")

for schema, table in cur.fetchall():
    print(f"Schema: {schema} | Table: {table}")

# 4) Explore the table column names
table_name = 'hypertension_study_main_20250408'

cur.execute(f"""
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = %s;
""", (table_name,))

for col, dtype in cur.fetchall():
    print(f"Column: {col} | Type: {dtype}")

# 5a1) Explore the data head (using psycopg2 and pandas, allowing table-like vis)
cur.execute("SELECT * FROM hypertension_study_main_20250408 LIMIT 2")
rows = cur.fetchall()
columns = [desc[0] for desc in cur.description]

df = pd.DataFrame(rows, columns=columns)
print(df)

# 5a2) Explore the data head (using a function for psycopg2 and pandas, allowing table-like vis)
def preview_table_safe(conn, table_name, limit):
    with conn.cursor() as cur:
        query = sql.SQL("SELECT * FROM {} LIMIT %s").format(sql.Identifier(table_name))
        cur.execute(query, (limit,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=columns)

df = preview_table_safe(conn, "hypertension_study_main_20250408", 2)
print(df)

# 5b) Explore the data head (using pandas only...note: only compatible with SELECT)
df = pd.read_sql("SELECT * FROM hypertension_study_main_20250408 LIMIT 2;", conn)
print(df.head())

# 5c) Explore the data head using psycopg2, seeing the data as a dictionary (no column headers)
cur.execute("SELECT * FROM hypertension_study_main_20250408 LIMIT 2;")
rows = cur.fetchall()
columns = [desc[0] for desc in cur.description]

for row in rows:
    row_dict = dict(zip(columns, row))
    print(row_dict)

# 5d) Explore the data head using psycopg2, seeing the data as rows (no column headers)
cur.execute("SELECT * FROM hypertension_study_main_20250408 LIMIT 2;")
rows = cur.fetchall()

for row in rows:
    print(row)

# 6) Selection based on specific columns and a condition; create a dataframe with the selection
query = """
    SELECT patient_id, month, race, sex, age_group, education, income, treatment, city, bmi, bmi_category, smoked_one_pack_10_years_or_more, alcohol_daily, hypertension 
    FROM hypertension_study_main_20250408
    WHERE treatment = %s
    """

params = ('Drug A',)

with conn.cursor() as cur:
    cur.execute(query, params)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

df = pd.DataFrame(rows, columns=columns)

# 7) Selection based on specific columns and multiple conditions; create a dataframe with the selection
query = """
    SELECT patient_id, month, race, sex, age_group, education, income, treatment, city, bmi, bmi_category, smoked_one_pack_10_years_or_more, alcohol_daily, hypertension 
    FROM hypertension_study_main_20250408
    WHERE treatment = %s
    """

# Treatments to query
treatment_groups = ['A', 'B'] # these are the multiple conditions to query by

# Dictionary to store dataframes
dfs = {}

# Loop over treatment values
for group in treatment_groups:
    with conn.cursor() as cur:
        cur.execute(query, (group,)) # group is the parameterized variable %s
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=columns)
    dfs[group] = df
    # Save query
    df.to_csv(f"treated_patients_{group}.csv", index=False)

# Optional: print confirmation
for key, df in dfs.items():
    print(f"Treatment {key} - shape: {df.shape}")