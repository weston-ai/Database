# MAIN OBJECTIVE: Conduct a stratified random sampling of a given number of patients from a longitudinal dataset, clean the column headers, create sql schemas, upload the sql table schema to a postgresql database, copy the sampled data to the postgresql table, and run verification to evaluate the characteristics of the copied table

# Import modules
import pandas as pd
pd.set_option('display.max_columns', None)
import os
import psycopg2
import psycopg2.sql as sql
import io

os.getcwd() # /home/cweston1/miniconda3/envs/PythonProject

# Upload big dataset to sample from
df = pd.read_csv("datasets/my_custom_datasets/diabetes_60k_pats_20250428.csv")

# ================================================================
### 1) FUNCTIONS USED IN THIS SCRIPT (ALWAYS AT THE TOP)
# ================================================================
# Create function that standardizes the column headers to a clean lowercase format
def clean_column(col, max_length=63): # the max length for column names is 63 characters in postgresql
    cleaned = (
        col.strip()
           .lower()
           .replace(" ", "_")
           .replace("-", "_")
           .replace(":", "_")
    )
    return cleaned[:max_length]

### Create function that will infer the SQL data type form pandas dtype
def infer_sql_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "INT"
    elif pd.api.types.is_float_dtype(series):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    else:
        return "TEXT"

### Create function to explore the data head (using a function for psycopg2 and pandas, allowing table-like vis)
def preview_table_safe(conn, table_name, limit):
    with conn.cursor() as cur:
        query = sql.SQL("SELECT * FROM {} LIMIT %s").format(sql.Identifier(table_name))
        cur.execute(query, (limit,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=columns)

# ================================================================
### 2a) Count unique number of patients within each treatment group
# ================================================================
patient_counts = df.groupby('Treatment')['Patient_ID'].nunique()
print(patient_counts)

# Treatment
# Drug A                   20000
# Drug A with Exercise     20000
# Placebo with Exercise    20000

# ================================================================
### 2b) Get stratified random sample of patients (e.g. 20% per treatment group)
# ================================================================
# Identify fraction to sample
sample_fraction = 0.2 # 20% per group

# Sample patients per group
sampled_patients = (
    df[['Patient_ID', 'Treatment']] # We use double brackets to return a dataframe
    .drop_duplicates()
    .groupby('Treatment')
    .sample(frac=sample_fraction, random_state=39) # uses the sample fraction to sample on the groupby
)

sampled_ids = sampled_patients['Patient_ID'].unique() # get unique IDs from sampled patients

# ================================================================
### 2c) Subset original DataFrame to include all rows for sampled patients
# ================================================================
df_sampled = df[df['Patient_ID'].isin(sampled_ids)] # create df w/the IDs in the sample

# ================================================================
### 2d) Sanity check of sample (make sure the sample number represents the sample fraction)
# ================================================================
df_sampled.groupby('Treatment')['Patient_ID'].nunique()

# Treatment
# Drug A                   4000
# Drug A with Exercise     4000
# Placebo with Exercise    4000

# ================================================================
### 3) CLEAN THE COLUMN NAMES OF THE SAMPLED DATA (NECESSARY FOR CLEAN UPDLOAD INTO DATABASE)
# ================================================================
## Simplify the dataframe name (df)
df = df_sampled
del df_sampled # free up memory

## --- Clean column names ---
df.columns = [clean_column(col) for col in df.columns]

## --- Verify column names are clean
df.columns
# Index(['patient_id', 'month', 'hba1c', 'fasting_glucose', 'fasting_insulin',
#        'homa_ir', 'race', 'sex', 'age_group', 'education', 'income',
#        'insurance', 'bmi', 'bmi_category', 'treatment', 'dropout_month',
#        'city', 'city_risk_tier', 'smoked_one_pack_10_years_or_more',
#        'alcohol_daily', 'hypertension', 'hyperlipidemia',
#        'hba1c_remission_month', 'glucose_remission_month',
#        'insulin_remission_month', 'homa_ir_remission_month',
#        'hba1c_remission_duration', 'glucose_remission_duration',
#        'insulin_remission_duration', 'homa_ir_remission_duration', 'ins',
#        'irs1', 'irs2', 'pparg', 'glut4', 'tnf_alpha', 'il6', 'foxo1', 'g6pc',
#        'pck1', 'ucp2', 'palmitoylcarnitine', 'oleoylcarnitine', 'c3_carnitine',
#        'c5_carnitine', 'c6_carnitine', 'c8_carnitine', 'adiponectin',
#        'resistin', 'crp', 'fetuina', 'leptin', 'fndc5', 'complement_c3',
#        'complement_c4', 'cer_d18_1_16_0', 'cer_d18_1_24_1', 'dag_18_0_18_1',
#        'palmitic_acid', 'oleic_acid', 'sm_d18_1_16_0', 'lpc_16_0', 'lpc_18_1',
#        'tg_54_2', 'tg_52_2', 'glucose_metabolite', 'lactate', 'pyruvate',
#        'alanine', 'leucine', 'isoleucine', 'valine', 'glutamine', 'glutamate',
#        'citrate', 'succinate', 'malate', 'fumarate', 'alpha_ketoglutarate',
#        'acetoacetate', 'beta_hydroxybutyrate', 'uric_acid',
#        'hydroxyisobutyrate', 'phenylalanine', 'tyrosine', 'serine', 'glycine'],
#       dtype='object')

## --- Save the sampled data as a CSV file ---
csv_path = "/datasets/my_custom_datasets/diabetes_12k_pats_20250611.csv"
# df.to_csv(csv_path)  We don't need to save the csv if we only want to update the database

# =====================================================================
### 4) CREATE SQL SCHEMA FROM A SINGLE DATAFRAME
# =====================================================================
# ---- Configuration ----
table_name = os.path.splitext(os.path.basename(csv_path))[0].lower() # uses the last split as the table name (i.e. diabetes_12K-pats_20250611)
output_dir = "/home/cweston1/miniconda3/envs/PythonProject/py_scripts/database/sql_schemas/diabetes_schema/" # ✅ Destination folder for .sql file

### ---- Generate SQL CREATE TABLE ----
# Infer SQL-compatible column definitions using pandas dtype inspection
    # Example output: ["patient_id INT", "hba1c FLOAT", "sex TEXT", ...]
column_defs = [
    f"{col} {infer_sql_type(df[col])}" for col in df.columns
]

# Join all column definitions with commas and newlines for formatting
column_def_str = ",\n  ".join(column_defs)

# Create the final SQL statement with `IF NOT EXISTS` for safe reruns
    # This avoids overwrite errors if the table already exists in PostgreSQ
sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n  {column_def_str}\n);\n"

# ---- Write to .sql file ----
os.makedirs(output_dir, exist_ok=True)
sql_filepath = os.path.join(output_dir, f"{table_name}.sql")
with open(sql_filepath, "w") as f:
    # Include a comment header in the SQL file for readability
    f.write(f"-- Table: {table_name}\n{sql}")
print(f"Created schema file: {sql_filepath}")

# =====================================================================
### 5) LOAD THE SQL SCHEMA INTO A TABLE WITHIN POSTGRESQL DATABASE
# =====================================================================
##### Set environment variables in the bashrc to avoid hardcoding sensitive info in the script
## From bash run "vim ~/.bashrc"
## Add these five environment variables to the bashrc and save
# export PG_DBNAME="actual db name"
# export PG_USER="actual username"
# export PG_PASSWORD="actual password"
# export PG_HOST="localhost"
# export PG_PORT="5432"
## Source the bash by running "source ~/.bashrc" from
## Restart pycharm so that pycharm can inherit the new environmental variables

### Check if the environment variables are available to python (problem if "none" is outputted)
print(os.environ.get("PG_DBNAME"))
print(os.environ.get("PG_USER"))
print(os.environ.get("PG_PASSWORD"))
print(os.environ.get("PG_HOST"))
print(os.environ.get("PG_PORT"))

### Establish connection to PostgreSQL, using environment variables set in the bashrc
conn = psycopg2.connect(
    dbname=os.environ.get("PG_DBNAME"),
    user=os.environ.get("PG_USER"),
    password=os.environ.get("PG_PASSWORD"),
    host=os.environ.get("PG_HOST"),
    port=os.environ.get("PG_PORT")
)

# Read the SQL file (note this will not loop over multiple .sql schemas)
with open(sql_filepath, 'r') as file:
    sql_commands = file.read()

# Create a cursor and execute the SQL commands
try:
    with conn: # "with conn:" performs conn.commit() if no exceptions occur
        with conn.cursor() as cur: # "with conn.cursor() as cur:" automatically closes the cursor when the block finishes
            cur.execute(sql_commands)
            print("Schema executed successfully")
except Exception as e:
    print("Error occurred:", e)

# =====================================================================
### 6) VERIFY THE TABLE STRUCTURE IN THE DATABASE
# =====================================================================
# List all databases (requires superuser access)
with conn.cursor() as cur:
    cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")

for db in cur.fetchall():
    print("Database:", db[0])

# List table names within the database
with conn.cursor() as cur:
    cur.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE'
    AND table_schema NOT IN ('pg_catalog', 'information_schema');
    """)

for schema, table in cur.fetchall():
    print(f"Schema: {schema} | Table: {table}")

# Explore the table column data types
with conn.cursor() as cur:
    cur.execute(f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = %s;
    """, (table_name,)) # table_name was defined in Step 3 configuration

for col, dtype in cur.fetchall():
    print(f"Column: {col} | Type: {dtype}")

# =====================================================================
### 7) LOAD THE RANDOMLY SAMPLED DATA INTO THE NEW TABLE
# =====================================================================
# Ensure that NaNs are converted to "\\N" in the dataframe
df = df.fillna("\\N") # "\\N" is the PostgreSQL default for NULL when COPY operations are performed. When the COPY operation attempts place NaN in a numeric field, an error will result.

# Create a buffer to store the csv-formatted content of the random sample
buffer = io.StringIO()
df.to_csv(buffer, index=False, header=False) # Header=False because table already created
buffer.seek(0) # this moves the cursor to the first element in the buffered object

# Open a cursor and use copy_from to upload the data
try:
    with conn:
        with conn.cursor() as cur:
            cur.copy_from(
                file=buffer,
                table=table_name, #table_name was defined in Step 3 configuration
                sep=","
            )
            print("Data uploaded successfully")
except Exception as e:
    print("Upload failed:, e")

# =====================================================================
### 8) INSPECT THE TABLE SHAPE AND COLUMN DATA TYPES
# =====================================================================
## Count the number of rows & columns in the table
with conn.cursor() as cur:
    # Query 1: count the rows in the table
    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cur.fetchone()[0]

    # Query 2: count the columns in the table
    cur.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = %s;
    """, (table_name,))
    col_count = cur.fetchone()[0]

print(f"Table shape: {row_count} rows x {col_count} columns") # Table shape: 286027 rows x 87 columns

# Compare with the random sample dataframe shape (they should be identical)
df.shape # (286027, 87)

## Explore the table column data types
with conn.cursor() as cur:
    cur.execute(f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = %s;
    """, (table_name,)) # table_name was defined in Step 3 configuration

for col, dtype in cur.fetchall():
    print(f"Column: {col} | Type: {dtype}")

### Explore the head(2) of the table, using a previewing function
df_head = preview_table_safe(conn, table_name, 2) #table_name was created during Step 3 configuration
print(df_head)

# =====================================================================
### 9) INSPECT THE DISK USAGE OF THE NEW TABLE
# =====================================================================
with conn.cursor() as cur:
    cur.execute("""
        SELECT pg_size_pretty(pg_total_relation_size(%s));
    """, (table_name,))
    table_size = cur.fetchone()[0]

print(f"Table '{table_name}' disk usage: {table_size}")  # 203 MB