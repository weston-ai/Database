# This file will dynamically create .sql files that have the appropriate sql scheme for creating tables in a postgreSQL database.

# 1a) Clean the column names of all files
# 1b) Clean the column names of a single csv file
# 2) Create sql schema for a batch of csv files
# 3a) Create sql schema for a single csv file (loading the entire file)
# 3b) Create sql schema for a single csv file (loading the first 100 rows)

import os
import pandas as pd
pd.set_option('display.max_columns', None) # see all columns when we call head()

# =============================================================
# 1a) CLEAN THE COLUMN NAMES OF ALL CSV FILES IN DIRECTORY
# =============================================================
# --- Configuration ---
csv_dir = "/home/cweston1/miniconda3/envs/PythonProject/datasets/kaggle_csv_datasets/"  # Your folder with CSVs

# --- Column name cleaning function ---
def clean_column(col):
    return (
        col.strip()
           .lower()
           .replace(" ", "_")
           .replace("-", "_")
           .replace(":", "_")
    )

# --- Loop through all CSVs in the folder ---
dataframes = {}
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_dir, filename)

        # Load CSV
        df = pd.read_csv(file_path)

        # Clean columns
        cleaned_columns = [clean_column(col) for col in df.columns]
        df.columns = cleaned_columns

        # Save cleaned version (overwrite original)
        df.to_csv(file_path, index=False)

        print(f"✅ Cleaned and saved: {filename}")

print("🎯 All CSVs cleaned and updated.")


# ================================================================
# 1b) CLEAN THE COLUMN NAMES OF A SINGLE CSV FILE IN DIRECTORY
# ================================================================
# --- Configuration ---
csv_path = "/datasets/my_custom_datasets/hypertension_study_main_20250408.csv"

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Clean column names ---
df.columns = [clean_column(col) for col in df.columns]

# --- Overwrite the original file ---
df.to_csv(csv_path, index=False)

print(f"✅ Cleaned and saved: {os.path.basename(csv_path)}")

# ==================================================
# 2) CREATE SQL SCHEMAS FOR BATCH OF CSV FILES
# ==================================================

# directory of csv files
csv_dir = "/home/cweston1/miniconda3/envs/PythonProject/datasets/kaggle_csv_datasets/"

dataframes = {}

# Loop through csv files and load them into pycharm
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        table_name = os.path.splitext(filename)[0] # splits at the last period in the string
        file_path = os.path.join(csv_dir, filename)
        dataframes[table_name] = pd.read_csv(file_path)

        # Print a message or use df as needed
        print(f"Loaded {table_name} with shape {dataframes[table_name].shape}")
        print(f"Columns: ({', '.join(dataframes[table_name].columns)})")
        print(dataframes[table_name].head(1))
        print("=" * 40)

# Print all of the column names from each dataframe in dataframes
for table_name, df in dataframes.items(): # loops through the two items stored in the dictionary
    print(f"{table_name}: ({', '.join(df.columns)})")

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

### Write one CREATE TABLE .sql file per table (for a batch of tables)
output_dir = "/home/cweston1/miniconda3/envs/PythonProject/py_scripts/database/sql_schemas/" # where .sql files will be saved

for table_name, df in dataframes.items():
    column_defs = [
        f"{col} {infer_sql_type(df[col])}" for col in df.columns
    ]
    column_def_str = ",\n  ".join(column_defs)
    sql = f"CREATE TABLE {table_name} (\n  {column_def_str}\n);\n"

    filepath = os.path.join(output_dir, f"{table_name}.sql")
    with open(filepath, "w") as f:
        f.write(f"-- Table: {table_name}\n{sql}")
    print(f"✅ Created {filepath}")

# =====================================================================
# 3a) CREATE SQL SCHEMA FOR A SINGLE CSV FILE (LOADING ENTIRE FILE)
# =====================================================================

# ---- Configuration ----
csv_path = "/datasets/my_custom_datasets/hypertension_study_main_20250408.csv"  # 🔁 change this to your actual CSV
table_name = os.path.splitext(os.path.basename(csv_path))[0].lower() # 🔁 choose your PostgreSQL table name
output_dir = "/home/cweston1/miniconda3/envs/PythonProject/py_scripts/database/sql_schemas/" # ✅ Destination folder for .sql file

# ---- Load the single CSV ----
df = pd.read_csv(csv_path)

# ---- Generate SQL CREATE TABLE ----
column_defs = [
    f"{col} {infer_sql_type(df[col])}" for col in df.columns
]
column_def_str = ",\n  ".join(column_defs)
sql = f"CREATE TABLE {table_name} (\n  {column_def_str}\n);\n"

# ---- Write to .sql file ----
os.makedirs(output_dir, exist_ok=True)
filepath = os.path.join(output_dir, f"{table_name}.sql")
with open(filepath, "w") as f:
    f.write(f"-- Table: {table_name}\n{sql}")
print(f"✅ Created {filepath}")

# ==================================================================================
# 3b) CREATE SQL SCHEMA FOR A SINGLE CSV FILE (LOADING ONLY THE FIRST 100 ROWS)
# ==================================================================================

# ---- Configuration ----
csv_path = "/datasets/my_custom_datasets/hypertension_study_main_20250408.csv"  # Set to your CSV path
table_name = os.path.splitext(os.path.basename(csv_path))[0].lower()  # ✅ Set the table name
output_dir = "/home/cweston1/miniconda3/envs/PythonProject/py_scripts/database/sql_schemas/" # ✅ Destination folder for .sql file

# ---- Load sample of first 100 rows ----
df_sample = pd.read_csv(csv_path, nrows=100)
df_sample = df_sample.infer_objects()  # Improve dtype inference from sample

# ---- Construct SQL statement ----
column_defs = [
    f"{col} {infer_sql_type(dtype)}" for col, dtype in df_sample.dtypes.items()
]
column_def_str = ",\n  ".join(column_defs)
sql = f"CREATE TABLE {table_name} (\n  {column_def_str}\n);\n"

# ---- Write to file ----
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"{table_name}.sql")
with open(filename, "w") as f:
    f.write(f"-- Table: {table_name}\n{sql}")

print(f"✅ Created SQL file: {filename}")