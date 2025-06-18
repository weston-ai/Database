"""
Script: process_diabetes_dataset.py

Description:
    This script connects to PostgreSQL databases ("mydb" and "otherdb") using SQLAlchemy
    to extract, process, and restructure a longitudinal diabetes dataset.
    It identifies time-invariant variables (e.g., race, sex, age, city) for each patient
    and separates them from time-variant variables (e.g., repeated health measurements over time).

    Two new tables are created:
        1. `diabetes_12k_pats_time_invariant_variables` — one row per patient
        2. `diabetes_12k_pats_time_variant_variables` — multiple rows per patient over time

    These tables are stored in the "otherdb" database. A primary key constraint is added
    to the time-invariant table on `patient_id`, and a foreign key constraint is added
    to the time-variant table, referencing the primary key in the invariant table.

Functionality:
    - Connects to PostgreSQL via SQLAlchemy with environment-secured credentials
    - Loads full dataset from "mydb" into a pandas DataFrame
    - Detects and extracts time-invariant vs. time-variant variables
    - Validates the row and patient-level integrity of the split datasets
    - Writes both subsets to "otherdb" with appropriate primary/foreign key constraints
    - Logs all major actions and exceptions to both console and log file

Usage:
    Run this script from the command line or a Python environment after setting the
    following environment variables:
        - PG_MYDB_URL
        - PG_OTHERDB_URL

Author: Chris Weston
Created: June 2025
"""

# Import modules
import pandas as pd
pd.set_option('display.max_columns', None)
from sqlalchemy import create_engine, inspect, text
import os
import logging

# ==========================
### Setup Logging
# ==========================
log_dir = "py_scripts/database/postgresql_psycopg2_sqlalchemy/logs"

if os.path.exists(log_dir):
    print(f"{log_dir} Exists")
else:
    print(f"{log_dir} Does not exist")

log_format = "%(asctime)s - %(levelname)s - %(message)s"
log_filepath = os.path.join(log_dir, "run_log.log")
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

# ==================================================================================
# Connect to the "mydb" database and the "otherdb" database with SQLAlchemy
# ==================================================================================

## In the bashrc, define PG_MYDB_URL and PG_OTHERDB_URL
# export PG_MYDB_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
# export PG_OTHERDB_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

# Load the database environment variables
mydb_url = os.getenv("PG_MYDB_URL") # database where we will load data from

if not mydb_url:
    raise ValueError("Environment variable 'PG_MYDB_URL' not found")

otherdb_url = os.getenv("PG_OTHERDB_URL") # database where we will save the data to

if not otherdb_url:
    raise ValueError("Environment variable 'PG_OTHERDB_URL' not found")

# Connect to the databases
mydb_engine = create_engine(mydb_url)
otherdb_engine = create_engine(otherdb_url)

# ===========================================
# VIEW THE TABLES IN MYDB
# ===========================================

# Inspect the tables within the source database
mydb_inspector = inspect(mydb_engine)

# List all tables in the database
tables = mydb_inspector.get_table_names()
print("Tables in database:")
for table in tables:
    print("-", table)

# Tables in database:
# - hypertension_study_main_20250408
# - diabetes_12k_pats_20250611

# ====================================================================
# VIEW THE COLUMNS OF "diabetes_12k_pats_20250611" using get_columns()
# ====================================================================

table_name = 'diabetes_12k_pats_20250611'  # Replace with your actual table name

columns = mydb_inspector.get_columns(table_name)
print(f"\nColumns in '{table_name}':")
for col in columns:
    print(f"- {col['name']} ({col['type']})")

# ===========================================================================================
# LOAD DATA FROM THE TABLE "diabetes_12k_pats_20250611" using PANDAS (supports SQLAlchemy)
# ===========================================================================================

# Use pandas to load all columns of the dataset
df = pd.read_sql("SELECT * FROM diabetes_12k_pats_20250611", mydb_engine)

# ==============================================
# IDENTIFY TIME-INVARIANT COLUMNS PER PATIENT
# ==============================================

# Group the dataframe by Patient_ID. For each group, count the number of unique values in column. Then find the maximum of those counts. If the maximum is 1, it means no patient ever had more than one value for that column, so it's time-invariant.

time_invariant_cols = []
patient_col = 'patient_id'  # Replace with your actual ID column

for col in df.columns:
    if col == patient_col:
        continue # don't waste time testing the condition for patient_id
    # Check if each patient only has one unique value for this column
    if df.groupby(patient_col)[col].nunique().max() == 1: # true if max obs per patient == 1
        time_invariant_cols.append(col)

# =====================================================
# CREATE THE TIME-INVARIANT and TIME-VARIANT TABLES
# =====================================================

# Create the time-invariant dataframe (keep only the patient ID and invariant columns, drop duplicates)
df_invariant = df[[patient_col] + time_invariant_cols].drop_duplicates(subset=patient_col)

# Create the time-variant dataframe (keep only the columns not in time_invariant_cols)
variant_cols = [col for col in df.columns if col not in time_invariant_cols]
df_variant = df[variant_cols]

# ===============================================================
# SANITY CHECKS FOR TIME-INVARIANT AND TIME-VARIANT DATAFRAMES
# ===============================================================

## TIME-INVARIANT DATAFRAME
# Confirm it's truly one row per patient in the time-invariant dataset
assert df_invariant[patient_col].is_unique, "Duplicate patients found"

## TIME-VARIANT DATAFRAME
# Confirm that the time-invariant dataframe has the same number of observations as the original "diabetes_12k_pats_20250611" dataset.
# Group by patient_id and count the size of each patient group, and then sort by index
df_original_counts = df.groupby('patient_id').size().sort_index()
df_time_variant_counts = df_variant.groupby('patient_id').size().sort_index()

# Check if the index (patient_id) in the time-variant data matches the index (patient_id) of the full dataframe
assert set(df_original_counts.index) == set(df_time_variant_counts.index), "Mismatched patient IDs"

# Check if the count of patient oservations is the same between time-variant and full dataframe
assert df_original_counts.equals(df_time_variant_counts), "Mismatched number of patient observations"

# ==============================================================
# SAVE THE TIME-INVARIANT and TIME-VARIANT TABLES TO OTHERDB
# ==============================================================
df_invariant.to_sql("diabetes_12k_pats_time_invariant_variables", otherdb_engine, index=False, if_exists="replace")

df_variant.to_sql("diabetes_12k_pats_time_variant_variables", otherdb_engine, index=False, if_exists="replace")

# =========================================================
# DROP AN UNNECESSARY TABLE THAT I FOUND IN OTHERDB
# =========================================================
# Create inspector
otherdb_inspector = inspect(otherdb_engine)

# List all tables in the database
otherdb_tables = otherdb_inspector.get_table_names()
print(otherdb_tables)

table_name = "diabetes_time_invariant_variables"

with otherdb_engine.begin() as conn: # we use begin() because it auto-commits the change
    conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
    print(f"✅ Table '{table_name}' dropped from 'otherdb'.")

# Recreate inspector
otherdb_inspector = inspect(otherdb_engine)

# List all tables in the database
otherdb_tables = otherdb_inspector.get_table_names()
print(otherdb_tables)

# ======================================================================
# Inspect the columns of the tables that were just loaded into otherdb
# ======================================================================

# Inspect all columns in "diabetes_12k_pats_time_invariant_variables"
table_name = 'diabetes_12k_pats_time_invariant_variables'  # Replace with your actual table name
columns = otherdb_inspector.get_columns(table_name)
print(f"\nColumns in '{table_name}':")
for col in columns:
    print(f"- {col['name']} ({col['type']})")

# Inspect all columns in "diabetes_12k_pats_time_variant_variables"
table_name = 'diabetes_12k_pats_time_variant_variables'  # Replace with your actual table name
columns = otherdb_inspector.get_columns(table_name)
print(f"\nColumns in '{table_name}':")
for col in columns:
    print(f"- {col['name']} ({col['type']})")

# =======================================================================================
# Assign primary key to the time-invariant data // Add foreign key to time-variant data
# =======================================================================================

# Define ALTER statements
alter_primary_key = """
ALTER TABLE diabetes_12k_pats_time_invariant_variables
ADD CONSTRAINT pk_patient_id PRIMARY KEY (patient_id);
"""

alter_foreign_key = """
ALTER TABLE diabetes_12k_pats_time_variant_variables
ADD CONSTRAINT fk_patient_id
FOREIGN KEY (patient_id)
REFERENCES diabetes_12k_pats_time_invariant_variables(patient_id);
"""

# Run ALTER TABLE commands (i.e. add the primary key and foreign key in the same session: good coding practice)
with otherdb_engine.begin() as conn:
    try:
        logging.info("ADDING PRIMARY KEY constraint TO diabetes_12k_pats_time_invariant_variables")
        conn.execute(text(alter_primary_key))
        logging.info("Primary key added")
    except Exception as e:
        logging.warning("Could not add primary key. It may already exist", exc_info=True)

    try:
        logging.info("ADDING FOREIGN KEY constraint to diabetes_12k_pats_time_variant_variables")
        conn.execute(text(alter_foreign_key))
        logging.info("Foreign key added")
    except Exception as e:
        logging.warning("Could not add foreign key. It may already exist or fail due to integrity issues", exc_info=True)

# ===========================================================================================
# Verify that the primary and foreign keys were added correctly
# ===========================================================================================

# Recreate the inspector for otherdb
otherdb_inspector = inspect(otherdb_engine)

# Inspect the primary key information in "diabetes_12k_pats_time_invariant_variables"
pk_columns = otherdb_inspector.get_pk_constraint("diabetes_12k_pats_time_invariant_variables")
print("Primary key:", pk_columns)

# Primary key: {'constrained_columns': ['patient_id'], 'name': 'pk_patient_id', 'comment': None, 'dialect_options': {'postgresql_include': []}}

# Inspect the foreign key information in "diabetes_12k_pats_time_variant_variables"
fk_constraints = otherdb_inspector.get_foreign_keys("diabetes_12k_pats_time_variant_variables")
print("Foreign keys:", fk_constraints)

# Foreign keys: [{'name': 'fk_patient_id', 'constrained_columns': ['patient_id'], 'referred_schema': None, 'referred_table': 'diabetes_12k_pats_time_invariant_variables', 'referred_columns': ['patient_id'], 'options': {}, 'comment': None}]

# ==================================================================================
# VIEW THE HEAD OF THE TIME-INVARIANT AND TIME-VARIANT TABLES IN OTHERDB
# ==================================================================================

query = pd.read_sql("SELECT * FROM diabetes_12k_pats_time_invariant_variables LIMIT 5", otherdb_engine)
print(query)

query = pd.read_sql("SELECT * FROM diabetes_12k_pats_time_variant_variables LIMIT 5", otherdb_engine)
print(query)

####################
#### END SCRIPT ####
####################