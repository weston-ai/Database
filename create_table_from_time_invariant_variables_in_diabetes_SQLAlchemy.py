# Script Objective: Use SQLAlchemy to load the diabetes longitudinal sample from my PostgreSQL database (mydb) and identify the time-invariant variables in the dataset (i.e. static variables such as birth date, sex, race, city, bmi, etc). Create a new table that has only these time-invariant variables assigned to patients. Then save this table to a different database called "otherdb". Lastly,  inspect the new table after saving it to otherdb, to ensure table validity. These time-invariant variables represent the type of data that would be available in an electronic health records dataset or demographics dataset.

# Import modules
import pandas as pd
pd.set_option('display.max_columns', None)
import os
import psycopg2
import psycopg2.sql as sql
import io
from sqlalchemy import create_engine, inspect

### Connect to the source database and the destination database with SQLAlchemy (this is necessary because pandas requires SQLAlchemy for loading data from a PostgreSQL database)

## Make sure that these have been defined with the "literal string" values in bashrc
# export PG_MYDB_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
# export PG_OTHERDB_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

# Load the database environment variables for SQLAlchemy
source_url = os.getenv("PG_MYDB_URL") # database where we will load data from
destination_url = os.getenv("PG_OTHERDB_URL") # database where we will save the data to

# Connect to the databases
source_engine = create_engine(source_url)
destination_engine = create_engine(destination_url)

# ===========================================
### VIEW THE TABLES IN MYDB, using SQLALCHEMY
# ===========================================

# Inspect the tables within the source database / create the inspector
inspector = inspect(source_engine)

# List all tables in the database
tables = inspector.get_table_names()
print("Tables in database:")
for table in tables:
    print("-", table)

# Tables in database:
# - hypertension_study_main_20250408
# - diabetes_12k_pats_20250611

# ====================================================================
### VIEW THE COLUMNS OF "diabetes_12k_pats_20250611" using SQLALCHEMY
# ====================================================================

table_name = 'diabetes_12k_pats_20250611'  # Replace with your actual table name

columns = inspector.get_columns(table_name)
print(f"\nColumns in '{table_name}':")
for col in columns:
    print(f"- {col['name']} ({col['type']})")

# ===========================================================================================
### LOAD DATA FROM THE TABLE "diabetes_12k_pats_20250611" using PANDAS (supports SQLAlchemy)
# ===========================================================================================

# Use pandas to load all columns of the dataset
df = pd.read_sql("SELECT * FROM diabetes_12k_pats_20250611", source_engine)

# ==============================================
### IDENTIFY TIME-INVARIANT COLUMNS PER PATIENT
# ==============================================

# Group the dataframe by Patient_ID. For each group, count the number of unique values in column col. Then find the maximum of those counts. If the maximum is 1, it means no patient ever had more than one value for that column—so it's time-invariant.

time_invariant_cols = []
patient_col = 'patient_id'  # Replace with your actual ID column

for col in df.columns:
    if col == patient_col:
        continue
    # Check if each patient only has one unique value for this column
    if df.groupby(patient_col)[col].nunique().max() == 1:
        time_invariant_cols.append(col)

# ==================================
### CREATE THE TIME-INVARIANT TABLE
# ==================================

# Keep only the patient ID and invariant columns, drop duplicates
df_invariant = df[[patient_col] + time_invariant_cols].drop_duplicates(subset=patient_col)

# Confirm it's truly one row per patient
assert df_invariant[patient_col].is_unique, "Duplicate patients found"

# ==========================================================
### SAVE THE TIME-INVARIANT DATA TO THE DESTINATION DATABASE
# ==========================================================

# Write to new table in other database
df_invariant.to_sql("diabetes_time_invariant_variables", destination_engine, index=False, if_exists="replace")

# Verify the transfer of data
print(f"Time-invariant columns: {time_invariant_cols}") # columns in time-invariant data
print(f"{len(df_invariant)} patients saved to 'patient_time_invariant' in {os.getenv('PG_OTHERDB_URL')}") # number of patients transferred to the table in the database

# ================================================
### VERIFY THE TABLES IN OTHERDB, using SQLALCHEMY
# ================================================

# Inspect the tables within the source database / create the inspector
inspector = inspect(destination_engine)

# List all tables in the database
tables = inspector.get_table_names()
print("Tables in database:")
for table in tables:
    print("-", table)

# Tables in database:
# - diabetes_time_invariant_variables

# ==========================================================================
### VIEW THE COLUMNS OF "diabetes_time_invariant_variables" using SQLALCHEMY
# ==========================================================================

table_name = 'diabetes_time_invariant_variables'  # Replace with your actual table name

columns = inspector.get_columns(table_name)
print(f"\nColumns in '{table_name}':")
for col in columns:
    print(f"- {col['name']} ({col['type']})")

# ==================================================================================
### VIEW THE HEAD OF "diabetes_time_invariant_variables" using SQLALCHEMY and PANDAS
# ==================================================================================

query = pd.read_sql("SELECT * FROM diabetes_time_invariant_variables LIMIT 3", destination_engine)
print(query)