"""
===============================================================================
Title: Create and Upload Longitudinal Diabetes Table to PostgreSQL
Author: Chris Weston
Date: 2025-06-26
===============================================================================

Purpose:
    This script implements a modular, production-grade ETL pipeline for processing
    a longitudinal diabetes dataset. It performs robust cleaning, stratified sampling,
    SQL schema inference, PostgreSQL table creation, and optimized data upload using
    memory-aware strategies.

    All input/output paths, database targets, and logging configurations are defined
    via generalized variables at the top of the script, enabling easy adaptation to
    new datasets, environments, and schemas. Utility functions are imported from
    dedicated helper modules to enforce clarity, reusability, and testability.

    By default, this script will *append* data to the target SQL table if it already
    exists. A pre-configured TRUNCATE operation is provided but commented out by
    default. This can be enabled in Block 13 to clear the table before inserting new
    data — a critical consideration for reproducibility and avoiding silent data growth.

-------------------------------------------------------------------------------

Key Workflow Steps:
    1. Load the full diabetes dataset from a user-defined CSV file.
    2. Perform stratified sampling (e.g., 20% per treatment group).
    3. Clean column names to conform with PostgreSQL constraints:
        - Max 63 characters
        - Lowercase
        - Alphanumeric with underscores
    4. Validate cleaned column names and compare to original.
    5. Generate SQL `CREATE TABLE` statement with inferred data types.
    6. Save the SQL schema to a reproducible `.sql` file.
    7. Establish a connection to the configured PostgreSQL database.
    8. Create the SQL table if it does not already exist.
    9. Upload the sampled DataFrame into PostgreSQL:
        - For DataFrames ≥ RAM threshold, use disk-based temporary CSVs.
        - For DataFrames < RAM threshold, use in-memory buffering (`StringIO`).
    10. Confirm the data load by logging:
        - Table row and column count
        - Column names and inferred SQL types
        - First 2 rows of the table
        - On-disk table size via `pg_size_pretty`

-------------------------------------------------------------------------------

Arguments and Configuration:
    All key variables are configured at the top of the script, including:

    csv_path (str):
        Path to the source CSV file to load.
    csv_sample_filename (str):
        File name for the stratified, cleaned sample saved locally.
    grouping_var (str):
        Name of the column to stratify by.
    unique_id (str):
        Unique subject identifier (used for sampling and grouping).
    sample_fraction (float):
        Sampling rate per strata (e.g., 0.2 for 20%).
    database (str):
        Name of the PostgreSQL target database.
    database_url (str):
        Name of environment variable holding SQLAlchemy database URL.
    log_filepath (str):
        Folder where logs are stored.
    log_filename (str):
        Log file name.
    sql_schema_dir (str):
        Directory to save the `.sql` schema file.
    RAM_buffer_limit (int):
        Determines threshold (in GB) for when to switch from RAM to disk upload.

-------------------------------------------------------------------------------

Data Upload Strategy:
    The script dynamically selects the best data upload method:

    - If DataFrame size ≥ RAM_buffer_limit:
        - Write to a temporary `.csv` file on disk.
        - Load into PostgreSQL using `COPY ... FROM STDIN`.

    - If DataFrame size < RAM_buffer_limit:
        - Stream from memory via `io.StringIO`.
        - Upload using `COPY` from RAM.

    Missing values are safely encoded using PostgreSQL’s null-friendly "\\N" escape.

    ⚠️ Note: If the destination table already exists, the new data will be appended.
    To avoid duplication, activate the optional TRUNCATE block in the data upload section.

-------------------------------------------------------------------------------

Logging and Debugging:
    - Modular logging is configured with both file and console output.
    - Log level is parameterized (default: `INFO`).
    - Logs include timestamps, event type, and detailed tracebacks for exceptions.
    - Every major processing and validation step is logged for reproducibility.

-------------------------------------------------------------------------------

Validation and Safeguards:
    - Table name is regex-validated to prevent SQL injection and name collisions.
    - Column names are rigorously cleaned and validated against PostgreSQL constraints.
    - SQL creation uses `IF NOT EXISTS` for rerun safety.
    - Post-upload validation includes table shape, schema inspection, preview, and disk usage.

-------------------------------------------------------------------------------

Output:
    - Cleaned, stratified sample saved to disk.
    - Reproducible SQL schema saved as `.sql` file.
    - Table created and populated in PostgreSQL.
    - Full audit trail written to the configured log file.

-------------------------------------------------------------------------------

Dependencies:
    - pandas
    - sqlalchemy
    - psycopg2 (via SQLAlchemy’s raw_connection)
    - logging
    - os, io, tempfile, re
    - custom utility modules (e.g., `database_utils`, `logging_utils`)

-------------------------------------------------------------------------------

Environment Variables:
    - PG_HEALTHDB_URL: SQLAlchemy-compatible PostgreSQL URL
      (e.g., postgresql+psycopg2://user:password@host:port/dbname)

-------------------------------------------------------------------------------

Usage Example:
    $ export PG_HEALTHDB_URL="postgresql+psycopg2://user:password@localhost:5432/healthdb"
    $ python create_table_from_longitudinal_data.py

    Adjust the variables at the top of the script to change:
        - Input data source
        - Sampling logic
        - Output locations
        - Target database/table

===============================================================================
"""

# ==============================================================================
### 1) IMPORT MODULES
# ==============================================================================
# ---- Import modules ----
import pandas as pd
pd.set_option('display.max_columns', None)
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError
import logging
from database_utils import clean_column, infer_sql_type #To import the utility modules in pycharm, you must set the script environment: File → Settings → Project: YourProject → Project Structure → Script Folder → Sources
import tempfile # for saving dataframe to temporary hard-disk (if dataframe is >= 2 GB RAM)
import os
import io # for buffering tabular data into RAM as stringIO (if dataframe is < 2 GB RAM)
import re # if we plan to remove data from pre-existing Postgres table before loading new data

# ===================================================================
### 2) CONFIGURATION (File Paths and Declared Variables)
# ===================================================================
### ------------- FILE PATHS ----------------
# Log filepath
log_filepath = "py_scripts/database/postgres_sqlalchemy/logs/"

# CSV filepath (used to construct SQL table)
csv_path = "datasets/my_custom_datasets/"

# Destination folder for SQL schema (.sql)
sql_schema_dir = "/home/cweston1/miniconda3/envs/PythonProject/py_scripts/database/sql_schemas/diabetes_schema/"

### -------- USER-DECLARED VARIABLES -----------
# CSV file name (this is the primary dataset to load)
csv_filename = "diabetes_60k_pats_20250428.csv"

# CSV file name (this is the dataset that you will create from stratified sampling)
csv_sample_filename = "diabetes_12k_pats_20250611.csv"

# Grouping variable in CSV data (important for the stratified sampling)
grouping_var = 'Treatment'

# Unique identifier in CSV data (important for stratified sampling)
unique_id = 'Patient_ID'

# Sampling fraction (percentage of samples to take per strata group)
sample_fraction = 0.2   # 20% per group

# Log file name (the logging file)
log_filename = "create_table_from_longitudinal_data.log"

# Database that you plan to make a connection with (i.e. where we will create the SQL table)
database = "healthdb"

# URL for the PostgreSQL database that you want to make connection with (derived from URL in bashrc)
database_url = "PG_HEALTHDB_URL"

# RAM buffer limit (Code block 13). Determine the maximum dataframe size (in gigabytes) for buffering the sample to in-memory StringIO object before uploading to the Postgres table. If the dataframe is less than the buffer limit, it will be buffered into RAM before upload to Postgres. If the dataframe is greater than or equal to the buffer limit, it will be temporarily saved to hard-disk before upload to Postgres.
RAM_buffer_limit = 3  # Gigabytes (GB)

### ---------- SPECIAL NOTES ------------
# This script is intended for a situation where the Postgres table does not exist yet. If the table already exists, this script will append new data onto the pre-existing rows. See Block 13 in the script if you want to ensure that all pre-existing rows are removed (truncated) before uploading the new data. Block 13 has two truncate steps that are commented out by default; you must uncomment them if you want to ensure that pre-existing rows are removed prior to uploading the new data.

# ==========================
### 3) LOGGING
# ==========================
# Configuration
log_format = "%(asctime)s - %(levelname)s - %(message)s" # date-time format
os.makedirs(os.path.join(log_filepath), exist_ok=True) # ensure log directory exists

# Clear existing handlers to avoid duplication or mismatches (important in interactive sessions)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create handlers
file_handler = logging.FileHandler(os.path.join(log_filepath, log_filename))
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Or DEBUG if you want debug tracking

# Set up root logger
logging.basicConfig(
    level=logging.INFO,  # Or DEBUG if you want debug tracking
    format=log_format,
    handlers=[file_handler, stream_handler]
)

logger = logging.getLogger(__name__) # creates an instance of the logger class from the logging module

## Log the log_path and the script objective
logger.info(f"These logs are saved in '{os.path.join(log_filepath, log_filename)}'")
script_objective = f"To create a table in '{database}' database that has a sample from the longitudinal diabetes dataset."
logger.info(f"SCRIPT OBJECTIVE: {script_objective}")

# ==================================================
# 4a) LOAD CSV DATA
# ==================================================
## Upload full diabetes dataset to sample from
try:
    logger.info("Loading data from '%s", os.path.join(csv_path, csv_filename))
    df = pd.read_csv(os.path.join(csv_path, csv_filename))
    logger.info("Data loaded successfully: %d rows, %d columns", *df.shape)
except FileNotFoundError:
    logger.error("File not found: %s", csv_path, exc_info=True)
    raise
except pd.errors.EmptyDataError:
    logger.error("CSV file is empty:", csv_path, exc_info=True)
    raise
except Exception as e:
    logger.error("Unexpected error loading CSV:", e, exc_info=True)
    raise

ncol_original_df = len(df.columns) # used downstream in the code

# ====================================================================
### 4b) Count unique number of patients within each treatment group
# ====================================================================
## Count the unique number of patients by treatment
patient_counts = df.groupby(grouping_var)[unique_id].nunique()
logger.info(f"Number of unique '{unique_id}s' per {grouping_var}:\n%s", patient_counts.to_string())

# ================================================================
### 4c) Get stratified random sample of patients (e.g. 20% per treatment group)
# ================================================================
total_patients = df[unique_id].nunique()  # total number of patients

logger.info("Starting stratified sample for %.1f%% of observations within each strata group", sample_fraction)

try:
    sampled_observations = (
        df[[unique_id, grouping_var]] # returns dataframe with these fields
        .drop_duplicates() # Keeps only one unique "patient_id / treatment" combination
        .groupby(grouping_var)
        .sample(frac=sample_fraction, random_state=39) # uses the sample fraction
    )
    logger.info("Successfully completed stratified sampling.")
except ValueError as e:
    logger.error("Sampling failed: %s", e, exc_info=True)
    raise

# ================================================================
### 4d) Subset original DataFrame to include all rows for sampled patients
# ================================================================
sampled_ids = sampled_observations[unique_id].unique() # Isolate the unique IDs from sampled patients

df_sampled = df[df[unique_id].isin(sampled_ids)] # create dataframe with the sampled IDs

logger.info("Filtered %d rows based on %d sampled IDs", len(df_sampled), len(sampled_ids))

# ================================================================
### 4e) VERIFY THAT THE APPROPRIATE FRACTION OF PATIENTS WERE SAMPLED
# ================================================================
# Count total patients per treatment in the full dataset and the sample
original_counts = df.groupby(grouping_var)[unique_id].nunique()  # creates pandas series
sampled_counts = df_sampled.groupby(grouping_var)[unique_id].nunique() # creates pandas series

# Compare expected vs actual
for treatment in original_counts.index: # indexed by the groupby()
    expected = int(original_counts[treatment] * sample_fraction) # n patients in original treatment * fraction
    actual = sampled_counts.get(treatment, 0) # uses dictionary logic to fetch the value for treatment key

    logger.info(
        f"{grouping_var}: %s | Expected Sampled: %d | Actual Sampled: %d | Δ: %d",
        treatment, expected, actual, actual - expected
    )

    # Optional: Add warning if there's a mismatch
    if abs(actual - expected) > 1:  # Allowing for rounding error
        logger.warning("Sampling mismatch for %s: expected %d, got %d",
                       treatment, expected, actual)

# ==============================================================================================
### 5) CLEAN THE COLUMN NAMES OF THE SAMPLED DATA (NECESSARY FOR CLEAN UPDLOAD INTO DATABASE)
# ==============================================================================================
df = df_sampled # Simplify the dataframe name
del df_sampled # free up memory

## --- Clean column names ---
try:
    logger.info("Initiate cleaning of column names")
    df.columns = [clean_column(col) for col in df.columns]
    logger.info("Column names cleaned.")
except Exception as e:
    logger.error("Cleaning of column names failed.", e, exc_info=True)
    raise

# ================================================================
### 6) VERIFY THAT THE COLUMN NAMES MEET POSTGRESQL CONSTRAINTS
# ================================================================
try:
    logger.info("Verifying that column names meet PostgreSQL constraints.")

    invalid_cols = [
        col for col in df.columns
        if len(col) > 63 or not col.replace("_", "").isalnum()
    ]

    if invalid_cols:
        logger.warning("Invalid PostgreSQL column names detected. \n Must be <= 63 chars, alpha-numeric, and underscores. Invalid columns: %s", invalid_cols)
    else:
        logger.info("All column names past PostgreSQL constraints.")

except Exception as e:
    logger.error("PostgreSQL column name validation failed: %s", e, exc_info=True)
    raise

# ============================================================================================
### 7) VERIFY THAT THE CLEANED DATAFRAME HAS THE SAME NUMBER OF COLUMNS AS THE ORIGINAL DATASET
# ============================================================================================
try:
    logger.info("Verifying that that the cleaned dataframe has the same number of columns as original dataset")
    if len(df.columns) == ncol_original_df:
        logger.info("Cleaned dataframe has the same number of columns as original dataset. Cleaned: %d. Original: %d. Good!", len(df.columns), ncol_original_df)
    else:
        logger.warning("Cleaned dataframe DOES NOT have the same number of columns as original dataset. Cleaned: %d. Original: %d. Bad!", len(df.columns), ncol_original_df)
except Exception as e:
    logger.error("Error occurred while validating the column count: %s", e, exc_info=True)
    raise

# ======================================================================
### 8) LOG THE NAMES OF ALL COLUMNS IN THE CLEANED DATAFRAME
# ======================================================================
logger.info("Cleaned dataframe column names: %s", df.columns.tolist())

# ======================================================================
### 9) SAVE THE CLEANED DATAFRAME TO DISK AS A CSV FILE
# ======================================================================
try:
    csv_path = os.path.join("datasets/my_custom_datasets/", csv_sample_filename)
    logger.info("Attempting to save cleaned dataframe to CSV file: '%s'.", csv_sample_filename)
    df.to_csv(csv_path)
    logger.info("Successfully saved dataframe as '%s' at path '%s'.", csv_sample_filename, csv_path)
except Exception as e:
    logger.error("Failed to save dataframe as '%s' at path '%s': '%s'", csv_sample_filename, csv_path, e, exc_info=True)
    raise

# =====================================================================
### 10) CREATE SQL SCHEMA FROM A DATAFRAME
# =====================================================================
## Configuration
os.makedirs(sql_schema_dir, exist_ok=True)
logger.info("SQL schema output directory exists or was created.")

## Create table name
try:
    logger.info("Attempting to create table name based on extraction from CSV filepath: '%s'", csv_path)
    table_name = os.path.splitext(os.path.basename(csv_path))[0].lower() # uses the last split as the table name (i.e. diabetes_12K-pats_20250611)
    logger.info("Successfully created table name: '%s'", table_name)
except Exception as e:
    logger.error("Failed to create table name: %s", e, exc_info=True)

## Create SQL schema
try:
    # Infer SQL-compatible column definitions using pandas dtype inspection
        # Example output: ["patient_id INT", "hba1c FLOAT", "sex TEXT", ...]
    column_defs = [f"{col} {infer_sql_type(df[col])}" for col in df.columns]
    logger.info("Inferred SQL data types for %d columns.", len(column_defs))

    # Join all column definitions with commas and newlines for formatting ----
    column_def_str = ",\n  ".join(column_defs)
    logger.debug("Column definitions:\n%s", column_def_str)

    # Create the final SQL statement with `IF NOT EXISTS` for safe reruns ----
        # This avoids overwrite errors if the table already exists in PostgreSQ
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n  {column_def_str}\n);\n"
    logger.info("Generated SQL CREATE TABLE statement for: '%s'", table_name)

    # ---- Write to .sql file ----
    sql_filepath = os.path.join(sql_schema_dir, f"{table_name}.sql")
    with open(sql_filepath, "w") as f:
        # Include a comment header in the SQL file for readability
        f.write(f"-- Table: {table_name}\n{sql}")
    logger.info("Successfully wrote SQL schema to file: '%s'", sql_filepath)

except Exception as e:
    logging.error("Failed to generate SQL schema for table '%s': %s", table_name, e, exc_info=True)
    raise

# ===============================================================
### 11) VERIFY THAT THE DATABASE URL IS VALID FOR SQLALCHEMY
# ===============================================================
try:
    url = os.environ[database_url]
    if make_url(url):
        logger.info("'%s' is a valid SQLAlchemy URL.", database_url)
except KeyError:
    logger.error("Environment variable '%s' is not set.", database_url, exc_info=True)
    raise
except Exception as e:
    logger.error("'%s' is not a valid SQLAlchemy URL: %s", database_url, e, exc_info=True)
    raise

# ======================================================
### 12) CREATE TABLE WITH SQL SCHEMA WITHIN DATABASE
# ======================================================
## Retrieve database URL environment
try:
    logger.info("Attempting to retrieve '%s' database URL.", database)
    url = os.getenv(database_url)
    logger.info("Successfully retrieved URL for '%s' database.", database)
except Exception as e:
    logger.error("Failed to retrieve '%s' database URL: '%s'", database, e, exc_info=True)
    raise

## Create database engine
try:
    logger.info("Attempting to create engine for '%s' database.", database)
    db_engine = create_engine(url)
    logger.info("Successfully created engine for '%s' database.", database)
except Exception as e:
    logger.error("Failed to create database engine: '%s'", e, exc_info=True)
    raise

## Load SQL schema into the python environment
try:
    logger.info("Attempting to load SQL schema for table '%s'.", table_name)
    with open(sql_filepath, 'r') as file:
        sql_commands = file.read() # CREATE TABLE commands
        logger.info("Successfully loaded SQL schema for table '%s'.", table_name)
except Exception as e:
    logger.error("Failed to load SQL schema for table '%s'.", table_name, exc_info=True)
    raise

## Execute SQL schema in database
try:
    with db_engine.begin() as conn: # begin() opens the connection and commits if successful or rolls back if failed
        logger.info("Attempting to execute SQL schema %s in the '%s' database.", os.path.basename(sql_filepath), database)
        conn.execute(text(sql_commands)) # Execute sql command
        logger.info("Successfully executed SQL schema for table '%s' in '%s' database.", table_name, database)
except SQLAlchemyError as e:
    logger.error("Error executing schema: '%s'", e, exc_info=True)
    raise

# ================================================================
### 13) Upload CSV data into the SQL table
# ================================================================
### Determine whether we will buffer the data into RAM, or if we need to save the data to disk
df = df.fillna("\\N") # Convert NaN to "\\N" for Postgres compatibility
mem_df = df.memory_usage(deep=True).sum() / 1000000000 # Calculate the size of the dataframe in gigabytes

### Upload data from dataframe in SQL table (includes a loop to determine temporary storage in RAM or hard-disk)
raw_conn = db_engine.raw_connection() # raw connection leverages psycopg2 to work @ low level (optimizes speed)
try:
    with raw_conn.cursor() as cur:
        # Make sure table name is safe
        logger.info("Ensuring that table name is safe (no semicolons, SQL keywords, or special characters).")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(f"Unsafe table name: {table_name}")
        logger.info("Table name safe.")

        # Check if dataframe is >= the RAM buffer limit, prior to uploading data to Postgres table
        if mem_df >= RAM_buffer_limit:
            logger.info("Attempting to copy large dataframe (%.2f GB) to temporary file on hard-disk.", mem_df)
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv') as tmp:
                df.to_csv(tmp, index=False, header=False)
                tmp.seek(0) # moves the file pointer to the beginning of the file
                ## If you plan to remove all data from the table before loading new data (uncomment next 3 blocks)
                # cur.execute(f"TRUNCATE TABLE {table_name};")
                # raw_conn.commit()
                # logger.info("Data successfully truncated from table '%s' in '%s' database.", table_name, database)
                cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV NULL '\\N'", tmp) # copy_expert() optimizes speed
                logger.info("Successfully copied large dataframe to table '%s' in '%s' database.", table_name, database)
        # If dataframe is < 2 GB, it will be buffered to RAM (threshold based on 32 GB RAM computer)
        else:
            buffer = io.StringIO()  # Create a RAM object that can receive StringIO for postgres upload
            df.to_csv(buffer, index=False, header=False)  # Buffer dataframe into RAM
            buffer.seek(0) # moves cursor to first position
            ## If you plan to remove all data from the table before loading new data (uncomment next 3 blocks)
            # cur.execute(f"TRUNCATE TABLE {table_name};")
            # raw_conn.commit()
            # logger.info("Data successfully truncated from table '%s' in '%s' database.", table_name, database)
            cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV NULL '\\N'", buffer) # copy_expert() optimizes speed
            logger.info("Successfully copied small dataframe to table '%s' in '%s' database using buffer.", table_name, database)

        # Commit changes to database
        raw_conn.commit() # Commit the data upload transaction
        logger.info("Successfully committed transaction.")
except Exception as e:
    raw_conn.rollback()
    logger.error("Failed to copy dataframe into '%s' table in '%s' database: %s", table_name, database, e, exc_info=True)
    raise

# ==========================================
### 14) INSPECT THE STRUCTURE OF THE TABLE
# ==========================================
## Table structure verification
with db_engine.connect() as conn:
    try:
        # Table Shape
        logger.info("Inspecting the shape of table '%s' in '%s' database.", table_name, database)
        row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};")).scalar()
        col_count = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.columns WHERE table_name = :table;
        """), {"table": table_name}).scalar()
        logger.info(f"Table shape: {row_count} rows x {col_count} columns")
    except Exception as e:
        logger.error("Failed to inspect table shape: %s", e, exc_info=True)

    try:
        # Table columns and data types
        logger.info("Inspecting the column names and data types in table '%s' in '%s' database.", table_name, database)
        result = conn.execute(text("""
            SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :table;
        """), {"table": table_name})
        for col, dtype in result:
            logger.info(f"Column: {col} | Type: {dtype}")
    except Exception as e:
        logger.error("Failed to inspect column names and data types: %s", e, exc_info=True)

    try:
        # Inspect the head (limit 2) of the table
        logger.info("Inspecting the head of table '%s' in '%s' database.", table_name, database)
        df_head = pd.DataFrame(
            conn.execute(text(f"SELECT * FROM {table_name} LIMIT 2")).fetchall(),
            columns=[col for col in df.columns]
        )
        logger.info("Head of the '%s' table in '%s' database:\n %s", table_name, database, df_head)
    except Exception as e:
        logger.error("Failed to inspect table head: %s", e, exc_info=True)

## Disk usage of table in database
with db_engine.connect() as conn:
    size = conn.execute(text("SELECT pg_size_pretty(pg_total_relation_size(:table));"),
                        {"table": table_name}).scalar()
    logger.info("Disk usage of table '%s' in '%s' database: %s", table_name, database, size)

##################################
######### END OF SCRIPT ##########
##################################