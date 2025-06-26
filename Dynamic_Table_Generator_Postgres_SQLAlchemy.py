"""
===============================================================================
Title: Create and Upload Longitudinal Diabetes Table to PostgreSQL
Author: Chris Weston
Date: 2025-06-26
===============================================================================

Purpose:
    This script processes a longitudinal diabetes dataset and performs an
    end-to-end pipeline that includes data cleaning, stratified sampling,
    schema generation, SQL table creation, and efficient data upload into
    a PostgreSQL database.

    It is designed for scalable workflows with built-in safeguards, memory-aware
    logic, and structured logging. The goal is to prepare and insert a cleaned,
    sampled version of a large CSV dataset into a PostgreSQL database in a
    robust and reproducible manner.

-------------------------------------------------------------------------------

Key Workflow Steps:
    1. Load the full diabetes dataset from a CSV file.
    2. Perform stratified sampling based on treatment group (e.g., 20% per group).
    3. Clean column names to conform with PostgreSQL constraints:
        - Max 63 characters
        - Lowercase
        - Alphanumeric with underscores
    4. Validate column names and log warnings for any violations.
    5. Generate a SQL `CREATE TABLE` statement with inferred data types.
    6. Save the SQL schema to a `.sql` file for reproducibility.
    7. Establish a connection to the specified PostgreSQL database.
    8. Create the target table if it does not already exist.
    9. Upload the data using memory-aware branching:
        - For DataFrames >= 2GB, write to a temporary `.csv` on disk.
        - For DataFrames < 2GB, buffer via RAM using `StringIO`.
    10. Confirm upload success by inspecting:
        - Table row and column count
        - Column names and data types
        - Preview of the first two rows
        - Total disk usage of the new table

-------------------------------------------------------------------------------

Arguments and Configuration:
    csv_path (str):
        Absolute or relative path to the input CSV file containing the full
        longitudinal dataset to be sampled and loaded.

    database (str):
        Name of the PostgreSQL database where the new table will be created.

    url (str):
        Name of the environment variable containing the SQLAlchemy-compatible
        PostgreSQL database URL (e.g., 'PG_HEALTHDB_URL').

    output_dir (str):
        Directory path where the generated SQL schema `.sql` file will be saved.

    log_filepath (str):
        Path to the file where execution logs will be stored.

-------------------------------------------------------------------------------

Data Upload Strategy:
    The script dynamically selects the appropriate upload method based on memory usage:

    - If the in-memory DataFrame exceeds 2 GB:
        - The data is written to a temporary `.csv` file on disk.
        - It is then loaded into PostgreSQL using `COPY ... FROM STDIN`.

    - If the in-memory DataFrame is under 2 GB:
        - The data is buffered via an in-memory `StringIO` stream.
        - The `COPY` command is used directly from RAM.

    All `NaN` values are converted to PostgreSQL-compatible nulls using `"\\N"`.

-------------------------------------------------------------------------------

Logging and Debugging:
    - Logs are written to both a file and console using the `logging` module.
    - Logging level is set to `INFO`, with full tracebacks included on exceptions.
    - Key events, warnings, and errors are timestamped and structured for traceability.

-------------------------------------------------------------------------------

Validation and Safeguards:
    - Table name is validated using a regex to prevent SQL injection or malformed identifiers.
    - Column names are cleaned and validated to meet PostgreSQL constraints:
        - ≤ 63 characters
        - Lowercase
        - Alphanumeric with optional underscores
    - The `CREATE TABLE` statement uses `IF NOT EXISTS` for safe reruns.

-------------------------------------------------------------------------------

Output:
    - Cleaned, stratified sample saved as a new `.csv` file.
    - SQL schema saved as a `.sql` file to the output directory.
    - Table created and populated in the target PostgreSQL database.
    - Table shape, column structure, preview, and disk usage are logged for validation.

-------------------------------------------------------------------------------

Dependencies:
    - pandas
    - sqlalchemy
    - psycopg2 (via SQLAlchemy’s raw_connection)
    - logging, os, io, tempfile, re

-------------------------------------------------------------------------------

Environment Variables:
    - PG_HEALTHDB_URL: SQLAlchemy-formatted database URL
      (e.g., postgresql+psycopg2://user:password@host:port/dbname)

-------------------------------------------------------------------------------

Usage Example:
    Run this script from a Python environment that has access to the required
    CSV file and the appropriate PostgreSQL credentials set in your environment:

    $ export PG_HEALTHDB_URL="postgresql+psycopg2://user:pass@localhost:5432/healthdb"
    $ python create_table_from_longitudinal_data.py

===============================================================================
"""

# ==============================================================================
### IMPORT MODULES
# ==============================================================================
# ---- Import modules ----
import pandas as pd
pd.set_option('display.max_columns', None)
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError
import logging
import tempfile # for saving dataframe to temporary hard-disk (if dataframe is >= 2 GB RAM)
import os
import io # for buffering tabular data into RAM as stringIO (if dataframe is < 2 GB RAM)
import re # if we plan to remove data from pre-existing Postgres table before loading new data

# ======================
### CONFIGURATION
# ======================
### Paths
# Log filepath
log_filepath = "py_scripts/database/postgres_sqlalchemy/logs/create_table_from_longitudinal_data.log"

# CSV filepath (used to construct SQL table)
csv_path = "datasets/my_custom_datasets/diabetes_60k_pats_20250428.csv"

# Destination folder for .sql file
output_dir = "/home/cweston1/miniconda3/envs/PythonProject/py_scripts/database/sql_schemas/diabetes_schema/"

### USER-defined variables
# Database that you plan to make a connection with (i.e. where we will create the SQL table)
database = "healthdb"

# URL for the PostgreSQL database that you want to make connection with (derived from URL in bashrc)
url = "PG_HEALTHDB_URL"

# ===============
### LOGGING
# ===============
log_format = "%(asctime)s - %(levelname)s - %(message)s" # date-time format

# Clear existing handlers to avoid duplication or mismatches (important in interactive sessions)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create handlers
file_handler = logging.FileHandler(log_filepath)
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

logger.info(f"These logs are saved in '{log_filepath}'")
script_objective = "To create a table in 'healthdb' that has the longitudinal diabetes sample"
logger.info(f"SCRIPT OBJECTIVE: {script_objective}")

# ================================================================
### 1) DEFINE FUNCTIONS USED IN THIS SCRIPT
# ================================================================
## Create function that standardizes the column headers to a clean lowercase format
def clean_column(col, max_length=63): # the max length for column names is 63 characters in postgresql
    cleaned = (
        col.strip()
           .lower()
           .replace(" ", "_")
           .replace("-", "_")
           .replace(":", "_")
    )
    return cleaned[:max_length]

## Create function that will infer the SQL data type form pandas dtype
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

# ==================================================
# LOAD DIABETES DATA
# ==================================================
## Upload full diabetes dataset to sample from
try:
    logger.info("Loading data from '%s", csv_path)
    df = pd.read_csv(csv_path)
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

# ================================================================
### 3a) Count unique number of patients within each treatment group
# ================================================================
## Count the unique number of patients by treatment
patient_counts = df.groupby('Treatment')['Patient_ID'].nunique()
logger.info(f"Number of unique patients per patient_counts:\n%s", patient_counts.to_string())

# Treatment
# Drug A                   20000
# Drug A with Exercise     20000
# Placebo with Exercise    20000

# ================================================================
### 3b) Get stratified random sample of patients (e.g. 20% per treatment group)
# ================================================================
sample_fraction = 0.2   # 20% per group
total_patients = df['Patient_ID'].nunique()  # total number of patients

logger.info("Starting stratified sample for %.1f%% of %d patients", sample_fraction, total_patients)

try:
    sampled_patients = (
        df[['Patient_ID', 'Treatment']] # returns dataframe with these fields
        .drop_duplicates() # Keeps only one unique "patient_id / treatment" combination
        .groupby('Treatment')
        .sample(frac=sample_fraction, random_state=39) # uses the sample fraction
    )
    logger.info("Stratified sampling completed.")
except ValueError as e:
    logger.error("Sampling failed: %e", e, exc_info=True)
    raise

# ================================================================
### 3c) Subset original DataFrame to include all rows for sampled patients
# ================================================================
sampled_ids = sampled_patients['Patient_ID'].unique() # Isolate the unique IDs from sampled patients

df_sampled = df[df['Patient_ID'].isin(sampled_ids)] # create dataframe with the sampled IDs

logger.info("Filtered %d rows based on %d sampled IDs", len(df_sampled), len(sampled_ids))

# ================================================================
### 3d) VERIFY THAT THE APPROPRIATE FRACTION OF PATIENTS WERE SAMPLED
# ================================================================
# Count total patients per treatment in the full dataset and the sample
original_counts = df.groupby('Treatment')['Patient_ID'].nunique()  # creates pandas series
sampled_counts = df_sampled.groupby('Treatment')['Patient_ID'].nunique() # creates pandas series

# Compare expected vs actual
for treatment in original_counts.index: # indexed by the groupby()
    expected = int(original_counts[treatment] * sample_fraction) # n patients in original treatment * fraction
    actual = sampled_counts.get(treatment, 0) # uses dictionary logic to fetch the value for treatment key

    logger.info(
        "Treatment: %s | Expected Sampled: %d | Actual Sampled: %d | Δ: %d",
        treatment, expected, actual, actual - expected
    )

    # Optional: Add warning if there's a mismatch
    if abs(actual - expected) > 1:  # Allowing for rounding error
        logger.warning("Sampling mismatch for %s: expected %d, got %d",
                       treatment, expected, actual)

# ==============================================================================================
### 4) CLEAN THE COLUMN NAMES OF THE SAMPLED DATA (NECESSARY FOR CLEAN UPDLOAD INTO DATABASE)
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
### 5) VERIFY THAT THE COLUMN NAMES MEET POSTGRESQL CONSTRAINTS
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
# 6) VERIFY THAT THE CLEANED DATAFRAME HAS THE SAME NUMBER OF COLUMNS AS THE ORIGINAL DATASET
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
### 7) LOG THE NAMES OF ALL COLUMNS IN THE CLEANED DATAFRAME
# ======================================================================
logger.info("Cleaned dataframe column names: %s", df.columns.tolist())

# ======================================================================
### 8) SAVE THE CLEANED DATAFRAME TO DISK AS A CSV FILE
# ======================================================================
try:
    csv_file = "diabetes_12k_pats_20250611.csv"
    csv_path = os.path.join("datasets/my_custom_datasets/", csv_file)
    logger.info("Attempting to save cleaned dataframe to CSV file: '%s'.", csv_file)
    df.to_csv(csv_path)
    logger.info("Successfully saved dataframe as '%s' at path '%s'.", csv_file, csv_path)
except Exception as e:
    logger.error("Failed to save dataframe as '%s' at path '%s': '%s'", csv_file, csv_path, e, exc_info=True)
    raise

# =====================================================================
### 9) CREATE SQL SCHEMA FROM A DATAFRAME
# =====================================================================
## Configuration
os.makedirs(output_dir, exist_ok=True)
logger.info("SQL schema output directory exists.")

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
    sql_filepath = os.path.join(output_dir, f"{table_name}.sql")
    with open(sql_filepath, "w") as f:
        # Include a comment header in the SQL file for readability
        f.write(f"-- Table: {table_name}\n{sql}")
    logger.info("Successfully wrote SQL schema to file: '%s'", sql_filepath)

except Exception as e:
    logging.error("Failed to generate SQL schema for table '%s': %s", table_name, e, exc_info=True)
    raise

# ===============================================================
### 10) VERIFY THAT THE DATABASE URL IS VALID FOR SQLALCHEMY
# ===============================================================
try:
    if make_url(os.environ[url]): # make_url checks if URL is valid for SQLALchemy
        logger.info("'%s' is a valid SQLAlchemy URL.", url)
except KeyError:
    logger.error("'%s' is not a valid SQLAlchemy URL.", url, exc_info=True)
    raise
except Exception as e:
    logger.error("%s is malformed: '%s'", url, e, exc_info=True)
    raise

# ======================================================
### 11) CREATE TABLE WITH SQL SCHEMA WITHIN DATABASE
# ======================================================
## Retrieve database URL environment
try:
    logger.info("Attempting to retrieve '%s' database URL.", database)
    healthdb_url = os.getenv("PG_HEALTHDB_URL")
    logger.info("Successfully retrieved URL for '%s' database.", database)
except Exception as e:
    logger.error("Failed to retrieve '%s' database URL: '%s'", database, e, exc_info=True)
    raise

## Create database engine
try:
    logger.info("Attempting to create engine for '%s' database.", database)
    healthdb_engine = create_engine(healthdb_url)
    logger.info("Successfully created engine for '%s' database.", database)
except Exception as e:
    logger.error("Failed to create healthdb engine: '%s'", e, exc_info=True)
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
    with healthdb_engine.begin() as conn: # begin() opens the connection and commits if successful or rolls back if failed
        logger.info("Attempting to execute SQL schema %s in the '%s' database.", os.path.basename(sql_filepath), database)
        conn.execute(text(sql_commands)) # Execute sql command
        logger.info("Successfully execute SQL schema for table '%s' in '%s' database.", table_name, database)
except SQLAlchemyError as e:
    logger.error("Error executing schema: '%s'", e, exc_info=True)
    raise

# ================================================================
### 12) Upload CSV data into the SQL table
# ================================================================
### Determine whether we will buffer the data into RAM, or if we need to save the data to disk
df = df.fillna("\\N") # Convert NaN to "\\N" for Postgres compatibility
mem_df = df.memory_usage(deep=True).sum() / 1000000000 # Calculate the size of the dataframe in gigabytes

### Upload data from dataframe in SQL table (includes a loop to determine temporary storage in RAM or hard-disk)
raw_conn = healthdb_engine.raw_connection() # raw connection leverages psycopg2 to work @ low level (optimizes speed)
try:
    with raw_conn.cursor() as cur:
        ## Validate table name to protect from SQL injection: necessary if truncating tables in this block
        logger.info("Ensuring that table name is safe (no semicolons, SQL keywords, or special characters).")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name): # Ensures no semicolons, SQL keywords, or special characters
            raise ValueError(f"Unsafe table name: {table_name}")
        logger.info("Table name safe.")

        # Check if dataframe is >= 2 GB "large dataframe" (threshold based on 32 GB RAM computer)
        if mem_df >= 2:
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

        raw_conn.commit() # Commit the data upload transaction
        logger.info("Successfully committed transaction for table '%s' in '%s' database.", table_name, database)
except Exception as e:
    raw_conn.rollback()
    logger.error("Failed to copy dataframe into '%s' table in '%s' database: %s", table_name, database, e, exc_info=True)
    raise

# =========================================================================
### 13) INSPECT THE STRUCTURE OF THE TABLE THAT WAS POPULATED WITH DATA
# =========================================================================
## Table structure verification
with healthdb_engine.connect() as conn:
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
with healthdb_engine.connect() as conn:
    size = conn.execute(text("SELECT pg_size_pretty(pg_total_relation_size(:table));"),
                        {"table": table_name}).scalar()
    logger.info("Disk usage of table '%s' in '%s' database: %s", table_name, database, size)

