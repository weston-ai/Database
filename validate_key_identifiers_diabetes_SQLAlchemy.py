"""
Script: validate_key_identifiers_diabetes_SQLAlchemy.py

Description:
    This script connects to a PostgreSQL database ("otherdb") and performs integrity checks
    between two relational tables derived from a longitudinal diabetes dataset:

        1. `diabetes_12k_pats_time_invariant_variables`: one row per patient (static attributes)
        2. `diabetes_12k_pats_time_variant_variables`: multiple rows per patient (longitudinal data)

    The purpose is to verify referential integrity between the two tables — ensuring that:
        - Every patient ID in the time-variant table has a matching ID in the time-invariant table.
        - There are no orphaned or unmatched IDs in either table.
        - The primary and foreign key design is respected before merging or analysis.

Features:
    - Uses raw SQL and SQLAlchemy to query table structures and contents.
    - Validates ID overlaps using SQL joins and subqueries.
    - Performs the same validation using pandas `set` logic and outer joins.
    - Reports on unmatched IDs, row counts, and mismatches between the two tables.
    - Useful for debugging relational consistency before merging or modeling.

Requirements:
    - PostgreSQL database with accessible tables
    - Environment variable `PG_OTHERDB_URL` must be defined
    - Python packages: `pandas`, `sqlalchemy`

Author: Chris Weston
Created: June 2025
"""

import pandas as pd
pd.set_option('display.max_columns', None)
import os
from sqlalchemy import create_engine, text, inspect

# ====================================================================
# INSPECT THE POSTGRESQL DATABASE (show available databases and list tables)
# ====================================================================

## Obtain the environment variables for postgres (so that we can build an SQLalchemy engine for postgres)
postgres_source = os.getenv("PG_POSTGRES_URL") # mydb database

## Create the sqlalchemy engine with the postgres connection
postgres_engine = create_engine(postgres_source)

## Use raw SQL to show all available databases (this is because SQLAlchemy lacks a built-in fxn for this)
with postgres_engine.connect() as conn:
    result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false;"))
    for row in result:
        print(row[0])

# postgres
# mydb
# otherdb

## Obtain the environment variables for mydb and otherdb (so that we can create SQLalchemy engines)
otherdb_source = os.getenv("PG_OTHERDB_URL")

## Create SQLalchemy engines for mydb and otherdb
otherdb_engine = create_engine(otherdb_source)

## List the tables in mydb and otherdb
inspector_otherdb = inspect(otherdb_engine)

tables_mydb = inspector_otherdb.get_table_names()
tables_mydb # ['diabetes_12k_pats_time_invariant_variables', 'diabetes_12k_pats_time_variant_variables']

# =================================================================
# SANITY CHECKS ON TABLES, USING SQL DIRECTLY IN THE OTHERDB DATABASE
# =================================================================

## COUNT THE NUMBER OF UNIQUE IDS IN THE TIME-INVARIANT AND TIME-VARIANT DATASETS
with otherdb_engine.connect() as conn:
    count_ids = conn.execute(text("""
        SELECT
            (SELECT COUNT(*)
             FROM diabetes_12k_pats_time_invariant_variables) AS count_invariant,
             
            (SELECT COUNT(*)
             FROM diabetes_12k_pats_time_variant_variables) AS count_variant;
        """)).fetchone()

count_invariant, count_variant = count_ids

count_invariant # 12,000 patients
count_variant # 286,027 patients

## ISOLATE THE UNMATCHED KEY IDENTIFIERS IN TIME-INVARIANT AND TIME-VARIANT DATASETS
with otherdb_engine.connect() as conn:
    unmatched_ids = conn.execute(text("""
    SELECT
        a.patient_id as invariant_id,
        b.patient_id as variant_id
    FROM 
        diabetes_12k_pats_time_invariant_variables AS a
    FULL OUTER JOIN    -- Full Join is necessary to indentify unmatching in both tables
        diabetes_12k_pats_time_variant_variables AS b
    ON a.patient_id = b.patient_id
    WHERE
        a.patient_id IS NULL OR b.patient_id IS NULL;
    """)).fetchall()

    print("\nUnmatched patient_ids:")
    for row in unmatched_ids:
        print(dict(row))     # No results: perfect matching!

## COUNT THE NUMBER OF UNMATCHED KEY IDENTIFIERS IN THE TIME-INVARIANT TABLE
with otherdb_engine.connect() as conn:
    count_unmatched_time_invariant = conn.execute(text("""
        SELECT COUNT(*) AS unmatched_in_invariant
        FROM diabetes_12k_pats_time_invariant_variables AS a   -- "a" is table alias; a is left table
        LEFT JOIN diabetes_12k_pats_time_variant_variables AS b   -- "b" is table alias; b is right table
        ON a.patient_id = b.patient_id
        WHERE b.patient_id IS NULL;
    """)).scalar()

print(f"Number unmatched time-invariant: {count_unmatched_time_invariant}") # Number unmatched: 0 (nice!)

## COUNT THE NUMBER OF UNMATCHED KEY IDENTIFIERS IN THE TIME-VARIANT TABLE
with otherdb_engine.connect() as conn:
    count_unmatched_time_variant = conn.execute(text("""
        SELECT COUNT(*) AS unmatched_in_variant
        FROM diabetes_12k_pats_time_variant_variables AS a   -- "a" is table alias; "a" will be left table
        LEFT JOIN diabetes_12k_pats_time_invariant_variables AS b   -- "b" is table alias; "a" will be right table
        ON a.patient_id = b.patient_id
        WHERE b.patient_id IS NULL;
    """)).scalar()

print(f"Number unmatched time-variant: {count_unmatched_time_invariant}") # Number unmatched: 0 (nice!)

## SIMPLIFY THE PREVIOUS TWO BLOCKS OF CODE INTO A SINGLE QUERY (USING TWO SUBQUERIES)
with otherdb_engine.connect() as conn:
    result = conn.execute(text("""
        SELECT
            -- First subquery
            (SELECT COUNT(*)
             FROM diabetes_12k_pats_time_invariant_variables AS a
             LEFT JOIN diabetes_12k_pats_time_variant_variables AS b
             ON a.patient_id = b.patient_id
             WHERE b.patient_id IS NULL) AS count_unmatched_invariant,
             
            -- Second subquery
            (SELECT COUNT(*)
             FROM diabetes_12k_pats_time_variant_variables AS a
             LEFT JOIN diabetes_12k_pats_time_invariant_variables AS b
             ON a.patient_id = b.patient_id
             WHERE b.patient_id IS NULL) AS count_unmatched_variant;
    """)).fetchone() # we use fetchone() because the main SELECT receives one row with two count values

unmatched_invariant, unmatched_variant = result # Assign tuple values to individual objects)

print(f"Count of unmatched keys in time-invariant data: {unmatched_invariant}")  # 0 (nice!)
print(f"Count of unmatched keys in time-variant data: {unmatched_variant}")  # 0 (nice!)

# =================================================================
# SANITY CHECK OF TABLES AFTER LOADING THEM INTO PANDAS DATAFRAMES
# =================================================================

## Load postgres tables into pandas
invariantDF = pd.read_sql("SELECT * FROM diabetes_12k_pats_time_invariant_variables", otherdb_engine)
variantDF = pd.read_sql("SELECT * FROM diabetes_12k_pats_time_variant_variables", otherdb_engine)

## Isolate unmatched patient IDs in the time-invariant and time-variant datasets
# Extract unique IDs and optimize speed (both are accomplished by using "set"; it removes duplicates)
invariant_IDs = set(invariantDF["patient_id"])
variant_IDs = set(variantDF["patient_id"])

# Count the number of unmatched IDs in each dataframe
only_in_invariant = invariant_IDs - variant_IDs
only_in_variant = variant_IDs - invariant_IDs

print(f"IDs only in time-invariant data: {len(only_in_invariant)}")  # 0 (nice!)
print(f"IDs only in time-variant data: {len(only_in_variant)}")  # 0 (nice!)

## Show full mismatch table using OUTER JOIN logic in pandas
df_invariant_IDs = pd.DataFrame({'patient_id': list(invariant_IDs)})
df_variant_IDs = pd.DataFrame({'patient_id': list(variant_IDs)})

# Full outer join of IDs
merged = pd.merge(df_invariant_IDs, df_variant_IDs, on='patient_id', how='outer', indicator=True) # indicator creates a column called "_merge", and it has the values "both", "left_only", and "right_only"

# Filter for mismatches
unmatched_df = merged[merged['_merge'] != 'both']
print(f"\nCount of mismatched patient_ids: {len(unmatched_df)}") # 0 (nice!)


