# Script Objective: use a production level method for creating a postgreSQL database with psycopg2.

# Import modules
import pandas as pd
pd.set_option('display.max_columns', None)
import psycopg2
import psycopg2.sql as sql
import logging
import os

os.getcwd() # '/home/cweston1/miniconda3/envs/PythonProject'

# ==============================================================
### STATE THE SCRIPT OBJECTIVE
# ==============================================================
script_objective = "To create a PostgreSQL database and assign user privileges"

# ==============================================================
### DEFINE LOGGING (i.e. save log to file and output in console)
# ==============================================================

# Define logging format
log_format = "%(asctime)s - %(levelname)s - %(message)s"
log_filepath = "py_scripts/database/postgresql_psycopg2/logs/otherdb_setup.log"

# Set up root logger
logging.basicConfig(
    level=logging.DEBUG, # Logs DEBUG, INFO, WARNING, ERROR, and CRITICAL (change to INFO if DEBUG is no longer wanted in logs)
    format=log_format, # Timestamp + log level + message
    handlers=[
        logging.FileHandler(log_filepath), # log file
        logging.StreamHandler() # console output
    ]
)

# Log the script objective
logging.info(f"SCRIPT OBJECTIVE: {script_objective}")

# Log the location of the log
logging.info(f"These logs are saved in '{log_filepath}'")

# ==================================================================
### CREATE FUNCTION THAT PRODUCES A DATABASE CALLED "OtherTablesDB"
# ==================================================================
fxn_name = "create_database_with_privileges()"

# Define the create_database function
def create_database_with_privileges(
        dbname: str,
        owner: str = None,
        template: str = "template1",
        encoding: str = "UTF8",
        conn_params: dict = None
):
    """
    Create a new PostgreSQL database using psycopg2.

    Parameters:
        dbname (str): Name of the new database
        owner (str): Optional owner of the new DB (must be a valid role)
        template (str): Template to use (default: template1)
        encoding (str): Character encoding (default: UTF8)
        conn_params (dict): Parameters for connecting to the 'postgres' system DB
    """

    if conn_params is None:
        raise ValueError("conn_params dictionary is required")

    try:
        logging.info(f"Connecting to database '{conn_params['dbname']}' to create '{dbname}'...")
        conn = psycopg2.connect(**conn_params) # Connect to 'postgres' DB using connection parameters for a pre-existing database
        conn.set_session(autocommit=True) # setting Autocommit=True to allow DB creation
        cur = conn.cursor() # create cursor

        # ===========================================
        # Build the CREATE DATABASE command
        # ===========================================
        query = sql.SQL("CREATE DATABASE {} WITH TEMPLATE {} ENCODING {}").format(
            sql.Identifier(dbname), # Identifier avoids SQL injection
            sql.Identifier(template), # Identifier avoids SQL injection
            sql.Literal(encoding) # Literal avoids SQL injection
        )

        # Add owner clause if provided
        if owner:
            query += sql.SQL(" OWNER {}").format(sql.Identifier(owner))

        cur.execute(query)
        logging.info(f"Database '{dbname}' created successfully.")

        # ===========================================
        # Assign privileges to the database
        # ===========================================
        if owner:
            # Now connect to the new database to GRANT privileges
            grant_conn_params = conn_params.copy()
            grant_conn_params["dbname"] = dbname

            grant_conn = psycopg2.connect(**grant_conn_params)
            grant_conn.set_session(autocommit=True)
            grant_cur = grant_conn.cursor()

            grant_query = sql.SQL("GRANT CONNECT, TEMP ON DATABASE {} TO {};").format(
                sql.Identifier(dbname),
                sql.Identifier(owner)
            )
            logging.info(f"Granting CONNECT and TEMP on '{dbname}' to '{owner}'...")
            grant_cur.execute(grant_query)
            logging.info(f"Granted CONNECT and TEMP on '{dbname}' to '{owner}'.")

    except Exception as e:
        logging.error(f"Error creating database or granting privileges: '{dbname}': {e}", exc_info=True)

    finally:
        for var_name in ["cur", "conn", "grant_cur", "grant_conn"]:
            obj = locals().get(var_name)
            if obj:
                try:
                    obj.close()
                except Exception:
                    logging.warning(f"Failed to close {var_name}.", exc_info=True)

        logging.debug("All PostgreSQL connections closed.")

# Log script completion
logging.info(f"{fxn_name} script completed.")

# Define the connection parameters (these must reflect a pre-existing database to allow connection)
conn_params = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "0821",
    "host": "localhost",
    "port": "5432"
}

# Run the create_database function to create a database called "otherdb" with owner "cweston1"
create_database_with_privileges("otherdb", owner = "cweston1", conn_params=conn_params)

# ==================================================================================
### VERIFY THAT THE DATABASE WAS CREATED SUCCESSFULLY IN THE POSTGRESQL CLUSTER
# ==================================================================================

## Update the bashrc with the following:
# export PG_OTHER_DBNAME="actual db name"
# export PG_OTHER_USER="actual user"
# export PG_OTHER_PASSWORD="actual password"
# export PG_OTHER_HOST="actual host"
# export PG_OTHER_PORT="actual port"

# Connect to the OtherTablesDB database
conn = psycopg2.connect(
    dbname=os.environ.get("PG_OTHER_DBNAME"),
    user=os.environ.get("PG_OTHER_USER"),
    password=os.environ.get("PG_OTHER_PASSWORD"),
    host=os.environ.get("PG_OTHER_HOST"),
    port=os.environ.get("PG_OTHER_PORT")
)

# List all databases (requires superuser access)
with conn:
    with conn.cursor() as cur:
        cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
        for db in cur.fetchall():
            print("Database:", db[0])

# Database: postgres
# Database: mydb
# Database: otherdb
