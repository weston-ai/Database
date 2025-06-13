#!/bin/bash

CSV_DIR="/home/cweston1/miniconda3/envs/PythonProject/datasets/kaggle_csv_datasets"
DB_NAME="mydb"
DB_USER="cweston1"

cd "$CSV_DIR" || exit

for file in *.csv; do
  table_name=$(basename "$file" .csv | tr '[:upper:]' '[:lower:]')
  echo "🚀 Loading $file into table $table_name..."

  psql -U "$DB_USER" -d "$DB_NAME" -c "\copy $table_name FROM '$CSV_DIR/$file' DELIMITER ',' CSV HEADER"

  if [ $? -eq 0 ]; then
    echo "Successfully loaded $table_name"
  else
    echo "Failed to load $table_name"
  fi
done
