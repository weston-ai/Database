#!/bin/bash

for file in *.sql; do
  echo "Running $file"
  psql -U cweston1 -d mydb -f "$file"
done
