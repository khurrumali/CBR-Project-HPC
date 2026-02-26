import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv('DATA_PATH')
db_path = os.path.join(os.path.dirname(os.getenv('MODELS_PATH')), 'eicu_demo.sqlite')

print(f"Reading CSVs from: {data_path}")
print(f"Creating SQLite DB at: {db_path}")

# Find all .csv.gz files in the data directory
all_files = [f for f in os.listdir(data_path) if f.endswith('.csv.gz')]
tables = [f.replace('.csv.gz', '') for f in all_files]

print(f"Found {len(tables)} tables to import.")

conn = sqlite3.connect(db_path)

for table in tables:
    csv_file = os.path.join(data_path, f"{table}.csv.gz")
    print(f"Importing {table}...")
    try:
        # Using low_memory=False to avoid DtypeWarnings for large clinical tables
        df = pd.read_csv(csv_file, low_memory=False)
        df.to_sql(table, conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"Error importing {table}: {e}")

conn.close()
print("Database setup complete.")
