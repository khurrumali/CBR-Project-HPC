import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv('DATABASE_URL')
print(f"Connecting to: {db_url}")

try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"))
        tables = [row[0] for row in result]
        print(f"Connection Successful!")
        print(f"Total tables found: {len(tables)}")
        print(f"Tables index: {', '.join(tables)}")
        
        # Test query on patient table
        result = conn.execute(text("SELECT COUNT(*) FROM patient;"))
        count = result.scalar()
        print(f"Patient count: {count}")
        
except Exception as e:
    print(f"Connection Failed: {e}")
