import sqlite3
import pandas as pd
import os

def inspect_database(db_path):
    """Inspect SQLite database structure and contents"""
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return
        
    print(f"Inspecting database: {db_path}")
    print(f"Database size: {os.path.getsize(db_path) / (1024*1024):.2f} MB")
    
    conn = sqlite3.connect(db_path)
    
    # Get list of all tables
    tables_query = """
    SELECT name FROM sqlite_master 
    WHERE type='table'
    ORDER BY name;
    """
    tables = pd.read_sql_query(tables_query, conn)
    
    if tables.empty:
        print("\nNo tables found in database.")
        conn.close()
        return
    
    print("\nDatabase Tables:")
    print("---------------")
    for table in tables['name']:
        print(f"\nTable: {table}")
        print("-" * (len(table) + 7))
        
        # Get schema for each table
        schema_query = f"PRAGMA table_info({table});"
        schema = pd.read_sql_query(schema_query, conn)
        print("\nColumns:")
        print(schema[['name', 'type']].to_string())
        
        # Get row count
        count_query = f"SELECT COUNT(*) as count FROM {table};"
        count = pd.read_sql_query(count_query, conn)
        print(f"\nNumber of rows: {count['count'].iloc[0]:,}")
        
        # Get sample data
        sample_query = f"SELECT * FROM {table} LIMIT 1;"
        sample = pd.read_sql_query(sample_query, conn)
        print("\nSample row:")
        print(sample.to_string())
        print("\n" + "="*50)
    
    conn.close()

if __name__ == "__main__":
    db_path = "/Users/olivernoonan/PythonProjects/NBA-Machine-Learning-Sports-Betting/Data/nba.sqlite"
    inspect_database(db_path)
