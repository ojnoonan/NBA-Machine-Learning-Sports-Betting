import sys
import os
import sqlite3
import time
import random
from datetime import datetime, timedelta
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.tools import get_json_data, to_data_frame

# Load configuration and connect to database
config = toml.load("../../config.toml")
url = config['data_url']
con = sqlite3.connect("../../Data/TeamData.sqlite")

# Helper function to get existing tables (dates)
def get_existing_dates(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return {table[0] for table in tables}

existing_tables = get_existing_dates(con)
current_year = datetime.now().year

# Iterate over datasets in config
for key, value in config['get-data'].items():
    start_date = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    end_date = datetime.strptime(value['end_date'], "%Y-%m-%d").date()
    date_pointer = start_date

    while date_pointer <= end_date:
        table_name = date_pointer.strftime("%Y-%m-%d")

        # Fetch data if missing or if it's for the current year
        if table_name not in existing_tables or (date_pointer.year == current_year and date_pointer.month == datetime.now().month):
            print(f"Fetching data for {table_name}")

            raw_data = get_json_data(
                url.format(date_pointer.month, date_pointer.day, value['start_year'], date_pointer.year, key))
            df = to_data_frame(raw_data)

            # Add a 'Date' column and write to the database
            df['Date'] = str(date_pointer)
            df.to_sql(table_name, con, if_exists="replace")
            
            time.sleep(random.randint(1, 3))  # Avoid overwhelming the server
        else:
            print(f"Data for {table_name} already exists, skipping.")

        date_pointer += timedelta(days=1)