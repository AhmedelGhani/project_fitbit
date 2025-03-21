import sqlite3
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import linregress
import time

#Section 4.1, Weight_log table modified to fill in missing values from WeightKg and drop fat column
connection = sqlite3.connect('fitbit_database.db')
cursor = connection.cursor()

update_query = """UPDATE weight_log SET WeightKg = WeightPounds / 2.20462262185 WHERE WeightKg IS NULL;"""
cursor.execute(update_query)
connection.commit()

#When class is run for the first class, fat column is eliminated. Otherwise, it is skipped over
cursor.execute("PRAGMA table_info(weight_log);")
columns = [col[1] for col in cursor.fetchall()]
if 'Fat' in columns:
    cursor.execute("ALTER TABLE weight_log DROP COLUMN Fat;")
    connection.commit()
connection.close()

#4.2: Functions for merging tables based on individual ID
def mergingtables(table1, table2, columns1, columns2, time_column1=None, time_column2=None, chunksize = 100000):
    connection = sqlite3.connect('fitbit_database.db')
    
    df1 = pd.read_sql_query(f"SELECT {', '.join(columns1)} FROM {table1}", connection)
    df2 = pd.read_sql_query(f"SELECT {', '.join(columns2)} FROM {table2}", connection)
    
    if time_column1 and time_column2:
        df1[time_column1] = pd.to_datetime(df1[time_column1], errors='coerce').dt.round('H')
        df2[time_column2] = pd.to_datetime(df2[time_column2], errors='coerce').dt.round('H')
        merged = pd.merge(df1, df2, how='inner', left_on=['Id', time_column1], right_on=['Id', time_column2])
    else:
        merged = pd.merge(df1, df2, on='Id', how='inner')

    connection.close()
    return merged

conn = sqlite3.connect('fitbit_database.db')
df = pd.read_sql_query("SELECT * FROM heart_rate", conn)
conn.close()

print(df)  

pd.set_option('display.float_format', '{:.0f}'.format)
merged_data = mergingtables('hourly_steps', 'minute_sleep', ['Id','ActivityHour', 'StepTotal'], ['Id','date', 'value'], 'ActivityHour', 'date')
print(merged_data)
