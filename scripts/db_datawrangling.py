
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
def mergingtables(table1, table2, time_column1=None, time_column2=None):
    connection = sqlite3.connect('fitbit_database.db')
    
    df1 = pd.read_sql_query(f"SELECT * FROM {table1}", connection)
    df2 = pd.read_sql_query(f"SELECT * FROM {table2}", connection)
    
    if time_column1 and time_column2:
        df1[time_column1] = pd.to_datetime(df1[time_column1], errors='coerce')
        df2[time_column2] = pd.to_datetime(df2[time_column2], errors='coerce')

        df1_resampled = (df1.groupby(['Id', pd.Grouper(key=time_column1, freq='1H')]).mean(numeric_only=True).reset_index())
        df2_resampled = (df2.groupby(['Id', pd.Grouper(key=time_column2, freq='1H')]).mean(numeric_only=True).reset_index())

        df1_resampled.rename(columns={time_column1: 'Date/Time'}, inplace=True)
        df2_resampled.rename(columns={time_column2: 'Date/Time'}, inplace=True)

        merged = pd.merge(df1_resampled, df2_resampled, how='inner', on=['Id', 'Date/Time'])
    else:
        merged = pd.merge(df1, df2, on='Id', how='inner')

    connection.close()
    return merged

pd.set_option('display.float_format', '{:.0f}'.format)

