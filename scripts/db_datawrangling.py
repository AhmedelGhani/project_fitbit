
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
#cursor.execute("PRAGMA table_info(weight_log);")
#columns = [col[1] for col in cursor.fetchall()]
#if 'Fat' in columns:
    #cursor.execute("ALTER TABLE weight_log DROP COLUMN Fat;")
    #connection.commit()
#connection.close()

#4.2: Functions for merging tables based on individual ID
def mergingtables(table1, table2, time_column1=None, time_column2=None):
    connection = sqlite3.connect('fitbit_database.db')

    df1 = table1 if isinstance(table1, pd.DataFrame) else pd.read_sql_query(f"SELECT * FROM {table1}", connection)
    df2 = table2 if isinstance(table2, pd.DataFrame) else pd.read_sql_query(f"SELECT * FROM {table2}", connection)

    connection.close()
    
    if time_column1 and time_column2:
        df1[time_column1] = pd.to_datetime(df1[time_column1], errors='coerce')
        df2[time_column2] = pd.to_datetime(df2[time_column2], errors='coerce')

        df1['Date/Time'] = df1[time_column1].dt.floor('D')
        df2['Date/Time'] = df2[time_column2].dt.floor('D')

        df1_resampled = (df1.groupby(['Id', 'Date/Time']).sum(numeric_only=True).reset_index())
        df2_resampled = (df2.groupby(['Id', 'Date/Time']).sum(numeric_only=True).reset_index())

        df1_resampled.rename(columns={time_column1: 'Date/Time'}, inplace=True)
        df2_resampled.rename(columns={time_column2: 'Date/Time'}, inplace=True)

        merged = pd.merge(df1_resampled, df2_resampled, how='left', on=['Id', 'Date/Time'])
    else:
        merged = pd.merge(df1, df2, on='Id', how='left')

    return merged

pd.set_option('display.float_format', '{:.0f}'.format)

def get_daily_sleep_minutes():
    connection = sqlite3.connect('fitbit_database.db')
    sleep_df = pd.read_sql_query("SELECT * FROM minute_sleep", connection)
    connection.close()

    sleep_df['timestamp'] = pd.to_datetime(sleep_df['date'], errors='coerce')
    sleep_df['Date'] = sleep_df['timestamp'].dt.floor('H')

    hourly_sleep = (sleep_df.groupby(['Id', 'Date']).size().reset_index(name='SleepMinutes'))

    return hourly_sleep

def get_hourly_active_minutes():
    connection = sqlite3.connect('fitbit_database.db')
    df = pd.read_sql_query("SELECT * FROM daily_activity", connection)
    connection.close()

    df['timestamp'] = pd.to_datetime(df['ActivityDate'], errors='coerce')
    df['Date'] = df['timestamp'].dt.floor('H')

    df['ActiveMinutes'] = (df['VeryActiveMinutes'] + df['FairlyActiveMinutes'] + df['LightlyActiveMinutes'])

    hourly_active = (df.groupby(['Id', 'Date'])['ActiveMinutes'].sum().reset_index(name='ActiveMinutes'))

    return hourly_active