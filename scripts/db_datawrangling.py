import sqlite3
import pandas as pd
import numpy as np


connection = sqlite3.connect('fitbit_database.db')
query = 'SELECT * FROM "weight_log";'
cursor = connection.cursor()
cursor.execute(query)
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
#print(df)
connection.close()

#print("Missing values before handling:")
#print(df.isnull().sum())

df["WeightKg"] = df["WeightKg"].fillna(df["WeightPounds"] / 2.20462262185)
df = df.drop(columns=['Fat']) 

#print("Missing values after handling:")
##print(df.isnull().sum())
#print(df)

def mergingtables(table1, table2, join_column='Id'):
    connection = sqlite3.connect('fitbit_database.db')
    cursor = connection.cursor()

    query_weight = f'SELECT * FROM "{table1}";'
    cursor.execute(query_weight)
    weight_rows = cursor.fetchall()
    weight_columns = [desc[0] for desc in cursor.description]
    df_weight = pd.DataFrame(weight_rows, columns=weight_columns)
    
    query_other = f'SELECT * FROM "{table2}";'
    cursor.execute(query_other)
    other_rows = cursor.fetchall()
    other_columns = [desc[0] for desc in cursor.description]
    df_other = pd.DataFrame(other_rows, columns=other_columns)

    connection.close()
    
    merged_df = pd.merge(df_weight, df_other, on=join_column, how='inner')
    
    return merged_df

merged_data = mergingtables('weight_log', 'hourly_steps')
print(merged_data)

def numeric_summary(df, columns=None, start_date = None, end_date = None, start_time = None, end_time = None):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')
        if start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%m/%d/%Y')
            df = df[df['Date'] >= start_date]
        if end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%m/%d/%Y')
            df = df[df['Date'] <= end_date]
    
    if start_time is not None:
        start_time = pd.to_datetime(start_time, format = '%I:%M:%S %p').time()
        df = df[df['Date'].dt.time >= start_time]
    if end_time is not None:
        end_time = pd.to_datetime(end_time, format = '%I:%M:%S %p').time()
        df = df[df['Date'].dt.time <= end_time]
    
    if columns is None:
        exclude = ['Id', 'LogId', 'IsManualReport', 'ActivityHour']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in numeric_cols if col not in exclude]
    else:
        if not isinstance(columns, list):
            columns = [columns]
        columns = [col for col in columns if col in df.columns]
    
    print("Rows after time filter:", len(df))
    grouped_stats = df.groupby('Id')[columns].agg(['median', 'std', 'min', 'max', 'count'])
    return grouped_stats

temp = merged_data.copy()

numeric_summary = numeric_summary (merged_data, 'StepTotal', None, None, '6:00:00 AM', '10:00:00 AM')
print (numeric_summary)