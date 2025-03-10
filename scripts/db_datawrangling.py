#Just to see if my branch works
import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Increase column width
import numpy as np


connection = sqlite3.connect('fitbit_database.db')
query = 'SELECT * FROM "weight_log";'
cursor = connection.cursor()
cursor.execute(query)
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
print(df)
connection.close()

print("Missing values before handling:")
print(df.isnull().sum())

df["WeightKg"] = df["WeightKg"].fillna(df["WeightPounds"] / 2.20462262185)
df = df.drop(columns=['Fat']) 

print("Missing values after handling:")
print(df.isnull().sum())
print(df)

def mergingtables(other_table, join_column='Id'):
    connection = sqlite3.connect('fitbit_database.db')
    cursor = connection.cursor()

    query_weight = 'SELECT * FROM "weight_log";'
    cursor.execute(query_weight)
    weight_rows = cursor.fetchall()
    weight_columns = [desc[0] for desc in cursor.description]
    df_weight = pd.DataFrame(weight_rows, columns=weight_columns)
    
    query_other = f'SELECT * FROM "{other_table}";'
    cursor.execute(query_other)
    other_rows = cursor.fetchall()
    other_columns = [desc[0] for desc in cursor.description]
    df_other = pd.DataFrame(other_rows, columns=other_columns)

    connection.close()
    
    merged_df = pd.merge(df_weight, df_other, on=join_column, how='inner')
    
    return merged_df

merged_data = mergingtables('daily_activity')
print(merged_data.head())