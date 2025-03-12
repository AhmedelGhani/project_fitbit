import sqlite3
import pandas as pd
import numpy as np

connection = sqlite3.connect('fitbit_database.db')
query_activity = 'SELECT * FROM "daily_activity";'
cursor = connection.cursor()
cursor.execute(query_activity)
rows = cursor.fetchall()
df_activity = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

id_counts = df_activity['Id'].value_counts().reset_index()
id_counts.columns = ['Id', 'Count']

def classify_user(count):
    if count <= 10:
        return 'Light user'
    elif 11 <= count <= 15:
        return 'Moderate user'
    else:
        return 'Heavy user'

id_counts['Class'] = id_counts['Count'].apply(classify_user)
user_class_df = id_counts[['Id', 'Class']]

query_sleep =  'SELECT * FROM "minute_sleep";'
cursor.execute(query_sleep)
rows = cursor.fetchall()
df_sleep = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
sleep_duration_df = df_sleep.groupby(['Id', 'logId']).size().reset_index(name = 'Sleepduration')

merged_df = pd.merge(sleep_duration_df, user_class_df, on = 'Id', how = 'left')

pd.set_option('display.float_format', '{:.0f}'.format)
print(merged_df)
connection.close()
