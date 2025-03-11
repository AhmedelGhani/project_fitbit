import sqlite3
import pandas as pd
import numpy as np

connection = sqlite3.connect('fitbit_database.db')
query = 'SELECT Id FROM "daily_activity";'
cursor = connection.cursor()
cursor.execute(query)
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

id_counts = df['Id'].value_counts().reset_index()
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
pd.set_option('display.float_format', '{:.0f}'.format)
print(user_class_df)
connection.close()
