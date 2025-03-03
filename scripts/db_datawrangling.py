#Just to see if my branch works
import sqlite3
import pandas as pd
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

print("Missing values after handling:")
print(df.isnull().sum())
print(df)