#Just to see if my branch works
import sqlite3
import pandas as pd
import numpy as np


conn = sqlite3.connect('fitbit_database.db')
query = 'SELECT * FROM "weight_log";'
df = pd.read_sql_query(query, conn)
conn.close()

print("Missing values before handling:")
print(df.isnull().sum())


