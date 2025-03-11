import sqlite3
import pandas as pd
import numpy as np

connection=sqlite3.connect('fitbit_database.db')
query='SELECT * FROM "daily_activity";'
cursor = connection.cursor()
cursor.execute(query)
rows=cursor.fetchall()
df = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
print(df)
connection.close
