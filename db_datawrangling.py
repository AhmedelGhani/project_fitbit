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
def mergingtables(table1, table2, join_column='Id', chunksize = 100000):
    connection = sqlite3.connect('fitbit_database.db')
    cursor = connection.cursor()

    #Error message displayed if tables inputted to function don't exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    available_tables = [row[0] for row in cursor.fetchall()]
    if table1 not in available_tables or table2 not in available_tables:
        connection.close()
        print(f"Error: Cannot merge tables. One or both table names ('{table1}', '{table2}') are incorrect.")
        connection.close()
        return None
    
    #Estimation of rows from each table
    table1_count = pd.read_sql_query(f"SELECT COUNT(*) AS count FROM {table1}", connection).iloc[0, 0]
    table2_count = pd.read_sql_query(f"SELECT COUNT(*) AS count FROM {table2}", connection).iloc[0, 0]
    print(f"Table '{table1}' has {table1_count} rows.")
    print(f"Table '{table2}' has {table2_count} rows.\n")
    
    #Joint table rows calculated and processed into chunks
    join_count_query = f"""SELECT COUNT(*) AS count FROM {table1} AS t1 JOIN {table2} AS t2 ON t1.{join_column} = t2.{join_column} """
    join_count = pd.read_sql_query(join_count_query, connection).iloc[0, 0]
    estimated_chunks = join_count // chunksize + (1 if join_count % chunksize else 0)
    print(f"Join result has approximately {join_count} rows, which will be split into ~{estimated_chunks} chunks (chunksize={chunksize}).\n")
    
    #Makes sure that joined column (Id) does not get duplicated
    table2_cols = pd.read_sql_query(f"PRAGMA table_info({table2})", connection)['name'].tolist()
    table2_cols = [col for col in table2_cols if col != join_column]
    query = f"""SELECT t1.*, {', '.join(f"t2.{col}" for col in table2_cols)} FROM {table1} AS t1 JOIN {table2} AS t2 ON t1.{join_column} = t2.{join_column}"""
    
    #Chunks processed and displayed with time spent
    chunks = []
    start_time = time.time()
    for i, chunk in enumerate(pd.read_sql_query(query, connection, chunksize=chunksize)):
        elapsed = time.time() - start_time
        print(f"Chunk {i} loaded with shape {chunk.shape} in {elapsed:.2f} seconds")
        start_time = time.time()
        chunks.append(chunk)
    
    merged_df = pd.concat(chunks, ignore_index=True)
    connection.close()

    return merged_df

pd.set_option('display.float_format', '{:.0f}'.format)
