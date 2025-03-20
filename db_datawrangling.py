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
    
    #Error message displayed if tables inputted to function don't exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    available_tables = [row[0] for row in cursor.fetchall()]
    if table1 not in available_tables or table2 not in available_tables:
        connection.close()
        print(f"Error: Cannot merge tables. One or both table names ('{table1}', '{table2}') are incorrect.")
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
    query = f"""SELECT t1.*, {', '.join(f"t2.{col}" for col in table2_cols)} FROM {table1} AS t1JOIN {table2} AS t2 ON t1.{join_column} = t2.{join_column}"""
    
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

merged_data = mergingtables('heart_rate', 'weight_log')
pd.set_option('display.float_format', '{:.0f}'.format)
print(merged_data)

#Add another column to compare
def numeric_summary(df, ids = None, columns=None, datetime_col = None, start_date = None, end_date = None, start_time = None, end_time = None):
    print("\n[DEBUG] -- Before any processing in numeric_summary --")
    print("df.columns:", df.columns)
    print("df.index.names:", df.index.names)
    print("First 5 rows of df:\n", df.head(), "\n")
    
    if 'Id' in df.index.names:
        df = df.reset_index()
        
    if datetime_col is None:
        raise ValueError("Please provide a valid datetime column name (datetime_col).")
    
    df['timestamp'] = pd.to_datetime (df[datetime_col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    if start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] >= start_date]
    if end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] <= end_date]
    
    if start_time is not None:
        start_time = pd.to_datetime(start_time, format = '%I:%M:%S %p').time()
        df = df[df['timestamp'].dt.time >= start_time]
    if end_time is not None:
        end_time = pd.to_datetime(end_time, format = '%I:%M:%S %p').time()
        df = df[df['timestamp'].dt.time <= end_time]
    
    if ids is not None:
        if not isinstance (ids,list):
            ids = [ids]
        df = df[df['Id'].isin(ids)]

    if columns is None:
        exclude = ['Id', 'LogId', 'IsManualReport', 'ActivityHour']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in numeric_cols if col not in exclude]
    else:
        if not isinstance(columns, list):
            columns = [columns]
        columns = [col for col in columns if col in df.columns]
    
    def IQR(series):
        return series.quantile(0.75) - series.quantile(0.25)

    def Q1(series):
        return series.quantile(0.25)

    def Q3(series):
        return series.quantile(0.75)

    print("Rows after time filter:", len(df))
    grouped_stats = df.groupby('Id')[columns].agg(['mean', 'std', 'min', Q1, 'median', Q3, 'max', IQR, 'count'])
    return grouped_stats

def scatter_plot(df, xcol, ycol, ids= None, datetime_col = None, start_date = None, end_date = None, start_time = None, end_time = None):
    
    if 'Id' in df.index.names:
        df = df.reset_index()
        
    if datetime_col is None:
        raise ValueError("Please provide a valid datetime column name (datetime_col).")
    
    df['timestamp'] = pd.to_datetime (df[datetime_col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    if start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] >= start_date]
    if end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] <= end_date]
    
    if start_time is not None:
        start_time = pd.to_datetime(start_time, format = '%I:%M:%S %p').time()
        df = df[df['timestamp'].dt.time >= start_time]
    if end_time is not None:
        end_time = pd.to_datetime(end_time, format = '%I:%M:%S %p').time()
        df = df[df['timestamp'].dt.time <= end_time]
    
    if ids is not None:
        if not isinstance (ids,list):
            ids = [ids]
        df = df[df['Id'].isin(ids)]
    
    aggregated = df.groupby('Id')[[xcol, ycol]].median().reset_index()  
    print("Number of individuals after filtering:", len(aggregated))

    regression = linregress(aggregated[xcol], aggregated[ycol])
    slope = regression.slope
    intercept = regression.intercept
    r_value = regression.rvalue
    r_squared = r_value ** 2

    plt.figure(figsize=(8, 6))
    plt.scatter(aggregated[xcol], aggregated[ycol], alpha=0.6, edgecolor='k')
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"Scatter Plot of {ycol} vs. {xcol}")

    x_values = np.linspace(aggregated[xcol].min(), aggregated[xcol].max(), 100)
    y_values = intercept + slope * x_values
    plt.plot(x_values, y_values, color='red', label=f"Trendline (R² = {r_squared:.2f})")

    plt.legend()
    plt.grid(True)
    plt.show()

def box_plot (df, ids = None, columns = None, datetime_col = None, start_date = None, end_date = None, start_time = None, end_time = None, ):
    
    if 'Id' in df.index.names:
        df = df.reset_index()
        
    if datetime_col is None:
        raise ValueError("Please provide a valid datetime column name (datetime_col).")
    
    df['timestamp'] = pd.to_datetime (df[datetime_col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    if start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] >= start_date]
    if end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] <= end_date]
    
    if start_time is not None:
        start_time = pd.to_datetime(start_time, format = '%I:%M:%S %p').time()
        df = df[df['timestamp'].dt.time >= start_time]
    if end_time is not None:
        end_time = pd.to_datetime(end_time, format = '%I:%M:%S %p').time()
        df = df[df['timestamp'].dt.time <= end_time]
    
    if ids is not None:
        if not isinstance (ids,list):
            ids = [ids]
        df = df[df['Id'].isin(ids)]
    
    if columns is None:
        exclude = ['Id', 'LogId', 'IsManualReport', 'ActivityHour']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in numeric_cols if col not in exclude]
    else:
        if not isinstance(columns, list):
            columns = [columns]
        columns = [col for col in columns if col in df.columns]

    num_cols = len(columns)
    fig, axes = plt.subplots(nrows = 1, ncols = num_cols, figsize = (6 * num_cols, 6), squeeze = False)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.boxplot(x = 'Id', y = col, data = df, ax = axes[i])
        axes[i].set_title(f"Boxplot of {col} by Id")
        axes[i].set_xlabel("Id")
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()

def timeseries_plot(df, col1, col2, ids = None, datetime_col = None, start_date=None, end_date=None, start_time=None, end_time=None, time_intervals = None):
    if 'Id' in df.index.names:
        df = df.reset_index()
        
    if datetime_col is None:
        raise ValueError("Please provide a valid datetime column name (datetime_col).")
    
    df['timestamp'] = pd.to_datetime (df[datetime_col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    
    if start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] >= start_date]
    if end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] <= end_date]
    
    if start_time is not None:
        start_time = pd.to_datetime(start_time, format = '%I:%M:%S %p').time()
        df = df[df['timestamp'].dt.time >= start_time]
    if end_time is not None:
        end_time = pd.to_datetime(end_time, format = '%I:%M:%S %p').time()
        df = df[df['timestamp'].dt.time <= end_time]
    
    if ids is not None:
        if not isinstance (ids,list):
            ids = [ids]
        df = df[df['Id'].isin(ids)]
    
    if time_intervals is not None:
        df_copy = df.copy()
        df_copy.set_index('timestamp', inplace=True)       
        df_copy = df_copy.groupby('Id').resample(time_intervals)[[col1, col2]].mean()
        df_copy.reset_index(inplace=True)
        df = df_copy

    unique_ids = df['Id'].unique()
    n_ids = len(unique_ids)
    fig, axes = plt.subplots(n_ids, 1, figsize=(12, 5 * n_ids), sharex=True)

    if n_ids == 1:
        axes = [axes]
    
    for ax, uid in zip(axes, unique_ids):
        df_uid = df[df['Id'] == uid].sort_values(by='timestamp')
        ax.plot(df_uid['timestamp'], df_uid[col1], label=col1)
        ax.plot(df_uid['timestamp'], df_uid[col2], label=col2)
        ax.set_title(f"Time Series for Id {uid}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

#----------------------------------------------------------
#merged_data['Date'] = pd.to_datetime(merged_data['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Define your specific ID (change this to the ID you want)
#specific_id = 8792009665
# Filter the DataFrame for rows with that ID and where the Date is on or before 4/1/2016
#filtered_data = merged_data[(merged_data['Id'] == specific_id) &
#                            (merged_data['Date'] <= pd.to_datetime('4/1/2016', format='%m/%d/%Y'))]

# Create a copy of just the TotalSteps column (adjust column name if needed)
#table_copy = filtered_data[['Value', 'Date']].copy()

# Print the resulting table
#print(table_copy.to_string())
#----------------------------------------------------------


#merged_data['ActiveMinutes'] = merged_data['VeryActiveMinutes'] + merged_data['FairlyActiveMinutes'] + merged_data['LightlyActiveMinutes']
print(merged_data.columns)
numeric_summary = numeric_summary (merged_data, None, ['Value', 'BMI'], 'Time', None, None, None, None)
print (numeric_summary)
scatter_plot(merged_data, 'Value', 'WeightKg', None, 'Time', None, None, None, None)
box_plot(merged_data, 8877689391, 'Value', 'Time', None, None, None, None)
timeseries_plot(merged_data, 'Value', 'WeightKg', 8877689391, 'Time', None, None, None, None, '2D')