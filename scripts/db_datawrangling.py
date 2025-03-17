import sqlite3
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import linregress


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

merged_data = mergingtables('minute_sleep', 'daily_activity')
pd.set_option('display.float_format', '{:.0f}'.format)
print(merged_data)

#Add another column to compare
def numeric_summary(df, ids = None, columns=None, start_date = None, 
                    end_date = None, start_time = None, end_time = None):
    df.rename(columns={'date': 'Date'}, inplace=True)
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

def scatter_plot(df, xcol, ycol, ids= None, start_date = None, end_date = None, start_time = None, end_time = None):
    df.rename(columns={'date': 'Date'}, inplace=True)

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

def box_plot (df, ids = None, columns = None, start_date = None, end_date = None, start_time = None, end_time = None, ):
    df.rename(columns={'date': 'Date'}, inplace=True)
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

def timeseries_plot(df, col1, col2, ids = None, start_date=None, end_date=None, start_time=None, end_time=None, time_intervals = None):
    df.rename(columns={'date': 'Date'}, inplace=True)
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
    
    if ids is not None:
        if not isinstance (ids,list):
            ids = [ids]
        df = df[df['Id'].isin(ids)]
    
    if time_intervals is not None:
        df_copy = df.copy()
        df_copy.set_index('Date', inplace=True)       
        df_copy = df_copy.groupby('Id').resample(time_intervals)[[col1, col2]].mean()
        df_copy.reset_index(inplace=True)
        df = df_copy

    unique_ids = df['Id'].unique()
    n_ids = len(unique_ids)
    fig, axes = plt.subplots(n_ids, 1, figsize=(12, 5 * n_ids), sharex=True)

    if n_ids == 1:
        axes = [axes]
    
    for ax, uid in zip(axes, unique_ids):
        df_uid = df[df['Id'] == uid].sort_values(by='Date')
        df_uid.interpolate(inplace=True)
        ax.plot(df_uid['Date'], df_uid[col1], label=col1)
        ax.plot(df_uid['Date'], df_uid[col2], label=col2)
        ax.set_title(f"Time Series for Id {uid}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
    
    plt.tight_layout()
    plt.show()




merged_data['ActiveMinutes'] = merged_data['VeryActiveMinutes'] + merged_data['FairlyActiveMinutes'] + merged_data['LightlyActiveMinutes']
print(merged_data.columns)
numeric_summary = numeric_summary (merged_data, None, ['ActiveMinutes', 'SedentaryMinutes'], None, None, None, None)
print (numeric_summary)
scatter_plot(merged_data, 'SedentaryMinutes', 'ActiveMinutes', None, None, None, None, None)
box_plot(merged_data, 8792009665, 'TotalSteps', None, '4/1/2016', None, None)
timeseries_plot(merged_data, 'FairlyActiveMinutes', 'TotalSteps', 8792009665, None, '4/1/2016', None, None, 'h')