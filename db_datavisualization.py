from db_datawrangling import mergingtables, merged_data
import sqlite3
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import linregress

def numeric_summary(df, ids = None, columns=None, start_date = None, end_date = None, start_time = None, end_time = None):
    if 'Id' in df.index.names:
        df = df.reset_index()
        
    df['timestamp'] = pd.to_datetime (df['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
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
    grouped_stats = df.groupby('Id')[columns].agg(['mean', 'std', 'min', Q1, 'median', Q3, 'max', IQR, 'count', 'sum'])
    return grouped_stats

def scatter_plot(df, xcol, ycol, ids= None, start_date = None, end_date = None, start_time = None, end_time = None):
    
    if 'Id' in df.index.names:
        df = df.reset_index()
        
    df['timestamp'] = pd.to_datetime (df['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    if start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] >= start_date]
    if end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%m/%d/%Y') + pd.Timedelta(days=1)
            df = df[df['timestamp'] < end_date]
    
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

def box_plot (df, ids = None, columns = None, start_date = None, end_date = None, start_time = None, end_time = None, ):
    
    if 'Id' in df.index.names:
        df = df.reset_index()
        
    df['timestamp'] = pd.to_datetime (df['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    if start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] >= start_date]
    if end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%m/%d/%Y') + pd.Timedelta(days=1)
            df = df[df['timestamp'] < end_date]
    
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

def timeseries_plot(df, col1, col2, ids = None, start_date=None, end_date=None, start_time=None, end_time=None, time_intervals = None):
    if 'Id' in df.index.names:
        df = df.reset_index()
    
    df['timestamp'] = pd.to_datetime (df['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    if start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%m/%d/%Y')
            df = df[df['timestamp'] >= start_date]
    if end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%m/%d/%Y') + pd.Timedelta(days=1)
            df = df[df['timestamp'] < end_date]
    
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
        ax2 = ax.twinx()

        ax.plot(df_uid['timestamp'], df_uid[col1], color = 'red', label=col1)
        ax.set_ylabel(col1, color = 'red')
        ax.tick_params(axis='y', labelcolor = 'red')

        ax2.plot(df_uid['timestamp'], df_uid[col2], color = 'blue', label=col2, marker = 'o')
        ax2.set_ylabel(col2, color = 'blue')
        ax2.tick_params(axis='y', labelcolor = 'blue')

        start_str = df_uid['timestamp'].dt.date.min().strftime('%Y-%m-%d') if not df_uid.empty else ''
        end_str = df_uid['timestamp'].dt.date.max().strftime('%Y-%m-%d') if not df_uid.empty else ''
        ax.set_title(f"{col1} and {col2} for Id {uid} on {start_str} and {end_str}")
        
        ax.set_xlabel("Time")
        ax.legend()
    
    plt.tight_layout()
    plt.show()


numeric_summary = numeric_summary (merged_data,  None, ['Value', 'Calories'], None, None, None, None)
print (numeric_summary)
scatter_plot(merged_data, 'Value', 'Calories', None, None, None, None, None)
box_plot(merged_data, 6962181067, 'Value', None, None, None, None)
timeseries_plot(merged_data, 'Value', 'Calories', 6962181067, '3/30/2016', '3/31/2016', None, None, '60min')