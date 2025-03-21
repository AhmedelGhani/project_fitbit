from scripts.db_datawrangling import mergingtables, merged_data
from db_datavisualization import numeric_summary, scatter_plot, box_plot, timeseries_plot
from scripts.data_loader import load_data
from scripts.data_analysis import plot_total_distance, plot_calories_per_day, get_unique_users, compute_total_distance


df = load_data('/Users/ahmedelghani/project_fitbit/data/daily_acivity.csv')

print(f'Number of unique users: {get_unique_users(df)}')
plot_total_distance(df)
plot_calories_per_day(df, user_id=1503960366, start_date='2016-03-12', end_date='2016-04-20')
