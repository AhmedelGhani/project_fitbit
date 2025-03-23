from scripts.db_datawrangling import mergingtables, merged_data
from scripts.db_datavisualization import numeric_summary, scatter_plot, box_plot, timeseries_plot
from scripts.data_loader import load_data
from scripts.data_analysis import plot_total_distance, plot_calories_per_day, get_unique_users, compute_total_distance, plot_workout_frequency, plot_regression_per_user, plot_distance_vs_calories, plot_distance_over_time


df = load_data('/Users/ahmedelghani/project_fitbit/data/daily_acivity.csv')

print(f'Number of unique users: {get_unique_users(df)}')
plot_total_distance(df)
plot_calories_per_day(df, user_id=1503960366, start_date='2016-03-12', end_date='2016-04-20')
plot_workout_frequency(df)
plot_regression_per_user(df, user_id=1624580081)
plot_distance_over_time(df)
plot_distance_vs_calories(df)
