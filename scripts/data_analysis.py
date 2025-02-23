import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime



def get_unique_users(df):
    return df['Id'].nunique()

def compute_total_distance(df):
    return df.groupby('Id', as_index=False)['TotalDistance'].sum()

def plot_total_distance(df):
    user_distance = compute_total_distance(df)
    user_distance.set_index('Id')['TotalDistance'].plot(kind='bar')
    plt.xlabel('User Id')
    plt.ylabel('Total Distance')
    plt.title('Total Distance per unique user')
    plt.show()

def plot_calories_per_day(df, user_id, start_date=None, end_date=None):
    df['ActivityDate']= pd.to_datetime(df['ActivityDate'])
    user_data = df[df['Id']==user_id]
    if start_date and end_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        user_data = user_data[(user_data['ActivityDate']>= start_date) & (user_data['ActivityDate']<= end_date)]
    plt.figure(figsize=(10,5))
    sns.lineplot(x='ActivityDate', y='Calories', data=user_data)
    plt.xlabel('Date')
    plt.ylabel('Calories burned')
    plt.title(f'Calories burned by user {user_id}')
    plt.xticks(rotation=45)
    plt.show()