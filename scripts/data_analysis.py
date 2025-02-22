import matplotlib.pyplot as plt
import pandas as pd

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