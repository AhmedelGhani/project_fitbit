import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf


def get_unique_users(df):
    return df['Id'].nunique()

def compute_total_distance(df):
    return df.groupby('Id', as_index=False)['TotalDistance'].sum()

def plot_total_distance(df):
    user_distance = compute_total_distance(df)
    plt.figure(figsize=(10, 5))
    user_distance.set_index('Id')['TotalDistance'].plot(kind='bar')
    plt.xlabel('User Id')
    plt.ylabel('Total Distance')
    plt.title('Total Distance per unique user')
    plt.subplots_adjust(bottom=0.3)
    plt.show()

def plot_calories_per_day(df, user_id, start_date=None, end_date=None):
    df['ActivityDate']= pd.to_datetime(df['ActivityDate'])
    user_data = df[df['Id']==user_id]
    if start_date and end_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        user_data = user_data[(user_data['ActivityDate']>= start_date) & (user_data['ActivityDate']<= end_date)]
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='ActivityDate', y='Calories', data=user_data, marker='o')
    plt.xlabel('Date')
    plt.ylabel('Calories burned')
    plt.title(f'Calories burned by user {user_id}')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def plot_workout_frequency(df):
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    df['DayOfWeek'] = df['ActivityDate'].dt.day_name()
    workout_frequency = df['DayOfWeek'].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=workout_frequency.index, y=workout_frequency.values, palette='crest')
    plt.xlabel('Day of the Week')
    plt.ylabel('Workout Frequency')
    plt.title('Workout Frequency Per Day of the Week')
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def run_regression(df):
    df["Id"] = df["Id"].astype('category')
    model = smf.ols('Calories ~ TotalSteps + C(Id)', data=df).fit()
    return model.summary()

def plot_regression_per_user(df, user_id):
    user_data = df[df['Id'] == user_id]
    plot = sns.lmplot(x='TotalSteps', y='Calories', data=user_data, height=5, aspect=2)

    model = smf.ols('Calories ~ TotalSteps', data=user_data).fit()
    intercept = model.params['Intercept']
    step_coef = model.params['TotalSteps']

    equation = f'Calories = {intercept:.2f} + ({step_coef:.4f} × TotalSteps)'

    plot.ax.text(0.05, 0.9, equation, transform=plot.ax.transAxes, fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.5), color='red')
    plt.xlabel('Total Steps')
    plt.ylabel('Calories Burned')
    plt.title(f'Calories vs Total Steps for User {user_id}')
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.show()

#Additional analysis:#
def plot_distance_over_time(df):
    #Plots total distance covered over time to observe seasonal trends.#
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    daily_distance = df.groupby('ActivityDate')['TotalDistance'].sum()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=daily_distance.index, y=daily_distance.values, marker='o')
    plt.xlabel('Date')
    plt.ylabel('Total Distance Covered')
    plt.title('Total Distance Covered Over Time')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.show()

def plot_distance_vs_calories(df):
    #Creates a scatter plot showing the relationship between Total Distance and Calories Burned.#
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='TotalDistance', y='Calories', data=df, alpha=0.7)
    plt.xlabel('Total Distance')
    plt.ylabel('Calories Burned')
    plt.title('Total Distance vs Calories Burned')
    plt.show()