import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Part 3.1: Compute sleep duration per logId for each individual
connection = sqlite3.connect('fitbit_database.db')
queryActivity = 'SELECT * FROM "daily_activity";'
cursor = connection.cursor()
cursor.execute(queryActivity)
rows = cursor.fetchall()
activity = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

ids = activity['Id'].value_counts().reset_index()
ids.columns = ['Id', 'Count']

def classifyUser(count):
    if count <= 10:
        return 'Light user'
    elif 11 <= count <= 15:
        return 'Moderate user'
    else:
        return 'Heavy user'

ids['Class'] = ids['Count'].apply(classifyUser)
userClass = ids[['Id', 'Class']]

querySleep =  'SELECT * FROM "minute_sleep";'
cursor.execute(querySleep)
rows = cursor.fetchall()
sleep = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
sleepDuration = sleep.groupby(['Id', 'logId']).size().reset_index(name = 'Sleepduration')

mergedClassSleep = pd.merge(userClass, sleepDuration, on = 'Id', how = 'left')

pd.set_option('display.float_format', '{:.0f}'.format)
print(mergedClassSleep)


# Part 3.2: Sleep duration compared to total active minutes with regression
sleep['date'] = pd.to_datetime(sleep['date'])
sleepPerDay = sleep.groupby(['Id', 'date']).size().reset_index(name='SleepMinutes')
activity['ActivityDate'] = pd.to_datetime(activity['ActivityDate'])
mergedActivitySleep = pd.merge(activity, sleepPerDay, left_on=['Id', 'ActivityDate'], right_on=['Id', 'date'], how='inner')
mergedActivitySleep['TotalActiveMinutes'] = mergedActivitySleep['VeryActiveMinutes'] + mergedActivitySleep['FairlyActiveMinutes'] + mergedActivitySleep['LightlyActiveMinutes']

x = mergedActivitySleep['TotalActiveMinutes']
y = mergedActivitySleep['SleepMinutes']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())


# Part 3.3: Sleep duration compared to sedentary activity with regression and verification for normality



# Part 3.4: Average steps, calories burned, and minutes of sleep per 4 hour time blocks



# Part 3.5: Heart rate and total intensity plot for given Id



# Part 3.6: Weather data analysis for Chicago



connection.close()
