import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

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

xActivity = mergedActivitySleep['TotalActiveMinutes']
ySleep = mergedActivitySleep['SleepMinutes']
xActivity = sm.add_constant(xActivity)
modelActivity = sm.OLS(ySleep, xActivity).fit()
print(modelActivity.summary())


# Part 3.3: Sleep duration compared to sedentary activity with regression and verification for normality
xSedentary = mergedActivitySleep['SedentaryMinutes']
xSedentary = sm.add_constant(xSedentary)
modelSedentary = sm.OLS(ySleep, xSedentary).fit()
print(modelSedentary.summary())
residualsSedentary = modelSedentary.resid

binsMain = np.linspace(-0.05, 0.01, 30)
binsOutlier = np.linspace(0.95, 1.05, 5)
fig, (axis1, axis2) = plt.subplots(1, 2, sharey=True, figsize=(9, 5), gridspec_kw={'width_ratios': [6, 1]})
axis1.hist(residualsSedentary, bins=binsMain, edgecolor='black')
axis1.set_xlim(-0.05, 0.01)
axis1.set_xlabel('Residual main')
axis1.set_ylabel('Frequency')
axis1.set_title('Main range for histogram of residuals')
axis2.hist(residualsSedentary, bins=binsOutlier, edgecolor='black')
axis2.set_xlim(0.9, 1.1)
axis2.set_xlabel('Residual outlier')
axis2.set_title('Outlier range')
axis1.spines['right'].set_visible(False)
axis2.spines['left'].set_visible(False)
axis1.tick_params(right=False)
axis2.tick_params(left=False)
plt.tight_layout()
plt.savefig('part3ResidualsHistogramSedentary.png')

plt.figure()
stats.probplot(residualsSedentary, dist="norm", plot=plt)
plt.title('Q-Q plot of residuals, Sedentary minutes against Sleep minutes)')
plt.savefig('part3ResidualsQQplotSedentary.png')


# Part 3.4: Average steps, calories burned, and minutes of sleep per 4 hour time blocks



# Part 3.5: Heart rate and total intensity plot for given Id



# Part 3.6: Weather data analysis for Chicago



connection.close()
