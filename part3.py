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
fig, (axis1Q3, axis2Q3) = plt.subplots(1, 2, sharey=True, figsize=(9, 5), gridspec_kw={'width_ratios': [6, 1]})
axis1Q3.hist(residualsSedentary, bins=binsMain, edgecolor='black', color = 'purple')
axis1Q3.set_xlim(-0.05, 0.01)
axis1Q3.set_xlabel('Residual main')
axis1Q3.set_ylabel('Frequency')
axis1Q3.set_title('Main range for histogram of residuals')
axis2Q3.hist(residualsSedentary, bins=binsOutlier, edgecolor='black', color = 'purple')
axis2Q3.set_xlim(0.9, 1.1)
axis2Q3.set_xlabel('Residual outlier')
axis2Q3.set_title('Outlier range')
axis1Q3.spines['right'].set_visible(False)
axis2Q3.spines['left'].set_visible(False)
axis1Q3.tick_params(right=False)
axis2Q3.tick_params(left=False)
plt.tight_layout()
plt.savefig('part3ResidualsHistogram.png')

plt.figure()
stats.probplot(residualsSedentary, dist="norm", plot=plt)
plt.title('Q-Q plot of residuals, Sedentary minutes against Sleep minutes)')
plt.savefig('part3ResidualsQQplot.png')


# Part 3.4: Average steps, calories burnt, and minutes of sleep per 4 hour time blocks
querySteps = 'SELECT * FROM "hourly_steps";'
cursor.execute(querySteps)
rows = cursor.fetchall()
steps = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
steps['Hour'] = pd.to_datetime(steps['ActivityHour']).dt.hour
steps['Block'] = pd.cut(steps['Hour'], bins=[0,4,8,12,16,20,24], labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'], right=False)
avgStepsBlock = steps.groupby('Block')['StepTotal'].mean().reset_index(name='AverageSteps')

queryCalories = 'SELECT * FROM "hourly_calories";'
cursor.execute(queryCalories)
rows = cursor.fetchall()
calories = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
calories['Hour'] = pd.to_datetime(calories['ActivityHour']).dt.hour
calories['Block'] = pd.cut(calories['Hour'], bins=[0,4,8,12,16,20,24], labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'], right=False)
avgCaloriesBlock = calories.groupby('Block')['Calories'].mean().reset_index(name='AverageCalories')

sleep['Hour'] = pd.to_datetime(sleep['date']).dt.hour
sleep['Block'] = pd.cut(sleep['Hour'], bins=[0,4,8,12,16,20,24], labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'], right=False)
avgSleepBlock = sleep.groupby('Block').size().reset_index(name='AverageSleepMinutes')
avgSleepBlock['AverageSleepMinutes'] = avgSleepBlock['AverageSleepMinutes'] / sleep['Id'].nunique()

fig, (axis1Q4, axis2Q4, axis3Q4) = plt.subplots(1, 3, figsize=(18, 5))
axis1Q4.bar(avgStepsBlock['Block'], avgStepsBlock['AverageSteps'], color='blue')
axis1Q4.set_title('Average steps per 4 hour block')
axis1Q4.set_xlabel('4 hour block')
axis1Q4.set_ylabel('Average steps')
axis2Q4.bar(avgCaloriesBlock['Block'], avgCaloriesBlock['AverageCalories'], color='red')
axis2Q4.set_title('Average calories burnt per 4 hour block')
axis2Q4.set_xlabel('4 hour block')
axis2Q4.set_ylabel('Average calories')
axis3Q4.bar(avgSleepBlock['Block'], avgSleepBlock['AverageSleepMinutes'], color='green')
axis3Q4.set_title('Average sleep minutes per 4 hour block')
axis3Q4.set_xlabel('4 hour block')
axis3Q4.set_ylabel('Average sleep minutes')
plt.tight_layout()
plt.savefig('part3Averages.png')


# Part 3.5: Heart rate and total intensity plot for given Id



# Part 3.6: Weather data analysis for Chicago



connection.close()
