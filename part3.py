import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.dates as mdates


# Part 3.1: Compute total sleep duration for each individual log Id
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
totalSleepPerId = sleepDuration.groupby('Id')['Sleepduration'].sum().reset_index(name='TotalSleepDuration')

mergedTotalSleep = pd.merge(userClass, totalSleepPerId, on='Id', how='left')
mergedTotalSleep['TotalSleepDuration'] = mergedTotalSleep['TotalSleepDuration'].apply(lambda x: f"{int(x)} min" if pd.notna(x) else 'Not available')

pd.set_option('display.float_format', '{:.0f}'.format)
with open("part3Q1TotalSleepPerId.txt", "w") as f:
    f.write("Part 3.1: Total sleep Duration per User\n\n")
    f.write(mergedTotalSleep.to_string(index=False))


# Part 3.2: Sleep duration compared to total active minutes with regression and a visualisation
sleep['date'] = pd.to_datetime(sleep['date']).dt.date
sleepPerDay = sleep.groupby(['Id', 'date']).size().reset_index(name='SleepMinutes')
activity['ActivityDate'] = pd.to_datetime(activity['ActivityDate']).dt.date

mergedActivitySleep = pd.merge(activity, sleepPerDay, left_on=['Id', 'ActivityDate'], right_on=['Id', 'date'], how='left')
mergedActivitySleep['TotalActiveMinutes'] = mergedActivitySleep['VeryActiveMinutes'] + mergedActivitySleep['FairlyActiveMinutes'] + mergedActivitySleep['LightlyActiveMinutes']
filtered = mergedActivitySleep.dropna(subset=['SleepMinutes', 'TotalActiveMinutes'])

xActivity = filtered['TotalActiveMinutes']
ySleep = filtered['SleepMinutes']
xActivity = sm.add_constant(xActivity)
modelActivity = sm.OLS(ySleep, xActivity).fit()

plt.figure(figsize=(8, 5))
plt.scatter(filtered['TotalActiveMinutes'], filtered['SleepMinutes'], alpha=0.7, label='Data points')
coefActivity = np.polyfit(filtered['TotalActiveMinutes'], filtered['SleepMinutes'], 1)
plt.plot(filtered['TotalActiveMinutes'],
         coefActivity[0] * filtered['TotalActiveMinutes'] + coefActivity[1],
         color='black', linestyle='--', label='Trend Line')
plt.title('Sleep minutes against total active minutes (per day)')
plt.xlabel('Total active minutes')
plt.ylabel('Sleep minutes')
plt.legend()
plt.tight_layout()
plt.savefig('part3Q2Graph.png')

with open("part3Q2OLS.txt", "w") as f:
    f.write("Part 3.2: Sleep minutes vs Total Active Minutes\n\n")
    f.write(str(modelActivity.summary()))


# Part 3.3: Sleep duration compared to sedentary activity with regression and verification for normality on the residuals
xSedentary = filtered['SedentaryMinutes']
xSedentary = sm.add_constant(xSedentary)
ySleep = filtered['SleepMinutes']
modelSedentary = sm.OLS(ySleep, xSedentary).fit()
residualsSedentary = modelSedentary.resid

plt.figure(figsize=(8, 5))
plt.hist(residualsSedentary, bins=30, edgecolor='black', color='purple')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of residuals, Sedentary minutes against sleep minutes)')
plt.tight_layout()
plt.savefig('part3Q3Histogram.png')

plt.figure()
stats.probplot(residualsSedentary, dist="norm", plot=plt)
plt.title('Q-Q plot of residuals, Sedentary minutes against Sleep minutes')
plt.savefig('part3Q3QQplot.png')

plt.figure(figsize=(8, 5))
plt.scatter(filtered['SedentaryMinutes'], filtered['SleepMinutes'], alpha=0.7, label='Data points')
coefSedentary = np.polyfit(filtered['SedentaryMinutes'], filtered['SleepMinutes'], 1)
plt.plot(filtered['SedentaryMinutes'],
         coefSedentary[0] * filtered['SedentaryMinutes'] + coefSedentary[1],
         color='black', linestyle='--', label='Trend Line')
plt.title('Sleep minutes against sedentary minutes (per day)')
plt.xlabel('Sedentary minutes')
plt.ylabel('Sleep minutes')
plt.legend()
plt.tight_layout()
plt.savefig('part3Q3Scatterplot.png')

with open("part3Q3OLS.txt", "w") as f:
    f.write("Part 3.3: Sleep minutes vs Total Active Minutes\n\n")
    f.write(str(modelSedentary.summary()))


# Part 3.4: Average steps, calories burnt, and minutes of sleep per 4 hour time blocks
querySteps = 'SELECT * FROM "hourly_steps";'
cursor.execute(querySteps)
rows = cursor.fetchall() 
steps = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
steps['Hour'] = pd.to_datetime(steps['ActivityHour']).dt.hour
steps['Block'] = pd.cut(steps['Hour'], bins=[0,4,8,12,16,20,24],
                        labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'], right=False)
avgStepsBlock = steps.groupby('Block')['StepTotal'].mean().reset_index(name='AverageSteps')

queryCalories = 'SELECT * FROM "hourly_calories";'
cursor.execute(queryCalories)
rows = cursor.fetchall()
calories = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
calories['Hour'] = pd.to_datetime(calories['ActivityHour']).dt.hour
calories['Block'] = pd.cut(calories['Hour'], bins=[0,4,8,12,16,20,24],
                           labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'], right=False)
avgCaloriesBlock = calories.groupby('Block')['Calories'].mean().reset_index(name='AverageCalories')

querySleepAgain = 'SELECT * FROM "minute_sleep";'
cursor.execute(querySleepAgain)
rows = cursor.fetchall()
sleepP3Q4 = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
sleepP3Q4['Hour'] = pd.to_datetime(sleepP3Q4['date']).dt.hour
sleepP3Q4['Block'] = pd.cut(sleepP3Q4['Hour'], bins=[0,4,8,12,16,20,24],
                            labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'], right=False)
avgSleepBlock = sleepP3Q4.groupby('Block').size().reset_index(name='AverageSleepMinutes')
avgSleepBlock['AverageSleepMinutes'] = avgSleepBlock['AverageSleepMinutes'] / sleepP3Q4['Id'].nunique()

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
plt.savefig('part3Q4Averages.png')


# Part 3.5: Heart rate and total intensity plot for a random Id, graphed for two consecutive days
queryHeart = 'SELECT * FROM "heart_rate";'
cursor.execute(queryHeart)
rows = cursor.fetchall() 
heart = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
heart['Datetime'] = pd.to_datetime(heart['Time'])

queryIntensity = 'SELECT * FROM "hourly_intensity";'
cursor.execute(queryIntensity)
rows = cursor.fetchall()
intensity = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
intensity['Datetime'] = pd.to_datetime(intensity['ActivityHour'])

uniqueIds = heart['Id'].unique()
np.random.shuffle(uniqueIds)
success = False

for selectedId in uniqueIds:
    heartId = heart[heart['Id'] == selectedId].copy()
    heartId['Date'] = heartId['Datetime'].dt.date
    intensityId = intensity[intensity['Id'] == selectedId].copy()
    intensityId['Date'] = intensityId['Datetime'].dt.date
    validDates = []

    for date in heartId['Date'].unique():
        nextDay = date + pd.Timedelta(days=1)
        heartDay1 = heartId[heartId['Date'] == date]
        heartDay2 = heartId[heartId['Date'] == nextDay]
        intensityDay1 = intensityId[intensityId['Date'] == date]
        intensityDay2 = intensityId[intensityId['Date'] == nextDay]

        if (heartDay1['Datetime'].dt.hour.nunique() >= 24 and
            heartDay2['Datetime'].dt.hour.nunique() >= 24 and
            intensityDay1['Datetime'].dt.hour.nunique() >= 24 and
            intensityDay2['Datetime'].dt.hour.nunique() >= 24):
            validDates.append(date)

    if len(validDates) > 0:
        date = np.random.choice(validDates)
        nextDate = date + pd.Timedelta(days=1)
        heart2Days = heartId[(heartId['Date'] == date) | (heartId['Date'] == nextDate)].copy()
        heart2Days60MinInterval = heart2Days.set_index('Datetime')['Value'].resample('60min').mean().reset_index()
        intensity2Days = intensityId[(intensityId['Date'] == date) | (intensityId['Date'] == nextDate)].copy()

        fig, axis1Q5 = plt.subplots(figsize=(14, 5))
        axis1Q5.plot(heart2Days60MinInterval['Datetime'], heart2Days60MinInterval['Value'], label='Heart rate', color='red')
        axis1Q5.set_xlabel('Time')
        axis1Q5.set_ylabel('Heart Rate', color='red')
        axis1Q5.tick_params(axis='y', labelcolor='red')
        dateFormat = mdates.DateFormatter('%H:%M (%m-%d)')
        axis1Q5.xaxis.set_major_formatter(dateFormat)
        plt.xticks(rotation=45)
        axis2Q5 = axis1Q5.twinx()
        axis2Q5.plot(intensity2Days['Datetime'], intensity2Days['TotalIntensity'], label='Total intensity', color='blue', marker='o')
        axis2Q5.set_ylabel('Total intensity', color='blue')
        axis2Q5.tick_params(axis='y', labelcolor='blue')
        plt.title(f'Heart rate and total intensity for Id {selectedId} on {date} and {nextDate}')
        plt.tight_layout()
        plt.savefig('part3Q5Graph.png')
        success = True
        break

if not success:
    print("No valid Id found with 2 full consecutive days of data")


# Part 3.6: Weather data analysis for Chicago using Total Active Minutes per day for 3 random users
weather = pd.read_csv("chicago 2016-03-11 to 2016-04-09.csv")
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather.rename(columns={'datetime': 'ActivityDate'}, inplace=True)
weather['ActivityDate'] = pd.to_datetime(weather['ActivityDate'])

queryIntensity = 'SELECT * FROM "hourly_intensity";'
cursor.execute(queryIntensity)
rows = cursor.fetchall()
intensity = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
intensity['ActivityHour'] = pd.to_datetime(intensity['ActivityHour'])
intensity['Datetime'] = intensity['ActivityHour']
intensity['ActivityDate'] = intensity['ActivityHour'].dt.floor('D')

uniqueIds = intensity['Id'].unique()
randomIds = np.random.choice(uniqueIds, size=3, replace=False)
colors = ['blue', 'green', 'red']

fig, axisQ6 = plt.subplots(2, 3, figsize=(18, 10))

with open("part3Q6OLS.txt", "w") as f:
    for idx, user_id in enumerate(randomIds):
        userData = intensity[intensity['Id'] == user_id].copy()
        totalIntensity = userData.groupby('ActivityDate')['TotalIntensity'].sum().reset_index(name='TotalIntensity')
        merged = pd.merge(totalIntensity, weather, on='ActivityDate', how='inner')

        xTemperature = sm.add_constant(merged['temp'])
        xPrecipitation = sm.add_constant(merged['precip'])
        y = merged['TotalIntensity']

        modelTemperature = sm.OLS(y, xTemperature).fit()
        modelPrecipitation = sm.OLS(y, xPrecipitation).fit()

        f.write("Part 3.6: Weather against total intensity\n\n")
        f.write(f"OLS Regression for ID {user_id}\n\n")
        f.write("Temperature against total intensity:\n\n")
        f.write(str(modelTemperature.summary()))
        f.write("\n\nPrecipitation against total intensity:\n\n")
        f.write(str(modelPrecipitation.summary()))
        f.write("\n\n\n\n")

        color = colors[idx]
        axisQ6[0, idx].scatter(merged['temp'], merged['TotalIntensity'], alpha=0.7, color=color)
        coefTemperature = np.polyfit(merged['temp'], merged['TotalIntensity'], 1)
        axisQ6[0, idx].plot(merged['temp'], coefTemperature[0]*merged['temp'] + coefTemperature[1], color='black', linestyle='--', label='Trend Line')
        axisQ6[0, idx].set_title(f"Temp against total intensity\nID {user_id}")
        axisQ6[0, idx].set_xlabel("Avg daily temperature (°F)")
        axisQ6[0, idx].set_ylabel("Total Intensity")
        axisQ6[0, idx].legend()

        axisQ6[1, idx].scatter(merged['precip'], merged['TotalIntensity'], alpha=0.7, color=color)
        coefPrecipitation = np.polyfit(merged['precip'], merged['TotalIntensity'], 1)
        axisQ6[1, idx].plot(merged['precip'], coefPrecipitation[0]*merged['precip'] + coefPrecipitation[1], color='black', linestyle='--', label='Trend Line')
        axisQ6[1, idx].set_title(f"Precipitaion against total intensity\nID {user_id}")
        axisQ6[1, idx].set_xlabel("Daily precipitation (inches)")
        axisQ6[1, idx].set_ylabel("Total Intensity")
        axisQ6[1, idx].legend()

plt.tight_layout()
plt.savefig("part3Q6Scatterplots.png")

fig_summary, (axisQ6Temperature, axisQ6Precipitation) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
date_format = mdates.DateFormatter('%m-%d')
locator = mdates.DayLocator()
axisQ6Temperature.plot(weather['ActivityDate'], weather['temp'], color='orange', marker='o', linestyle='-')
axisQ6Temperature.set_title("Average daily temperature in Chicago")
axisQ6Temperature.set_ylabel("Temperature (°F)")
axisQ6Temperature.set_xlabel("Date")
axisQ6Temperature.xaxis.set_major_formatter(date_format)
axisQ6Temperature.xaxis.set_major_locator(locator)
axisQ6Temperature.tick_params(labelbottom=True) 
axisQ6Temperature.grid(True)
plt.setp(axisQ6Temperature.get_xticklabels(), rotation=45, ha='right')
axisQ6Precipitation.bar(weather['ActivityDate'], weather['precip'], width=0.8, color='skyblue', edgecolor='black')
axisQ6Precipitation.set_title("Daily precipitation in Chicago")
axisQ6Precipitation.set_xlabel("Date")
axisQ6Precipitation.set_ylabel("Precipitation (inches)")
axisQ6Precipitation.xaxis.set_major_formatter(date_format)
axisQ6Precipitation.xaxis.set_major_locator(locator)
axisQ6Precipitation.grid(True)
plt.setp(axisQ6Precipitation.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("part3Q6WeatherSummary.png")


connection.close()
