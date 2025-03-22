# Fitbit Data Visualization Dashboard

This Streamlit dashboard visualizes Fitbit activity data for a group of participants. It provides an interactive way to explore individual and collective trends in steps, calories, activity, sleep, and more.

## 📊 Features

### ✅ Home Dashboard
- **KPIs**: Total participants, average steps, calories burned, active minutes, and sedentary time.
- **Regression Analysis**: 
  - Linear regression between **Total Steps** and **Calories Burned**.
  - Summary of R², intercept, and slope with p-values.
  - A scatterplot of **Total Distance vs. Calories**.
- **Average Metrics by Time Block**: 
  - Bar charts showing average steps, calories, and sleep per 4-hour block throughout the day.
- **User Classification Pie Chart**:
  - Users categorized as Light, Moderate, or Heavy based on activity data.

### 👤 Individual Stats
- Filter by user ID and select a custom date range.
- Compare up to 4 daily metrics per user.
- Bar charts and statistical summaries for each metric.
- Optional regression analysis between any two selected metrics.
- Weight information and BMI health status when available.

### 🛌 Sleep Analysis
- ....

---

## 📂 Project Structure

```
project_fitbit/
│
├── part5_dashboard.py          # Main Streamlit dashboard
├── data_loader.py              # CSV data loading and preprocessing
├── db_datawrangling.py         # Data cleaning & transformation
├── db_datavisualization.py     # Custom visualizations (bar charts, pie charts, etc.)
├── data_analysis.py            # Statistical computations (regression, p-values, etc.)
├── part3.py                    # Script for database creation
├── fitbit_database.db          # SQLite database with Fitbit data
├── Fitbit_logo_2016.webp       # Sidebar logo
└── README.md                   # Project overview and documentation
```

---

## 🛠️ Technologies Used

- **Python** (Pandas, NumPy, Matplotlib, Statsmodels)
- **Streamlit** (for dashboard UI)
- **SQLite** (for data storage)
- **PIL** (for image rendering)
- **streamlit-option-menu** (for sidebar navigation)

---

## 🚀 Getting Started

### 1. Install Dependencies

Make sure you have the required Python packages:

```bash
pip install streamlit pandas matplotlib statsmodels Pillow streamlit-option-menu
```

### 2. Run the Dashboard

Navigate to the project directory and run:

```bash
streamlit run part5_dashboard.py
```

### 3. Navigate the App

Use the sidebar to switch between:
- **Home**
- **Individual Stats**
- **Sleep Analysis**

---

## 📌 Notes

- The data is loaded from `fitbit_database.db`, which must be present in the root directory.
- All data processing is modularized across separate files for maintainability.
- The dashboard layout is styled with custom HTML and CSS using Streamlit's markdown injection.

---

## ✍️ Authors

Alec Evers, Ahmed el Ghani and Maxim Morales Nassiboulina – *Fitbit Group 6*

---

