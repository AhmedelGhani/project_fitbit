import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fitbit Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image("Fitbit_logo_2016.webp", use_container_width=True)
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Individual Stats", "Sleep Analysis"],
        icons=["house", "person", "bar-chart"],
        menu_icon="cast",
        default_index=0
    )

connection = sqlite3.connect('fitbit_database.db')
activity = pd.read_sql_query("SELECT * FROM daily_activity", connection)
connection.close()
activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"])

if selected == "Home":
    st.title("Fitbit Data Dashboard")
    st.subheader("Overview of Research Statistics")
    st.markdown("This dashboard presents key statistics based on Fitbit activity, sleep, and other daily metrics.")

    st.markdown("Filter by Date")
    min_date, max_date = activity["ActivityDate"].min(), activity["ActivityDate"].max()
    selected_dates = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)

    if isinstance(selected_dates, list) and len(selected_dates) == 2:
        activity_filtered = activity[(activity["ActivityDate"] >= pd.to_datetime(selected_dates[0])) & (activity["ActivityDate"] <= pd.to_datetime(selected_dates[1]))]
    else:
        activity_filtered = activity

    st.markdown("Numerical Summary")
    avg_stats = {
        "Average Total Active Minutes": activity_filtered["VeryActiveMinutes"].add(activity_filtered["FairlyActiveMinutes"]).add(activity_filtered["LightlyActiveMinutes"]).mean(),
        "Average Sedentary Minutes": activity_filtered["SedentaryMinutes"].mean(),
        "Average Calories Burnt": activity_filtered["Calories"].mean(),
        "Average Steps": activity_filtered["TotalSteps"].mean()
    }
    summary_df = pd.DataFrame(avg_stats, index=["Average Value"]).T
    st.dataframe(summary_df.style.format("{:.2f}"))

    st.markdown("Graphical Summary")
    st.markdown("Below is a graphical summary showing average steps, calories burnt, and sleep minutes across different 4-hour blocks.")
    image = Image.open("part3Q4Averages.png")
    st.image(image, caption="Average per 4-hour block", use_container_width=True)

if selected == "Individual Stats":
    import matplotlib.pyplot as plt

    st.title("Individual Fitbit Statistics")
    st.sidebar.header("Select Individual ID")
    unique_ids = activity["Id"].unique()
    selected_id = st.sidebar.selectbox("Choose an ID to view individual statistics", unique_ids)

    min_date, max_date = activity["ActivityDate"].min(), activity["ActivityDate"].max()
    selected_dates = st.sidebar.date_input(
        "Filter by date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) == 2:
            start_date, end_date = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
        else:
            start_date = end_date = pd.to_datetime(selected_dates[0])
    else:
        start_date = end_date = pd.to_datetime(selected_dates)

    individual_data = activity[activity["Id"] == selected_id].copy()
    individual_data = individual_data[
        (individual_data["ActivityDate"] >= start_date) &
        (individual_data["ActivityDate"] <= end_date)
    ]

    st.sidebar.header("Select up to 4 Metrics")
    metrics_map = {
        "Total Steps": "TotalSteps",
        "Total Distance": "TotalDistance",
        "Total Active Minutes": "TotalActiveMinutes",  
        "Total Sedentary Minutes": "SedentaryMinutes",
        "Total Calories": "Calories",
        "Total Minutes of Sleep": "TotalSleepMinutes",  
        "Total Intensity": "TotalIntensity"  
    }
    display_map = {
        "Total Steps": "Steps",
        "Total Distance": "Distance",
        "Total Active Minutes": "Active Minutes",
        "Total Sedentary Minutes": "Sedentary Minutes",
        "Total Calories": "Calories",
        "Total Minutes of Sleep": "Minutes of Sleep",
        "Total Intensity": "Intensity"
    }

    selected_metrics = st.sidebar.multiselect(
        "Choose metrics for histograms",
        options=list(metrics_map.keys()),
        default=["Total Steps", "Total Calories"]
    )

    if len(selected_metrics) > 4:
        st.sidebar.warning("Please select at most 4 metrics.")
        selected_metrics = selected_metrics[:4]

    individual_data["TotalActiveMinutes"] = (
        individual_data["VeryActiveMinutes"] +
        individual_data["FairlyActiveMinutes"] +
        individual_data["LightlyActiveMinutes"]
    )

    if "Total Minutes of Sleep" in selected_metrics:
        conn = sqlite3.connect('fitbit_database.db')
        sleep = pd.read_sql_query("SELECT * FROM minute_sleep", conn)
        conn.close()
        sleep['date'] = pd.to_datetime(sleep['date']).dt.date
        sleep_daily = sleep[sleep["Id"] == selected_id].groupby("date").size().reset_index(name="TotalSleepMinutes")

        individual_data["ActivityDate_date"] = individual_data["ActivityDate"].dt.date
        individual_data = pd.merge(
            individual_data,
            sleep_daily,
            left_on="ActivityDate_date",
            right_on="date",
            how="left"
        )
        individual_data["TotalSleepMinutes"] = individual_data["TotalSleepMinutes"].fillna(0)

    if "Total Intensity" in selected_metrics:
        conn = sqlite3.connect('fitbit_database.db')
        intensity = pd.read_sql_query("SELECT * FROM hourly_intensity", conn)
        conn.close()
        intensity['ActivityHour'] = pd.to_datetime(intensity['ActivityHour'])
        intensity['Date'] = intensity['ActivityHour'].dt.date
        intensity_daily = (
            intensity[intensity["Id"] == selected_id]
            .groupby("Date")["TotalIntensity"].sum()
            .reset_index()
        )

        individual_data["ActivityDate_date"] = individual_data["ActivityDate"].dt.date
        individual_data = pd.merge(
            individual_data,
            intensity_daily,
            left_on="ActivityDate_date",
            right_on="Date",
            how="left"
        )
        individual_data["TotalIntensity"] = individual_data["TotalIntensity"].fillna(0)

    if not individual_data.empty:
        st.markdown(
            f"**Statistics for ID:** `{selected_id}` "
            f"between **{start_date.date()}** and **{end_date.date()}**"
        )

        unique_days = individual_data["ActivityDate"].nunique()

        kpi_cols = st.columns(len(selected_metrics))
        for i, metric in enumerate(selected_metrics):
            col_name = metrics_map[metric]
            display_name = display_map.get(metric, metric)

            total_val = individual_data[col_name].sum()
            avg_val = total_val / unique_days if unique_days > 0 else 0

            kpi_cols[i].markdown(
                f"""
                <div style="background-color: #002a3a; border-radius: 10px; 
                            padding: 15px; margin-bottom: 10px;">
                    <h4 style="color: white; margin: 0px;">{display_name}</h4>
                    <p style="color: white; margin: 0px;">
                        Sum: {total_val:.0f}
                    </p>
                    <p style="color: white; margin: 0px;">
                        Avg/Day: {avg_val:.2f}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Daily Histograms")
        if selected_metrics:
            num_metrics = len(selected_metrics)
            rows_subplot = (num_metrics + 1) // 2
            fig, axs = plt.subplots(rows_subplot, 2, figsize=(16, 4 * rows_subplot))
            axs = axs.flatten()

            individual_data = individual_data.sort_values("ActivityDate")
            x_vals = individual_data["ActivityDate"]

            for idx, metric in enumerate(selected_metrics):
                col_name = metrics_map[metric]
                axs[idx].bar(x_vals, individual_data[col_name], color='#00B5B8')
                axs[idx].set_title(f"{display_map.get(metric, metric)} per Day")
                axs[idx].tick_params(axis='x', rotation=45)

            for j in range(idx + 1, len(axs)):
                axs[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.warning("No data available for this ID and selected date range.")


elif selected == "Sleep Analysis":
    st.title("Sleep Duration Analysis")
    st.markdown("This page will contain sleep duration statistics and visualizations (to be implemented next).")