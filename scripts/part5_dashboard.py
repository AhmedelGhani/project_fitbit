import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import data_analysis


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
    st.write("<style>h1 { text-align: center; }</style>", unsafe_allow_html=True)
    st.title("Fitbit Data Dashboard")

    num_participants = activity["Id"].nunique()
    avg_steps = activity["TotalSteps"].mean()
    avg_calories = activity["Calories"].mean()
    avg_active = (activity["VeryActiveMinutes"] + activity["FairlyActiveMinutes"] + activity["LightlyActiveMinutes"]).mean()
    avg_sedentary = activity["SedentaryMinutes"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Participants", num_participants)
    with col2:
        st.metric("Avg Steps", f"{avg_steps:.0f}")
    with col3:
        st.metric("Avg Calories", f"{avg_calories:.0f} kcal")
    with col4:
        st.metric("Avg Active Min", f"{avg_active:.1f} min")
    with col5:
        st.metric("Avg Sedentary Min", f"{avg_sedentary:.1f} min")

    st.markdown("---")

    st.subheader("Calories vs Total Steps Regression")
    model = data_analysis.run_regression(activity)
    r2 = model.rsquared
    intercept = model.params[0]
    coef = model.params[1]
    pval_intercept = model.pvalues[0]
    pval_slope = model.pvalues[1]

    colR1, colR2 = st.columns(2)
    with colR1:
        st.markdown(f"""
        <div style='background-color:#002a3a; padding:15px; border-radius:10px; color:white;'>
        <h4>OLS Regression Summary</h4>
        <p><strong>R-squared:</strong> {r2:.3f}</p>
        <p><strong>Intercept:</strong> {intercept:.2f} (p = {pval_intercept:.2e})</p>
        <p><strong>TotalSteps Coef:</strong> {coef:.4f} (p = {pval_slope:.2e})</p>
        </div>
        """, unsafe_allow_html=True)

    with colR2:
        st.subheader("Distance vs Calories Plot")
        fig_dist = data_analysis.plot_distance_vs_calories(activity)
        st.pyplot(fig_dist)


    st.markdown("---")

    conn = sqlite3.connect('fitbit_database.db')
    steps = pd.read_sql_query("SELECT * FROM hourly_steps;", conn)
    steps['Hour'] = pd.to_datetime(steps['ActivityHour']).dt.hour
    steps['Block'] = pd.cut(
        steps['Hour'], 
        bins=[0,4,8,12,16,20,24],
        labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'],
        right=False
    )
    avgStepsBlock = steps.groupby('Block')['StepTotal'].mean().reset_index(name='AverageSteps')

    calories = pd.read_sql_query("SELECT * FROM hourly_calories;", conn)
    calories['Hour'] = pd.to_datetime(calories['ActivityHour']).dt.hour
    calories['Block'] = pd.cut(
        calories['Hour'], 
        bins=[0,4,8,12,16,20,24],
        labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'],
        right=False
    )
    avgCaloriesBlock = calories.groupby('Block')['Calories'].mean().reset_index(name='AverageCalories')

    sleepDF = pd.read_sql_query("SELECT * FROM minute_sleep;", conn)
    conn.close()
    sleepDF['Hour'] = pd.to_datetime(sleepDF['date']).dt.hour
    sleepDF['Block'] = pd.cut(
        sleepDF['Hour'], 
        bins=[0,4,8,12,16,20,24],
        labels=['0 – 4','4 – 8','8 – 12','12 – 16','16 – 20','20 – 24'],
        right=False
    )
    avgSleepBlock = sleepDF.groupby('Block').size().reset_index(name='TotalMinutes')
    num_part = sleepDF['Id'].nunique()
    avgSleepBlock['AverageSleepMinutes'] = avgSleepBlock['TotalMinutes'] / num_part

    st.subheader("Averages per 4h Block")
    fig_block, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
    ax1.bar(avgStepsBlock['Block'], avgStepsBlock['AverageSteps'], color='#00B5B8')
    ax1.set_title("Avg Steps")
    ax1.set_xlabel("4h Block")
    ax1.set_ylabel("Steps")
    ax1.tick_params(axis='x', labelrotation=45)

    ax2.bar(avgCaloriesBlock['Block'], avgCaloriesBlock['AverageCalories'], color='#00B5B8')
    ax2.set_title("Avg Calories")
    ax2.set_xlabel("4h Block")
    ax2.set_ylabel("kcal")
    ax2.tick_params(axis='x', labelrotation=45)

    ax3.bar(avgSleepBlock['Block'], avgSleepBlock['AverageSleepMinutes'], color='#00B5B8')
    ax3.set_title("Avg Sleep")
    ax3.set_xlabel("4h Block")
    ax3.set_ylabel("Minutes")
    ax3.tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    st.pyplot(fig_block)

    st.markdown("---")

    counts = activity["Id"].value_counts()
    def classify(count):
        if count <= 10:
            return "Light"
        elif count <= 15:
            return "Moderate"
        else:
            return "Heavy"
    classifications = counts.apply(classify)
    classification_counts = classifications.value_counts()

    st.subheader("User Activity Classification")

    col_pie, _ = st.columns([1, 2])  # Control layout with padding
    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(2.5, 2.5))  # Smaller size
        ax_pie.pie(
            classification_counts,
            labels=classification_counts.index,
            autopct="%1.1f%%",
            colors=['#00B5B8', '#66c2a5', '#3288bd'],
            startangle=140,
            textprops={'fontsize': 9}
    )
    ax_pie.set_title("User Activity Classification", fontsize=11)
    st.pyplot(fig_pie)



if selected == "Individual Stats":
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import numpy as np

    st.write("""<style>h1 { text-align: center; }</style>""", unsafe_allow_html=True)
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
        "Steps": "TotalSteps",
        "Distance": "TotalDistance",
        "Active Minutes": "TotalActiveMinutes",
        "Sedentary Minutes": "SedentaryMinutes",
        "Calories": "Calories",
        "Minutes of Sleep": "TotalSleepMinutes",
        "Intensity": "TotalIntensity"
    }

    display_map = {
        "Steps": "Steps",
        "Distance": "Distance",
        "Active Minutes": "Active Minutes",
        "Sedentary Minutes": "Sedentary Minutes",
        "Calories": "Calories",
        "Minutes of Sleep": "Minutes of Sleep",
        "Intensity": "Intensity"
    }

    units_map = {
        "Steps": "",
        "Distance": " (km)",
        "Active Minutes": " (min)",
        "Sedentary Minutes": " (min)",
        "Calories": " (kcal)",
        "Minutes of Sleep": " (min)",
        "Intensity": ""
    }

    selected_metrics = st.sidebar.multiselect(
        "Choose metrics for histograms",
        options=list(metrics_map.keys()),
        default=["Steps", "Calories"]
    )

    if len(selected_metrics) > 4:
        st.sidebar.warning("Please select at most 4 metrics.")
        selected_metrics = selected_metrics[:4]

    individual_data["TotalActiveMinutes"] = (
        individual_data["VeryActiveMinutes"]
        + individual_data["FairlyActiveMinutes"]
        + individual_data["LightlyActiveMinutes"]
    )

    if "Minutes of Sleep" in selected_metrics:
        conn = sqlite3.connect('fitbit_database.db')
        sleep = pd.read_sql_query("SELECT * FROM minute_sleep", conn)
        conn.close()
        sleep["date"] = pd.to_datetime(sleep["date"]).dt.date
        sleep_daily = (
            sleep[sleep["Id"] == selected_id]
            .groupby("date")
            .size()
            .reset_index(name="TotalSleepMinutes")
        )
        individual_data["ActivityDate_date"] = individual_data["ActivityDate"].dt.date
        individual_data = pd.merge(
            individual_data,
            sleep_daily,
            left_on="ActivityDate_date",
            right_on="date",
            how="left"
        )
        individual_data["TotalSleepMinutes"] = individual_data["TotalSleepMinutes"].fillna(0)

    if "Intensity" in selected_metrics:
        conn = sqlite3.connect('fitbit_database.db')
        intensity = pd.read_sql_query("SELECT * FROM hourly_intensity", conn)
        conn.close()
        intensity["ActivityHour"] = pd.to_datetime(intensity["ActivityHour"])
        intensity["Date"] = intensity["ActivityHour"].dt.date
        intensity_daily = (
            intensity[intensity["Id"] == selected_id]
            .groupby("Date")["TotalIntensity"]
            .sum()
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
            unit = units_map.get(metric, "")
            total_val = individual_data[col_name].sum()
            avg_val = total_val / unique_days if unique_days > 0 else 0
            kpi_cols[i].markdown(
                f"""
                <div style="background-color: #002a3a; border-radius: 10px; 
                            padding: 15px; margin-bottom: 10px; text-align: center;">
                    <h4 style="color: white; margin: 0px; text-align: center;">{display_name}{unit}</h4>
                    <p style="color: white; margin: 0px;">
                        Total: {total_val:.0f}{unit}
                    </p>
                    <p style="color: white; margin: 0px;">
                        Avg/Day: {avg_val:.2f}{unit}
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
                unit = units_map.get(metric, "")
                axs[idx].bar(x_vals, individual_data[col_name], color='#00B5B8')
                axs[idx].set_title(f"{display_map.get(metric, metric)}{unit} per Day")
                axs[idx].set_xlabel("Date")
                axs[idx].set_ylabel(f"{display_map.get(metric, metric)}{unit}")
                axs[idx].tick_params(axis='x', rotation=45)

            for j in range(idx + 1, len(axs)):
                axs[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("### Correlation Analysis")
        st.sidebar.header("Correlation Analysis")
        analysis_metrics = st.sidebar.multiselect(
            "Pick 2 metrics to see correlation",
            selected_metrics,
            default=selected_metrics[:2]
        )

        if len(analysis_metrics) != 2:
            st.sidebar.warning("Please select exactly 2 metrics for correlation analysis.")
        else:
            x_metric, y_metric = analysis_metrics
            x_col = metrics_map[x_metric]
            y_col = metrics_map[y_metric]
            df_scatter = individual_data[[x_col, y_col]].dropna()

            if not df_scatter.empty:
                col_scatter, col_summary = st.columns(2)

                with col_scatter:
                    fig_corr, ax_corr = plt.subplots(figsize=(8,4))
                    ax_corr.scatter(df_scatter[x_col], df_scatter[y_col], color='#002a3a', alpha=0.7)
                    X_ols = sm.add_constant(df_scatter[x_col])
                    model_ols = sm.OLS(df_scatter[y_col], X_ols).fit()
                    df_scatter["pred"] = model_ols.predict(X_ols)
                    ax_corr.plot(df_scatter[x_col], df_scatter["pred"], color='red', linewidth=2)
                    ax_corr.set_xlabel(f"{x_metric}{units_map.get(x_metric, '')}")
                    ax_corr.set_ylabel(f"{y_metric}{units_map.get(y_metric, '')}")
                    st.pyplot(fig_corr)

                with col_summary:
                    r2 = model_ols.rsquared
                    beta0 = model_ols.params[0]
                    beta1 = model_ols.params[1]
                    pval0 = model_ols.pvalues[0]
                    pval1 = model_ols.pvalues[1]
                    st.markdown(
                        f"""
                        <div style="background-color: #002a3a; border-radius: 10px; 
                                    padding: 15px; margin-bottom: 10px; text-align: center;">
                            <h4 style="color: white; margin: 0px; text-align: center;">OLS Summary</h4>
                            <p style="color: white; margin: 0px;">
                                R-squared: {r2:.3f}
                            </p>
                            <p style="color: white; margin: 0px;">
                                Intercept (p-value): {beta0:.3f} ({pval0:.3f})
                            </p>
                            <p style="color: white; margin: 0px;">
                                Slope (p-value): {beta1:.3f} ({pval1:.3f})
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown("### Weight Data")
        conn = sqlite3.connect('fitbit_database.db')
        weight_data = pd.read_sql_query("SELECT * FROM weight_log WHERE Id = ?", conn, params=[selected_id])
        conn.close()

        if weight_data.empty:
            st.markdown("Weight data not available")
        else:
            row = weight_data.iloc[0]
            weight_kg = row["WeightKg"] if pd.notna(row["WeightKg"]) else None
            fat_pct = row.get("Fat") if pd.notna(row.get("Fat")) else None
            bmi_val = row["BMI"] if pd.notna(row["BMI"]) else None

            col_w, col_f, col_b = st.columns(3)

            if weight_kg is not None:
                col_w.markdown(
                    f"""
                    <div style="background-color: #002a3a; border-radius: 10px; 
                                padding: 15px; text-align: center;">
                        <h4 style="color: white; margin: 0px;">Weight (kg)</h4>
                        <p style="color: white; margin: 0px;">{weight_kg:.1f} kg</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                col_w.markdown(
                    f"""
                    <div style="background-color: #002a3a; border-radius: 10px; 
                                padding: 15px; text-align: center;">
                        <h4 style="color: white; margin: 0px;">Weight (kg)</h4>
                        <p style="color: white; margin: 0px;">Not available</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if fat_pct is not None:
                col_f.markdown(
                    f"""
                    <div style="background-color: #002a3a; border-radius: 10px; 
                                padding: 15px; text-align: center;">
                        <h4 style="color: white; margin: 0px;">Fat %</h4>
                        <p style="color: white; margin: 0px;">{fat_pct:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                col_f.markdown(
                    f"""
                    <div style="background-color: #002a3a; border-radius: 10px; 
                                padding: 15px; text-align: center;">
                        <h4 style="color: white; margin: 0px;">Fat %</h4>
                        <p style="color: white; margin: 0px;">Not available</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if bmi_val is not None:
                if bmi_val < 18.5:
                    healthy_status = "Underweight"
                    status_color = "red"
                elif 18.5 <= bmi_val <= 24.9:
                    healthy_status = "Healthy"
                    status_color = "green"
                else:
                    healthy_status = "Overweight"
                    status_color = "red"

                col_b.markdown(
                    f"""
                    <div style="background-color: #002a3a; border-radius: 10px; 
                                padding: 15px; text-align: center;">
                        <h4 style="color: white; margin: 0px;">BMI</h4>
                        <p style="color: white; margin: 0px;">{bmi_val:.1f}</p>
                        <p style="color: white; margin: 0px;">Healthy BMI range: 18.5-24.9</p>
                        <p style="color: {status_color}; margin: 0px;">Status: {healthy_status}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                col_b.markdown(
                    f"""
                    <div style="background-color: #002a3a; border-radius: 10px; 
                                padding: 15px; text-align: center;">
                        <h4 style="color: white; margin: 0px;">BMI</h4>
                        <p style="color: white; margin: 0px;">Not available</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("No data available for this ID and selected date range.")


elif selected == "Sleep Analysis":
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from db_datawrangling import mergingtables, get_daily_sleep_minutes, get_hourly_active_minutes
    from db_datavisualization import timeseries_plot
    st.title("Sleep Duration Analysis")

    min_date, max_date = activity["ActivityDate"].min(), activity["ActivityDate"].max()
    selected_dates = st.sidebar.date_input("Filter by date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date)

    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) == 2:
            start_date, end_date = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
        else:
            start_date = end_date = pd.to_datetime(selected_dates[0])
    else:
        start_date = end_date = pd.to_datetime(selected_dates)

    st.sidebar.header("Select A Metric")
    metric_options = {
    "Total Steps": "TotalSteps",
    "Total Distance": "TotalDistance",
    "Active Minutes": "TotalActiveMinutes",
    "Sedentary Minutes": "SedentaryMinutes",
    "Total Calories": "Calories",
    "Total Intensity": "TotalIntensity"
    }

    selected_metric_label = st.sidebar.selectbox(
    "Choose a metric to compare graphically against Sleep Duration.",
    options=list(metric_options.keys()),
    index=0  
    )
    selected_metric = metric_options[selected_metric_label]

    if selected_metric_label == "Active Minutes":
        active_minutes = get_hourly_active_minutes()
        active_minutes.rename(columns={"Date": "ActivityHour", "ActiveMinutes": "TotalActiveMinutes"}, inplace=True)
        merged_metric = mergingtables("daily_activity", active_minutes, time_column1="ActivityDate", time_column2="ActivityHour")
    elif selected_metric_label == "Total Intensity":
        merged_metric = mergingtables("daily_activity", "hourly_intensity", time_column1="ActivityDate", time_column2="ActivityHour")
    else: 
        merged_metric = activity.copy() 
        merged_metric["ActivityDate"] = pd.to_datetime(merged_metric["ActivityDate"], errors="coerce")
        merged_metric["Date/Time"] = merged_metric["ActivityDate"].dt.floor("D")

    if selected_metric in merged_metric.columns:
        merged_metric = merged_metric[merged_metric[selected_metric].notna()]

    sleep_df = get_daily_sleep_minutes()
    final_merged = mergingtables(merged_metric, sleep_df, time_column1="Date/Time", time_column2="Date")
    final_merged = final_merged[(final_merged["Date/Time"] >= start_date.floor("D")) &
        (final_merged["Date/Time"] <= end_date.floor("D"))].copy()

    candidate_ids = final_merged["Id"].unique()
    valid_ids = []
    for uid in candidate_ids:
        subset = final_merged[final_merged["Id"] == uid]
        if not subset.empty and subset["SleepMinutes"].sum() > 0:
            valid_ids.append(uid)

    st.sidebar.header("Select Individual ID")
    selected_id = st.sidebar.selectbox("Choose an ID to view individual statistics", valid_ids)

    final_data = final_merged[final_merged["Id"] == selected_id]
    if final_data.empty:
        st.error("No data can be displayed during the given dateframe, please extend your date length.")
    else:
        st.subheader(f"{selected_metric_label} vs. Sleep Duration Over Time (Dual Y-Axis Graph)")
        fig = timeseries_plot(final_data, selected_metric, "SleepMinutes")
        st.pyplot(fig)

    st.markdown("### Correlation Analysis")

    if final_data.empty or selected_metric not in final_data.columns:
        st.error("Insufficient data for correlation analysis.")
    else:
        final_data["SleepMinutes"] = final_data["SleepMinutes"].fillna(0)
        df_corr = final_data[["SleepMinutes", selected_metric]].copy()
        df_corr.dropna(subset=[selected_metric], inplace=True)
        if df_corr.empty:
            st.error("No valid data available for correlation analysis.")
        else:
            fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
            ax_corr.scatter(df_corr[selected_metric], df_corr["SleepMinutes"], color='#002a3a', alpha=0.7)
            X_ols = sm.add_constant(df_corr[selected_metric])
            model_ols = sm.OLS(df_corr["SleepMinutes"], X_ols).fit()
            df_corr["pred"] = model_ols.predict(X_ols)
            ax_corr.plot(df_corr[selected_metric], df_corr["pred"], color='red', linewidth=2)
            ax_corr.set_xlabel(selected_metric_label)
            ax_corr.set_ylabel("Sleep Duration")
            st.pyplot(fig_corr)
        
            st.markdown(
            f"""
            <div style="background-color: #002a3a; border-radius: 10px; 
                padding: 15px; text-align: center; margin-top: 15px;">
            <h4 style="color: white; margin: 0px;">OLS Summary</h4>
            <p style="color: white; margin: 0px;">
            R-squared: {model_ols.rsquared:.3f}
            </p>
            <p style="color: white; margin: 0px;">
            Intercept (p-value): {model_ols.params[0]:.3f} ({model_ols.pvalues[0]:.3f})
            </p>
            <p style="color: white; margin: 0px;">
            Slope (p-value): {model_ols.params[1]:.3f} ({model_ols.pvalues[1]:.3f})
            </p>
            </div>
            """,
            unsafe_allow_html=True
            )
