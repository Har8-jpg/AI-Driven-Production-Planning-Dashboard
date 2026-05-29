# =========================================================
# AI-DRIVEN PRODUCTION PLANNING DASHBOARD
# STREAMLIT + LSTM + WORKFORCE ANALYSIS
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Driven Production Planning Dashboard",
    page_icon="🏭",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🏭 AI-Driven Production Planning Dashboard")

st.markdown("""
This dashboard helps manufacturing companies:
- Forecast weekly production demand using LSTM
- Analyze workforce allocation
- Monitor production capacity
- Support decision analysis under uncertainty
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard Overview",
        "Demand Forecasting (LSTM)",
        "Workforce Planning",
        "Capacity Analysis",
        "EDA Visualization",
        "Scenario Simulation"
    ]
)

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

# =========================================================
# READ FILE
# =========================================================

if uploaded_file is not None:

    file_name = uploaded_file.name

    # READ CSV
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    # READ EXCEL
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)

    st.success(f"Dataset Uploaded Successfully: {file_name}")

    # =====================================================
    # DATE PROCESSING
    # =====================================================

# CREATE DATE COLUMN IF NOT EXISTS

    if "Date" not in df.columns:

        # Create artificial date using Year + Month
        df["Date"] = pd.to_datetime(
            df["Year"].astype(str) + "-"
            + df["Month"].astype(str) + "-1"
        )

    else:

        df["Date"] = pd.to_datetime(df["Date"])

    # =====================================================
    # DASHBOARD OVERVIEW
    # =====================================================

    if page == "Dashboard Overview":

        st.header("📊 Dashboard Overview")

        total_demand = df["Total Demand (Adjusted)"].sum()

        total_production = df[
            "Total Actual Production (Adjusted)"
        ].sum()

        avg_capacity = df["Total Capacity"].mean()

        avg_headcount = df["Total Headcount"].mean()

        # KPI CARDS
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Demand",
            f"{total_demand:,.0f}"
        )

        col2.metric(
            "Total Production",
            f"{total_production:,.0f}"
        )

        col3.metric(
            "Average Capacity",
            f"{avg_capacity:,.0f}"
        )

        col4.metric(
            "Average Headcount",
            f"{avg_headcount:,.0f}"
        )

        # DEMAND TREND
        st.subheader("📈 Weekly Demand Trend")

        fig = px.line(
            df,
            x="Date",
            y="Total Demand (Adjusted)",
            title="Weekly Production Demand"
        )

        st.plotly_chart(fig, use_container_width=True)

        # DEMAND VS PRODUCTION
        st.subheader("🏭 Demand vs Actual Production")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Total Demand (Adjusted)"],
            mode='lines',
            name='Demand'
        ))

        fig2.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Total Actual Production (Adjusted)"],
            mode='lines',
            name='Actual Production'
        ))

        st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # LSTM FORECASTING
    # =====================================================

    elif page == "Demand Forecasting (LSTM)":

        st.header("📈 LSTM Demand Forecasting")

        # ==========================================
        # PREPARE DATA
        # ==========================================

        demand_data = df[
            "Total Demand (Adjusted)"
        ].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_data = scaler.fit_transform(demand_data)

        # ==========================================
        # CREATE SEQUENCES
        # ==========================================

        sequence_length = 4

        X = []
        y = []

        for i in range(sequence_length, len(scaled_data)):

            X.append(
                scaled_data[i-sequence_length:i, 0]
            )

            y.append(
                scaled_data[i, 0]
            )

        X = np.array(X)
        y = np.array(y)

        X = np.reshape(
            X,
            (X.shape[0], X.shape[1], 1)
        )

        # ==========================================
        # BUILD MODEL
        # ==========================================

        model = Sequential()

        model.add(
            LSTM(
                units=50,
                return_sequences=False,
                input_shape=(X.shape[1], 1)
            )
        )

        model.add(Dense(1))

        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )

        # ==========================================
        # TRAIN MODEL
        # ==========================================

        with st.spinner("Training LSTM Model..."):

            model.fit(
                X,
                y,
                epochs=20,
                batch_size=8,
                verbose=0
            )

        st.success("LSTM Model Trained Successfully!")

        # ==========================================
        # PREDICTION
        # ==========================================

        predictions = model.predict(X)

        predictions = scaler.inverse_transform(
            predictions
        )

        actual_values = scaler.inverse_transform(
            y.reshape(-1, 1)
        )

        # ==========================================
        # EVALUATION
        # ==========================================

        mae = mean_absolute_error(
            actual_values,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual_values,
                predictions
            )
        )

        # KPI METRICS
        col1, col2 = st.columns(2)

        col1.metric(
            "MAE",
            f"{mae:.2f}"
        )

        col2.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        # ==========================================
        # RESULT DATAFRAME
        # ==========================================

        forecast_df = pd.DataFrame({
            "Actual Demand": actual_values.flatten(),
            "Predicted Demand": predictions.flatten()
        })

        st.subheader("Forecast Results")

        st.dataframe(forecast_df.head(20))

        # ==========================================
        # FORECAST PLOT
        # ==========================================

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=forecast_df["Actual Demand"],
            mode='lines',
            name='Actual Demand'
        ))

        fig.add_trace(go.Scatter(
            y=forecast_df["Predicted Demand"],
            mode='lines',
            name='Predicted Demand'
        ))

        fig.update_layout(
            title="Actual vs Predicted Demand",
            xaxis_title="Time",
            yaxis_title="Demand"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # WORKFORCE PLANNING
    # =====================================================

    elif page == "Workforce Planning":

        st.header("👷 Workforce Planning")

        STANDARD_OUTPUT_PER_OPERATOR = 60

        df["Required Headcount"] = np.ceil(
            df["Total Demand (Adjusted)"]
            / STANDARD_OUTPUT_PER_OPERATOR
        )

        df["Headcount Gap"] = (
            df["Required Headcount"]
            - df["Total Headcount"]
        )

        df["Workforce Status"] = np.where(
            df["Headcount Gap"] > 0,
            "Shortage",
            np.where(
                df["Headcount Gap"] < 0,
                "Excess",
                "Sufficient"
            )
        )

        # BAR CHART
        workforce_count = (
            df["Workforce Status"]
            .value_counts()
        )

        fig = px.bar(
            x=workforce_count.index,
            y=workforce_count.values,
            labels={
                "x": "Workforce Status",
                "y": "Count"
            },
            title="Workforce Status Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        # TABLE
        st.subheader("Workforce Planning Table")

        st.dataframe(df[
            [
                "Date",
                "Total Demand (Adjusted)",
                "Total Headcount",
                "Required Headcount",
                "Headcount Gap",
                "Workforce Status"
            ]
        ])

    # =====================================================
    # CAPACITY ANALYSIS
    # =====================================================

    elif page == "Capacity Analysis":

        st.header("🏭 Capacity Analysis")

        df["Capacity Utilization"] = (
            df["Total Actual Production (Adjusted)"]
            / df["Total Capacity"]
        ) * 100

        # UTILIZATION TREND
        fig = px.line(
            df,
            x="Date",
            y="Capacity Utilization",
            title="Capacity Utilization (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # DEMAND VS CAPACITY
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Total Demand (Adjusted)"],
            mode='lines',
            name='Demand'
        ))

        fig2.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Total Capacity"],
            mode='lines',
            name='Capacity'
        ))

        st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # EDA VISUALIZATION
    # =====================================================

    elif page == "EDA Visualization":

        st.header("📊 Exploratory Data Analysis")

        # CORRELATION HEATMAP
        st.subheader("Correlation Heatmap")

        corr_features = [
            "Total Demand (Adjusted)",
            "Total Actual Production (Adjusted)",
            "Total Headcount",
            "Total Capacity"
        ]

        corr_matrix = df[corr_features].corr()

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

        # BOXPLOT
        st.subheader("Outlier Detection")

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        sns.boxplot(
            data=df[corr_features],
            ax=ax2
        )

        st.pyplot(fig2)

    # =====================================================
    # SCENARIO SIMULATION
    # =====================================================

    elif page == "Scenario Simulation":

        st.header("⚙️ Scenario Simulation")

        demand_increase = st.slider(
            "Increase Demand (%)",
            0,
            100,
            10
        )

        workforce_increase = st.slider(
            "Increase Workforce (%)",
            0,
            100,
            5
        )

        simulated_demand = (
            df["Total Demand (Adjusted)"].mean()
            * (1 + demand_increase / 100)
        )

        simulated_workforce = (
            df["Total Headcount"].mean()
            * (1 + workforce_increase / 100)
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Simulated Demand",
            f"{simulated_demand:,.0f}"
        )

        col2.metric(
            "Simulated Workforce",
            f"{simulated_workforce:,.0f}"
        )

        if simulated_demand > df["Total Capacity"].mean():

            st.error(
                "⚠️ Demand exceeds current production capacity."
            )

        else:

            st.success(
                "✅ Current capacity is sufficient."
            )

# =========================================================
# NO FILE UPLOADED
# =========================================================

else:

    st.warning(
        "Please upload a CSV or Excel dataset to continue."
    )