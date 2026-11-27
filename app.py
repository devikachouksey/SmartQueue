from pathlib import Path
import subprocess
import sys
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SmartQueue", page_icon="⏳", layout="wide")

DATA_PATH = Path("data/queue_data.csv")
MODEL_PATH = Path("models/queue_model.joblib")
DB_PATH = "data/queue_history.db"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        subprocess.run([sys.executable, "generate_data.py"], check=True)
        subprocess.run([sys.executable, "train_model.py"], check=True)
    return joblib.load(MODEL_PATH)

def setup_database():
    Path("data").mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                service_type TEXT,
                people_waiting INTEGER,
                active_counters INTEGER,
                avg_service_time REAL,
                hour INTEGER,
                day_of_week INTEGER,
                predicted_wait REAL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_number INTEGER UNIQUE,
                customer_name TEXT,
                service_type TEXT,
                created_at TEXT,
                status TEXT DEFAULT 'Waiting'
            )
        """)
def save_prediction(row, prediction):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO predictions
            (created_at, service_type, people_waiting, active_counters,
             avg_service_time, hour, day_of_week, predicted_wait)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            row["service_type"], int(row["people_waiting"]),
            int(row["active_counters"]), float(row["avg_service_time"]),
            int(row["hour"]), int(row["day_of_week"]), float(prediction)
        ))

setup_database()
model = load_model()

st.title("⏳ SmartQueue")
st.caption("AI-Based Queue Waiting Time Predictor | College Office Demo")
st.info("Demo notice: this project currently uses a simulated training dataset. Predictions are estimates, not guaranteed waiting times.")


page = st.sidebar.radio(
    "Navigate",
    [
        "Predict Waiting Time",
        "Dashboard & History",
        "Peak Hours & Best Time",
        "Service-wise Analytics",
        "Admin Login",
     "Live Token Queue",
        "About Project"
    ]
)


if page == "Predict Waiting Time":
    st.subheader("Enter current queue details")
    col1, col2 = st.columns(2)

    with col1:
        service_type = st.selectbox(
            "Select service counter",
            ["Fees", "Examination", "Admission", "Library"]
        )
        people_waiting = st.number_input(
            "People currently waiting", min_value=0, max_value=300, value=12, step=1
        )
        active_counters = st.number_input(
            "Number of active counters", min_value=1, max_value=20, value=2, step=1
        )

    with col2:
        avg_service_time = st.number_input(
            "Average service time per person (minutes)",
            min_value=0.5, max_value=60.0, value=4.0, step=0.5
        )
        current_hour = st.slider("Current hour (24-hour format)", 9, 17, 11)
        day_name = st.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_name)

    if st.button("Predict Waiting Time", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "people_waiting": people_waiting,
            "active_counters": active_counters,
            "avg_service_time": avg_service_time,
            "hour": current_hour,
            "day_of_week": day_of_week,
            "service_type": service_type
        }])
        prediction = max(0, float(model.predict(input_data)[0]))
        save_prediction(input_data.iloc[0], prediction)

        st.success("Prediction generated!")
        st.metric("Estimated waiting time", f"{prediction:.1f} minutes")
        if prediction < 10:
            st.write("🟢 Low waiting time")
        elif prediction < 30:
            st.write("🟡 Moderate waiting time")
        else:
            st.write("🔴 High waiting time")
        st.caption("This is an ML estimate from simulated demo data. Actual waiting time may differ.")

elif page == "Dashboard & History":
    st.subheader("Prediction dashboard")
    with sqlite3.connect(DB_PATH) as con:
        history = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC", con
        )

    if history.empty:
        st.warning("No predictions yet. Go to Predict Waiting Time and make your first prediction.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", len(history))
        c2.metric("Average predicted wait", f"{history['predicted_wait'].mean():.1f} min")
        c3.metric("Longest predicted wait", f"{history['predicted_wait'].max():.1f} min")

        chart = px.bar(
            history.head(20).sort_values("id"),
            x="id", y="predicted_wait", color="service_type",
            title="Recent predictions (latest 20)"
        )
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(history, use_container_width=True)
        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button("Download history as CSV", csv, "queue_history.csv", "text/csv")

elif page == "Peak Hours & Best Time":
    st.subheader("⏰ Peak Hour Detection")
    st.write("Analyze your saved queue predictions to identify busy and less busy hours.")

    with sqlite3.connect(DB_PATH) as con:
        history = pd.read_sql_query(
            "SELECT hour, predicted_wait, service_type FROM predictions",
            con
        )

    if history.empty:
        st.warning("No prediction history available. Generate some predictions first.")
    else:
        hourly_data = (
            history.groupby("hour")["predicted_wait"]
            .mean()
            .reset_index()
            .sort_values("hour")
        )

        busiest = hourly_data.loc[hourly_data["predicted_wait"].idxmax()]
        best = hourly_data.loc[hourly_data["predicted_wait"].idxmin()]

        c1, c2 = st.columns(2)

        c1.metric(
            "Busiest Recorded Hour",
            f"{int(busiest['hour']):02d}:00",
            f"{busiest['predicted_wait']:.1f} min average wait"
        )

        c2.metric(
            "Best Recorded Hour",
            f"{int(best['hour']):02d}:00",
            f"{best['predicted_wait']:.1f} min average wait"
        )

        st.subheader("Average Waiting Time by Hour")

        chart = px.bar(
            hourly_data,
            x="hour",
            y="predicted_wait",
            title="Average Predicted Waiting Time",
            labels={
                "hour": "Hour of Day",
                "predicted_wait": "Average Waiting Time (minutes)"
            },
            color="predicted_wait",
            color_continuous_scale="RdYlGn_r"
        )

        st.plotly_chart(chart, use_container_width=True)

        st.subheader("📌 Best Time Recommendation")

        st.success(
            f"Based on recorded predictions, {int(best['hour']):02d}:00 "
            f"has the lowest average predicted waiting time."
        )

        st.info(
            "This recommendation is based on your saved prediction history, "
            "not live queue monitoring. More observations can improve its usefulness."
        )

elif page == "Service-wise Analytics":
    st.subheader("📊 Service-wise Analytics")
    st.write("Compare waiting times across different college service counters.")

    with sqlite3.connect(DB_PATH) as con:
        history = pd.read_sql_query(
            "SELECT service_type, predicted_wait FROM predictions",
            con
        )

    if history.empty:
        st.warning("No prediction history available. Generate predictions first.")
    else:
        service_stats = (
            history.groupby("service_type")["predicted_wait"]
            .agg(["mean", "max", "count"])
            .reset_index()
        )

        service_stats.columns = [
            "Service Type",
            "Average Wait",
            "Maximum Wait",
            "Total Predictions"
        ]

        c1, c2 = st.columns(2)

        busiest_service = service_stats.loc[
            service_stats["Average Wait"].idxmax()
        ]

        fastest_service = service_stats.loc[
            service_stats["Average Wait"].idxmin()
        ]

        c1.metric(
            "Highest Average Waiting",
            busiest_service["Service Type"],
            f"{busiest_service['Average Wait']:.1f} minutes"
        )

        c2.metric(
            "Lowest Average Waiting",
            fastest_service["Service Type"],
            f"{fastest_service['Average Wait']:.1f} minutes"
        )

        st.subheader("Average Waiting Time by Service")

        chart = px.bar(
            service_stats,
            x="Service Type",
            y="Average Wait",
            color="Service Type",
            title="Service Counter Comparison"
        )

        st.plotly_chart(chart, use_container_width=True)

        st.subheader("Detailed Service Statistics")
        st.dataframe(service_stats, use_container_width=True)

        st.caption(
            "Statistics are calculated from saved predictions, "
            "not actual measured waiting times."
        )

elif page == "Admin Login":
    st.subheader("🔐 Admin Login")

    if not st.session_state.get("admin_logged_in", False):
        st.write("Please enter admin credentials to access the dashboard.")

        username = st.text_input("Admin Username")
        password = st.text_input("Admin Password", type="password")

        if st.button("Login", type="primary"):
            try:
                admin_username = st.secrets["admin"]["username"]
                admin_password = st.secrets["admin"]["password"]

                if username == admin_username and password == admin_password:
                    st.session_state.admin_logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            except (KeyError, FileNotFoundError):
                st.error("Admin credentials are not configured.")

    else:
        st.success("Welcome, Admin! 👋")

        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.subheader("📊 Admin Dashboard")

        with sqlite3.connect(DB_PATH) as con:
            admin_history = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY id DESC",
                con
            )

        if admin_history.empty:
            st.warning("No prediction records available.")
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Predictions",
                len(admin_history)
            )

            col2.metric(
                "Average Waiting Time",
                f"{admin_history['predicted_wait'].mean():.1f} min"
            )

            col3.metric(
                "Maximum Waiting Time",
                f"{admin_history['predicted_wait'].max():.1f} min"
            )

            st.subheader("📈 Service-wise Prediction Count")

            service_counts = admin_history["service_type"].value_counts().reset_index()
            service_counts.columns = ["Service", "Predictions"]

            fig = px.pie(
                service_counts,
                names="Service",
                values="Predictions",
                title="Prediction Distribution by Service",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🕒 Recent Activity")

            st.dataframe(
                admin_history.head(20),
                use_container_width=True
            )

            csv_data = admin_history.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download All Records",
                csv_data,
                "admin_queue_records.csv",
                "text/csv"
            )

            st.subheader("⚠️ Database Management")

            confirm_delete = st.checkbox(
                "I understand that deleting history will permanently remove all prediction records."
            )

            if st.button("🗑️ Clear All Prediction History", disabled=not confirm_delete):
                with sqlite3.connect(DB_PATH) as con:
                    con.execute("DELETE FROM predictions")

                st.success("All prediction history deleted successfully!")
                st.rerun()

elif page == "Live Token Queue":
    st.subheader("🎟️ Live Queue & Token Management")
    st.write("Register customers and generate automatic tokens.")

    with st.form("token_form"):
        customer_name = st.text_input("Customer Name")

        token_service = st.selectbox(
            "Select Service",
            ["Fees", "Examination", "Admission", "Library"]
        )

        submitted = st.form_submit_button("Generate Token")

    if submitted:
        if not customer_name.strip():
            st.error("Please enter customer name.")
        else:
            with sqlite3.connect(DB_PATH) as con:
                next_token = con.execute(
                    "SELECT COALESCE(MAX(token_number), 0) + 1 FROM tokens"
                ).fetchone()[0]

                con.execute("""
                    INSERT INTO tokens
                    (token_number, customer_name, service_type, created_at, status)
                    VALUES (?, ?, ?, ?, 'Waiting')
                """, (
                    next_token,
                    customer_name.strip(),
                    token_service,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))

            st.success(f"Token generated! Your token number is: {next_token}")

    st.divider()
    st.subheader("Current Queue")

    with sqlite3.connect(DB_PATH) as con:
        queue = pd.read_sql_query(
            "SELECT token_number, customer_name, service_type, created_at, status "
            "FROM tokens ORDER BY id ASC",
            con
        )

    total_customers = len(queue)
    waiting_count = (queue["status"] == "Waiting").sum()
    serving_count = (queue["status"] == "Serving").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", total_customers)
    col2.metric("Currently Waiting", int(waiting_count))
    col3.metric("Currently Serving", int(serving_count))
    
    if queue.empty:
        st.info("No customers registered yet.")
    else:
        st.dataframe(queue, use_container_width=True)

    st.divider()
    st.subheader("Counter Controls")

    if st.button("▶️ Call Next Token"):
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "UPDATE tokens SET status = 'Completed' WHERE status = 'Serving'"
            )

            next_customer = con.execute(
                "SELECT id FROM tokens WHERE status = 'Waiting' ORDER BY id ASC LIMIT 1"
            ).fetchone()

            if next_customer:
                con.execute(
                    "UPDATE tokens SET status = 'Serving' WHERE id = ?",
                    (next_customer[0],)
                )
                st.success("Next customer called!")
            else:
                st.warning("No waiting customers.")

        st.rerun()

    if st.button("✅ Complete Current Token"):
        with sqlite3.connect(DB_PATH) as con:
            current = con.execute(
                "SELECT id FROM tokens WHERE status = 'Serving' LIMIT 1"
            ).fetchone()

            if current:
                con.execute(
                    "UPDATE tokens SET status = 'Completed' WHERE id = ?",
                    (current[0],)
                )
                st.success("Customer service completed!")
            else:
                st.warning("No customer is currently being served.")

        st.rerun()

elif page == "About Project":
    st.subheader("Project overview")
    st.write("""
    **Problem:** People often do not know how long they may wait at college service counters.

    **Solution:** SmartQueue uses a machine-learning regression model to estimate waiting time
    from queue length, active counters, average service time, time, day and service type.

    **Machine learning algorithm:** Random Forest Regressor.

    **Technology:** Python, Pandas, NumPy, Scikit-learn, Streamlit, SQLite and Plotly.

    **Current limitation:** The included dataset is synthetic and intended for demonstrating
    the complete application workflow. Collect real queue observations before making real-world
    accuracy claims.
    """)
    st.markdown("**Main workflow:** Input queue details → ML preprocessing → Random Forest prediction → Result and saved history.")
