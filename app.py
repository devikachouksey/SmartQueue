
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import create_client

st.set_page_config(
    page_title="SmartQueue",
    page_icon="⏳",
    layout="wide"
)

# ---------------- SUPABASE CONNECTION ----------------

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- FILE PATHS ----------------

DATA_PATH = Path("data/queue_data.csv")
MODEL_PATH = Path("models/queue_model.joblib")


# ---------------- MACHINE LEARNING MODEL ----------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        subprocess.run(
            [sys.executable, "generate_data.py"],
            check=True
        )
        subprocess.run(
            [sys.executable, "train_model.py"],
            check=True
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# ---------------- DATABASE FUNCTIONS ----------------

def save_prediction(row, prediction):
    supabase.table("predictions").insert({
        "created_at": datetime.now().astimezone().isoformat(),
        "service_type": row["service_type"],
        "people_waiting": int(row["people_waiting"]),
        "active_counters": int(row["active_counters"]),
        "avg_service_time": float(row["avg_service_time"]),
        "hour": int(row["hour"]),
        "day_of_week": int(row["day_of_week"]),
        "predicted_wait": float(prediction)
    }).execute()


def get_predictions():
    response = (
        supabase.table("predictions")
        .select("*")
        .order("id", desc=True)
        .execute()
    )

    return pd.DataFrame(response.data)


def get_tokens():
    response = (
        supabase.table("tokens")
        .select("*")
        .order("id")
        .execute()
    )

    return pd.DataFrame(response.data)


def is_admin():
    return st.session_state.get("admin_logged_in", False)


# ---------------- NAVIGATION ----------------

st.title("⏳ SmartQueue")
st.caption(
    "AI-Based Queue Waiting Time Predictor | College Office"
)

st.info(
    "Demo notice: This project uses a simulated training dataset. "
    "Predictions are estimates, not guaranteed waiting times."
)

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


# ---------------- PREDICT WAITING TIME ----------------

if page == "Predict Waiting Time":

    st.subheader("Enter current queue details")

    col1, col2 = st.columns(2)

    with col1:

        service_type = st.selectbox(
            "Select service counter",
            ["Fees", "Examination", "Admission", "Library"]
        )

        people_waiting = st.number_input(
            "People currently waiting",
            min_value=0,
            max_value=300,
            value=12,
            step=1
        )

        active_counters = st.number_input(
            "Number of active counters",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

    with col2:

        avg_service_time = st.number_input(
            "Average service time per person (minutes)",
            min_value=0.5,
            max_value=60.0,
            value=4.0,
            step=0.5
        )

        current_hour = st.slider(
            "Current hour (24-hour format)",
            9, 17, 11
        )

        day_name = st.selectbox(
            "Day of week",
            [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ]
        )

        day_of_week = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ].index(day_name)

    if st.button(
        "Predict Waiting Time",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{
            "people_waiting": people_waiting,
            "active_counters": active_counters,
            "avg_service_time": avg_service_time,
            "hour": current_hour,
            "day_of_week": day_of_week,
            "service_type": service_type
        }])

        prediction = max(
            0,
            float(model.predict(input_data)[0])
        )

        try:
            save_prediction(input_data.iloc[0], prediction)

            st.success("Prediction generated and saved successfully!")

            st.metric(
                "Estimated waiting time",
                f"{prediction:.1f} minutes"
            )

            if prediction < 10:
                st.write("🟢 Low waiting time")

            elif prediction < 30:
                st.write("🟡 Moderate waiting time")

            else:
                st.write("🔴 High waiting time")

            st.caption(
                "This is an ML estimate from simulated demo data. "
                "Actual waiting time may differ."
            )

        except Exception as e:
            st.error(f"Could not save prediction: {e}")


# ---------------- DASHBOARD & HISTORY ----------------

elif page == "Dashboard & History":

    if not is_admin():
        st.warning("Please login as Admin to access prediction history.")

    else:

        st.subheader("Prediction Dashboard")

        try:
            history = get_predictions()

            if history.empty:

                st.warning("No predictions available yet.")

            else:

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Predictions",
                    len(history)
                )

                c2.metric(
                    "Average Predicted Wait",
                    f"{history['predicted_wait'].mean():.1f} min"
                )

                c3.metric(
                    "Longest Predicted Wait",
                    f"{history['predicted_wait'].max():.1f} min"
                )

                chart = px.bar(
                    history.head(20).sort_values("id"),
                    x="id",
                    y="predicted_wait",
                    color="service_type",
                    title="Recent Predictions (Latest 20)"
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

                st.dataframe(
                    history,
                    use_container_width=True
                )

                csv = history.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download History as CSV",
                    csv,
                    "queue_history.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Could not load prediction history: {e}")


# ---------------- PEAK HOURS ----------------

elif page == "Peak Hours & Best Time":

    if not is_admin():
        st.warning("Please login as Admin to access analytics.")

    else:

        st.subheader("⏰ Peak Hour Detection")

        try:
            history = get_predictions()

            if history.empty:

                st.warning("No prediction history available.")

            else:

                hourly_data = (
                    history.groupby("hour")["predicted_wait"]
                    .mean()
                    .reset_index()
                    .sort_values("hour")
                )

                busiest = hourly_data.loc[
                    hourly_data["predicted_wait"].idxmax()
                ]

                best = hourly_data.loc[
                    hourly_data["predicted_wait"].idxmin()
                ]

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

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

                st.subheader("📌 Best Time Recommendation")

                st.success(
                    f"Based on recorded predictions, "
                    f"{int(best['hour']):02d}:00 has the lowest "
                    f"average predicted waiting time."
                )

                st.info(
                    "This recommendation is based on saved prediction "
                    "history, not live queue monitoring."
                )

        except Exception as e:
            st.error(f"Could not load peak hour analytics: {e}")


# ---------------- SERVICE-WISE ANALYTICS ----------------

elif page == "Service-wise Analytics":

    if not is_admin():
        st.warning("Please login as Admin to access analytics.")

    else:

        st.subheader("📊 Service-wise Analytics")

        try:
            history = get_predictions()

            if history.empty:

                st.warning("No prediction history available.")

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

                busiest_service = service_stats.loc[
                    service_stats["Average Wait"].idxmax()
                ]

                fastest_service = service_stats.loc[
                    service_stats["Average Wait"].idxmin()
                ]

                c1, c2 = st.columns(2)

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

                chart = px.bar(
                    service_stats,
                    x="Service Type",
                    y="Average Wait",
                    color="Service Type",
                    title="Service Counter Comparison"
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

                st.subheader("Detailed Service Statistics")

                st.dataframe(
                    service_stats,
                    use_container_width=True
                )

                st.caption(
                    "Statistics are calculated from saved predictions, "
                    "not actual measured waiting times."
                )

        except Exception as e:
            st.error(f"Could not load service analytics: {e}")


# ---------------- ADMIN LOGIN ----------------

elif page == "Admin Login":

    st.subheader("🔐 Admin Login")

    if not is_admin():

        st.write("Enter your admin credentials.")

        username = st.text_input("Admin Username")

        password = st.text_input(
            "Admin Password",
            type="password"
        )

        if st.button("Login", type="primary"):

            try:
                admin_username = st.secrets["admin"]["username"]
                admin_password = st.secrets["admin"]["password"]

                if (
                    username == admin_username
                    and password == admin_password
                ):

                    st.session_state.admin_logged_in = True

                    st.success("Login successful!")

                    st.rerun()

                else:

                    st.error("Invalid username or password.")

            except Exception:
                st.error("Admin credentials are not configured.")

    else:

        st.success("Welcome, Admin! 👋")

        if st.button("Logout"):

            st.session_state.admin_logged_in = False

            st.rerun()

        st.subheader("📊 Admin Dashboard")

        try:
            admin_history = get_predictions()

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

                service_counts = (
                    admin_history["service_type"]
                    .value_counts()
                    .reset_index()
                )

                service_counts.columns = [
                    "Service", "Predictions"
                ]

                fig = px.pie(
                    service_counts,
                    names="Service",
                    values="Predictions",
                    title="Prediction Distribution by Service",
                    hole=0.4
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("🕒 Recent Activity")

                st.dataframe(
                    admin_history.head(20),
                    use_container_width=True
                )

                csv_data = (
                    admin_history.to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    "📥 Download All Records",
                    csv_data,
                    "admin_queue_records.csv",
                    "text/csv"
                )

                st.subheader("⚠️ Database Management")

                confirm_delete = st.checkbox(
                    "I understand that deleting history will permanently "
                    "remove all prediction records."
                )

                if st.button(
                    "🗑️ Clear All Prediction History",
                    disabled=not confirm_delete
                ):

                    supabase.table("predictions").delete().gte(
                        "id", 0
                    ).execute()

                    st.success(
                        "All prediction history deleted successfully!"
                    )

                    st.rerun()

        except Exception as e:
            st.error(f"Admin dashboard error: {e}")


# ---------------- LIVE TOKEN QUEUE ----------------

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

        elif len(customer_name.strip()) > 80:

            st.error("Customer name must be 80 characters or fewer.")

        else:

            try:
                existing = (
                    supabase.table("tokens")
                    .select("token_number")
                    .order("token_number", desc=True)
                    .limit(1)
                    .execute()
                )

                if existing.data:
                    next_token = existing.data[0]["token_number"] + 1
                else:
                    next_token = 1

                supabase.table("tokens").insert({
                    "token_number": next_token,
                    "customer_name": customer_name.strip(),
                    "service_type": token_service,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "status": "Waiting"
                }).execute()

                st.success(
                    f"Token generated! Your token number is: {next_token}"
                )

            except Exception as e:
                st.error(f"Could not generate token: {e}")

    st.divider()

    st.subheader("Current Queue")

    try:
        queue = get_tokens()

        if queue.empty:

            st.info("No customers registered yet.")

        else:

            total_customers = len(queue)

            waiting_count = (
                queue["status"] == "Waiting"
            ).sum()

            serving_count = (
                queue["status"] == "Serving"
            ).sum()

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Customers", total_customers)

            col2.metric(
                "Currently Waiting",
                int(waiting_count)
            )

            col3.metric(
                "Currently Serving",
                int(serving_count)
            )

            # Public queue view hides customer names.
            public_columns = [
                "token_number",
                "service_type",
                "created_at",
                "status"
            ]

            st.dataframe(
                queue[public_columns],
                use_container_width=True
            )

            if is_admin():

                with st.expander("Admin: Customer Details"):
                    st.dataframe(
                        queue,
                        use_container_width=True
                    )

    except Exception as e:

        st.error(f"Could not load queue: {e}")

    st.divider()

    if is_admin():

        st.subheader("Counter Controls")

        if st.button("▶️ Call Next Token"):

            try:
                # Complete the currently serving token.
                supabase.table("tokens").update({
                    "status": "Completed"
                }).eq("status", "Serving").execute()

                next_customer = (
                    supabase.table("tokens")
                    .select("id")
                    .eq("status", "Waiting")
                    .order("id")
                    .limit(1)
                    .execute()
                )

                if next_customer.data:

                    customer_id = next_customer.data[0]["id"]

                    supabase.table("tokens").update({
                        "status": "Serving"
                    }).eq("id", customer_id).execute()

                    st.success("Next customer called!")

                else:

                    st.warning("No waiting customers.")

            except Exception as e:

                st.error(f"Could not call next token: {e}")

            st.rerun()

        if st.button("✅ Complete Current Token"):

            try:
                current = (
                    supabase.table("tokens")
                    .select("id")
                    .eq("status", "Serving")
                    .order("id")
                    .limit(1)
                    .execute()
                )

                if current.data:

                    current_id = current.data[0]["id"]

                    supabase.table("tokens").update({
                        "status": "Completed"
                    }).eq("id", current_id).execute()

                    st.success("Customer service completed!")

                else:

                    st.warning("No customer is currently being served.")

            except Exception as e:

                st.error(f"Could not complete token: {e}")

            st.rerun()

    else:

        st.info("Admin login is required to access counter controls.")


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.subheader("Project Overview")

    st.write("""
    **Project:** SmartQueue – AI-Based Queue Waiting Time Predictor

    **Problem:** People often do not know how long they may wait
    at college service counters.

    **Solution:** SmartQueue uses a machine-learning regression model
    to estimate waiting time from queue length, active counters,
    average service time, time, day and service type.

    **Machine Learning Algorithm:** Random Forest Regressor.

    **Technology:** Python, Pandas, NumPy, Scikit-learn,
    Streamlit, Supabase and Plotly.

    **Database:** Supabase PostgreSQL.

    **Current Limitation:** The included dataset is synthetic and
    intended for demonstrating the application workflow.
    Collect real queue observations before making real-world
    accuracy claims.
    """)

    st.markdown(
        "**Main workflow:** Input queue details → ML preprocessing "
        "→ Random Forest prediction → Result and saved history."
    )
