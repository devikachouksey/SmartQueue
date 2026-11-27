import numpy as np
import pandas as pd
from pathlib import Path

# Creates a clearly labelled synthetic dataset for the college demo.
# Replace this with real observations when available.
rng = np.random.default_rng(42)
n = 1200

people_waiting = rng.integers(0, 61, n)
active_counters = rng.integers(1, 6, n)
avg_service_time = rng.uniform(1.5, 8.0, n)  # minutes per person
hour = rng.integers(9, 18, n)
day_of_week = rng.integers(0, 7, n)  # Monday=0
service_type = rng.choice(["Fees", "Examination", "Admission", "Library"], n)

# Simple simulated relationship: workload divided across active counters.
rush_factor = np.where((hour >= 11) & (hour <= 14), 1.15, 1.0)
service_factor = pd.Series(service_type).map({
    "Fees": 1.0, "Examination": 1.2, "Admission": 1.3, "Library": 0.7
}).to_numpy()

wait_time = (
    people_waiting * avg_service_time / active_counters
) * rush_factor * service_factor
wait_time += rng.normal(0, 2.0, n)
wait_time = np.clip(wait_time, 0, None)

df = pd.DataFrame({
    "people_waiting": people_waiting,
    "active_counters": active_counters,
    "avg_service_time": avg_service_time.round(2),
    "hour": hour,
    "day_of_week": day_of_week,
    "service_type": service_type,
    "waiting_time": wait_time.round(1)
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/queue_data.csv", index=False)
print("Created data/queue_data.csv with", len(df), "synthetic rows.")
