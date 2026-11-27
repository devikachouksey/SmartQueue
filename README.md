# SmartQueue – AI-Based Queue Waiting Time Predictor

A beginner-friendly final-year project prototype built with Python and Streamlit.

## Features
- Predict estimated waiting time from queue details
- Supports Fees, Examination, Admission and Library service types
- Dashboard with recent predictions
- SQLite prediction history
- CSV download of prediction history

## Important data note
The included training dataset is generated synthetically for demonstration. It is not real institutional queue data. Do not claim real-world accuracy based on this dataset. For a stronger project, collect actual observations and retrain the model.

## Setup (Windows)
1. Install Python 3.10 or newer.
2. Extract this ZIP.
3. Open Command Prompt inside the `SmartQueue_Project` folder.
4. Create a virtual environment:

   `python -m venv venv`

5. Activate it:

   `venv\\Scripts\\activate`

6. Install packages:

   `pip install -r requirements.txt`

7. Generate sample data:

   `python generate_data.py`

8. Train the ML model:

   `python train_model.py`

9. Run the app:

   `streamlit run app.py`

The app opens in your browser, usually at `http://localhost:8501`.

## Project structure
- `app.py` – Streamlit user interface and prediction workflow
- `generate_data.py` – creates labelled synthetic demo data
- `train_model.py` – trains and evaluates the Random Forest model
- `data/queue_data.csv` – generated training data
- `models/queue_model.joblib` – trained model (created after training)
- `data/queue_history.db` – prediction history database (created when app runs)

## Suggested 4-day schedule
- Day 1: Install Python packages, generate data, train model.
- Day 2: Run the app and understand each input and output.
- Day 3: Test multiple cases, collect real queue observations if possible, retrain.
- Day 4: Prepare report, PPT, screenshots and viva answers.
