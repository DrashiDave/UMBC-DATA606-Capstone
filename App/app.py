# App/app.py

import os
import sys
import json
import datetime as dt

# Keep Spark driver & workers on the same Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.classification import LogisticRegressionModel

# ───────────────────────────── Helpers ─────────────────────────────
def format_time_12h(t: dt.time) -> str:
    return t.strftime("%I:%M %p").lstrip("0")

def pretty_mins(x: float) -> str:
    return f"{x:.1f}"

def parse_hhmm(s: str) -> dt.time | None:
    try:
        return dt.datetime.strptime(s.strip(), "%H:%M").time()
    except Exception:
        return None

# ───────────────────────── Page settings ───────────────────────────
st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Flight Delay Predictor")

# ───────────────────────── Spark session ───────────────────────────
spark = SparkSession.builder.appName("FlightDelayApp").getOrCreate()

# ─────────────────────── Load models & metadata ────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
pp_model   = PipelineModel.load(os.path.join(MODELS_DIR, "pp_model"))
reg_model  = LinearRegressionModel.load(os.path.join(MODELS_DIR, "regression_elasticnet"))
clf_model  = LogisticRegressionModel.load(os.path.join(MODELS_DIR, "classifier_lr"))

with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    META = json.load(f)

THRESHOLD = float(META.get("threshold", 0.50))

CRS_META      = META.get("crs_from_distance", None)
CRS_SLOPE     = float(CRS_META.get("slope", 0.12)) if CRS_META else 0.12
CRS_INTERCEPT = float(CRS_META.get("intercept", 20.0)) if CRS_META else 20.0

# ─────────────────────────── Inputs ────────────────────────────────
airline_options = {
    "AA": "American Airlines", "AS": "Alaska Airlines",   "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",    "F9": "Frontier Airlines", "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",  "MQ": "Envoy Air",         "NK": "Spirit Airlines",
    "OH": "PSA Airlines",       "OO": "SkyWest Airlines",  "UA": "United Airlines",
    "WN": "Southwest Airlines", "YX": "Republic Airways",  "9E": "Endeavor Air"
}
airline_display = [f"{code} – {name}" for code, name in airline_options.items()]

with st.form("flight_form"):
    left, right = st.columns([1.2, 1.2])

    with left:
        sel = st.selectbox("Airline", airline_display, index=0)
        airline = sel.split(" – ")[0]
        origin  = st.text_input("Origin (IATA)", "JFK")
        dest    = st.text_input("Destination (IATA)", "LAX")
        flight_date = st.date_input("Flight Date", value=dt.date.today())

    with right:
        # Primary widget: exact minutes via 1-minute step
        dep_time = st.time_input(
            "Scheduled Departure Time",
            value=dt.time(10, 0),
            step=dt.timedelta(minutes=1)  # ← precise minutes in the picker
        )

        # Optional fallback for browsers that still render 15-min jumps:
        manual = st.checkbox("Enter time manually (HH:MM)", value=False)
        if manual:
            t_str = st.text_input("Time (24h HH:MM)", "10:00")
            t_parsed = parse_hhmm(t_str)
            if t_parsed is None:
                st.info("Format must be HH:MM (e.g., 10:45). Using 10:00 for now.")
                dep_time = dt.time(10, 0)
            else:
                dep_time = t_parsed

        distance = st.number_input("Distance (miles)", value=2475.0, step=1.0)

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ─────────────────────────── Predict ───────────────────────────────
if submitted:
    # Derive Month and DOT DayOfWeek (1=Mon .. 7=Sun)
    month_derived = int(flight_date.month)
    dow_derived   = int(flight_date.isoweekday())

    # UI allows minute-precision; model uses only the hour
    dep_hour = dep_time.hour

    # Derive CRS Elapsed Time; proxy the post-flight fields to match training schema
    crs_et = CRS_SLOPE * float(distance) + CRS_INTERCEPT
    actual_elapsed_proxy = float(crs_et)
    airtime_proxy        = max(float(crs_et) - 30.0, 0.0)

    schema = StructType([
        StructField("Reporting_Airline", StringType(), True),
        StructField("Origin_S",          StringType(), True),
        StructField("Dest_S",            StringType(), True),
        StructField("Month",             IntegerType(), True),
        StructField("DayOfWeek",         IntegerType(), True),
        StructField("DepHour",           IntegerType(), True),
        StructField("Distance",          DoubleType(), True),
        StructField("CRSElapsedTime",    DoubleType(), True),
        StructField("ActualElapsedTime", DoubleType(), True),
        StructField("AirTime",           DoubleType(), True),
    ])

    row = [(
        airline, origin, dest,
        month_derived, dow_derived, int(dep_hour),
        float(distance), float(crs_et),
        float(actual_elapsed_proxy), float(airtime_proxy)
    )]

    input_df = spark.createDataFrame(row, schema=schema)

    # Preprocess → features
    feat_df = pp_model.transform(input_df).select("features")

    # Step 1: classification (Delayed / On-Time)
    prob_vec = clf_model.transform(feat_df).select("probability").first()[0]
    p_delay = float(prob_vec[1])
    is_delayed = (p_delay >= THRESHOLD)

    # Step 2: regression (minutes), only if delayed
    est_delay_min = 0.0
    if is_delayed:
        pred = reg_model.transform(feat_df).select("prediction").first()[0]
        est_delay_min = max(0.0, float(pred))

    # ───────────────────────── Results ─────────────────────────────
    st.subheader("Result")
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.metric("Flight Status", "🟠 Delayed" if is_delayed else "🟢 On-Time")
    with c2:
        st.metric("Estimated Delay (min)", pretty_mins(est_delay_min))
    with c3:
        st.metric("Scheduled Departure", format_time_12h(dep_time))

    st.divider()
    