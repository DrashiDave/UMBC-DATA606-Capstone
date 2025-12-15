# UMBC DATA606 Capstone – Flight Delay Prediction Proposal

**Project Title:** Predicting U.S. Flight Delays and Delay Duration  
**Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  
**Author:** Drashi Dave  
**GitHub Repository:** [https://github.com/DrashiDave/UMBC-DATA606-Capstone/Data/Airline_Delay_Cause.csv](https://github.com/DrashiDave/UMBC-DATA606-Capstone/Data/Airline_Delay_Cause.csv)  
**LinkedIn Profile:** [linkedin.com/in/drashi-d](https://www.linkedin.com/in/drashi-d)  
**PowerPoint Presentation:** [https://umbc-my.sharepoint.com/:p:/g/personal/drashid1_umbc_edu/IQCoLbsvpiUAQ44JQFHHRqQgAYwxCQjTmzcD5KOySLzvdB4?e=8fWlfL](https://umbc-my.sharepoint.com/:p:/g/personal/drashid1_umbc_edu/IQCoLbsvpiUAQ44JQFHHRqQgAYwxCQjTmzcD5KOySLzvdB4?e=8fWlfL)

**Youtube Link:** [https://www.youtube.com/watch?v=la8tpCWnTOQ](https://www.youtube.com/watch?v=la8tpCWnTOQ)

---

## Background

Flight delays impact travelers, airlines, and airport operations through missed connections, crew reassignments, and cascading congestion.  
This project analyzes U.S. domestic flight performance data to understand **when/where delays occur** and to build **machine learning models** that predict delay outcomes using features available before departure.

---

## Data Sources

- **Primary dataset:** U.S. DOT / BTS On-Time Performance (combined multi-year CSV exported into a single file)
- **Local file used:** `combined_flight_data.csv`
- **Cleaned dataset output:** `clean_flights.parquet` (saved at the end of the EDA notebook)

> Note: Cancelled and diverted flights were excluded to avoid distorting delay measures.

---

## Data Elements

### Target Variables (used in notebooks)

- **Regression target (minutes):**
  - `DepDelayMinutes` (departure delay minutes) used in the ML notebook regression section
  - `ArrDelayMinutes` analyzed in EDA (arrival delay minutes)
- **Classification target (binary):**
  - `IsDelayed = 1` if `DepDelayMinutes > 15`, else `0`
  - The 15-minute threshold follows the common DOT “delayed flight” interpretation.

### Key Features Used

Features selected were restricted to those reasonably available **before departure**:

- **Time & schedule:** `Month`, `DayOfWeek`, `CRSDepTime` → derived `DepHour`, `CRSElapsedTime`
- **Route:** `Origin`, `Dest`
- **Airline:** `Reporting_Airline`
- **Distance / flight characteristics:** `Distance`, `AirTime`, `ActualElapsedTime` _(note: some of these may not be known pre-departure depending on system availability)_

---

## Data Cleaning and Preparation

Major steps:

1. Filtered to **completed flights**: `Cancelled == 0` and `Diverted == 0`
2. Dropped non-informative or post-event fields (including diversion-related `Div*` fields)
3. Handled missing values:
   - Dropped rows missing critical identifiers (`Reporting_Airline`, `Origin`, `Dest`, `FlightDate`)
   - Filled missing numeric operational fields with `0` where appropriate
4. Saved cleaned output as Parquet for efficient downstream modeling:
   - `clean_flights.parquet`

---

## Exploratory Data Analysis Results

### Delay Distribution

- Departure and arrival delays show **heavy-tailed** distributions: many small delays with a few extreme outliers.
- Departure delays and arrival delays tend to move together: late departures often lead to late arrivals.

### Seasonal Trends (Month)

- Delays increase during **summer (June–July)** and **winter (January)**, consistent with demand peaks and weather impacts.
- Fall months (around **Oct–Nov**) tend to show smoother operations.

### Day of Week

- Average delays remain fairly consistent across weekdays with only modest differences, suggesting weekday alone is a weak predictor.

### Time of Day (Hour)

- Delays vary across the day and can show spikes in early morning / later periods due to operational waves and propagation.

### Airline & Airport Effects

- Carrier-level differences exist: some airlines consistently show higher average delay minutes than others.
- Certain airports show higher delayed-flight percentages, suggesting localized congestion, weather, or operational constraints.

### Correlation Insight

- Strong relationship between departure and arrival delays was observed in the EDA correlation heatmap:
  - `DepDelay` and `ArrDelay` were highly correlated (near 1.0 in the sampled correlation view).
- Distance-related features such as `Distance` and `AirTime` are also strongly correlated.

---

## Machine Learning Results

### Modeling Setup

- **Train/test split:** 80/20 on the prepared modeling dataset (with a fixed seed for reproducibility).
- **Preprocessing:** categorical features were indexed and assembled into a single `features` vector; models were trained using Spark ML.

---

### Regression Model: Predict Departure Delay Minutes (`DepDelayMinutes`)

- **Model:** Elastic Net Linear Regression
- **Goal:** Predict **continuous departure delay time (minutes)**

**Results (test set):**

- **RMSE:** 52.99
- **MAE:** 22.16
- **R²:** 0.013

> Low R² is common in delay regression tasks without real-time operational data.

**Interpretation:**  
The model’s average absolute error is about **22 minutes**, but the very low **R² (1.3%)** indicates that schedule/route-based features alone explain only a small portion of delay variability (large delays are driven heavily by factors not captured in the dataset, such as weather, ATC restrictions, and operational disruptions).

---

### Classification Model: Predict Whether a Flight Is Delayed (`IsDelayed`)

- **Definition:** `IsDelayed = 1` if `DepDelayMinutes > 15`, else `0`
- **Model:** Weighted Logistic Regression (to address class imbalance)

**Results (test set):**

- **Accuracy:** 60.06%
- **F1-score:** 0.641
- **ROC-AUC:** 0.641

**Interpretation:**  
The classifier performs **moderately better than random**, with ROC-AUC ≈ **0.64**, meaning it captures some predictive signal but still struggles to cleanly separate delayed vs. on-time flights using only the available features.

---

## Conclusion

This project demonstrates that:

- Flight delays show clear patterns by **season**, **time of day**, **airline**, and **airport**.
- Simple models using schedule- and route-based features can detect some predictive signal, but **delay prediction remains inherently uncertain** without richer operational context.
- Regression performance was modest (low R²), while classification achieved moderate discrimination (ROC-AUC ~0.64).

---

## Limitations

- **Feature availability:** Several strong predictors of delay are missing or not modeled (weather, air traffic constraints, runway/airport capacity, maintenance events).
- **Target ambiguity:** Delay minutes can include extreme outliers; even with filtering, heavy tails can hurt regression stability.
- **Sampling:** Local training used a fraction of the dataset, which may affect generalization.
- **Cardinality constraints:** Simplifying airports to Top-50 + Other reduces detail and may hide localized effects.

---

## Future Research Directions

- Add external features:
  - Weather data (precipitation, visibility, wind, storms)
  - Airport congestion signals (hourly departures/arrivals, historical queue indicators)
- Improve modeling:
  - Quantile regression or robust loss functions for heavy-tailed delays
  - Gradient boosted trees with careful handling of categorical features
- Target refinement:
  - Predict **arrival delay minutes** directly (or both dep/arr) with separate models
  - Multi-task learning to jointly learn dep + arr delay minutes
- Evaluation improvements:
  - Time-based split (train on earlier months/years, test on later) for realistic deployment

---
