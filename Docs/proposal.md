# UMBC DATA606 Capstone – Flight Delay Prediction Proposal

**Project Title:** Predicting U.S. Flight Delays and Delay Duration  
**Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  
**Author:** Drashi Dave  
**GitHub Repository:** [https://github.com/DrashiDave/UMBC-DATA606-Capstone/Data/Airline_Delay_Cause.csv](https://github.com/DrashiDave/UMBC-DATA606-Capstone/Data/Airline_Delay_Cause.csv)  
**LinkedIn Profile:** [linkedin.com/in/drashi-d](https://www.linkedin.com/in/drashi-d)  
**PowerPoint Presentation:** [https://umbc-my.sharepoint.com/:p:/g/personal/drashid1_umbc_edu/Eagtuy-mJQBDjglAUcdGpCABBogjmE2vjhWwZzeK9uoaYA?e=VZWo6P](https://umbc-my.sharepoint.com/:p:/g/personal/drashid1_umbc_edu/Eagtuy-mJQBDjglAUcdGpCABBogjmE2vjhWwZzeK9uoaYA?e=VZWo6P)

<!-- **YouTube Video:** *TBD* -->

## Background

Flight delays are a common issue affecting passengers, airlines, and the broader air transportation system. They increase operational costs, reduce passenger satisfaction, and cause cascading disruptions across connecting flights. The U.S. Bureau of Transportation Statistics (BTS) collects detailed information on flight operations, including arrival delays and their causes.

**Project Objective:**  
This project aims to predict whether a flight will be delayed (classification) and, if delayed, estimate the length of the delay in minutes (regression).

**Why it Matters:**  
Accurate predictions of flight delays help:

- Passengers plan their travel more efficiently
- Airlines optimize schedules, staffing, and resource allocation
- Airports and air traffic controllers manage congestion

**Research Questions:**

1. Can we predict whether a flight will be delayed by more than 15 minutes?
2. If a flight is delayed, can we predict the expected delay duration in minutes?
3. Which factors (airline, airport, time of year, weather, etc.) contribute most to delays?

## Data

**Data Source:**

- U.S. Bureau of Transportation Statistics - On-Time Performance Dataset (Open-source, publicly available):
  [https://www.transtats.bts.gov/](https://www.transtats.bts.gov/)
- Local Copy (for convenience): Available in this GitHub repository  
  [https://github.com/DrashiDave/UMBC-DATA606-Capstone/tree/main/data/Airline_Delay_Cause.csv](https://github.com/DrashiDave/UMBC-DATA606-Capstone/tree/main/data/Airline_Delay_Cause.csv)

**Data Overview:**

- Dataset size: 16.5 MB
- Shape: ~100,447 rows × 21 columns
- Time period: January 2021 – May 2025
- Each row represents aggregated statistics for one airline at one airport during one month, including flight counts, delays, and causes of delays.

**Column Data Types and Dictionary:**

| Column Name         | Data Type | Definition                                                                                     | Example Values                                     |
| ------------------- | --------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| year                | int       | Year in YYYY format                                                                            | 2025                                               |
| month               | int       | Month in MM format (1-12)                                                                      | 5                                                  |
| carrier             | object    | Code assigned by US DOT to identify a unique airline carrier                                   | 9E                                                 |
| carrier_name        | object    | Unique airline holding and reporting under the same DOT certificate                            | Endeavor Air Inc.                                  |
| airport             | object    | Three-character airport code issued by US DOT                                                  | ABE                                                |
| airport_name        | object    | Full name of airport including location                                                        | Allentown/Bethlehem/Easton, PA: Lehigh Valley Intl |
| arr_flights         | float     | Total number of arriving flights                                                               | 92.0                                               |
| arr_del15           | float     | Arrival Delay Indicator ≥ 15 min. Difference between actual and scheduled arrival time         | 17.0                                               |
| carrier_ct          | float     | Number of delays caused by the airline                                                         | 4.8                                                |
| weather_ct          | float     | Number of delays caused by weather                                                             | 1.24                                               |
| nas_ct              | float     | Number of delays caused by National Air System                                                 | 4.29                                               |
| security_ct         | float     | Number of delays caused by security                                                            | 0.0                                                |
| late_aircraft_ct    | float     | Number of delays caused by late aircraft                                                       | 6.67                                               |
| arr_cancelled       | float     | Number of cancelled flights                                                                    | 4.0                                                |
| arr_diverted        | float     | Number of diverted flights                                                                     | 2.0                                                |
| arr_delay           | float     | Difference in minutes between scheduled and actual arrival time. Early arrivals show negative. | 1834.0                                             |
| carrier_delay       | float     | Average delay time caused by carrier (minutes)                                                 | 517.0                                              |
| weather_delay       | float     | Average delay time caused by weather (minutes)                                                 | 555.0                                              |
| nas_delay           | float     | Average delay time caused by NAS (minutes)                                                     | 283.0                                              |
| security_delay      | float     | Average delay time caused by security (minutes)                                                | 0.0                                                |
| late_aircraft_delay | float     | Average delay time caused by late aircraft (minutes)                                           | 479.0                                              |

<!--**Categorical Variables:**

- `carrier_name`: 23 unique values
- `airport`: 385 unique values-->

**Target Variables and Feature Candidates:**

**Classification:** `Delayed` – 1 if `arr_delay > 15`, else 0.

Flight is marked as delayed if difference in the scheduled and actual arrival time (`arr_delay`) is more than 15 minutes, otherwise on-time.

> **Note:** The U.S. Bureau of Transportation Statistics defines a flight as “delayed” if it arrives 15 minutes or more after the scheduled time. This threshold is the standard in aviation and ensures your model aligns with real-world definitions.

**Regression:** `arr_delay` – continuous delay time in minutes.

Regression predicts the actual delay of a flight in minutes.

**Features:**  
These features capture the airline, airport, time, flight volume, and causes of delays, which will help the model predict whether a flight will be delayed and estimate the delay duration.

- Airline (`carrier_name`) – patterns of delay for each airline
- Airport (`airport`) – delays specific to each airport
- Time (`year`, `month`) – seasonal and yearly trends
- Flight counts (`arr_flights`) – volume of flights that may influence delays
- Cause counts (`carrier_ct`, `weather_ct`, `nas_ct`, `security_ct`, `late_aircraft_ct`) – number of delays due to each cause
- Cancellation/diversion counts (`arr_cancelled`, `arr_diverted`) – disruptions that affect delay patterns

> **Explanation:** These features will allow the model to predict both whether a flight is delayed and the expected delay duration.
