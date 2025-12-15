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
This project aims to predict the flight delay in minutes (regression) which will include both the Departure delay and Arrival Delay.

**Why it Matters:**  
Accurate predictions of flight delays help:

- Passengers plan their travel more efficiently
- Airlines optimize schedules, staffing, and resource allocation
- Airports and air traffic controllers manage congestion

**Research Questions:**

1. Can we predict whether a flight will be delayed by more than 15 minutes?
2. If a flight is delayed, can we predict the expected delay duration in minutes?
3. Which factors (airline, airport, time of year, weather, etc.) contribute most to delays?

<br>

## Data

**Data Source:**

The dataset was created by downloading monthly flight on-time performance data from the U.S. Department of Transportation’s Bureau of Transportation Statistics (BTS) for the period January 2023 – May 2025.

- U.S. Bureau of Transportation Statistics - On-Time Performance Dataset (Open-source, publicly available):
  [https://www.transtats.bts.gov/](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
- Local Copy (for convenience): Available in this GitHub repository  
  [https://github.com/DrashiDave/UMBC-DATA606-Capstone/blob/main/Data/combined_flight_data.csv](https://github.com/DrashiDave/UMBC-DATA606-Capstone/blob/main/Data/combined_flight_data.csv)

Each monthly ZIP file was extracted and merged using a Python script to create one consolidated dataset for analysis and modeling.

**Data Overview:**

- Time Period: January 2023 – May 2025
- Rows: ~4.2 million
- Columns: 29
- File Size: ~1.2 GB
- Each record represents an individual flight’s departure and arrival details with delay indicators and related attributes.

**Column Data Types and Dictionary:**

| Column Name         | Data Type | Description                                           | Example Value |
| ------------------- | --------- | ----------------------------------------------------- | ------------- |
| `Year`              | Integer   | Year of the flight                                    | 2024          |
| `Month`             | Integer   | Month of the flight (1–12)                            | 5             |
| `DayofMonth`        | Integer   | Day of the month                                      | 14            |
| `DayOfWeek`         | Integer   | Day of the week (1 = Monday, 7 = Sunday)              | 3             |
| `FlightDate`        | Date      | Flight date                                           | 2024-05-14    |
| `Carrier`           | String    | Airline carrier code                                  | AA            |
| `Origin`            | String    | Departure airport code                                | JFK           |
| `Dest`              | String    | Arrival airport code                                  | LAX           |
| `DepTime`           | Float     | Actual departure time (HHMM, local)                   | 1820          |
| `CRSDepTime`        | Float     | Scheduled departure time (HHMM, local)                | 1800          |
| `ArrTime`           | Float     | Actual arrival time (HHMM, local)                     | 2135          |
| `CRSArrTime`        | Float     | Scheduled arrival time (HHMM, local)                  | 2105          |
| `DepDelay`          | Float     | Departure delay in minutes (negative = early)         | 20.0          |
| `ArrDelay`          | Float     | Arrival delay in minutes (negative = early)           | 30.0          |
| `Cancelled`         | Integer   | Flight cancelled (1 = Yes, 0 = No)                    | 0             |
| `Diverted`          | Integer   | Flight diverted (1 = Yes, 0 = No)                     | 0             |
| `Distance`          | Float     | Distance between origin and destination (miles)       | 2475.0        |
| `AirTime`           | Float     | Actual flight time in minutes                         | 305.0         |
| `TaxiOut`           | Float     | Taxi-out time before takeoff                          | 18.0          |
| `TaxiIn`            | Float     | Taxi-in time after landing                            | 10.0          |
| `CRSElapsedTime`    | Float     | Scheduled total flight duration (minutes)             | 330.0         |
| `ActualElapsedTime` | Float     | Actual total flight duration (minutes)                | 335.0         |
| `CarrierDelay`      | Float     | Delay minutes caused by the airline                   | 12.0          |
| `WeatherDelay`      | Float     | Delay minutes caused by weather                       | 5.0           |
| `NASDelay`          | Float     | Delay minutes caused by National Air System           | 8.0           |
| `SecurityDelay`     | Float     | Delay minutes caused by security checks or procedures | 0.0           |
| `LateAircraftDelay` | Float     | Delay due to previous aircraft arriving late          | 15.0          |
| `DepDelay15`        | Integer   | 1 if departure delay > 15 minutes, else 0             | 1             |
| `ArrDelay15`        | Integer   | 1 if arrival delay > 15 minutes, else 0               | 1             |

<!--**Categorical Variables:**

- `carrier_name`: 23 unique values
- `airport`: 385 unique values-->
<br>

## Target Variables and Feature Candidates:

1. Will the flight be delayed?
   → Binary classification (ArrDelay15 and/or DepDelay15)

2. If delayed, by how many minutes?
   → Regression (ArrDelay and/or DepDelay)

<br>

| **Type**                                                                                                                                                                                                      | **Variable(s)**                                                                                                                                                                                               | **Description**                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------                                                                                                                                                    |
| **Target (Classification)**                                                                                                                                                                                   | `DepDelay15`, `ArrDelay15`                                                                                                                                                                                    | 1 if delay > 15 minutes for departure or arrival, else 0   |
| **Target (Regression)**                                                                                                                                                                                       | `DepDelay`, `ArrDelay`                                                                                                                                                                                        | Continuous delay time in minutes for departure and arrival |
| **Predictors (Features)**                                                                                                                                                                                     | `AirTime`, `Distance`, `TaxiOut`, `TaxiIn`, `CRSElapsedTime`, `ActualElapsedTime`, `CarrierDelay`, `WeatherDelay`, `NASDelay`, `LateAircraftDelay`, `Year`, `Month`, `DayOfWeek`, `Carrier`, `Origin`, `Dest` | Used to predict both arrival and departure delays          |
