Ride Demand Prediction Using Weather and Time Features

Project Overview

This project aims to analyze and predict ride-sharing demand based on weather conditions and time-based features. By integrating ride data with weather data, we explore how factors like temperature, precipitation, and wind speed impact ride demand.

The project includes data cleaning, feature engineering, exploratory data analysis (EDA), and building a predictive model to forecast ride counts.
 
Objectives

1.Integrate and clean ride-sharing and weather datasets.
2.Perform exploratory data analysis (EDA) to identify patterns in ride demand.
3.Develop a predictive model to estimate ride counts using weather and time features.
4.Visualize insights and evaluate model performance.

Data Sources

1.Ride Data: Pre-collected ride data (cleaned_data.csv).
2.Weather Data: Collected using the Visual Crossing Weather API for New York City.

Steps Followed

1. Data Collection
Ride data (cleaned_data.csv) was provided.

Weather data for New York City (2022) was fetched via the Visual Crossing Weather API and saved as New York City Weather.csv.

3. Data Cleaning and Merging
Merged ride data with weather data on the pickup_datetime column.
Kept relevant columns like temp, precip, windspeed, humidity, and conditions.

4. Handling Missing Data
Missing weather values were handled using linear interpolation.

5. Feature Engineering
New features were added:

Time-Based Features:
month: Month of the ride.
is_weekend: Indicates whether the ride occurred on a weekend.
Weather Flags:
is_rainy: Precipitation > 0.1
is_windy: Wind speed > 20 km/h
is_humid: Humidity > 70%

5. Exploratory Data Analysis (EDA)
Visualized the data to understand patterns:

  1.Monthly Ride Demand: Number of rides per month.
  2.Rainy vs Non-Rainy Days: Comparison of ride demand on rainy and clear days.
  3.Weekend vs Weekday Demand: Ride trends on weekends vs weekdays.
  4.Temperature and Wind Impact:
   Scatter plots showing the relationship between temp, windspeed, and ride counts.

6. Predictive Modeling
   
 Built a Linear Regression model using:
   Features: is_weekend, is_rainy, is_windy, is_humid, month
   Target: ride_count
 Evaluated the model using:
   Mean Absolute Error (MAE)
   Root Mean Squared Error (RMSE)
Visualized Actual vs Predicted Ride Counts.



* Visualizations
  
1. Monthly Ride Demand  ![Figure_1](https://github.com/user-attachments/assets/962d03ac-f348-43b1-901f-060f229fcfb1)

2. Ride Demand: Rainy vs Non-Rainy Days   ![Figure_6](https://github.com/user-attachments/assets/11c544be-7ece-447d-a74f-338dc4da3fec)

3. Ride Demand: Weekend vs Weekday  ![Figure_5](https://github.com/user-attachments/assets/10320f68-fa74-4795-bc0e-951bec8a9ee5)

4. Ride Demand: weekend vs Weekday(Rainy vs Non-Rainy)  ![Figure_7](https://github.com/user-attachments/assets/f7112a17-1b1b-41a7-8ef8-c19ec7caf3dd)
   
5. Actual vs Predicted Ride Counts   ![Final](https://github.com/user-attachments/assets/435fc950-8a7f-4e39-ab8e-3baf9bdf2a43)


Project Structure

The project directory is organized as follows:

 Ride_Demand_Prediction/

   cleaned_data.csv                 *Original Ride Data  
   
   New York City Weather.csv        * Weather Data collected via API
   
   refined_merged_data.csv          * Merged and Cleaned Data
   
   interpolated_data.csv            * Data with Missing Values Filled
   
   feature_engineered_data.csv      * Dataset with New Features
   
   ride_demand_analysis.py          * Complete Python Code


How to Run the Project
1. Dependencies
Install the required libraries using pip:

pip install pandas matplotlib scikit-learn requests

2. Run the Script

Run the ride_demand_analysis.py script to execute all steps:

 
python ride_demand_analysis.py
3. Outputs
Visualizations will be saved in the Figures/ folder.
Cleaned datasets will be saved as intermediate files:
refined_merged_data.csv
interpolated_data.csv
feature_engineered_data.csv


*Model Performance

Mean Absolute Error (MAE): ~58.39
Root Mean Squared Error (RMSE): ~69.05
The model provides a reasonable estimate of ride counts based on weather and time features.

*Next Steps for Improvement

Explore advanced models like Random Forest or XGBoost for better accuracy.
Incorporate additional features like traffic data or holidays.
Use hourly weather data to improve time-based granularity.

Contributors
Your Name Dilpreet Badhan




PYTHON CODE 

# Ride Demand Prediction Project: Complete Python Code

import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# 1. LOAD AND EXPLORE DATA
# -------------------------------
# Load Ride Data

print("Step 1: Loading Ride Data...")
ride_data = pd.read_csv("cleaned_data.csv")  # Original ride data
print("Ride Data Loaded Successfully!")
print("\nRide Data Preview:")
print(ride_data.head())

# Load Weather Data

print("\nLoading Weather Data...")
weather_data = pd.read_csv("New York City Weather.csv")  # Weather data collected via API
print("Weather Data Loaded Successfully!")
print("\nWeather Data Preview:")
print(weather_data.head())

# Check basic statistics

print("\nRide Data Summary:")
print(ride_data.describe())

print("\nWeather Data Summary:")
print(weather_data.describe())











# -------------------------------
# 2. FETCH WEATHER DATA FROM API
# -------------------------------

def fetch_weather_data():
    print("\nStep 2: Fetching Weather Data from API...")
    # API endpoint and parameters
    api_key = "YOUR_API_KEY"  # Replace with your Visual Crossing API key
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/New+York+City,USA/2022-01-01/2022-12-31?unitGroup=metric&key={api_key}&contentType=csv"
    
    # Fetch data
    response = requests.get(url)
    if response.status_code == 200:
        with open("New_York_City_Weather.csv", "wb") as f:
            f.write(response.content)
        print("Weather data saved as 'New_York_City_Weather.csv'.")
    else:
        print(f"Failed to fetch weather data. Status Code: {response.status_code}")

# Uncomment the following line to fetch weather data via API
# fetch_weather_data()



# -------------------------------
# 3. DATA CLEANING AND MERGING
# -------------------------------


print("\nStep 3: Cleaning and Merging Data...")
# Merge the two datasets on pickup_datetime
merged_data = pd.merge(ride_data, weather_data, left_on="pickup_datetime", right_on="datetime", how="left")

# Keep only necessary columns
columns_to_keep = ["pickup_datetime", "PUlocationID", "temp", "precip", "windspeed", "humidity", "conditions"]
merged_data = merged_data[columns_to_keep]

# Save the merged and cleaned data
merged_data.to_csv("refined_merged_data.csv", index=False)
print("Merged and cleaned data saved as 'refined_merged_data.csv'.")


# -------------------------------
# 4. HANDLING MISSING DATA

# -------------------------------
print("\nStep 4: Handling Missing Data...")
# Load the merged data
data = pd.read_csv("refined_merged_data.csv")

# Convert pickup_datetime to datetime
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

# Resample data monthly and interpolate missing values
data_resampled = data.resample("M", on="pickup_datetime").sum()
data_resampled = data_resampled.interpolate(method="linear")

# Save the interpolated data
data_resampled.to_csv("interpolated_data.csv", index=False)
print("Missing values filled and saved as 'interpolated_data.csv'.")



# -------------------------------
# 5. FEATURE ENGINEERING
# -------------------------------


print("\nStep 5: Feature Engineering...")
# Reload the data
data = pd.read_csv("interpolated_data.csv")

# Add month feature
data["month"] = pd.to_datetime(data["pickup_datetime"]).dt.month

# Add is_weekend feature
data["day_of_week"] = pd.to_datetime(data["pickup_datetime"]).dt.day_name()
data["is_weekend"] = data["day_of_week"].isin(["Saturday", "Sunday"])

# Add weather flags
data["is_rainy"] = data["precip"] > 0.1
data["is_windy"] = data["windspeed"] > 20
data["is_humid"] = data["humidity"] > 70

# Save feature-engineered data
data.to_csv("feature_engineered_data.csv", index=False)
print("Feature-engineered data saved as 'feature_engineered_data.csv'.")


# -------------------------------
# 6. EXPLORATORY DATA ANALYSIS
# -------------------------------

print("\nStep 6: Exploratory Data Analysis...")
# Load feature-engineered data
data = pd.read_csv("feature_engineered_data.csv")

# Monthly Ride Demand
monthly_demand = data.groupby("month")["PUlocationID"].count()
monthly_demand.plot(kind="bar", color="skyblue", figsize=(8, 5))
plt.title("Monthly Ride Demand")
plt.xlabel("Month")
plt.ylabel("Number of Rides")
plt.savefig("Figures/monthly_demand.png")
plt.show()

# Ride Demand on Rainy vs Non-Rainy Days
rainy_impact = data.groupby("is_rainy")["PUlocationID"].count()
rainy_impact.plot(kind="bar", color=["orange", "blue"], figsize=(6, 4))
plt.title("Ride Demand on Rainy vs Non-Rainy Days")
plt.xlabel("Rainy (True/False)")
plt.ylabel("Number of Rides")
plt.xticks([0, 1], ["Non-Rainy", "Rainy"], rotation=0)
plt.savefig("Figures/rainy_vs_nonrainy.png")
plt.show()



# -------------------------------
# 7. PREDICTIVE MODELING
# -------------------------------


print("\nStep 7: Predictive Modeling...")
# Define Features and Target
X = data[["is_weekend", "is_rainy", "is_windy", "is_humid", "month"]]
y = data["PUlocationID"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)



# Predict on Test Data
y_pred = model.predict(X_test)

# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Visualize Actual vs Predicted Ride Counts
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Ride Count")
plt.ylabel("Predicted Ride Count")
plt.title("Actual vs Predicted Ride Counts")
plt.savefig("Figures/actual_vs_predicted.png")
plt.show()




