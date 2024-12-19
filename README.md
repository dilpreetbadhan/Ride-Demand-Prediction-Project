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
  
1. Monthly Ride Demand ![Figure_1](https://github.com/user-attachments/assets/962d03ac-f348-43b1-901f-060f229fcfb1)

2. Ride Demand: Rainy vs Non-Rainy Days  ![Figure_6](https://github.com/user-attachments/assets/11c544be-7ece-447d-a74f-338dc4da3fec)

3. Ride Demand: Weekend vs Weekday ![Figure_5](https://github.com/user-attachments/assets/10320f68-fa74-4795-bc0e-951bec8a9ee5)

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
