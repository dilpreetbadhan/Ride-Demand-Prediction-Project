Ride Demand Prediction Using Weather and Time Features
Project Overview
This project aims to analyze and predict ride-sharing demand based on weather conditions and time-based features. By integrating ride data with weather data, we explore how factors like temperature, precipitation, and wind speed impact ride demand.

The project includes data cleaning, feature engineering, exploratory data analysis (EDA), and building a predictive model to forecast ride counts.

Objectives
Integrate and clean ride-sharing and weather datasets.
Perform exploratory data analysis (EDA) to identify patterns in ride demand.
Develop a predictive model to estimate ride counts using weather and time features.
Visualize insights and evaluate model performance.
Data Sources
Ride Data: Pre-collected ride data (cleaned_data.csv).
Weather Data: Collected using the Visual Crossing Weather API for New York City.
Steps Followed
1. Data Collection
Ride data (cleaned_data.csv) was provided.
Weather data for New York City (2022) was fetched via the Visual Crossing Weather API and saved as New York City Weather.csv.
2. Data Cleaning and Merging
Merged ride data with weather data on the pickup_datetime column.
Kept relevant columns like temp, precip, windspeed, humidity, and conditions.
3. Handling Missing Data
Missing weather values were handled using linear interpolation.
4. Feature Engineering
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

Monthly Ride Demand: Number of rides per month.
Rainy vs Non-Rainy Days: Comparison of ride demand on rainy and clear days.
Weekend vs Weekday Demand: Ride trends on weekends vs weekdays.
Temperature and Wind Impact:
Scatter plots showing the relationship between temp, windspeed, and ride counts.
6. Predictive Modeling
Built a Linear Regression model using:
Features: is_weekend, is_rainy, is_windy, is_humid, month
Target: ride_count
Evaluated the model using:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Visualized Actual vs Predicted Ride Counts.
Visualizations
1. Monthly Ride Demand

2. Ride Demand: Rainy vs Non-Rainy Days

3. Ride Demand: Weekend vs Weekday

4. Temperature vs Ride Count

5. Wind Speed vs Ride Count

6. Actual vs Predicted Ride Counts

Project Structure
The project directory is organized as follows:

bash
Copy code
Ride_Demand_Prediction/
│
├── cleaned_data.csv                 # Original Ride Data
├── New York City Weather.csv        # Weather Data collected via API
├── refined_merged_data.csv          # Merged and Cleaned Data
├── interpolated_data.csv            # Data with Missing Values Filled
├── feature_engineered_data.csv      # Dataset with New Features
├── ride_demand_analysis.py          # Complete Python Code
├── Figures/                         # Folder for Visualizations
│   ├── monthly_demand.png           # Monthly Demand Plot
│   ├── rainy_vs_nonrainy.png        # Rainy vs Non-Rainy Plot
│   ├── weekend_vs_weekday.png       # Weekend vs Weekday Plot
│   ├── temp_vs_rides.png            # Temperature Impact
│   ├── windspeed_vs_rides.png       # Wind Speed Impact
│   ├── actual_vs_predicted.png      # Actual vs Predicted Ride Counts
│
└── README.md                        # Project Documentation
How to Run the Project
1. Dependencies
Install the required libraries using pip:

bash
Copy code
pip install pandas matplotlib scikit-learn requests
2. Run the Script
Run the ride_demand_analysis.py script to execute all steps:

bash
Copy code
python ride_demand_analysis.py
3. Outputs
Visualizations will be saved in the Figures/ folder.
Cleaned datasets will be saved as intermediate files:
refined_merged_data.csv
interpolated_data.csv
feature_engineered_data.csv
Model Performance
Mean Absolute Error (MAE): ~58.39
Root Mean Squared Error (RMSE): ~69.05
The model provides a reasonable estimate of ride counts based on weather and time features.

Next Steps for Improvement
Explore advanced models like Random Forest or XGBoost for better accuracy.
Incorporate additional features like traffic data or holidays.
Use hourly weather data to improve time-based granularity.
Contributors
Your Name Dilpreet Badhan
