 # Ride Demand Prediction Using Weather and Time Features

## Project Overview
This project explores the influence of weather conditions and time-based factors on ride-sharing demand. By integrating and analyzing ride data with weather data, we aim to predict ride demand and understand how environmental factors impact it. The project encompasses data cleaning, exploratory data analysis (EDA), feature engineering, and predictive modeling.

## Objectives
1. Integrate and clean ride-sharing and weather datasets.
2. Conduct exploratory data analysis (EDA) to uncover patterns in ride demand.
3. Develop a predictive model to forecast ride counts using weather and time features.
4. Visualize insights and evaluate the performance of the model.

## Data Sources
1. **Ride Data**: Pre-collected ride data (`cleaned_data.csv`).
2. **Weather Data**: Weather data for New York City, collected using the Visual Crossing Weather API (`New York City Weather.csv`).

## Steps Followed
### Data Collection
- **Ride Data**: Loaded from `cleaned_data.csv`.
- **Weather Data**: Fetched for New York City (2022) via the Visual Crossing Weather API and saved as `New York City Weather.csv`.

### Data Cleaning and Merging
- Merged ride and weather data on `pickup_datetime`.
- Retained relevant columns like `temp`, `precip`, `windspeed`, `humidity`, and `conditions`.

### Handling Missing Data
- Employed linear interpolation to manage missing weather values.

### Feature Engineering
- Added time-based and weather flags:
  - **Time-Based Features**: `month`, `is_weekend`.
  - **Weather Flags**: `is_rainy`, `is_windy`, `is_humid`.

### Exploratory Data Analysis (EDA)
- Visualized data to assess:
  - Monthly Ride Demand
  - Ride Demand on Rainy vs. Non-Rainy Days
  - Weekend vs. Weekday Demand
  - Impact of Temperature and Wind on Ride Counts

### Predictive Modeling
- Built a Linear Regression model using:
  - **Features**: `is_weekend`, `is_rainy`, `is_windy`, `is_humid`, `month`.
  - **Target**: `ride_count`.
- Evaluated the model using MAE and RMSE.
- Visual comparisons of Actual vs. Predicted Ride Counts.

## Visualizations
1. **Monthly Ride Demand**  
   ![Monthly Ride Demand](https://github.com/user-attachments/assets/962d03ac-f348-43b1-901f-060f229fcfb1)

2. **Ride Demand: Rainy vs Non-Rainy Days**  
   ![Ride Demand Rainy vs Non-Rainy Days](https://github.com/user-attachments/assets/11c544be-7ece-447d-a74f-338dc4da3fec)

3. **Weekend vs Weekday Ride Demand**  
   ![Weekend vs Weekday](https://github.com/user-attachments/assets/10320f68-fa74-4795-bc0e-951bec8a9ee5)

4. **Weekend vs Weekday (Rainy vs Non-Rainy)**  
   ![Weekend vs Weekday Rainy vs Non-Rainy](https://github.com/user-attachments/assets/f7112a17-1b1b-41a7-8ef8-c19ec7caf3dd)

5. **Actual vs Predicted Ride Counts**  
   ![Actual vs Predicted](https://github.com/user-attachments/assets/435fc950-8a7f-4e39-ab8e-3baf9bdf2a43)


