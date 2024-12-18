# Ride Demand Prediction Project

## **Project Overview**
This project focuses on predicting ride-sharing demand using weather and time-based features. The dataset contains ride information, weather conditions, and other time-based attributes for New York City.

## **Objectives**
1. Explore the impact of weather and time-based features on ride demand.
2. Build a predictive model to estimate ride demand based on the available data.

## **Steps Followed**
1. **Data Preparation**:
   - Cleaned and preprocessed data to handle missing values.
   - Added new features like `month` and weather-based categories (`is_rainy`, `is_humid`).

2. **Exploratory Data Analysis**:
   - Analyzed the impact of weather conditions (rainy days, windy days) and day type (weekends vs weekdays).

3. **Modeling**:
   - Used Linear Regression for prediction.
   - Evaluated the model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## **Key Findings**
- **Weekends vs Weekdays**:
  - Ride demand is higher on weekends but drops significantly on rainy weekends.
- **Weather Impact**:
  - Non-rainy days see significantly higher demand compared to rainy days.
- **Final Model Performance**:
  - MAE: 58.39
  - RMSE: 69.05

## **Results Visualization**
![Actual vs Predicted](Figure_10.png)

## **Next Steps**
- Enhance the model by exploring advanced algorithms like Decision Trees or Random Forests.
- Include additional features like `season` or time-specific metrics.

## **How to Run**
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
