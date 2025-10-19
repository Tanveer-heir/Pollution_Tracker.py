# Air Quality Index (AQI) Prediction Using Random Forest

This project predicts the Air Quality Index (AQI) using a Random Forest regression model trained on real-world city pollution data (`city_day.csv`). The model analyzes various air pollutant features and provides interactive predictions based on user input.

---

## Table of Contents

- [Requirements](#requirements)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Making Predictions](#making-predictions)
- [Cross-Validation](#cross-validation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Author](#author)

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib


---

## How It Works

1. **Data Loading**: Reads city pollution data from `city_day.csv`.
2. **Data Cleaning**: Drops rows with missing values in essential features and AQI.
3. **Feature Engineering**: Selected pollutant features are used for prediction.
4. **Model Selection & Training**: Uses `RandomForestRegressor` with grid search and cross-validation to select the best hyperparameters.
5. **Evaluation**: Reports metrics including MAE, RMSE, MSE, and R² score.
6. **Visualization**: Plots actual vs predicted AQI.
7. **User Interaction**: Prompts user for input values to predict AQI for new samples.
8. **Model Export/Import**: Saves the trained model for later use.

---

## Usage

1. Place `city_day.csv` in the project directory.
2. Run the script:

---

## Model Training and Evaluation

- **Grid Search:** Optimizes `n_estimators` and `max_depth`.
- **Metrics:** Outputs MAE, RMSE, MSE, and R² after training.
- **Feature Importance:** Displays key pollutant contributors to AQI.

---

## Making Predictions

- Enter feature values separated by commas when prompted.
- The order must be:
['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

- You will receive the predicted AQI as output.
  
---

## Cross-Validation

Performs 5-fold cross-validation and displays the mean absolute error (MAE) across folds.

---

## Saving and Loading the Model

- The trained model is saved as `rf_aqi_model.joblib`.
- To use the model later, load with joblib


---

## Author

Created by Tanveer Singh
If you have questions or suggestions, feel free to raise an issue or contribute!

---

