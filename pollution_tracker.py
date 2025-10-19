import pandas as pd # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
import matplotlib.pyplot as plt
import joblib

file_path = "city_day.csv"
data = pd.read_csv(file_path)

feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
data_clean = data.dropna(subset=feature_cols + ['AQI'])

print("\nDataset Shape:", data_clean.shape)
print("\nMissing values:\n", data_clean[feature_cols + ['AQI']].isnull().sum())
print("\nFeature Summary:\n", data_clean[feature_cols].describe())

X = data_clean[feature_cols]
y = data_clean['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {"n_estimators": [100, 200], "max_depth": [None, 7, 15]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("\nBest Model Params:", grid_search.best_params_)

best_model.fit(X_train, y_train)

importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:\n", importances)

predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

plt.figure(figsize=(7,4))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()

joblib.dump(best_model, "rf_aqi_model.joblib")
print("\nModel saved as rf_aqi_model.joblib")

print("\nEnter feature values for prediction (comma separated):")
user_input = input(f"{feature_cols}\n").split(',')
sample_data = [float(i) for i in user_input]
predicted_aqi = best_model.predict([sample_data])
print("Predicted AQI for input data:", predicted_aqi[0])

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validated MAE: {-cv_scores.mean():.2f}")

if hasattr(best_model, 'estimators_'):
    all_preds = np.array([estimator.predict([sample_data]) for estimator in best_model.estimators_])
    print("Prediction mean ± std:", all_preds.mean(), "±", all_preds.std())
