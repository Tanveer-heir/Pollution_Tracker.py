import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_path = "city_day.csv"
data = pd.read_csv(file_path)

feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

data_clean = data.dropna(subset=feature_cols + ['AQI'])

X = data_clean[feature_cols]
y = data_clean['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")


sample_data = [[50, 60, 10, 20, 15, 1, 0.5, 5, 30, 0.1, 0.05, 0.02]]  
predicted_aqi = model.predict(sample_data)
print("Predicted AQI for sample data:", predicted_aqi[0])
