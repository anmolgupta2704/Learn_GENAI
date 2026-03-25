# ----------------------------------------
# HOUSE + CAR PRICE PREDICTION (IMPROVED)
# ----------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ========================================
# PART 1: HOUSE PRICE PREDICTION
# ========================================

print("\n===== HOUSE PRICE PREDICTION =====")

# Step 1: Dataset
house_data = {
    "area": [800, 1200, 1500, 1800, 2200, 2600, 3000, 3200, 3500],
    "bedrooms": [2, 2, 3, 3, 4, 4, 5, 5, 6],
    "age": [20, 15, 10, 8, 5, 3, 1, 2, 1],
    "price": [15, 25, 35, 45, 55, 65, 75, 82, 90]
}

house_df = pd.DataFrame(house_data)

# Step 2: Features & target
X_house = house_df[["area", "bedrooms", "age"]]
y_house = house_df["price"]

# Step 3: Model
house_model = LinearRegression()

# Step 4: Train on FULL data (better for small dataset)
house_model.fit(X_house, y_house)

# Step 5: Predict on same data (demo purpose)
y_pred = house_model.predict(X_house)

print("Actual:", y_house.values)
print("Predicted:", y_pred)

# Step 6: Evaluation
mse = mean_squared_error(y_house, y_pred)
r2 = r2_score(y_house, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Step 7: Predict new house (FIXED WARNING)
new_house = pd.DataFrame([[2000, 3, 7]],
                         columns=["area", "bedrooms", "age"])

predicted_price = house_model.predict(new_house)

print("Predicted Price:", predicted_price[0], "lakhs")


# ========================================
# PART 2: CAR PRICE PREDICTION
# ========================================

print("\n===== CAR PRICE PREDICTION =====")

# Step 1: Dataset
car_data = {
    "year": [2012, 2015, 2018, 2020, 2016, 2019, 2021, 2022, 2023],
    "mileage": [70000, 50000, 30000, 20000, 45000, 25000, 15000, 10000, 8000],
    "engine": [1000, 1200, 1500, 1800, 1300, 1600, 2000, 2200, 2500],
    "price": [3, 5, 8, 12, 6, 9, 14, 16, 18]
}

car_df = pd.DataFrame(car_data)

# Step 2: Features & target
X_car = car_df[["year", "mileage", "engine"]]
y_car = car_df["price"]

# Step 3: Model
car_model = LinearRegression()

# Step 4: Train
car_model.fit(X_car, y_car)

# Step 5: Predict
y_pred_c = car_model.predict(X_car)

print("Actual:", y_car.values)
print("Predicted:", y_pred_c)

# Step 6: Evaluation
mse_c = mean_squared_error(y_car, y_pred_c)
r2_c = r2_score(y_car, y_pred_c)

print("Car MSE:", mse_c)
print("Car R2 Score:", r2_c)

# Step 7: Predict new car (FIXED WARNING)
new_car = pd.DataFrame([[2019, 30000, 1500]],
                       columns=["year", "mileage", "engine"])

predicted_car_price = car_model.predict(new_car)

print("Predicted Car Price:", predicted_car_price[0], "lakhs")