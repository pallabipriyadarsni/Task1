'''Task 1:-
Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.'''

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load or Create Sample Data
data = {
    'square_feet': [1500, 1800, 2400, 3000, 3500, 1200, 2000, 2700, 3200, 4000],
    'bedrooms': [3, 4, 3, 5, 4, 2, 3, 4, 5, 4],
    'bathrooms': [2, 2, 3, 4, 3, 1, 2, 3, 4, 3],
    'price': [400000, 500000, 600000, 700000, 750000, 300000, 450000, 620000, 720000, 800000]
}
df = pd.DataFrame(data)

# Show dataset info
print("Dataset:\n", df.head())

# Features and Target
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Visualize Predictions
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.show()

# Predict a new house price
new_house = pd.DataFrame([[2600, 4, 3]], columns=['square_feet', 'bedrooms', 'bathrooms'])
predicted_price = model.predict(new_house)
print(f"\n🏠 Predicted price for house (2600 sqft, 4 beds, 3 baths): ${predicted_price[0]:,.2f}")
