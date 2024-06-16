import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
actual_price = pd.read_csv("sample_submission.csv")

train_data = train_data[['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','SalePrice']]
test_data = test_data[['Id','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr']]
test_data = test_data.dropna()

actual_price = actual_price[actual_price['Id'].isin(test_data['Id'])]
actual_price = actual_price[['SalePrice']]

test_data = test_data.drop('Id', axis=1)

X = train_data.drop('SalePrice', axis=1)
y = train_data[['SalePrice']]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(test_data)

mse = mean_squared_error(actual_price, predictions)
mae = mean_absolute_error(actual_price, predictions)
print('Mean Squared Error: ', mse)
print('Mean Absolute Error: ', mae)

plt.figure(figsize=(10, 6))
plt.plot(actual_price.values, label='Actual Sale Price', color='blue', linewidth=2)
plt.plot(predictions, label='Predicted Sale Price', color='red', linestyle='--', linewidth=2)
plt.xlabel('Sample Index')
plt.ylabel('Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.legend()
plt.show()
