import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 1. Data collection and preprocessing
data = pd.read_csv("tesla_annual_gross_profit.csv")

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Normalize the data
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

Y = scalerY.fit_transform(data["Profit (Million of US Dollar)"].values.reshape(-1, 1))

X = scalerX.fit_transform(data["Year"].values.reshape(-1, 1))


look_back = 1

# 2. Designing and training the neural network
model = Sequential()
model.add(
    LSTM(units=50, return_sequences=True, input_shape=(1, look_back), activation="relu")
)
model.add(LSTM(units=50, activation="relu"))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

arr = [i for i in range(2018, 2031)]
arr_scaled = scalerX.transform(arr)
scaled_future_profit = model.predict(arr_scaled)

predicted_future_profit = scalerY.inverse_transform(scaled_future_profit)

for year, profit in zip(arr, predicted_future_profit.flatten()):
    print(year, profit)


# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(data["Year"], data["Profit (Million of US Dollar)"], label="Actual Profit")
plt.plot(data["Year"], data["Profit (Million of US Dollar)"], label="Actual Profit")
plt.plot(arr, predicted_future_profit.flatten(), label="Predicted Profit (Future)")
plt.xlabel("Year")
plt.ylabel("Profit (Million of US Dollar)")
plt.title("Tesla's Annual Gross Profit")
plt.legend()
plt.show()
