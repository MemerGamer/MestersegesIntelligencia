import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 1. Data collection and preprocessing
data = pd.read_csv("hungarian_population_romania.csv")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
population_scaled = scaler.fit_transform(data["Population"].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(population_scaled) * 0.8)
train_data = population_scaled[0:train_size, :]
test_data = population_scaled[train_size : len(population_scaled), :]


# Create the training data structure
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i : (i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 1
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Reshape the input data
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 2. Designing and training the neural network
model = Sequential()
model.add(
    LSTM(units=50, return_sequences=True, input_shape=(1, look_back), activation="relu")
)
model.add(LSTM(units=50, activation="relu"))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# 3. Making predictions and evaluating the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Extend the prediction until the future years
last_test_year = data["Year"].iloc[train_size + len(test_predict) - 1]
future_start_year = last_test_year + 1
future_years = np.arange(future_start_year, 2051 + 10)  # Extend to 2060

# Make predictions for the extended years
extended_population = []
last_input = test_data[
    -look_back:
]  # Use the last look_back population values from the test data as input
for _ in range(len(future_years)):
    extended_pred = model.predict(np.reshape(last_input, (1, 1, look_back)))
    extended_population.append(extended_pred[0, 0])
    last_input = np.array(
        [extended_pred[0, 0]]
    )  # Update the input for the next prediction

# Invert the predictions back to the original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])
extended_population = scaler.inverse_transform(
    np.array(extended_population).reshape(-1, 1)
)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data["Year"], data["Population"], label="Actual Population")
plt.plot(
    data["Year"][look_back:train_size],
    train_predict.flatten(),
    label="Predicted Population (Training)",
)
plt.plot(
    data["Year"][train_size : train_size + len(test_predict)],
    test_predict.flatten(),
    label="Predicted Population (Testing)",
)
plt.plot(
    future_years,
    extended_population.flatten(),
    label="Predicted Population (Extended)",
)
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Hungarian Population in Romania over Time")
plt.legend()
plt.show()
