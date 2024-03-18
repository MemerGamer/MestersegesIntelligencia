import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout
from keras.optimizers import Adam

# 1. Data collection and preprocessing
data = pd.read_csv("romania_birth_rate.csv")

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
birth_rate_scaled = scaler.fit_transform(data["Birth Rate"].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(birth_rate_scaled) * 0.8)
train_data = birth_rate_scaled[0:train_size, :]
test_data = birth_rate_scaled[train_size : len(birth_rate_scaled), :]


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
model.add(
    LSTM(units=50, return_sequences=True, input_shape=(1, look_back), activation="tanh")
)
model.add(LSTM(units=50, activation="tanh"))
model.add(Dropout(0.2))


model.compile(loss="mean_squared_error", optimizer=Adam(clipvalue=1.0))
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# 3. Making predictions and evaluating the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Extend the prediction until the future years
last_test_year = data["Year"].iloc[-1]  # Use the last year from the original data
future_start_year = last_test_year + 1
future_years = np.arange(future_start_year, 2051)  # Extend to 2050

# Make predictions for the extended years
extended_birth_rate = np.zeros(len(future_years))
last_input = test_data[
    -look_back:
]  # Use the last look_back birth rate values from the test data as input
for i in range(len(future_years)):
    extended_pred = model.predict(np.reshape(last_input, (1, 1, look_back)))
    extended_birth_rate[i] = extended_pred[0, 0]
    last_input = np.append(
        last_input[1:], extended_pred[0, 0]
    )  # Update the input for the next prediction

# Check for infinite or large values
if np.isinf(extended_birth_rate).any() or np.isneginf(extended_birth_rate).any():
    raise ValueError("Predicted values contain infinity.")
if np.abs(extended_birth_rate).max() > np.finfo(np.float32).max:
    raise ValueError("Predicted values are too large for dtype('float32').")

# Invert the predictions back to the original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])
extended_birth_rate = scaler.inverse_transform(
    np.array(extended_birth_rate).reshape(-1, 1)
)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data["Year"], data["Birth Rate"], label="Actual Birth Rate")
plt.plot(
    data["Year"][look_back:train_size],
    train_predict.flatten()[: train_size - look_back],
    label="Predicted Birth Rate (Training)",
)

# Couldn't plot it
# Plot the testing set predictions
# plt.plot(
#     data["Year"][train_size : train_size + len(test_predict)],
#     test_predict.flatten(),
#     label="Predicted Birth Rate (Testing)",
# )

plt.plot(
    future_years,
    extended_birth_rate.flatten(),
    label="Predicted Birth Rate (Extended)",
)
plt.xlabel("Year")
plt.ylabel("Birth Rate")
plt.title("Romania's Birth Rate over Time")
plt.legend()
plt.show()
