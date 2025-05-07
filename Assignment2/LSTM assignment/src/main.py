#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

#dataset
data = pd.read_csv('pss10.csv')

target_column = 'TEMP'  

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[[target_column]])

#create sequences for LSTM/BiLSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

#Sequence length (month)
SEQ_LENGTH = 30

X, y = create_sequences(scaled_data, SEQ_LENGTH)

#training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#LSTM Model

lstm_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

#Train LSTM model
lstm_history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

#BiLSTM Model

bilstm_model = Sequential([
    Bidirectional(LSTM(64), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

bilstm_model.compile(optimizer='adam', loss='mse')

# Train BiLSTM model
bilstm_history = bilstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate and Plot

# Predict with BiLSTM 
predictions = bilstm_model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

#results
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Soil Temperature')
plt.plot(predictions, label='Predicted Soil Temperature')
plt.title('Actual vs Predicted Soil Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()