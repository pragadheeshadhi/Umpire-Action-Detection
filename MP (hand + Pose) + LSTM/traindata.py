import numpy as np
import pandas as pd
import tensorflow as tf
import os
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Directory where data is stored
data_dir = r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\Data"

# List of umpire signals
#signals = [
 #   "Out", "No-ball", "Wide ball", "Four runs", "Six Runs", "Byes",
  #  "Dead Ball", "Short Run", "Penalty Runs", "Revoke Decision"
#]

signals = [ "Leg Byes" ]

X, y = [], []
no_of_timesteps = 10  # LSTM time steps

# Load and process data for each signal
for idx, signal in enumerate(signals):
    file_path = os.path.join(data_dir, f"{signal}.csv")
    
    # Read CSV file
    df = pd.read_csv(file_path, header=None)
    dataset = df.values  # Convert to NumPy array
    
    n_samples = len(dataset)
    
    # Create sequences for LSTM
    for i in range(no_of_timesteps, n_samples):
        X.append(dataset[i - no_of_timesteps:i, :])  # Take 10 consecutive frames
        y.append(idx)  # Assign class index

# Convert lists to NumPy arrays
X, y = np.array(X), np.array(y)

# One-hot encode labels
onehot_encoder = OneHotEncoder(sparse_output=False)
y = onehot_encoder.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(len(signals), activation="softmax")
])

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model in directory
model_dir = r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\Models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "Cricket_LSTM_Model.h5")
model.save(model_path)

print(f"Model saved successfully at: {model_path}")
