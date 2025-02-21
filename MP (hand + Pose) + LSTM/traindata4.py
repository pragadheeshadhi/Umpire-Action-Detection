import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# 📌 Define Umpire Signals
signals = ["No Action", "No Ball","Four","Wide","Out","Revoke","Penality","Bye","Leg Bye","Short Run","Six","Dead Ball"]  # Add more labels if needed

X, y = [], []
no_of_timesteps = 7  # 📌 Defines how many frames define one action

# 📌 Load Data
for idx, signal in enumerate(signals):
    file_path = f"Data4/{signal}.csv"
    df = pd.read_csv(file_path)
    dataset = df.iloc[:, :-1].values  # 📌 Exclude label column
    n_sample = len(dataset)

    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i-no_of_timesteps:i, :])
        y.append(idx)  

X, y = np.array(X), np.array(y)

# 📌 Convert Labels to One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
y = onehot_encoder.fit_transform(y.reshape(-1, 1))

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Build LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(len(signals), activation="softmax")
])

# 📌 Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 📌 Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 📌 Evaluate Model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 📌 Confusion Matrix & Accuracy
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
acc = accuracy_score(y_true_classes, y_pred_classes)
print(f"Accuracy: {acc}")
print("Confusion Matrix:")
print(conf_matrix)

# 📌 Plot Confusion Matrix
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=signals, yticklabels=signals)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 📌 Save Model
os.makedirs("Models4", exist_ok=True)
model.save("Models4/LSTM_Model_T7.h5")
print("Model saved successfully.")
