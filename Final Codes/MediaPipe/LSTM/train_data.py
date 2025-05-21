import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os

# 📌 Define Umpire Signals
#signals = ["No Action", "No Ball", "Four", "Wide", "Out",
 #           "Revoke", "Penality", "Bye", "Leg Bye",
  #            "Short Run", "Six", "Dead Ball"]
#signals=["No Action","Out","Wide","Four","Six"]
signals = ["No Action", "No Ball", "Four", "Wide", "Out", "Revoke", "Penality", "Bye", "Leg Bye", "Short Run", "Six", "Dead Ball"]
X, y = [], []
no_of_timesteps = 10  # 📌 Defines how many frames define one action

# 📌 Load Data
for idx, signal in enumerate(signals):
    file_path = f"C:\PROJECT\MediaPipe\MP (hand + Pose) + LSTM\Data Blue Umpire\{signal}.csv"
    df = pd.read_csv(file_path)
    dataset = df.iloc[:, :-1].values  # 📌 Exclude label column
    #df = pd.read_csv(file_path, low_memory=False)
    #dataset = df.iloc[1:, :-1]  # Skip first row and last column
    #df.reset_index(drop=True, inplace=True)  # Reset index
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
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 📌 Evaluate Model
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 📌 Confusion Matrix & Accuracy
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
acc = accuracy_score(y_true_classes, y_pred_classes)

print(f"\n🔹 Train Accuracy: {train_acc:.4f}")
print(f"🔹 Test Accuracy: {test_acc:.4f}")
print(f"🔹 Overall Accuracy: {acc:.4f}")

# 📌 Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_true_classes, y_pred_classes, target_names=signals)
print("\n🔹 Classification Report:\n")
print(class_report)

# 📌 Per-Class Accuracy
per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
for i, signal in enumerate(signals):
    print(f"🔹 {signal}: {per_class_acc[i]:.4f}")

# 📌 Plot Confusion Matrix
plt.figure(figsize=(8,8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=signals, yticklabels=signals)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 📌 Plot Accuracy and Loss Graphs
plt.figure(figsize=(12, 5))

# 🔹 Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()

# 🔹 Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()

plt.show()

# 📌 Save Model
os.makedirs("Models4", exist_ok=True)
model.save(r"MediaPipe/MP (hand + Pose) + LSTM/Models4/DataBlueUmpire_A1M1.h5")
print("\n✅ Model saved successfully.")
