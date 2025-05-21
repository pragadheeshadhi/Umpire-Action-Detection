import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import timm  # EfficientNet model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dir = r"C:\PROJECT\Datasets\Final Dataset Duplicate - Copy\train"
val_dir = r"C:\PROJECT\Datasets\Final Dataset Duplicate - Copy\valid"

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pre-trained EfficientNet model
model = timm.create_model("efficientnet_b0", pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 3)  # 3 classes: umpire, non-umpire, pitch
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
train_acc_list, train_loss_list = [], []

for epoch in range(num_epochs):
    print(f"Training Epoch:{epoch + 1}")
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100

    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Save model
torch.save(model.state_dict(), r"C:\PROJECT\DenseNet\Models\denseNet_classifier_1.pth")
print("Model saved successfully!")

# Evaluation on validation set
model.eval()
all_preds, all_labels = [], []
correct, total = 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Overall Accuracy
accuracy = correct / total * 100
print(f"\nValidation Accuracy: {accuracy:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
classes = ["umpire", "non-umpire", "pitch"]

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report (Precision, Recall, F1-score)
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=classes))

# Per-Class Accuracy
class_correct = [0] * len(classes)
class_total = [0] * len(classes)

for i in range(len(all_labels)):
    label = all_labels[i]
    pred = all_preds[i]
    if label == pred:
        class_correct[label] += 1
    class_total[label] += 1

for i in range(len(classes)):
    class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"Accuracy for class {classes[i]}: {class_acc:.2f}%")

# Plot Accuracy & Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(train_acc_list, label="Train Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_loss_list, label="Train Loss", marker="o", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()
