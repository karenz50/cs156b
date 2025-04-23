import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, 
    roc_curve, accuracy_score, balanced_accuracy_score, confusion_matrix,
    log_loss, matthews_corrcoef
)
import matplotlib.pyplot as plt

from ..data_processing import ChestScanDataset, Rescale, RandomCrop, CenterCrop, Normalize, ToTensor
from ..stratified_split import stratified_train_val_split
from torchvision import transforms
import numpy as np
import pandas as pd

# paths
csv_file = "/groups/CS156b/data/student_labels/train.csv"
root_dir = "/groups/CS156b/data"
img_dir = "/groups/CS156b/2025/ChexMix/kz_models/imgs/"
model_save_path = "/groups/CS156b/2025/ChexMix/kz_models/pneumonia/resnet_p.pt"
label_col = "Pneumonia"
abbr = "p"

# transforms
transform = transforms.Compose([
    Rescale(256),
    RandomCrop(224),
    ToTensor()
])

# load dataset and subset
full_dataset = ChestScanDataset(csv_file=csv_file, root_dir=root_dir, label_col=label_col, transform=transform)
small_size = 100
train_size = int(0.8 * small_size)
val_size = small_size - train_size
small_dataset, _ = random_split(full_dataset, [small_size, len(full_dataset) - small_size])
train_dataset, val_dataset = random_split(small_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

from torchvision import models

class ResNetMeta(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(base.children())[:-1])   # 512‑D
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

# simple CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.image_features = nn.Linear(32 * 56 * 56, 64)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.image_features(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetMeta().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# save trained model weights
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to: {model_save_path}")

# evaluation
model.eval()
y_true = []
y_scores = []

with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze()
        y_scores.extend(probs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# metrics
threshold = 0.5
y_pred = [1 if p > threshold else 0 for p in y_scores]

auc = roc_auc_score(y_true, y_scores)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Positive labels in validation set: {sum(y_true)} / {len(y_true)}")

print(f"\nEvaluation Metrics on Validation Set:")
print(f"AUC-ROC:  {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(img_dir + "p_resnet_roc_curve.png")

# precision recall curve
prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
plt.figure()
plt.plot(rec_curve, prec_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(img_dir + "p_resnet_pr_curve.png")

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    log_loss, matthews_corrcoef
)

# More evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
mcc = matthews_corrcoef(y_true, y_pred)

try:
    ll = log_loss(y_true, y_scores)
except:
    ll = float('nan')

print("\nAdditional Evaluation Metrics:")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Balanced Accuracy:  {balanced_acc:.4f}")
print(f"Specificity (TNR):  {specificity:.4f}")
print(f"Matthews Corr Coef: {mcc:.4f}")
print(f"Log Loss:           {ll:.4f}")

# print confusion matrix
print(f"\nConfusion Matrix:\n[[TN FP]\n [FN TP]]\n{confusion_matrix(y_true, y_pred)}")

print("\n----------------------pneumonia complete----------------------")