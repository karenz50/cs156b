import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from kz_models.data_processing import ChestScanDataset, Rescale, RandomCrop, ToTensor
from kz_models.lung_opacity.full_lo import SmallCNN
from kz_models.pneumonia.resnet_p import ResNetMeta

TEST_CSV = "/groups/CS156b/data/student_labels/test_ids.csv"
DATA_ROOT = "/groups/CS156b/data"
MODEL_DIR = "/groups/CS156b/2025/ChexMix/final_predictions/final_models/"
OUTPUT_CSV = "/groups/CS156b/2025/ChexMix/final_predictions/final_predictions.csv"

# labels to predict
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# transforms
transform = transforms.Compose([
    Rescale(256),
    RandomCrop(224),
    ToTensor()
])

# load test dataset
test_dataset = ChestScanDataset(csv_file=TEST_CSV, root_dir=DATA_ROOT, label_col=None, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# collect predictions for all labels
all_predictions = {label: [] for label in LABELS}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for label in LABELS:
    model_path = os.path.join(MODEL_DIR, f"{label.lower().replace(' ', '_')}.pt")

    if not os.path.exists(model_path):
        print(f"Skipping {label}: model not found at {model_path}")
        all_predictions[label] = [0.0] * len(test_dataset)
        continue

    if label == "Lung Opacity":
        model = SmallCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model for {label}")
    elif label == "Pneumonia" or label == "Pleural Effusion":
        model = ResNetMeta().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model for {label}")
    else:
        model = None # add elif/else statements as necessary to make it work with your model

    preds = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            if probs.ndim == 0:
                preds.append(float(probs))
            else:
                preds.extend(probs.tolist())

    all_predictions[label] = preds

# get test IDs from test CSV 
ids = pd.read_csv(TEST_CSV).iloc[:, 0]

# build data frame
pred_df = pd.DataFrame({
    "Id": ids
})
for label in LABELS:
    pred_df[label] = all_predictions[label]

# write to CSV
pred_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nPredictions saved to: {OUTPUT_CSV}")
