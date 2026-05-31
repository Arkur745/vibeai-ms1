# scripts/train_resnet_emotion.py

import random
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)
from torchvision import (
    transforms,
    models
)

import torchaudio.transforms as T

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

import mlflow
import mlflow.pytorch

import matplotlib.pyplot as plt


# ---------------------------------
# Reproducibility
# ---------------------------------

SEED = 42

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)

random.seed(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


# ---------------------------------
# Config
# ---------------------------------

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 3e-4

PATIENCE = 3


# ---------------------------------
# Dataset
# ---------------------------------

class EmotionDataset(Dataset):

    def __init__(

        self,

        dataframe,

        transform=None
    ):

        self.df = dataframe

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(

            row["image_path"]

        ).convert("RGB")

        valence = torch.tensor(

            row["valence"],

            dtype=torch.float32
        )

        arousal = torch.tensor(

            row["arousal"],

            dtype=torch.float32
        )

        if self.transform:

            image = self.transform(image)

        return image, valence, arousal


# ---------------------------------
# Emotion ResNet
# ---------------------------------

class EmotionResNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = resnet18(

            weights=ResNet18_Weights.DEFAULT
        )

        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        # ---------------------------------
        # Shared Representation
        # ---------------------------------

        self.shared = nn.Sequential(

            nn.Linear(
                in_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3)
        )

        # ---------------------------------
        # Valence Head
        # ---------------------------------

        self.valence_head = nn.Linear(
            256,
            1
        )

        # ---------------------------------
        # Arousal Head
        # ---------------------------------

        self.arousal_head = nn.Linear(
            256,
            1
        )

    def forward(self, x):

        features = self.backbone(x)

        shared = self.shared(features)

        valence = self.valence_head(shared)

        arousal = self.arousal_head(shared)

        return valence, arousal


# ---------------------------------
# Main
# ---------------------------------

def main():

    print("\nUsing Device:")
    print(DEVICE)

    Path("models").mkdir(
        exist_ok=True
    )

    Path("artifacts").mkdir(
        exist_ok=True
    )

    # ---------------------------------
    # Load Dataset
    # ---------------------------------

    df = pd.read_csv(
        "data/processed/deam_spectrograms.csv"
    )

    # ---------------------------------
    # Split
    # ---------------------------------

    train_df, test_df = train_test_split(

        df,

        test_size=0.2,

        random_state=SEED
    )

    # ---------------------------------
    # Augmentations
    # ---------------------------------

    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]
        ),

        T.FrequencyMasking(
            freq_mask_param=15
        ),

        T.TimeMasking(
            time_mask_param=35
        )
    ])

    test_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]
        )
    ])

    # ---------------------------------
    # Datasets
    # ---------------------------------

    train_dataset = EmotionDataset(

        train_df,

        transform=train_transform
    )

    test_dataset = EmotionDataset(

        test_df,

        transform=test_transform
    )

    # ---------------------------------
    # Dataloaders
    # ---------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=True
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        num_workers=0,

        pin_memory=True
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = EmotionResNet().to(DEVICE)

    criterion = nn.HuberLoss()

    optimizer = optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )

    best_loss = float("inf")

    epochs_without_improvement = 0

    # ---------------------------------
    # MLflow
    # ---------------------------------

    with mlflow.start_run(

        run_name="resnet_emotion_model"
    ):

        mlflow.log_param(
            "architecture",
            "resnet18_emotion"
        )

        mlflow.log_param(
            "epochs",
            EPOCHS
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE
        )

        # ---------------------------------
        # Training Loop
        # ---------------------------------

        for epoch in range(EPOCHS):

            model.train()

            running_loss = 0.0

            for images, valence, arousal in train_loader:

                images = images.to(DEVICE)

                valence = valence.to(DEVICE)

                arousal = arousal.to(DEVICE)

                optimizer.zero_grad()

                pred_valence, pred_arousal = model(images)

                pred_valence = pred_valence.view(-1)


                pred_arousal = pred_arousal.view(-1)

                valence_loss = criterion(

                    pred_valence,

                    valence
                )

                arousal_loss = criterion(

                    pred_arousal,

                    arousal
                )

                loss = (

                    valence_loss

                    + arousal_loss
                )

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = (

                running_loss /

                len(train_loader)
            )

            # ---------------------------------
            # Validation
            # ---------------------------------

            model.eval()

            val_losses = []

            true_valence = []
            pred_valence_all = []

            true_arousal = []
            pred_arousal_all = []

            with torch.no_grad():

                for images, valence, arousal in test_loader:

                    images = images.to(DEVICE)

                    valence = valence.to(DEVICE)

                    arousal = arousal.to(DEVICE)

                    pred_valence, pred_arousal = model(images)

                    pred_valence = pred_valence.squeeze()

                    pred_arousal = pred_arousal.squeeze()

                    valence_loss = criterion(

                        pred_valence,

                        valence
                    )

                    arousal_loss = criterion(

                        pred_arousal,

                        arousal
                    )

                    loss = (

                        valence_loss

                        + arousal_loss
                    )

                    val_losses.append(
                        loss.item()
                    )

                    true_valence.extend(

                        valence.cpu().numpy()
                    )

                    pred_valence_all.extend(

                        pred_valence.cpu().numpy()
                    )

                    true_arousal.extend(

                        arousal.cpu().numpy()
                    )

                    pred_arousal_all.extend(

                        pred_arousal.cpu().numpy()
                    )

            avg_val_loss = np.mean(
                val_losses
            )

            valence_mae = mean_absolute_error(

                true_valence,

                pred_valence_all
            )

            arousal_mae = mean_absolute_error(

                true_arousal,

                pred_arousal_all
            )

            # ---------------------------------
            # MLflow Metrics
            # ---------------------------------

            mlflow.log_metric(

                "train_loss",

                avg_train_loss,

                step=epoch
            )

            mlflow.log_metric(

                "val_loss",

                avg_val_loss,

                step=epoch
            )

            mlflow.log_metric(

                "valence_mae",

                valence_mae,

                step=epoch
            )

            mlflow.log_metric(

                "arousal_mae",

                arousal_mae,

                step=epoch
            )

            # ---------------------------------
            # Save Best Model
            # ---------------------------------

            if avg_val_loss < best_loss:

                best_loss = avg_val_loss

                epochs_without_improvement = 0

                torch.save({

                    "model_state_dict": (
                        model.state_dict()
                    )

                },

                    "models/best_resnet_emotion_model.pth")

                print("\nBest model updated.")

            else:

                epochs_without_improvement += 1

                print(

                    f"\nNo improvement for "

                    f"{epochs_without_improvement} epoch(s)"
                )

                if (
                    epochs_without_improvement
                    >= PATIENCE
                ):

                    print(
                        "\nEarly stopping triggered."
                    )

                    break

            # ---------------------------------
            # Epoch Summary
            # ---------------------------------

            print(

                f"\nEpoch {epoch+1}/{EPOCHS}"

                f"\nTrain Loss: {avg_train_loss:.4f}"

                f"\nValidation Loss: {avg_val_loss:.4f}"

                f"\nValence MAE: {valence_mae:.4f}"

                f"\nArousal MAE: {arousal_mae:.4f}"
            )

        # ---------------------------------
        # Scatter Plot
        # ---------------------------------

        plt.figure(figsize=(8, 8))

        plt.scatter(

            true_valence,

            pred_valence_all,

            alpha=0.6
        )

        plt.xlabel("True Valence")

        plt.ylabel("Predicted Valence")

        plt.title(
            "Valence Prediction Scatter"
        )

        plt.savefig(
            "artifacts/valence_scatter.png"
        )

        plt.close()

        # ---------------------------------
        # Log Model
        # ---------------------------------

        mlflow.pytorch.log_model(

            model,

            "emotion_resnet_model"
        )

        print("\nTraining Complete")


if __name__ == "__main__":

    main()
