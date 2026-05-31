import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader
)

from torchvision import (
    transforms,
    models
)

import torchaudio.transforms as T

from PIL import Image

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.pytorch

import joblib


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

LEARNING_RATE = 1e-4

PATIENCE = 3


# ---------------------------------
# Dataset
# ---------------------------------

class SpectrogramDataset(Dataset):

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

        label = row["label"]

        if self.transform:

            image = self.transform(image)

        return image, label


# ---------------------------------
# ResNet Transfer Learning Model
# ---------------------------------

class GenreResNet(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.model = models.resnet18(

            pretrained=True
        )

        # ---------------------------------
        # Grayscale Input
        # ---------------------------------


        # ---------------------------------
        # Replace Final Layer
        # ---------------------------------

        in_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(

                in_features,

                num_classes
            )
        )

    def forward(self, x):

        return self.model(x)


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
        "data/processed/gtzan_spectrograms.csv"
    )

    encoder = LabelEncoder()

    df["label"] = encoder.fit_transform(
        df["genre"]
    )

    # ---------------------------------
    # Save Label Encoder
    # ---------------------------------

    joblib.dump(

        encoder,

        "models/resnet_genre_label_encoder.pkl"
    )

    # ---------------------------------
    # Train/Test Split
    # ---------------------------------

    train_df, test_df = train_test_split(

        df,

        test_size=0.2,

        random_state=SEED,

        stratify=df["label"]
    )

    # ---------------------------------
    # Augmentation
    # ---------------------------------

    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.5,0.5,0.5],

            std=[0.5,0.5,0.5]
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

            mean=[0.5],

            std=[0.5]
        )
    ])

    # ---------------------------------
    # Datasets
    # ---------------------------------

    train_dataset = SpectrogramDataset(

        train_df,

        transform=train_transform
    )

    test_dataset = SpectrogramDataset(

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

        pin_memory=True,

        drop_last=True
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

    model = GenreResNet(

        num_classes=len(encoder.classes_)
    ).to(DEVICE)

    # ---------------------------------
    # Freeze Backbone
    # ---------------------------------


    # ---------------------------------
    # Train Final Layer Only
    # ---------------------------------


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )

    best_accuracy = 0.0

    epochs_without_improvement = 0

    # ---------------------------------
    # MLflow
    # ---------------------------------

    with mlflow.start_run(

        run_name="resnet18_transfer_learning"
    ):

        mlflow.log_param(
            "architecture",
            "resnet18_transfer"
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE
        )

        mlflow.log_param(
            "epochs",
            EPOCHS
        )

        # ---------------------------------
        # Training Loop
        # ---------------------------------

        for epoch in range(EPOCHS):

            # ---------------------------------
            # Training
            # ---------------------------------

            model.train()

            running_loss = 0.0

            train_predictions = []

            train_actuals = []

            for images, labels in train_loader:

                images = images.to(DEVICE)

                labels = labels.to(

                    DEVICE,

                    dtype=torch.long
                )

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                preds = torch.argmax(

                    outputs,

                    dim=1
                )

                train_predictions.extend(

                    preds.cpu().numpy()
                )

                train_actuals.extend(

                    labels.cpu().numpy()
                )

            avg_loss = (

                running_loss /

                len(train_loader)
            )

            train_accuracy = accuracy_score(

                train_actuals,

                train_predictions
            )

            # ---------------------------------
            # Validation
            # ---------------------------------

            model.eval()

            val_predictions = []

            val_actuals = []

            with torch.no_grad():

                for images, labels in test_loader:

                    images = images.to(DEVICE)

                    labels = labels.to(

                        DEVICE,

                        dtype=torch.long
                    )

                    outputs = model(images)

                    preds = torch.argmax(

                        outputs,

                        dim=1
                    )

                    val_predictions.extend(

                        preds.cpu().numpy()
                    )

                    val_actuals.extend(

                        labels.cpu().numpy()
                    )

            val_accuracy = accuracy_score(

                val_actuals,

                val_predictions
            )

            # ---------------------------------
            # MLflow Metrics
            # ---------------------------------

            mlflow.log_metric(

                "train_loss",

                avg_loss,

                step=epoch
            )

            mlflow.log_metric(

                "train_accuracy",

                train_accuracy,

                step=epoch
            )

            mlflow.log_metric(

                "val_accuracy",

                val_accuracy,

                step=epoch
            )

            # ---------------------------------
            # Save Best Model
            # ---------------------------------

            if val_accuracy > best_accuracy:

                best_accuracy = val_accuracy

                epochs_without_improvement = 0

                torch.save({

                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "label_encoder_classes": (
                        encoder.classes_
                    ),

                    "num_classes": (
                        len(encoder.classes_)
                    )

                },

                    "models/best_resnet_genre_classifier.pth")

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

                f"\nTrain Loss: {avg_loss:.4f}"

                f"\nTrain Accuracy: "
                f"{train_accuracy:.4f}"

                f"\nValidation Accuracy: "
                f"{val_accuracy:.4f}"
            )

        # ---------------------------------
        # Final Metrics
        # ---------------------------------

        mlflow.log_metric(

            "best_val_accuracy",

            best_accuracy
        )

        # ---------------------------------
        # Classification Report
        # ---------------------------------

        print("\nClassification Report")
        print("--------------------------")

        print(

            classification_report(

                val_actuals,

                val_predictions,

                target_names=encoder.classes_
            )
        )

        # ---------------------------------
        # Confusion Matrix
        # ---------------------------------

        cm = confusion_matrix(

            val_actuals,

            val_predictions
        )

        plt.figure(figsize=(10, 8))

        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            xticklabels=encoder.classes_,

            yticklabels=encoder.classes_
        )

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.title(
            "ResNet Genre Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            "artifacts/resnet_confusion_matrix.png"
        )

        plt.close()

        # ---------------------------------
        # Log Model
        # ---------------------------------

        mlflow.pytorch.log_model(

            model,

            "resnet_genre_model"
        )

        print("\nTraining Complete")


if __name__ == "__main__":

    main()
