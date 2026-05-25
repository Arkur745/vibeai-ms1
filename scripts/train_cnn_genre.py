import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import (
    Dataset,
    DataLoader
)

from torchvision import transforms

from PIL import Image

import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)

from sklearn.metrics import (
    accuracy_score
)

import mlflow
import mlflow.pytorch
import random
import numpy as np
import torchaudio.transforms as T
# ---------------------------------
# Config
# ---------------------------------
SEED = 42
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)

random.seed(SEED)
DEVICE = torch.device(

    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.001


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
        ).convert("L")

        label = row["label"]

        if self.transform:

            image = self.transform(image)

        return image, label


# ---------------------------------
# CNN Model
# ---------------------------------

class GenreCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 32, kernel_size=3, padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, kernel_size=3, padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 28 * 28,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ---------------------------------
# Main
# ---------------------------------

def main():

    print("\nUsing Device:")
    print(DEVICE)

    df = pd.read_csv(
        "data/processed/gtzan_spectrograms.csv"
    )

    encoder = LabelEncoder()

    df["label"] = encoder.fit_transform(
        df["genre"]
    )
    Path("models").mkdir(
        exist_ok=True
    )
    joblib.dump(

        encoder,

        "models/cnn_genre_label_encoder.pkl"
    )

    train_df, test_df = train_test_split(

        df,

        test_size=0.2,

        random_state=42,

        stratify=df["label"]
    )

    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.5],

            std=[0.5]
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

    train_dataset = SpectrogramDataset(

        train_df,

        transform=train_transform
    )

    test_dataset = SpectrogramDataset(

        test_df,

        transform=test_transform
    )

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

    model = GenreCNN(
        num_classes=len(encoder.classes_)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )

    with mlflow.start_run(
        run_name="cnn_genre_classifier"
    ):
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

        best_accuracy = 0.0
        patience = 3


        epochs_without_improvement = 0


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
            # MLflow Logging
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

                    "model_state_dict": model.state_dict(),

                    "label_encoder_classes": encoder.classes_,

                    "num_classes": len(encoder.classes_)

                }, "models/best_cnn_genre_classifier.pth")

                print(
                    "\nBest model updated."
                )
            else:

                epochs_without_improvement += 1

                print(
                
                    f"\nNo improvement for "

                    f"{epochs_without_improvement} epoch(s)"
                )

                if epochs_without_improvement >= patience:
                
                    print("\nEarly stopping triggered.")

                    break

            # ---------------------------------
            # Epoch Summary
            # ---------------------------------

            print(
            
                f"\nEpoch {epoch+1}/{EPOCHS}"

                f"\nTrain Loss: {avg_loss:.4f}"

                f"\nTrain Accuracy: {train_accuracy:.4f}"

                f"\nValidation Accuracy: {val_accuracy:.4f}"
            )
        mlflow.log_metric(
            "best_val_accuracy",
            best_accuracy
        )

if __name__ == "__main__":

    main()
