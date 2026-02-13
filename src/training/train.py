import mlflow
from model import build_model
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def train_one_epoch(model, dataloader, criterion, optimizer):
    loss_history = []

    for batch_idx, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx} Loss:", loss.item())
        loss_history.append(loss.item())

        mlflow.log_metric("loss", loss.item(), step=batch_idx)

        # keep training short for now
        if batch_idx == 5:
            break

    return loss_history


def train():
    print("Starting training pipeline...")

    # ---- Load dataset (beginner version) ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root="data/raw/PetImages",
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("Dataset loaded. Total images:", len(dataset))

    # Build model
    model = build_model()
    print("Model loaded")

        # ---- Define loss and optimizer ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Loss function and optimizer ready")


    # ---- Test one forward pass ----
    images, labels = next(iter(dataloader))
    print("Batch shape:", images.shape)    

    # Start MLflow run
    mlflow.start_run()

    # ---- Small training loop (beginner version) ----
    print("Starting small training loop...")
    loss_history = train_one_epoch(model, dataloader, criterion, optimizer)

    print("Training loop finished")

    # Log some example parameters
    mlflow.log_param("model", "resnet18")
    mlflow.log_param("num_classes", 2)

    print("MLflow logging started")

    # Save model locally
    torch.save(model.state_dict(), "model.pt")
    print("Model saved locally")

    # Log model artifact to MLflow
    mlflow.log_artifact("model.pt")
    print("Model logged to MLflow")

    # Log a dummy metric (we will replace with real accuracy later)
    mlflow.log_metric("accuracy", 0.80)
    print("Metric logged")

    # ---- Create real loss curve ----
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Batch")
    plt.ylabel("Loss")

    plot_path = "training_loss.png"
    plt.savefig(plot_path)

    print("Training loss curve saved")

    mlflow.log_artifact(plot_path)
    print("Training loss curve logged to MLflow")
    print("Loss curve saved")

    # Log plot to MLflow
    mlflow.log_artifact(plot_path)
    print("Loss curve logged to MLflow")


    mlflow.end_run()
    print("Training pipeline finished")


if __name__ == "__main__":
    train()
