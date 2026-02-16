import mlflow
import mlflow.pytorch
from model import build_model
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------
# Train One Epoch
# ---------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, max_batches, epoch):
    loss_history = []

    model.train()

    for batch_idx, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # ---- Accuracy ----
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == labels).float().mean()

        print(f"Epoch {epoch+1} Batch {batch_idx} Loss:", loss.item())

        loss_history.append(loss.item())

        # unique step id across epochs
        step_id = epoch * max_batches + batch_idx

        mlflow.log_metric("loss", loss.item(), step=step_id)
        mlflow.log_metric("train_accuracy", accuracy.item(), step=step_id)

        if batch_idx >= max_batches:
            break

    return loss_history


# ---------------------------------------------------
# Evaluate on Test Set
# ---------------------------------------------------
def evaluate(model, dataloader, max_eval_batches=20):
    model.eval()

    correct = 0
    total = 0

    cat_preds = 0
    dog_preds = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            cat_preds += (preds == 0).sum().item()
            dog_preds += (preds == 1).sum().item()

            # limit evaluation speed
            if batch_idx >= max_eval_batches:
                break

    accuracy = correct / total

    print("\n===== Evaluation Summary =====")
    print(f"Total Samples Evaluated: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cat Predictions: {cat_preds}")
    print(f"Dog Predictions: {dog_preds}")

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("cat_predictions", cat_preds)
    mlflow.log_metric("dog_predictions", dog_preds)

    return accuracy

def evaluate_old(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    mlflow.log_metric("test_accuracy", accuracy)

    return accuracy


# ---------------------------------------------------
# Main Training Function
# ---------------------------------------------------
def train():
    print("Starting training pipeline...")

    # ---- Hyperparameters ----
    learning_rate = 0.0003
    batch_size = 4
    max_batches = 20
    num_epochs = 3

    # ---- Dataset ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root="data/raw/PetImages",
        transform=transform
    )

    # ---- Train/Test Split ----
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {train_size}, Test size: {test_size}")

    # ---- Model ----
    model = build_model()
    print("Model loaded")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Loss function and optimizer ready")

    # ---- MLflow Start ----
    run_name = f"resnet18_lr{learning_rate}_bs{batch_size}"

    mlflow.start_run(run_name=run_name)

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("max_batches", max_batches)
    mlflow.log_param("num_epochs", num_epochs)

    # ---- Training Loop ----
    print("Starting training loop...")

    all_loss_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        epoch_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            max_batches,
            epoch
        )

        all_loss_history.extend(epoch_loss)

    print("Training loop finished")

    # ---- Evaluation ----
    print("Evaluating model on test set...")
    evaluate(model, test_loader)

    # ---- Save Model ----
    torch.save(model.state_dict(), "model.pt")
    mlflow.log_artifact("model.pt")
    mlflow.pytorch.log_model(model, "model")

    print("Model saved and logged to MLflow")

    # ---- Plot Loss Curve ----
    plt.figure()
    plt.plot(all_loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plot_path = "training_loss.png"
    plt.savefig(plot_path)

    mlflow.log_artifact(plot_path)

    print("Loss curve saved and logged")

    mlflow.end_run()
    print("Training pipeline finished")


if __name__ == "__main__":
    train()
