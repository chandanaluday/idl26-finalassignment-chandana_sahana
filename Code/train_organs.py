"""
MAI/IDL SS26 - Final assignment.
Data-Scarcity task: compares training from scratch vs. transfer learning
on the low-sample 'organs' dataset (500 train / 200 test images, 11 classes).
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loaders
import models
from fit import Trainer

IN_CHANNELS = 1
NUM_CLASSES = 11


def train_from_scratch(train_loader, val_loader, device, config, epochs=10):
    print("\n--- Strategy A: Training from scratch ---")
    model = models.ResNet18(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", "ReLU")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=epochs)

    return model, trainer.history

def train_with_transfer_learning(train_loader, val_loader, device, config, epochs=10):
    print("\n--- Strategy B: Transfer learning (fine-tuning from orgs checkpoint) ---")
    model = models.ResNet18(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", "ReLU")).to(device)

    checkpoint_path = f"checkpoints/orgs_ResNet18_epoch{config['EPOCHS']}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded pretrained weights from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()
    fine_tune_lr = config["LEARNING_RATE"] * 0.1  # added: smaller learning rate to gently adjust, not overwrite, existing knowledge
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=epochs)

    return model, trainer.history

def evaluate_on_test(model, test_loader, device):
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    return accuracy, f1


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")

    train_loader, val_loader, test_loader = get_loaders(data="organs", data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    scratch_model, scratch_history = train_from_scratch(train_loader, val_loader, device, config, epochs=10)
    scratch_acc, scratch_f1 = evaluate_on_test(scratch_model, test_loader, device)
    print(f"\n[Scratch] Test Accuracy: {scratch_acc:.2f}% | Test F1: {scratch_f1:.2f}%")

    transfer_model, transfer_history = train_with_transfer_learning(train_loader, val_loader, device, config, epochs=10)
    transfer_acc, transfer_f1 = evaluate_on_test(transfer_model, test_loader, device)
    print(f"\n[Transfer] Test Accuracy: {transfer_acc:.2f}% | Test F1: {transfer_f1:.2f}%")

    results = {
        "scratch": {"test_accuracy": scratch_acc, "test_f1": scratch_f1, "history": scratch_history},
        "transfer_learning": {"test_accuracy": transfer_acc, "test_f1": transfer_f1, "history": transfer_history}
    }

    with open("logs/organs_scratch_vs_transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to logs/organs_scratch_vs_transfer_results.json")


if __name__ == "__main__":
    main()