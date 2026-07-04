"""
MAI/IDL SS26 - Final assignment.
Evaluation script: loads trained checkpoints and evaluates them on the held-out test set.
"""
import json

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data import get_loaders
import models

TARGET_ACCURACY = {
    "cells": 90,
    "chest": 87,
    "lesions": 67,
    "orgs": 83
}


def evaluate_on_test(model, test_loader, device):
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
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    return accuracy, precision, recall, f1


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    results = []

    for run in config["RUNS"]:
        print(f"\nEvaluating | Dataset: {run['DATA']} | Model: {run['MODEL']}")

        _, _, test_loader = get_loaders(data=run["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

        model_class = getattr(models, run["MODEL"])
        model = model_class(in_channels=run["CHANNELS"], num_classes=run["NUM_CLASSES"], drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", "ReLU")).to(device)

        checkpoint_path = f"checkpoints/{run['DATA']}_{run['MODEL']}_epoch{config['EPOCHS']}.pt"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        accuracy, precision, recall, f1 = evaluate_on_test(model, test_loader, device)

        target = TARGET_ACCURACY.get(run["DATA"], 0)
        status = "PASS " if accuracy >= target else "FAIL "

        print(f"  Test Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")
        print(f"  Target: {target}% | Status: {status}")

        results.append({
            "DATA": run["DATA"],
            "MODEL": run["MODEL"],
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "target_accuracy": target,
            "status": "PASS" if accuracy >= target else "FAIL"
        })

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll results saved to evaluation_results.json")


if __name__ == "__main__":
    main()