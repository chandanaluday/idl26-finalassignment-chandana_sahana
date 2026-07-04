"""
MAI/IDL SS26 - Final assignment.
Green Initiative efficiency profiling: compares GreenNet against the best original
model per dataset on parameter count, inference latency, and (where measurable) memory.
"""
import json
import time

import torch

from data import get_loaders
import models

# Best original model per dataset, based on REPORT.md benchmark results
BEST_ORIGINAL_MODEL = {
    "cells": "ResNet18",
    "chest": "VGG16",
    "lesions": "AlexNet",
    "orgs": "ResNet18"
}

CHANNELS_CLASSES = {
    "cells": (3, 8),
    "chest": (1, 2),
    "lesions": (3, 7),
    "orgs": (1, 11)
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_inference_latency(model, test_loader, device, num_samples=100):
    model.eval()
    single_image_loader = iter(test_loader)
    images, _ = next(single_image_loader)
    images = images.to(device)

    # Warm-up (first run is often slower due to backend setup, so we don't time it)
    with torch.no_grad():
        _ = model(images[:1])

    timings = []
    with torch.no_grad():
        for i in range(min(num_samples, images.size(0))):
            single_image = images[i:i+1]
            start = time.time()
            _ = model(single_image)
            timings.append(time.time() - start)

    avg_latency_ms = (sum(timings) / len(timings)) * 1000
    return avg_latency_ms

def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Profiling on device: {device}")

    results = []

    for dataset in ["cells", "chest", "lesions", "orgs"]:
        in_channels, num_classes = CHANNELS_CLASSES[dataset]
        _, _, test_loader = get_loaders(data=dataset, data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

        for model_name in [BEST_ORIGINAL_MODEL[dataset], "GreenNet"]:
            print(f"\nProfiling | Dataset: {dataset} | Model: {model_name}")

            model_class = getattr(models, model_name)
            model = model_class(in_channels=in_channels, num_classes=num_classes, drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", "ReLU")).to(device)

            checkpoint_path = f"checkpoints/{dataset}_{model_name}_epoch{config['EPOCHS']}.pt"
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

            param_count = count_parameters(model)
            latency_ms = measure_inference_latency(model, test_loader, device)

            print(f"  Parameters: {param_count:,} | Inference Latency: {latency_ms:.3f} ms/sample")

            results.append({
                "DATA": dataset,
                "MODEL": model_name,
                "role": "best_original" if model_name != "GreenNet" else "green",
                "parameter_count": param_count,
                "inference_latency_ms": round(latency_ms, 3)
            })

    with open("efficiency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll efficiency results saved to efficiency_results.json")


if __name__ == "__main__":
    main()