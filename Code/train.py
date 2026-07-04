"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training executing on device: {device}")
    
    for run in config["RUNS"]:
        print(f"\n{'='*50}")
        print(f"Training | Dataset: {run['DATA']} | Model: {run['MODEL']}")
        print(f"{'='*50}")


        #train_loader, val_loader, _ = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
        train_loader, val_loader, _ = get_loaders(data=run["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
        
        #model_class = getattr(models, config["MODEL"])
        model_class = getattr(models, run["MODEL"])

        #model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=0.99, activation_str=None).to(device)
        model = model_class(in_channels=run["CHANNELS"], num_classes=run["NUM_CLASSES"], drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", "ReLU")).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

        trainer = Trainer(model, criterion, optimizer, device)
        trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
        
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/{run['DATA']}_{run['MODEL']}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()