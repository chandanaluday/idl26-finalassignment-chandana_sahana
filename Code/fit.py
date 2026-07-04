"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""

import copy #added
import torch

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.best_val_acc = -1.0 #added
        self.best_state_dict = None #added
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []} #added

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        #correct, sum = 0, 0
        correct, total = 0, 0 #added
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()  # added
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            #sum += labels.size(0)
            total += labels.size(0) #added
            correct += predicted.eq(labels).sum().item()
            
        #return running_loss / sum, (correct / sum) * 100
        return running_loss / total, (correct / total) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        #return running_loss / total, (correct / total) * 100
        return running_loss / total, (correct / total) * 100


    def fit(self, train_loader, val_loader, epochs):
        print("\n Starting Training Routine...")
        print("-" * 50)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            self.history["train_loss"].append(train_loss) #added
            self.history["train_acc"].append(train_acc) #added
            self.history["val_loss"].append(val_loss) #added
            self.history["val_acc"].append(val_acc) #added

            if val_acc > self.best_val_acc: #added
                self.best_val_acc = val_acc #added
                self.best_state_dict = copy.deepcopy(self.model.state_dict()) #added
            
            #print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | " "Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | " f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            marker = " *" if val_acc == self.best_val_acc else ""
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%{marker}")
        
        print("-" * 50)
        #print("Training Complete!")
        print(f"Training Complete! Best Val Acc: {self.best_val_acc:.2f}%")

        if self.best_state_dict is not None: #added
            self.model.load_state_dict(self.best_state_dict) #added
