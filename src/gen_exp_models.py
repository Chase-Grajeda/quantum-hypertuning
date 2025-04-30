import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from lstnet import LSTNet
from torch.nn import Module, MSELoss
from torch.optim import Optimizer, Adam
import gc
import yfinance as yf
import seaborn as sns


class USDJPY(Dataset):
    def __init__(self, data: np.ndarray, period: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.period = period
    
    def __len__(self):
        return len(self.data) - self.period

    def __getitem__(self, index):
        x = self.data[index: index + self.period]
        y = self.data[index + self.period]
        return x, y


def create_fnamne(period: int, batch: int, epoch: int, ending: str):
    return f"{period}-{batch}-{epoch}{ending}"


def train(loader: DataLoader, model: Module, optimizer: Optimizer, loss_fn):
    model.train()
    total_loss = 0
    for seqs, tgts in loader:
        optimizer.zero_grad()
        preds = model(seqs)
        loss = loss_fn(preds, tgts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return np.squeeze(avg_loss)


if __name__ == "__main__":
    # Param list
    period_params = [10, 20, 30]
    batch_params = [16, 32, 64]
    epoch_params = [100, 500, 1000]
    param_schedule = list(itertools.product(period_params, batch_params, epoch_params))
    
    # Download dataset
    historical_data = yf.download("JPY=X", start="2022-01-01", end="2025-01-01")
    historical_data = pd.DataFrame(historical_data)
    historical_data.to_csv("data/usdjpy.csv")
    
    # Transform data to make stationary
    log_data = np.log(historical_data["Close"].to_numpy())
    log_return_data = log_data[1:] - log_data[:-1]
    print("Data shape:", log_return_data.shape)
    
    # Generate models
    for param_set in param_schedule:
        period, batch_size, epochs = param_set
        print("Params:", param_set)
        
        # Create param-wise data loader
        train_dataset = USDJPY(log_return_data[:-100], period=period) # reserve last 100 steps for forecasting
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train
        model = LSTNet(period=period, num_features=1, rnn_dim=64, cnn_dim=64, skip_dim=32)
        loss_fn = MSELoss()
        optim = Adam(model.parameters(), lr=1e-6)
        loss_history = []
        for epoch in range(epochs):
            loss = train(train_loader, model, optim, loss_fn)
            loss_history.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} --- loss: {loss:.6f}")

        # Save loss history
        plt.plot(loss_history, color="blue", label="train loss")
        plt.title(create_fnamne(period, batch_size, epochs, "") + " Loss History")
        plt.legend()
        plt.savefig(f"models/{create_fnamne(period, batch_size, epochs, '.png')}")
        plt.close()
        
        # Save model
        torch.save(model.state_dict(), f"models/{create_fnamne(period, batch_size, epochs, '.pth')}")
        
        # Clear lingering gradients
        del model
        torch.cuda.empty_cache()