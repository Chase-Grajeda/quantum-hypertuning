import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from lstnet2 import LSTNet
from torch.nn import Module, MSELoss
from torch.optim import Optimizer, Adam
import yfinance as yf


class USDJPY(Dataset):
    def __init__(self, data: np.ndarray, period: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.period = period
    
    def __len__(self):
        return len(self.data) - self.period

    def __getitem__(self, index):
        x = self.data[index: index + self.period]
        y = self.data[index + self.period][0]
        return x, y


def create_fnamne(rsi_period: int, wp_period: int, adx_period: int, ending: str):
    return f"r{rsi_period}-w{wp_period}-a{adx_period}{ending}"


def train(loader: DataLoader, model: Module, optimizer: Optimizer, loss_fn):
    model.train()
    total_loss = 0
    for seqs, tgts in loader:
        optimizer.zero_grad()
        preds = model(seqs)
        loss = loss_fn(torch.squeeze(preds), tgts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return np.squeeze(avg_loss)


def get_rsi(historical: pd.DataFrame, period: int):
    close = historical["Close"].to_numpy()
    delta = np.diff(close, axis=0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.ones_like(close)
    avg_loss = np.ones_like(close)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    
    rsi = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
    return rsi


def get_williamsr(historical: pd.DataFrame, period: int):
    high = historical["High"].to_numpy()
    low = historical["Low"].to_numpy()
    close = historical["Close"].to_numpy()
    williams = np.full_like(close, -50.)
    
    for i in range(period - 1, len(close)):
        max_high = np.max(high[i - period + 1: i + 1])
        min_low = np.min(low[i - period + 1: i + 1])
        williams[i] = -100 * (max_high - close[i]) / (max_high - min_low)
    
    return williams


def get_adx(historical: pd.DataFrame, period: int):
    high = historical["High"].to_numpy()
    low = historical["Low"].to_numpy()
    close = historical["Close"].to_numpy()
    
    tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr14 = np.ones_like(close)
    plus_dm14 = np.ones_like(close)
    minus_dm14 = np.ones_like(close)

    tr14[period] = np.sum(tr[:period])
    plus_dm14[period] = np.sum(plus_dm[:period])
    minus_dm14[period] = np.sum(minus_dm[:period])

    for i in range(period + 1, len(close)):
        tr14[i] = tr14[i - 1] - (tr14[i - 1] / period) + tr[i - 1]
        plus_dm14[i] = plus_dm14[i - 1] - (plus_dm14[i - 1] / period) + plus_dm[i - 1]
        minus_dm14[i] = minus_dm14[i - 1] - (minus_dm14[i - 1] / period) + minus_dm[i - 1]

    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)

    adx = np.full_like(close, 50.0)
    adx[period * 2] = np.mean(dx[period: period * 2])
    for i in range(period * 2 + 1, len(close)):
        adx[i] = ((adx[i - 1] * (period - 1)) + dx[i - 1]) / period

    return adx


if __name__ == "__main__":
    # Param list
    rsi_params = [7, 14, 21]
    wil_params = [7, 14, 21]
    adx_params = [7, 14, 21]
    param_schedule = list(itertools.product(rsi_params, wil_params, adx_params))
    
    # Download dataset
    historical_data = yf.download("JPY=X", start="2021-01-01", end="2025-01-01")
    historical_data = pd.DataFrame(historical_data)
    historical_data.to_csv("data/usdjpy_long.csv")
    
    # Generate models
    horizon = 20
    batch_size = 32
    epochs = 1000
    for param_set in param_schedule:
        rsi_period, wil_period, adx_period = param_set
        print("Params:", param_set)
        
        # Compute features, normalize
        l1_return = np.diff(historical_data["Close"].to_numpy(), axis=0)   
        rsi_data = get_rsi(historical_data, period=rsi_period) / 100.0
        wil_data = get_williamsr(historical_data, period=wil_period) / -100.0
        adx_data = get_adx(historical_data, period=adx_period) / 100.0
        features = np.concatenate((l1_return[199:], rsi_data[200:], wil_data[200:], adx_data[200:]), axis=-1)
        
        # Create param-wise data loader
        train_dataset = USDJPY(features[:-100], period=horizon) # reserve last 100 steps for forecasting
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train (using best params from Experiment 2)
        model = LSTNet(period=horizon, num_features=4, rnn_dim=64, cnn_dim=64, skip_dim=32)
        loss_fn = MSELoss()
        optim = Adam(model.parameters(), lr=1e-3)
        loss_history = []
        for epoch in range(epochs):
            loss = train(train_loader, model, optim, loss_fn)
            loss_history.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} --- loss: {loss:.6f}")

        # Save loss history
        plt.plot(loss_history, color="blue", label="train loss")
        plt.title(create_fnamne(rsi_period, wil_period, adx_period, "") + " Loss History")
        plt.legend()
        plt.savefig(f"models2/{create_fnamne(rsi_period, wil_period, adx_period, '.png')}")
        plt.close()
        
        # Save model
        torch.save(model.state_dict(), f"models2/{create_fnamne(rsi_period, wil_period, adx_period, '.pth')}")
        
        # Clear lingering gradients
        del model
        torch.cuda.empty_cache()