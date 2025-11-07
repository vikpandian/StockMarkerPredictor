# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error

# %%
device = torch.device('cpu')
device

# %%
ticker = 'AAPL'
df = yf.download(ticker,'2020-01-01')
dff = yf.download(ticker,'2020-01-01')

# %%
df

# %%
df.Close.plot()

# %%
scaler = StandardScaler()

df['Close'] = scaler.fit_transform(df['Close'])

# %%
df

# %%
dff

# %% [markdown]
# Can see Close price is now scaled

# %% [markdown]
# Goal is to be able to use data over any period of time to predict the stock price. So LSTM is best option

# %%
seq_length = 30
data = []

for i in range(len(df) - seq_length):
    data.append(df.Close[i:i+seq_length])

data = np.array(data)

# %% [markdown]
# Each window has 30 days. It is a sliding window of stock Close values starting from day 0 to day 29. Keep doing this until there is no more room.

# %%
type(data)

# %%
train_size = int(0.8 * len(data))

X_train = torch.from_numpy(data[:train_size, : -1, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

# %%
y_train

# %% [markdown]
# Have our data in tensors

# %%
class PredictionModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel,self).__init__()

        self.num_layers =num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, (hn,cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])

        return out

# %%
model = PredictionModel(input_dim=1, hidden_dim= 32, num_layers=2, output_dim=1).to(device)

# %%
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# %%
num_epochs = 200

for i in range(num_epochs):
    y_train_pred = model(X_train)

    loss = criterion(y_train_pred, y_train)

    if i % 25 == 0:
        print(i,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# %%
model.eval()

y_test_pred = model(X_test)

# %%
print(y_test_pred)
print(y_test_pred.size())

# %% [markdown]
# Need to inverse the scaler transformation and turn the tensor back into anumpy array, so we can compare the predictated data to the actual data

# %%
y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())


# %%
y_test_pred.shape

# %%
train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:,0])
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])

# %%
train_rmse

# %%
test_rmse

# %%
graph = plt.figure(figsize=(12,10))

gs = graph.add_gridspec(4,1)
axl = graph.add_subplot(gs[:3,0])

axl.plot(df.iloc[-len(y_test):].index, y_test, color = 'blue', label = 'Actual Price')
axl.plot(df.iloc[-len(y_test):].index, y_test_pred, color = 'green', label = 'Prediction Price')
axl.legend()

plt.title(f"{ticker} Stock Price Predictor")
plt.xlabel('Date')
plt.ylabel('Price')

ax2 = graph.add_subplot(gs[3,0])
ax2.axhline(test_rmse, color = 'black', linestyle = '--', label = 'RMSE')
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), 'red', label = 'Prediction Error')
ax2.legend()
plt.title('Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error')

plt.tight_layout()
plt.show()
