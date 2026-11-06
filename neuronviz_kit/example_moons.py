import torch, torch.nn as nn
from sklearn.datasets import make_moons
import neuronviz as nv

X, y = make_moons(500, noise=0.2, random_state=0)
X, y = torch.tensor(X).float(), torch.tensor(y).float()
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(),
                      nn.Linear(16, 16), nn.ReLU(),
                      nn.Linear(16, 1))

nv.attach(model, X, y)          # <- the only line you add

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(500):
    loss = loss_fn(model(X).squeeze(1), y)
    opt.zero_grad(); loss.backward()
    opt.step()
