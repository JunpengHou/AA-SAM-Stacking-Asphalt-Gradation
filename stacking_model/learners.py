"""Candidate regressors and finite search spaces. Feature order is preserved."""
from itertools import product
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

SPACES = {
    "RF": {"n_estimators": [64, 128, 256], "max_depth": [4, 8, 16],
           "min_samples_leaf": [1, 3, 8], "max_features": [0.5, 0.75, 1.0]},
    "XGBoost": {"n_estimators": [60, 120, 240], "max_depth": [2, 3, 5],
                "learning_rate": [0.025, 0.06, 0.12], "subsample": [0.75, 1.0]},
    "DT": {"max_depth": [2, 4, 8, 16], "min_samples_leaf": [1, 3, 8, 16]},
    "SVR": {"C": [0.3, 1, 3, 10, 30], "epsilon": [0.02, 0.08, 0.2],
            "gamma": [0.005, 0.02, 0.1, 0.5]},
    "KNN": {"n_neighbors": [1, 3, 5, 9], "p": [1, 2], "weights": ["uniform", "distance"]},
    "MLP": {"hidden_layer_sizes": [(16,), (32,), (32, 16)], "alpha": [0.001, 0.01, 0.1],
            "learning_rate_init": [0.001, 0.003, 0.01]},
    "LSTM": {"hidden": [8, 16, 32], "epochs": [30, 60], "lr": [0.001, 0.003, 0.01]},
}

class TabularLSTM(RegressorMixin, BaseEstimator):
    """One time step per image, all selected descriptors as channels; no invented time axis."""
    def __init__(self, hidden=16, epochs=30, lr=0.003, seed=42):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32).unsqueeze(1)
        target = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1)
        self.cell_ = nn.LSTM(x.shape[-1], self.hidden, batch_first=True)
        self.head_ = nn.Linear(self.hidden, 1)
        opt = torch.optim.Adam(list(self.cell_.parameters()) + list(self.head_.parameters()), lr=self.lr)
        for _ in range(self.epochs):
            for batch in torch.randperm(len(x)).split(64):
                opt.zero_grad()
                output, _ = self.cell_(x[batch])
                loss = nn.functional.mse_loss(self.head_(output[:, -1]).flatten(), target[batch])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X), dtype=torch.float32).unsqueeze(1)
            output, _ = self.cell_(x)
            return self.head_(output[:, -1]).flatten().numpy()

def make_model(name, params, seed):
    if name == "RF":
        return RandomForestRegressor(**params, random_state=seed, n_jobs=1)
    if name == "DT":
        return DecisionTreeRegressor(**params, random_state=seed)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params, random_state=seed, n_jobs=1, objective="reg:squarederror", tree_method="hist")
    if name == "SVR":
        model = SVR(**params)
    elif name == "KNN":
        model = KNeighborsRegressor(**params)
    elif name == "MLP":
        model = MLPRegressor(**params, max_iter=300, random_state=seed)
    else:
        model = TabularLSTM(**params, seed=seed)
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), model), transformer=StandardScaler())

def parameter_grid(name, min_train):
    space = SPACES[name]
    grid, coordinates = [], []
    for ix in product(*(range(len(v)) for v in space.values())):
        pars = {k: v[j] for (k, v), j in zip(space.items(), ix)}
        if name == "KNN" and pars["n_neighbors"] > min_train:
            continue
        grid.append(pars)
        coordinates.append([j / max(1, len(v)-1) for j, v in zip(ix, space.values())])
    return grid, np.asarray(coordinates)
