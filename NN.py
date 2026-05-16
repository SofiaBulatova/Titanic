import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Создаем нейросеть
class MyNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.network(x)

# Определяем обучение и пресказание
class NeuralNet(BaseEstimator, ClassifierMixin):

    def __init__(self, hidden_dims, dropout, lr, epochs, batch_size, random_state, patience, val_fraction, step_size, gamma):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.patience = patience
        self.val_fraction = val_fraction
        self.step_size = step_size
        self.gamma = gamma

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        # Масштабирование
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_fraction,
                                                          random_state=self.random_state, stratify=y)
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)
        X_val_scaled = self.scaler_.transform(X_val)

        input_dim = X_train_scaled.shape[1]
        self.model_ = MyNN(input_dim=input_dim, hidden_dims=self.hidden_dims, dropout=self.dropout)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_model = nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        X_val_t = torch.tensor(X_val_scaled)
        y_val_t = torch.tensor(y_val)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = loss_model(outputs, y_batch)
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
             # Ранняя остановка
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_val_t)
                val_loss = loss_model(val_outputs, y_val_t).item()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, val_loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.tensor(X_scaled)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)