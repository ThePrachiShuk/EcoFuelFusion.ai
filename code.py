import os
import time
import warnings
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import optuna

# ========== CONFIGURATION ==========
CONFIG = {
    "SEED": 42,
    "N_SPLITS": 5,
    "N_PROPERTIES": 10,
    "N_COMPONENTS": 5,
    "BATCH_SIZE": 128,
    "EPOCHS": 100,
    "PATIENCE": 10,
    "N_TRIALS": 20,
    "TRAIN_PATH": "train.csv",
    "TEST_PATH": "test.csv",
    "OUTPUT_CSV": "submission.csv"
}

# ========== LOGGING SETUP ==========
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger()

# ========== SEED & DEVICE ==========
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["SEED"])

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
log.info(f"Using device: {device}")

# ========== DATA LOADING ==========
def load_data(train_path, test_path):
    df = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    ids = df_test.pop("ID")
    return df, df_test, ids

# ========== FEATURE ENGINEERING ==========
def feature_engineering(df, df_test, n_components, n_properties):
    for i in range(1, n_properties + 1):
        weighted_sum = np.zeros(df.shape[0])
        weighted_sum_test = np.zeros(df_test.shape[0])
        comp_props = []

        for j in range(1, n_components + 1):
            frac_col = f"Component{j}_fraction"
            prop_col = f"Component{j}_Property{i}"
            w_col = f"Component{j}_Weighted_Property{i}"

            df[w_col] = df[frac_col] * df[prop_col]
            df_test[w_col] = df_test[frac_col] * df_test[prop_col]

            weighted_sum += df[w_col]
            weighted_sum_test += df_test[w_col]
            comp_props.append(df[prop_col])

        df[f"Weighted_Property_{i}"] = weighted_sum
        df_test[f"Weighted_Property_{i}"] = weighted_sum_test

        df[f"Prop{i}_mean"] = np.mean(comp_props, axis=0)
        df[f"Prop{i}_std"] = np.std(comp_props, axis=0)
        df_test[f"Prop{i}_mean"] = np.mean([df_test[f"Component{j}_Property{i}"] for j in range(1, n_components + 1)], axis=0)
        df_test[f"Prop{i}_std"] = np.std([df_test[f"Component{j}_Property{i}"] for j in range(1, n_components + 1)], axis=0)

    return df, df_test

# ========== ANN MODEL ==========
class ANN(nn.Module):
    def __init__(self, input_dim, params):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, params['units1']),
            nn.ReLU(),
            nn.BatchNorm1d(params['units1']),
            nn.Dropout(params['dropout1']),
            nn.Linear(params['units1'], params['units2']),
            nn.ReLU(),
            nn.Dropout(params['dropout2']),
            nn.Linear(params['units2'], 1)
        )

    def forward(self, x):
        return self.net(x)

def train_ann(model, optimizer, criterion, train_loader, X_val, y_val):
    best_loss = np.inf
    patience_counter = 0
    best_model = model.state_dict()

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                break

    model.load_state_dict(best_model)
    return model

# ========== OPTUNA TUNING ==========
def tune_tree_model(model_class, param_space, X, y, kf, fit_args=None):
    def objective(trial):
        params = {k: v(trial) for k, v in param_space.items()}
        params.update(fit_args or {})
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = model_class(**params)
            model.fit(X[train_idx], y[train_idx].ravel())
            preds = model.predict(X[val_idx]).reshape(-1, 1)
            scores.append(r2_score(y[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CONFIG["N_TRIALS"])
    return study.best_params

def tune_ann(X_scaled, y_scaled, kf, input_dim):
    def objective(trial):
        params = {
            'units1': trial.suggest_int('units1', 64, 256),
            'units2': trial.suggest_int('units2', 32, 128),
            'dropout1': trial.suggest_float('dropout1', 0.2, 0.5),
            'dropout2': trial.suggest_float('dropout2', 0.1, 0.3),
            'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        }
        scores = []

        for train_idx, val_idx in kf.split(X_scaled):
            X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32).to(device)
            y_train = torch.tensor(y_scaled[train_idx], dtype=torch.float32).to(device)
            X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32).to(device)
            y_val = torch.tensor(y_scaled[val_idx], dtype=torch.float32).to(device)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
            model = ANN(input_dim, params).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            criterion = nn.MSELoss()

            model = train_ann(model, optimizer, criterion, train_loader, X_val, y_val)
            with torch.no_grad():
                preds = model(X_val).cpu().numpy()

            scores.append(r2_score(y_scaler.inverse_transform(y_val.cpu().numpy()),
                                   y_scaler.inverse_transform(preds)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CONFIG["N_TRIALS"])
    return study.best_params

# ========== MAIN PIPELINE ==========
if __name__ == "__main__":
    df, df_test, ids = load_data(CONFIG["TRAIN_PATH"], CONFIG["TEST_PATH"])
    df, df_test = feature_engineering(df, df_test, CONFIG["N_COMPONENTS"], CONFIG["N_PROPERTIES"])
    predictions = pd.DataFrame({"ID": ids})

    for i in range(1, CONFIG["N_PROPERTIES"] + 1):
        start_time = time.time()
        target = f"BlendProperty{i}"
        log.info(f"📌 Training for {target}")

        input_cols = [col for col in df.columns if "BlendProperty" not in col and col != "ID"]
        X = df[input_cols].values
        y = df[target].values.reshape(-1, 1)
        X_test = df_test[input_cols].values

        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        X_scaled, y_scaled = x_scaler.fit_transform(X), y_scaler.fit_transform(y)
        X_test_scaled = x_scaler.transform(X_test)

        kf = KFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["SEED"])

        # --- Define search spaces ---
        xgb_space = {
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
            "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0)
        }
        xgb_args = {"n_estimators": 300, "verbosity": 0}

        lgb_space = {
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
            "num_leaves": lambda t: t.suggest_int("num_leaves", 20, 150),
            "feature_fraction": lambda t: t.suggest_float("feature_fraction", 0.5, 1.0),
        }
        lgb_args = {"n_estimators": 300}

        cat_space = {
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.1),
            "depth": lambda t: t.suggest_int("depth", 4, 10),
            "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
        cat_args = {"iterations": 300, "verbose": 0}

        ann_params = tune_ann(X_scaled, y_scaled, kf, X_scaled.shape[1])
        xgb_params = tune_tree_model(XGBRegressor, xgb_space, X_scaled, y_scaled, kf, xgb_args)
        lgb_params = tune_tree_model(LGBMRegressor, lgb_space, X_scaled, y_scaled, kf, lgb_args)
        cat_params = tune_tree_model(CatBoostRegressor, cat_space, X_scaled, y_scaled, kf, cat_args)

        oof_preds, test_preds = [], []

        for train_idx, val_idx in tqdm(list(kf.split(X_scaled)), desc=f"{target} CV"):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

            # ANN
            model = ANN(X_train.shape[1], ann_params).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=ann_params['lr'])
            criterion = nn.MSELoss()

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train).float().to(device), torch.tensor(y_train).float().to(device)),
                batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

            model = train_ann(model, optimizer, criterion, train_loader,
                              torch.tensor(X_val).float().to(device),
                              torch.tensor(y_val).float().to(device))

            y_val_ann = y_scaler.inverse_transform(model(torch.tensor(X_val).float().to(device)).cpu().detach().numpy())
            y_test_ann = y_scaler.inverse_transform(model(torch.tensor(X_test_scaled).float().to(device)).cpu().detach().numpy())

            # Other models
            models = {
                "xgb": XGBRegressor(**xgb_params, **xgb_args),
                "lgb": LGBMRegressor(**lgb_params, **lgb_args),
                "cat": CatBoostRegressor(**cat_params, **cat_args),
                "svr": SVR(C=5),
                "lin": LinearRegression()
            }

            stacked_val, stacked_test = [y_val_ann], [y_test_ann]

            for name, m in models.items():
                m.fit(X_train, y_train.ravel())
                pred_val = y_scaler.inverse_transform(m.predict(X_val).reshape(-1, 1))
                pred_test = y_scaler.inverse_transform(m.predict(X_test_scaled).reshape(-1, 1))
                stacked_val.append(pred_val)
                stacked_test.append(pred_test)

            stacked_val = np.hstack(stacked_val)
            stacked_test = np.hstack(stacked_test)

            meta = GradientBoostingRegressor(n_estimators=100)
            meta.fit(stacked_val, y[val_idx])
            y_val_meta = meta.predict(stacked_val)
            y_test_meta = meta.predict(stacked_test)

            oof_preds.append(y_val_meta)
            test_preds.append(y_test_meta.reshape(-1, 1))

            del model
            torch.cuda.empty_cache()

        final_val = np.concatenate(oof_preds)
        final_test = np.mean(test_preds, axis=0)
        score = r2_score(y, final_val)
        log.info(f"✅ {target} R2 Score: {score:.4f} | Time: {time.time() - start_time:.2f}s")
        predictions[target] = final_test.flatten()

    predictions.to_csv(CONFIG["OUTPUT_CSV"], index=False)
    log.info(f"🎯 All done! Predictions saved to {CONFIG['OUTPUT_CSV']}")
