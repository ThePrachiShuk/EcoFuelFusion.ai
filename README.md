# EcoFuelFusion.ai
# 📌 Blend Property Prediction using Stacking Ensemble and Neural Networks

## 🧠 Approach Summary

This project aims to predict **10 target Blend Properties** from input features describing mixtures of 5 components, each with their own properties and fractions.

We developed a **stacked ensemble regression pipeline** comprising:

### 1. Base Models:
- Artificial Neural Network (PyTorch)
- XGBoost
- LightGBM
- CatBoost
- Support Vector Regressor (SVR)
- Linear Regression

### 2. Meta Model:
- `GradientBoostingRegressor` to combine predictions from the base models

Each model was optimized using **Optuna** with cross-validation-based hyperparameter tuning.

---

## ⚙️ Feature Engineering

For each target property (`BlendProperty1` to `BlendProperty10`), we engineered the following:

### 🔹 Weighted Properties
- For each property _i_, calculated:
  ```
  Weighted_Property_i = Σ (Component_j_fraction × Component_j_Property_i) for j = 1 to 5
  ```

### 🔹 Statistical Features
- **Mean**: Average of Component_j_Property_i across j = 1 to 5
- **Standard Deviation**: Std deviation of Component_j_Property_i across j = 1 to 5

All features were engineered identically on training and test sets to avoid data leakage.

---

## 🛠️ Tools and Libraries Used

| Category       | Library / Tool                                  |
|----------------|--------------------------------------------------|
| Programming    | Python 3.13.5                                    |
| ML Libraries   | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `SVR` |
| Deep Learning  | `PyTorch`                                        |
| Optimization   | `Optuna`                                         |
| Utility        | `pandas`, `numpy`, `tqdm`, `logging`             |
| Evaluation     | R² Score (`scikit-learn`)                        |

---

## 🧪 Model Optimization

- Employed **5-fold cross-validation** across all models.
- **ANN, XGBoost, LightGBM, and CatBoost** were optimized using **Optuna** with 20 trials per model per target property.
- Final ensemble predictions were generated using the **GradientBoostingRegressor** meta-model trained on out-of-fold predictions.

---

<!-- ## 📁 Files Included

| File Name      | Description                              |
|----------------|------------------------------------------|
| `new.ipynb`    | Complete training and modeling notebook  |
| `train.csv`    | Provided training dataset                |
| `test.csv`     | Test dataset with ID column              |
| `submission.csv` | Predicted values for the test set      |
| `README.txt`   | Original project documentation           |

--- -->

## 📈 Output Format

Each row in `submission.csv` contains:
- `ID`: Sample identifier (from `test.csv`)
- `BlendProperty1` to `BlendProperty10`: Predicted values

---
