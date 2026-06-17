import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.enums import ModelType
from ml.features import FEATURES_CLF, FEATURES_REG, TARGET_CLF


@dataclass
class RegressionResult:
    metrics:  pd.DataFrame
    models:   dict
    best:     Pipeline
    features: list[str]
    X_test:   pd.DataFrame
    y_test:   pd.Series


def _build_pipeline(model_type: ModelType) -> Pipeline:
    if model_type == ModelType.RANDOM_FOREST:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)),
        ])
    if model_type == ModelType.RIDGE:
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])


def _regression_metrics(y_true, y_pred, model_type: ModelType) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {
        "model": model_type.value,
        "MAE":   round(mae, 2),
        "MSE":   round(mse, 2),
        "RMSE":  round(rmse, 2),
        "R²":    round(r2, 4),
    }


def train_regression(df: pd.DataFrame, target: str, label: str) -> RegressionResult:
    features = [f for f in FEATURES_REG if f != target]
    features += ["npk_total", "npk_ratio_nk"]
    features  = [f for f in features if f in df.columns]

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rows, trained = [], {}
    for model_type in ModelType:
        pipe = _build_pipeline(model_type)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        m = _regression_metrics(y_test, y_pred, model_type)
        rows.append(m)
        trained[model_type] = pipe
        print(f"  [{label}][{model_type.value}] R²={m['R²']}  RMSE={m['RMSE']}  MAE={m['MAE']}")

    best_row  = max(rows, key=lambda x: x["R²"])
    best_type = next(mt for mt in ModelType if mt.value == best_row["model"])

    return RegressionResult(
        metrics=pd.DataFrame(rows),
        models=trained,
        best=trained[best_type],
        features=features,
        X_test=X_test,
        y_test=y_test,
    )


def train_classification(df: pd.DataFrame, le: LabelEncoder):
    X = df[FEATURES_CLF]
    y = df[TARGET_CLF]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n  [Classification] Accuracy: {acc:.4f}")
    return clf, acc
