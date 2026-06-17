import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from config import MODEL_PATH, MODELS_DIR
from ml.features import FEATURES_CLF, TARGET_FER, TARGET_IRR, load_data, prepare
from ml.training import RegressionResult, train_classification, train_regression


@dataclass
class PipelineResult:
    irr:     RegressionResult
    fer:     RegressionResult
    clf:     RandomForestClassifier
    acc_clf: float
    df:      pd.DataFrame
    le:      LabelEncoder


def run_pipeline() -> PipelineResult:
    MODELS_DIR.mkdir(exist_ok=True)
    print("=== FarmTech ML Pipeline ===\n")

    df = load_data()
    df, le = prepare(df)
    print(f"Dataset: {len(df)} samples | {df['cultura'].nunique()} crops\n")

    print("--- Regression 1: Irrigation (chuva_mm) ---")
    irr = train_regression(df, TARGET_IRR, "Irrigation")

    print("\n--- Regression 2: Fertilization (nitrogenio) ---")
    fer = train_regression(df, TARGET_FER, "Fertilization")

    print("\n--- Classification: Crop Recommendation ---")
    clf, acc_clf = train_classification(df, le)

    joblib.dump({
        "irrigation":     {"model": irr.best, "features": irr.features},
        "fertilization":  {"model": fer.best, "features": fer.features},
        "classification": {"model": clf, "features": FEATURES_CLF, "le": le},
    }, MODEL_PATH)
    print(f"\nModels saved to {MODEL_PATH}")

    return PipelineResult(irr=irr, fer=fer, clf=clf, acc_clf=acc_clf, df=df, le=le)


if __name__ == "__main__":
    run_pipeline()
