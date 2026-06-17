import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from db.connection import connection

FEATURES_REG = [
    "nitrogenio", "fosforo", "potassio",
    "temperatura", "umidade", "ph", "chuva_mm", "cultura_cod",
]
FEATURES_CLF = ["nitrogenio", "fosforo", "potassio", "temperatura", "umidade", "ph", "chuva_mm"]
TARGET_IRR   = "chuva_mm"
TARGET_FER   = "nitrogenio"
TARGET_CLF   = "cultura_cod"


def load_data() -> pd.DataFrame:
    with connection() as conn:
        return pd.read_sql_query(
            """
            SELECT ls.*, c.nome AS cultura
            FROM leituras_sensor ls
            JOIN culturas c ON c.id = ls.cultura_id
            """,
            conn,
        )


def prepare(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df["cultura_cod"]  = le.fit_transform(df["cultura"])
    df["npk_total"]    = df["nitrogenio"] + df["fosforo"] + df["potassio"]
    df["npk_ratio_nk"] = df["nitrogenio"] / (df["potassio"] + 1)
    return df, le
