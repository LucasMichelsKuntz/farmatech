import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import CSV_PATH
from db.connection import connection


def ingest():
    df = pd.read_csv(CSV_PATH)
    print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Crops: {sorted(df['label'].unique().tolist())}\n")

    with connection() as conn:
        cur = conn.cursor()
        cur.executescript("""
            DROP TABLE IF EXISTS leituras_sensor;
            DROP TABLE IF EXISTS culturas;

            CREATE TABLE culturas (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT    NOT NULL UNIQUE
            );

            CREATE TABLE leituras_sensor (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                cultura_id INTEGER NOT NULL,
                nitrogenio REAL,
                fosforo    REAL,
                potassio   REAL,
                temperatura REAL,
                umidade    REAL,
                ph         REAL,
                chuva_mm   REAL,
                FOREIGN KEY (cultura_id) REFERENCES culturas(id)
            );
        """)

        crop_names = sorted(df["label"].unique())
        for name in crop_names:
            cur.execute("INSERT INTO culturas (nome) VALUES (?)", (name,))
        conn.commit()

        cur.execute("SELECT nome, id FROM culturas")
        crop_id_map = dict(cur.fetchall())

        rows = [
            (
                crop_id_map[row["label"]],
                row["N"], row["P"], row["K"],
                row["temperature"], row["humidity"],
                row["ph"], row["rainfall"],
            )
            for _, row in df.iterrows()
        ]
        cur.executemany(
            """
            INSERT INTO leituras_sensor
            (cultura_id, nitrogenio, fosforo, potassio, temperatura, umidade, ph, chuva_mm)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    print(f"Database ready: {len(rows)} rows inserted")


if __name__ == "__main__":
    ingest()
