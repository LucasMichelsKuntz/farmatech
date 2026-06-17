import sqlite3
from contextlib import contextmanager
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / "farmtech.db"


@contextmanager
def connection():
    conn = sqlite3.connect(_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
