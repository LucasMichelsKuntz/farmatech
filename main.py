import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH
from db.ingest import ingest


def main():
    if not DB_PATH.exists():
        print("Database not found — running initial setup...")
        ingest()

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
