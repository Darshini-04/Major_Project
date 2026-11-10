from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_PATH = Path(os.getenv("PULSEIQ_DATA_PATH", ROOT / "data/raw/online_retail_II.xlsx")).expanduser()
OUTPUT_DIR = ROOT / "outputs"
