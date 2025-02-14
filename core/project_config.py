from pathlib import Path

DEVICE = "cuda:0"
CACHE_DIR = "/share/u/models"

ROOT_DIR = Path(__file__).parent.parent
INPUT_DIR = ROOT_DIR / "artifacts" / "input"
INTERIM_DIR = ROOT_DIR / "artifacts" / "interim"
RESULT_DIR = ROOT_DIR / "artifacts" / "result"

with open(INPUT_DIR / "ant.txt", "r") as f:
    ANT = f.read().strip()
