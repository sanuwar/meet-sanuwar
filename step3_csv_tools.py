import csv, datetime
from pathlib import Path

# ---- paths (root-level) ----
PROJECT_ROOT = Path.cwd().resolve()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LEADS_CSV = DATA_DIR / "leads.csv"
UNKNOWN_CSV = DATA_DIR / "unknown_questions.csv"

def _write_row(path: Path, headers, row: dict):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(headers)
        w.writerow([row.get(h, "") for h in headers])

def record_user_details(name="", email = ""):
    _write_row(
        LEADS_CSV, ["timestamp", "name", "email"], 
        {"timestamp": datetime.datetime.utcnow().isoformat(), "name": name, "email": email}
    )

def log_unknown(question: str):
    _write_row(
        UNKNOWN_CSV, ["timestamp", "question"],
        {"timestamp": datetime.datetime.utcnow().isoformat(), "question": question}
    )


# quick smoke test (creates the files)
record_user_details("Jon Doe", "jon@doe.com")
log_unknown("tell me the name of Sanuwar's first car")
print("leads CSV:", LEADS_CSV)
print("unknown CSV:", UNKNOWN_CSV)
