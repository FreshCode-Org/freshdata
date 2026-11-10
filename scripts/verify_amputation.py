from pathlib import Path
import sys

FORBIDDEN_PATHS = [
    "src/freshdata/compliance",
    "src/freshdata/enterprise",
    "src/freshdata/parsers",
    "src/freshdata/domains",
    "src/freshdata/simple.py",
    "src/freshdata/context",
    "src/freshdata/learning",
    "src/freshdata/engine/backends/_spark.py",
    "src/freshdata/engine/backends/_freshcore.py",
    "tests/compliance",
    "tests/domains",
    "tests/parsers",
    "tests/context",
    "tests/learning",
]

repo_root = Path(__file__).resolve().parent.parent
found = [p for p in FORBIDDEN_PATHS if (repo_root / p).exists()]
if found:
    print(f"FAILED: Found amputated paths: {found}")
    sys.exit(1)
print("SUCCESS: All bloated paths removed.")
