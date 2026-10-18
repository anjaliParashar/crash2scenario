"""Repository paths. Nothing in this package depends on a machine-specific directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_crash_csv() -> Path:
    return ROOT / "data" / "Incident_Report_2021.csv"


def scenarios_dir() -> Path:
    return ROOT / "scenarios"
