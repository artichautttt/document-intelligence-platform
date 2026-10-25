"""Agent de données text-to-SQL sur une base SQLite publique et locale.

La base (`nyc_taxi_2025_01.sqlite`) agrège les trajets NYC Yellow Taxi de
janvier 2025 (données publiques de la TLC) ; elle est produite par
`scripts/build_taxi_db.py`.
"""

from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).with_name("nyc_taxi_2025_01.sqlite")

__all__ = ["DEFAULT_DB_PATH"]
