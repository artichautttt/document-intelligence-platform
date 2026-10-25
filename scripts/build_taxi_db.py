"""Construit la base SQLite agrégée de l'agent de données (NYC Yellow Taxi, janvier 2025).

Source publique : NYC Taxi & Limousine Commission (TLC), Trip Record Data.
- trajets : https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
- zones   : https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

Le fichier brut (≈ 3,5 millions de trajets, 59 Mo) n'est pas versionné : seule
la base agrégée produite ici l'est. Règles de nettoyage, appliquées dans cet
ordre et comptées dans `build_info` :
1. prise en charge dans [2025-01-01, 2025-02-01) ;
2. durée (dépose − prise en charge) dans [0, 24 h] ;
3. distance, tarif (`fare_amount`) et montant total positifs ou nuls ;
4. distance au plus 100 miles : au-delà, des valeurs de saisie aberrantes
   (jusqu'à 276 424 miles) ; 162 trajets bruts dépassent 100 miles mais
   représentent près de la moitié de la distance totale du fichier.

    uv run python scripts/build_taxi_db.py --trips yellow_tripdata_2025-01.parquet \
        --zones taxi_zone_lookup.csv

Dépendances : pandas et pyarrow (présents dans l'environnement de développement,
non requis à l'exécution : la base se lit avec le module `sqlite3` standard).
"""

import argparse
import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT = Path("src/document_intelligence/data_agent/nyc_taxi_2025_01.sqlite")
MONTH_START, MONTH_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")
MAX_DISTANCE_MI = 100

# Dictionnaire de données TLC (Yellow Trips Data Dictionary, champ payment_type).
PAYMENT_TYPES = [
    (0, "Flex Fare trip"),
    (1, "Credit card"),
    (2, "Cash"),
    (3, "No charge"),
    (4, "Dispute"),
    (5, "Unknown"),
    (6, "Voided trip"),
]

SCHEMA = """
CREATE TABLE zones (
    location_id INTEGER PRIMARY KEY,
    borough TEXT NOT NULL,
    zone TEXT NOT NULL,
    service_zone TEXT NOT NULL
);
CREATE TABLE payment_types (
    payment_type INTEGER PRIMARY KEY,
    label TEXT NOT NULL
);
CREATE TABLE daily_zone_trips (
    trip_date TEXT NOT NULL,
    pickup_location_id INTEGER NOT NULL REFERENCES zones(location_id),
    trip_count INTEGER NOT NULL,
    passenger_total INTEGER NOT NULL,
    distance_mi_total REAL NOT NULL,
    fare_usd_total REAL NOT NULL,
    tip_usd_total REAL NOT NULL,
    tolls_usd_total REAL NOT NULL,
    amount_usd_total REAL NOT NULL,
    PRIMARY KEY (trip_date, pickup_location_id)
);
CREATE TABLE hourly_trips (
    trip_date TEXT NOT NULL,
    pickup_hour INTEGER NOT NULL,
    trip_count INTEGER NOT NULL,
    distance_mi_total REAL NOT NULL,
    duration_min_total REAL NOT NULL,
    amount_usd_total REAL NOT NULL,
    PRIMARY KEY (trip_date, pickup_hour)
);
CREATE TABLE payment_daily (
    trip_date TEXT NOT NULL,
    payment_type INTEGER NOT NULL REFERENCES payment_types(payment_type),
    trip_count INTEGER NOT NULL,
    tip_usd_total REAL NOT NULL,
    amount_usd_total REAL NOT NULL,
    PRIMARY KEY (trip_date, payment_type)
);
CREATE TABLE od_monthly (
    pickup_location_id INTEGER NOT NULL REFERENCES zones(location_id),
    dropoff_location_id INTEGER NOT NULL REFERENCES zones(location_id),
    trip_count INTEGER NOT NULL,
    distance_mi_total REAL NOT NULL,
    amount_usd_total REAL NOT NULL,
    PRIMARY KEY (pickup_location_id, dropoff_location_id)
);
CREATE TABLE build_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def clean(trips: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Applique les règles de nettoyage et compte les lignes écartées par règle."""
    counts = {"raw_rows": len(trips)}
    pickup = trips["tpep_pickup_datetime"]
    trips = trips[(pickup >= MONTH_START) & (pickup < MONTH_END)]
    counts["dropped_outside_month"] = counts["raw_rows"] - len(trips)

    duration = (
        trips["tpep_dropoff_datetime"] - trips["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    before = len(trips)
    trips = trips.assign(duration_min=duration)[(duration >= 0) & (duration <= 1440)]
    counts["dropped_bad_duration"] = before - len(trips)

    before = len(trips)
    trips = trips[
        (trips["trip_distance"] >= 0)
        & (trips["fare_amount"] >= 0)
        & (trips["total_amount"] >= 0)
    ]
    counts["dropped_negative_amounts"] = before - len(trips)

    before = len(trips)
    trips = trips[trips["trip_distance"] <= MAX_DISTANCE_MI]
    counts["dropped_distance_over_100mi"] = before - len(trips)
    counts["kept_rows"] = len(trips)
    return trips, counts


def build(trips_path: Path, zones_path: Path, output: Path) -> dict[str, int]:
    trips = pd.read_parquet(
        trips_path,
        columns=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "PULocationID",
            "DOLocationID",
            "payment_type",
            "fare_amount",
            "tip_amount",
            "tolls_amount",
            "total_amount",
        ],
    )
    trips, counts = clean(trips)
    trips = trips.assign(
        trip_date=trips["tpep_pickup_datetime"].dt.strftime("%Y-%m-%d"),
        pickup_hour=trips["tpep_pickup_datetime"].dt.hour,
        passenger_count=trips["passenger_count"].fillna(0).astype("int64"),
    )

    zones = pd.read_csv(zones_path).rename(
        columns={
            "LocationID": "location_id",
            "Borough": "borough",
            "Zone": "zone",
            "service_zone": "service_zone",
        }
    )
    zones = zones.fillna("Unknown")

    daily_zone = (
        trips.groupby(["trip_date", "PULocationID"])
        .agg(
            trip_count=("fare_amount", "size"),
            passenger_total=("passenger_count", "sum"),
            distance_mi_total=("trip_distance", "sum"),
            fare_usd_total=("fare_amount", "sum"),
            tip_usd_total=("tip_amount", "sum"),
            tolls_usd_total=("tolls_amount", "sum"),
            amount_usd_total=("total_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"PULocationID": "pickup_location_id"})
    )
    hourly = (
        trips.groupby(["trip_date", "pickup_hour"])
        .agg(
            trip_count=("fare_amount", "size"),
            distance_mi_total=("trip_distance", "sum"),
            duration_min_total=("duration_min", "sum"),
            amount_usd_total=("total_amount", "sum"),
        )
        .reset_index()
    )
    payment = (
        trips.groupby(["trip_date", "payment_type"])
        .agg(
            trip_count=("fare_amount", "size"),
            tip_usd_total=("tip_amount", "sum"),
            amount_usd_total=("total_amount", "sum"),
        )
        .reset_index()
    )
    od = (
        trips.groupby(["PULocationID", "DOLocationID"])
        .agg(
            trip_count=("fare_amount", "size"),
            distance_mi_total=("trip_distance", "sum"),
            amount_usd_total=("total_amount", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "PULocationID": "pickup_location_id",
                "DOLocationID": "dropoff_location_id",
            }
        )
    )
    for df in (daily_zone, hourly, payment, od):
        for column in df.columns:
            if column.endswith("_total") and df[column].dtype.kind == "f":
                df[column] = df[column].round(2)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    digest = hashlib.sha256(trips_path.read_bytes()).hexdigest()
    with sqlite3.connect(output) as conn:
        conn.executescript(SCHEMA)
        zones[["location_id", "borough", "zone", "service_zone"]].to_sql(
            "zones", conn, if_exists="append", index=False
        )
        conn.executemany("INSERT INTO payment_types VALUES (?, ?)", PAYMENT_TYPES)
        daily_zone.to_sql("daily_zone_trips", conn, if_exists="append", index=False)
        hourly.to_sql("hourly_trips", conn, if_exists="append", index=False)
        payment.to_sql("payment_daily", conn, if_exists="append", index=False)
        od.to_sql("od_monthly", conn, if_exists="append", index=False)
        info = {
            "source": "NYC TLC Yellow Taxi Trip Records, 2025-01",
            "source_file": trips_path.name,
            "source_sha256": digest,
            **{k: str(v) for k, v in counts.items()},
        }
        conn.executemany("INSERT INTO build_info VALUES (?, ?)", info.items())
    with sqlite3.connect(output) as conn:
        conn.execute("VACUUM")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trips", type=Path, required=True)
    parser.add_argument("--zones", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    counts = build(args.trips, args.zones, args.output)
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"{args.output} : {args.output.stat().st_size / 1e6:.2f} Mo")


if __name__ == "__main__":
    main()
