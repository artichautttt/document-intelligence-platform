"""Description des tables de la base taxi, pour le modèle et pour les clients MCP.

Rédigée à la main (le modèle n'a pas accès à `sqlite_master`, refusé par
l'autorisateur) ; `tests/data_agent/test_schema.py` vérifie qu'elle correspond
exactement aux colonnes réelles de la base.
"""

from pydantic import BaseModel


class ColumnDoc(BaseModel):
    name: str
    type: str
    description: str


class TableDoc(BaseModel):
    name: str
    description: str
    columns: list[ColumnDoc]


class DatabaseSchema(BaseModel):
    description: str
    tables: list[TableDoc]


def _table(name: str, description: str, *columns: tuple[str, str, str]) -> TableDoc:
    return TableDoc(
        name=name,
        description=description,
        columns=[ColumnDoc(name=n, type=t, description=d) for n, t, d in columns],
    )


SCHEMA = DatabaseSchema(
    description=(
        "Trajets des taxis jaunes de New York en janvier 2025 (données publiques "
        "NYC TLC), agrégés. Aucune ligne par trajet : seulement des totaux. "
        "Montants en dollars US, distances en miles. Dates au format 'AAAA-MM-JJ' "
        "(texte), toutes en janvier 2025. Les moyennes se calculent en divisant des "
        "totaux (ex. SUM(amount_usd_total) / SUM(trip_count))."
    ),
    tables=[
        _table(
            "zones",
            "Les 265 zones de taxi de la TLC.",
            ("location_id", "INTEGER", "identifiant de zone (clé primaire)"),
            (
                "borough",
                "TEXT",
                "arrondissement : Manhattan, Brooklyn, Queens, Bronx, "
                "Staten Island, EWR (aéroport de Newark) ou Unknown",
            ),
            ("zone", "TEXT", "nom de la zone (ex. 'JFK Airport', 'Midtown Center')"),
            (
                "service_zone",
                "TEXT",
                "type de zone (Yellow Zone, Boro Zone, Airports...)",
            ),
        ),
        _table(
            "payment_types",
            "Libellés des modes de paiement (dictionnaire TLC).",
            ("payment_type", "INTEGER", "code du mode de paiement (clé primaire)"),
            (
                "label",
                "TEXT",
                "libellé : 'Flex Fare trip', 'Credit card', 'Cash', 'No charge', "
                "'Dispute', 'Unknown', 'Voided trip'",
            ),
        ),
        _table(
            "daily_zone_trips",
            "Totaux par jour et par zone de prise en charge.",
            ("trip_date", "TEXT", "jour de prise en charge"),
            ("pickup_location_id", "INTEGER", "zone de départ (zones.location_id)"),
            ("trip_count", "INTEGER", "nombre de trajets"),
            (
                "passenger_total",
                "INTEGER",
                "somme des passagers déclarés (valeur absente comptée 0)",
            ),
            ("distance_mi_total", "REAL", "distance totale (miles)"),
            ("fare_usd_total", "REAL", "total des tarifs au compteur (fare_amount)"),
            ("tip_usd_total", "REAL", "total des pourboires"),
            ("tolls_usd_total", "REAL", "total des péages"),
            ("amount_usd_total", "REAL", "total payé par les clients (total_amount)"),
        ),
        _table(
            "hourly_trips",
            "Totaux par jour et par heure de prise en charge (toutes zones).",
            ("trip_date", "TEXT", "jour de prise en charge"),
            ("pickup_hour", "INTEGER", "heure de prise en charge, 0 à 23"),
            ("trip_count", "INTEGER", "nombre de trajets"),
            ("distance_mi_total", "REAL", "distance totale (miles)"),
            ("duration_min_total", "REAL", "durée totale des trajets (minutes)"),
            ("amount_usd_total", "REAL", "total payé par les clients (total_amount)"),
        ),
        _table(
            "payment_daily",
            "Totaux par jour et par mode de paiement (toutes zones).",
            ("trip_date", "TEXT", "jour de prise en charge"),
            (
                "payment_type",
                "INTEGER",
                "mode de paiement (payment_types.payment_type)",
            ),
            ("trip_count", "INTEGER", "nombre de trajets"),
            ("tip_usd_total", "REAL", "total des pourboires"),
            ("amount_usd_total", "REAL", "total payé par les clients (total_amount)"),
        ),
        _table(
            "od_monthly",
            "Totaux du mois par couple (zone de départ, zone d'arrivée).",
            ("pickup_location_id", "INTEGER", "zone de départ (zones.location_id)"),
            ("dropoff_location_id", "INTEGER", "zone d'arrivée (zones.location_id)"),
            ("trip_count", "INTEGER", "nombre de trajets"),
            ("distance_mi_total", "REAL", "distance totale (miles)"),
            ("amount_usd_total", "REAL", "total payé par les clients (total_amount)"),
        ),
    ],
)


def schema_as_text(schema: DatabaseSchema = SCHEMA) -> str:
    """Schéma lisible, inséré dans le prompt système de l'agent."""
    lines = [schema.description, ""]
    for table in schema.tables:
        lines.append(f"Table {table.name} : {table.description}")
        lines.extend(
            f"  - {c.name} ({c.type}) : {c.description}" for c in table.columns
        )
        lines.append("")
    return "\n".join(lines).rstrip()
