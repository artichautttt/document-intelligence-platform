# Jeu d'évaluation text-to-SQL (`taxi_sql_eval.jsonl`)

**Jeu écrit par l'auteur avec l'aide de Claude**, sur la base agrégée NYC Yellow
Taxi de janvier 2025 (`data_agent/nyc_taxi_2025_01.sqlite`, construite par
`scripts/build_taxi_db.py` depuis les données publiques de la TLC). Ce n'est
pas un benchmark public : les questions ont été rédigées en connaissant le
schéma, ce qui peut les rendre plus faciles qu'une question d'utilisateur réel.

## Contenu

40 questions :

| Niveau | Nombre | Ce qu'il exige |
|---|---|---|
| `simple` | 10 | une table, une agrégation ou un filtre |
| `intermediaire` | 12 | une jointure, un regroupement trié, un filtre sur date, une moyenne pondérée |
| `difficile` | 12 | jointures multiples (zones de départ et d'arrivée), fonctions de fenêtre (`RANK`, `LAG`, somme cumulée), ratios conditionnels, CTE |
| `sans_reponse` | 6 | donnée absente de la base (météo, février, chauffeurs, taxis verts, surtaxe de congestion, répartition des passagers) |

## Format (une ligne JSON par question)

- `question`, `difficulty`, `provenance` ;
- `answerable` : `false` pour les questions sans réponse possible (refus attendu),
  avec `why_unanswerable` ;
- `reference_sql` : requête de référence ; `expected` : **sa valeur calculée en
  l'exécutant sur la base**, jamais écrite à la main ;
- `answer_type` (`integer`, `number`, `text`) et `decimals` (précision demandée
  dans la question) : règle de comparaison de la réponse ;
- `tie_check_sql` (questions de classement) : les deux premières valeurs du
  classement, qui doivent différer (pas d'ex æquo).

`tests/data_agent/test_sql_eval_dataset.py` rejoue en CI chaque requête de
référence et chaque contrôle d'ex æquo sur la base versionnée.

## Questions écartées ou reformulées pendant la rédaction (avant tout run du modèle)

- « rang de la zone Midtown Center » : rang 1, doublon de la question sur la zone
  la plus fréquentée ; remplacée par le rang de LaGuardia Airport (14) ;
- « zone d'arrivée la plus fréquente depuis JFK » : la réponse était JFK Airport
  elle-même (trajets internes à la zone) ; reformulée pour exclure la zone de départ ;
- « nombre total de passagers » : écartée, car les trajets sans nombre de
  passagers renseigné sont comptés pour 0, ce qui rend le total ambigu ;
- vérifications faites sans changement : aucun jour n'a exactement 120 000 trajets
  (seuil de `i04`), et le seuil de 50 % de `d11` est franchi nettement (48,5 % à
  14 zones, 50,9 % à 15).

La règle « distance ≤ 100 miles » de la construction de la base a été ajoutée
pendant cette rédaction : 162 trajets bruts aberrants (jusqu'à 276 424 miles)
pesaient près de la moitié de la distance totale et faussaient les questions de
distance et de vitesse.
