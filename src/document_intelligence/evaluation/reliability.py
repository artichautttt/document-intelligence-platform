"""Statistiques d'accord entre notes ordinales (0, 1, 2) : juge vs juge, juge vs humain.

Fonctions pures, sans dépendance externe. Une métrique non définie (aucune
paire, variance nulle, accord attendu nul) vaut `None` : elle n'est jamais
remplacée par une valeur arbitraire.
"""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations

LEVELS = (0, 1, 2)

Metric = Callable[[Sequence[int], Sequence[int]], float | None]


def _check_pairs(a: Sequence[int], b: Sequence[int]) -> None:
    if len(a) != len(b):
        raise ValueError(f"séries de longueurs différentes : {len(a)} et {len(b)}")


def exact_agreement(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Part des items notés à l'identique ; `None` sans item."""
    _check_pairs(a, b)
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def quadratic_weighted_kappa(
    a: Sequence[int], b: Sequence[int], levels: Sequence[int] = LEVELS
) -> float | None:
    """Kappa de Cohen à pondération quadratique.

    κ = 1 − Σ w·O / Σ w·E, avec w(i, j) = (i − j)² / (k − 1)², O la matrice
    des proportions observées et E le produit des marges. `None` si Σ w·E = 0
    (par exemple quand les deux séries sont constantes et égales).

    Raises:
        ValueError: longueurs différentes ou note hors de `levels`.
    """
    _check_pairs(a, b)
    if not a or len(levels) < 2:
        return None
    index = {level: i for i, level in enumerate(levels)}
    unknown = sorted({x for x in (*a, *b) if x not in index})
    if unknown:
        raise ValueError(f"notes hors échelle {list(levels)} : {unknown}")

    k, n = len(levels), len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1 / n
    rows = [sum(row) for row in observed]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:
        return (i - j) ** 2 / (k - 1) ** 2

    disagreement = sum(
        weight(i, j) * observed[i][j] for i in range(k) for j in range(k)
    )
    expected = sum(weight(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k))
    if expected <= 1e-12:
        return None
    return 1 - disagreement / expected


def _average_ranks(values: Sequence[int]) -> list[float]:
    """Rangs (à partir de 1), ex aequo au rang moyen."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = rank
        start = end + 1
    return ranks


def spearman(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Corrélation de Spearman (Pearson sur rangs moyens) ; `None` si une série est constante."""
    _check_pairs(a, b)
    if len(a) < 2:
        return None
    ra, rb = _average_ranks(a), _average_ranks(b)
    mean_a, mean_b = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def all_passes_agree(passes: Sequence[Sequence[int]]) -> float | None:
    """Part des items dont toutes les passes donnent la même note."""
    if not passes or not passes[0]:
        return None
    n = len(passes[0])
    if any(len(p) != n for p in passes):
        raise ValueError("toutes les passes doivent noter les mêmes items")
    return sum(len({p[i] for p in passes}) == 1 for i in range(n)) / n


def pairwise(metric: Metric, passes: Sequence[Sequence[int]]) -> list[float | None]:
    """`metric` sur chaque paire de passes (1-2, 1-3, 2-3...)."""
    return [metric(x, y) for x, y in combinations(passes, 2)]


def mean_if_all_defined(values: Sequence[float | None]) -> float | None:
    """Moyenne, ou `None` si une valeur est non définie (rien n'est masqué)."""
    defined = [v for v in values if v is not None]
    if not values or len(defined) != len(values):
        return None
    return sum(defined) / len(defined)


def median_score(scores: Sequence[int]) -> int:
    """Note médiane d'un nombre impair de passes."""
    if len(scores) % 2 == 0:
        raise ValueError("la médiane exige un nombre impair de passes")
    return sorted(scores)[len(scores) // 2]


@dataclass(frozen=True)
class BootstrapInterval:
    """Intervalle percentile ; `defined` = rééchantillons où la métrique existe."""

    low: float
    high: float
    level: float
    resamples: int
    defined: int


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Percentile par interpolation linéaire (convention de numpy par défaut)."""
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def bootstrap_ci(
    a: Sequence[int],
    b: Sequence[int],
    metric: Metric,
    *,
    resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> BootstrapInterval | None:
    """IC bootstrap percentile de `metric`, items rééchantillonnés avec remise.

    Les rééchantillons où la métrique est non définie sont écartés et comptés
    (`resamples - defined`). `None` si aucun rééchantillon n'est exploitable.
    """
    _check_pairs(a, b)
    if not a:
        return None
    rng = random.Random(seed)
    n = len(a)
    values: list[float] = []
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        value = metric([a[i] for i in picks], [b[i] for i in picks])
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    tail = (1 - level) / 2
    return BootstrapInterval(
        low=_percentile(values, tail),
        high=_percentile(values, 1 - tail),
        level=level,
        resamples=resamples,
        defined=len(values),
    )
