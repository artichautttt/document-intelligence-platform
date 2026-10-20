"""Statistiques d'accord : valeurs connues (calculées à la main, vérifiées avec scipy/sklearn)."""

import pytest

from document_intelligence.evaluation.reliability import (
    all_passes_agree,
    bootstrap_ci,
    exact_agreement,
    mean_if_all_defined,
    median_score,
    pairwise,
    quadratic_weighted_kappa,
    spearman,
)


class TestExactAgreement:
    def test_share_of_identical_scores(self) -> None:
        assert exact_agreement([0, 1, 2, 2], [0, 1, 1, 2]) == 0.75

    def test_empty_is_not_computable(self) -> None:
        assert exact_agreement([], []) is None

    def test_different_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            exact_agreement([0, 1], [0])


class TestQuadraticWeightedKappa:
    def test_perfect_agreement(self) -> None:
        assert quadratic_weighted_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(
            1.0
        )

    def test_known_value_swapped_levels(self) -> None:
        # Σw·O = (0 + 1/4 + 1/4) / 3 = 1/6 ; Σw·E = 3/9 = 1/3 → κ = 1 − 1/2.
        assert quadratic_weighted_kappa([0, 1, 2], [0, 2, 1]) == pytest.approx(0.5)

    def test_known_value_six_items(self) -> None:
        # sklearn cohen_kappa_score(weights="quadratic") donne 0.25.
        assert quadratic_weighted_kappa(
            [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0]
        ) == pytest.approx(0.25)

    def test_maximal_disagreement(self) -> None:
        assert quadratic_weighted_kappa([0, 2], [2, 0]) == pytest.approx(-1.0)

    def test_constant_identical_series_is_not_computable(self) -> None:
        assert quadratic_weighted_kappa([2, 2, 2], [2, 2, 2]) is None

    def test_score_outside_scale_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="hors échelle"):
            quadratic_weighted_kappa([0, 3], [0, 1])


class TestSpearman:
    def test_known_value_with_ties(self) -> None:
        # Exemple de la documentation scipy.stats.spearmanr.
        assert spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(
            0.8207826816681233
        )

    def test_known_value_on_rubric_scale(self) -> None:
        # scipy.stats.spearmanr donne 0.75.
        assert spearman([0, 1, 2, 2, 1, 0], [0, 1, 1, 2, 2, 0]) == pytest.approx(0.75)

    def test_perfect_inverse(self) -> None:
        assert spearman([0, 1, 2], [2, 1, 0]) == pytest.approx(-1.0)

    def test_constant_series_is_not_computable(self) -> None:
        assert spearman([1, 1, 1], [0, 1, 2]) is None
        assert spearman([1], [2]) is None


class TestPasses:
    def test_all_passes_agree(self) -> None:
        passes = [[0, 1, 2, 2], [0, 1, 1, 2], [0, 1, 2, 2]]
        assert all_passes_agree(passes) == 0.75

    def test_pairwise_and_mean(self) -> None:
        passes = [[0, 1, 2], [0, 1, 2], [0, 2, 1]]
        kappas = pairwise(quadratic_weighted_kappa, passes)
        assert kappas == [
            pytest.approx(1.0),
            pytest.approx(0.5),
            pytest.approx(0.5),
        ]
        assert mean_if_all_defined(kappas) == pytest.approx(2 / 3)

    def test_mean_is_not_computable_if_one_pair_is_undefined(self) -> None:
        assert mean_if_all_defined([0.5, None, 1.0]) is None

    def test_median_of_three_passes(self) -> None:
        assert median_score([2, 0, 1]) == 1
        assert median_score([2, 2, 0]) == 2
        with pytest.raises(ValueError):
            median_score([1, 2])


class TestBootstrap:
    def test_identical_series_give_degenerate_interval(self) -> None:
        ci = bootstrap_ci([0, 1, 2, 1], [0, 1, 2, 1], exact_agreement, resamples=200)
        assert ci is not None
        assert (ci.low, ci.high, ci.defined) == (1.0, 1.0, 200)

    def test_reproducible_with_same_seed(self) -> None:
        a, b = [0, 1, 2, 2, 1, 0, 2, 1], [0, 1, 1, 2, 2, 0, 2, 0]
        first = bootstrap_ci(a, b, quadratic_weighted_kappa, resamples=300, seed=7)
        second = bootstrap_ci(a, b, quadratic_weighted_kappa, resamples=300, seed=7)
        assert first == second
        assert first is not None
        assert first.low <= quadratic_weighted_kappa(a, b) <= first.high  # type: ignore[operator]

    def test_undefined_resamples_are_counted_not_replaced(self) -> None:
        # Spearman n'est défini que si les deux séries varient dans le rééchantillon.
        ci = bootstrap_ci([0, 2], [0, 2], spearman, resamples=400, seed=0)
        assert ci is not None
        assert 0 < ci.defined < 400
        assert (ci.low, ci.high) == (pytest.approx(1.0), pytest.approx(1.0))

    def test_never_defined_is_not_computable(self) -> None:
        assert bootstrap_ci([1, 1, 1], [1, 1, 1], spearman, resamples=50) is None
