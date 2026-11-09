"""Tests des calculs statistiques (lancer : pytest)."""

import numpy as np
import pytest

from src import analysis as an


@pytest.fixture(scope="module")
def tab():
    return an.regression_table(an.load_data())


def test_chargement(tab):
    assert len(tab) == 29
    assert list(tab.columns) == ["Species", "BOW", "BRW"]
    assert tab.isna().sum().sum() == 0


def test_retrait_espece_atypique(tab):
    tab2 = an.without_species(tab)
    assert len(tab2) == 28
    assert an.ATYPICAL_SPECIES not in set(tab2["Species"])


def test_espece_introuvable(tab):
    with pytest.raises(ValueError):
        an.without_species(tab, "Batman")


def test_modele_1(tab):
    s = an.summarize(an.fit_ols(tab))
    assert s.slope == pytest.approx(9.00, abs=0.01)
    assert s.intercept == pytest.approx(623.45, abs=0.01)
    assert s.r2 == pytest.approx(0.950, abs=0.001)
    assert s.p_slope < 1e-10
    assert not s.residuals_normal  # hypothèse violée avec l'espèce atypique


def test_modele_2_meilleur(tab):
    s1 = an.summarize(an.fit_ols(tab))
    s2 = an.summarize(an.fit_ols(an.without_species(tab)))
    assert s2.slope == pytest.approx(14.51, abs=0.01)
    assert s2.r2 > s1.r2
    assert s2.rmse < s1.rmse
    assert s2.residuals_normal and s2.homoscedastic


def test_espece_atypique_la_plus_influente(tab):
    infl = an.influence_table(an.fit_ols(tab), tab)
    assert infl.iloc[0]["Species"] == an.ATYPICAL_SPECIES
    assert infl.iloc[0]["distance_cook"] > 4 / len(tab)


def test_loi_echelle(tab):
    s = an.summarize(an.fit_loglog(tab))
    assert s.slope == pytest.approx(0.692, abs=0.001)
    assert s.slope_ci[1] < 1  # exposant significativement inférieur à 1
    assert s.residuals_normal


def test_qq_points_longueur(tab):
    th, ob = an.qq_points(an.fit_ols(tab).resid.to_numpy())
    assert len(th) == len(ob) == 29
    assert np.all(np.diff(th) > 0)


def test_format_francais():
    assert an.fr(1234.5, 1) == "1 234,5"
    assert an.fmt_p(0.0001) == "< 0,001"
