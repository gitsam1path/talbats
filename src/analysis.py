"""
analysis.py — toute la partie statistique du projet, sans interface.

Séparer les calculs (ici) de l'affichage (app.py) permet de :
  - tester les calculs automatiquement (tests/test_analysis.py) ;
  - réutiliser les mêmes fonctions dans un notebook ou un script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tabBats.txt"
ATYPICAL_SPECIES = "Pteropus vampyrus"
ALPHA = 0.05  # seuil de significativité utilisé dans tout le projet

DIET_LABELS = {1: "Phytophage", 2: "Gleaner", 3: "Insectivore aérien", 4: "Vampire"}
VARIABLES = {
    "BOW": "Masse corporelle",
    "BRW": "Masse du cerveau",
    "AUD": "Volume d'une structure cérébrale (AUD)",
    "MOB": "Volume d'une structure cérébrale (MOB)",
    "HIP": "Volume d'une structure cérébrale (HIP)",
}


# ─────────────────────────────────────────────────────────────
# Partie A — chargement et préparation
# ─────────────────────────────────────────────────────────────
def load_data(source=DATA_PATH) -> pd.DataFrame:
    """Charge tabBats.txt (colonnes séparées par des espaces, noms entre guillemets)."""
    df = pd.read_csv(source, sep=r"\s+", quotechar='"')
    expected = {"Species", "BOW", "BRW"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier : {', '.join(sorted(missing))}")
    return df


def regression_table(df: pd.DataFrame) -> pd.DataFrame:
    """Variables utiles à la régression : BOW (X), BRW (y) et Species pour repérer les points atypiques."""
    return df[["Species", "BOW", "BRW"]].copy()


def without_species(df: pd.DataFrame, species: str = ATYPICAL_SPECIES) -> pd.DataFrame:
    """Tableau tab2 : le même jeu de données sans l'espèce indiquée."""
    mask = df["Species"].str.contains(species, case=False, na=False)
    if not mask.any():
        raise ValueError(f"Espèce introuvable : {species}")
    return df.loc[~mask].copy()


# ─────────────────────────────────────────────────────────────
# Partie B — régression linéaire simple
# ─────────────────────────────────────────────────────────────
def fit_ols(df: pd.DataFrame, x: str = "BOW", y: str = "BRW"):
    """Régression des moindres carrés ordinaires : y = b0 + b1 * x."""
    X = sm.add_constant(df[x])
    return sm.OLS(df[y], X).fit()


def fit_loglog(df: pd.DataFrame):
    """Régression log10(BRW) = b0 + b1 * log10(BOW) (loi d'échelle, ou allométrie)."""
    logged = pd.DataFrame({"logBOW": np.log10(df["BOW"]), "logBRW": np.log10(df["BRW"])})
    return fit_ols(logged, "logBOW", "logBRW")


@dataclass
class ModelSummary:
    n: int
    intercept: float
    slope: float
    p_intercept: float
    p_slope: float
    slope_ci: tuple[float, float]
    r2: float
    r2_adj: float
    rmse: float
    shapiro_p: float  # normalité des résidus (H0 : résidus normaux)
    breusch_pagan_p: float  # homoscédasticité (H0 : variance constante)
    durbin_watson: float  # indépendance (≈ 2 : pas d'autocorrélation)

    @property
    def residuals_normal(self) -> bool:
        return self.shapiro_p > ALPHA

    @property
    def homoscedastic(self) -> bool:
        return self.breusch_pagan_p > ALPHA

    @property
    def independent(self) -> bool:
        return 1.5 < self.durbin_watson < 2.5


def summarize(model) -> ModelSummary:
    """Extrait d'un modèle statsmodels tous les indicateurs commentés dans l'application."""
    params = model.params.to_numpy()
    pvalues = model.pvalues.to_numpy()
    ci = model.conf_int().to_numpy()[1]
    resid = model.resid
    return ModelSummary(
        n=int(model.nobs),
        intercept=float(params[0]),
        slope=float(params[1]),
        p_intercept=float(pvalues[0]),
        p_slope=float(pvalues[1]),
        slope_ci=(float(ci[0]), float(ci[1])),
        r2=float(model.rsquared),
        r2_adj=float(model.rsquared_adj),
        rmse=float(np.sqrt(model.mse_resid)),
        shapiro_p=float(stats.shapiro(resid).pvalue),
        breusch_pagan_p=float(het_breuschpagan(resid, model.model.exog)[1]),
        durbin_watson=float(durbin_watson(resid)),
    )


def correlation_test(df: pd.DataFrame, x: str = "BOW", y: str = "BRW") -> tuple[float, float]:
    """Coefficient de Pearson et p-value du test (H0 : pas de corrélation)."""
    r, p = stats.pearsonr(df[x], df[y])
    return float(r), float(p)


# ─────────────────────────────────────────────────────────────
# Diagnostics : points influents
# ─────────────────────────────────────────────────────────────
def influence_table(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque espèce : résidu studentisé, levier et distance de Cook.
    Un point est jugé influent si sa distance de Cook dépasse 4/n (règle usuelle).
    """
    infl = model.get_influence()
    n = int(model.nobs)
    table = pd.DataFrame(
        {
            "Species": df["Species"].to_numpy(),
            "BOW": df["BOW"].to_numpy(),
            "BRW": df["BRW"].to_numpy(),
            "residu_studentise": infl.resid_studentized_external,
            "levier": infl.hat_matrix_diag,
            "distance_cook": infl.cooks_distance[0],
        }
    )
    table["influent"] = table["distance_cook"] > 4 / n
    return table.sort_values("distance_cook", ascending=False).reset_index(drop=True)


def qq_points(residuals) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles théoriques (loi normale) et quantiles observés des résidus standardisés."""
    standardized = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)
    (theoretical, observed), _ = stats.probplot(standardized, dist="norm")
    return theoretical, observed


# ─────────────────────────────────────────────────────────────
# Commentaires générés à partir des résultats
# ─────────────────────────────────────────────────────────────
def fr(x: float, decimals: int = 2) -> str:
    """Nombre au format français : virgule décimale, espace pour les milliers."""
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else fr(p, 3)


def comment_model(s: ModelSummary, x_label: str = "BOW", y_label: str = "BRW") -> list[str]:
    """Phrases d'interprétation d'un modèle, prêtes à afficher."""
    sig = "significative" if s.p_slope < ALPHA else "non significative"
    lines = [
        f"**Pente** : quand {x_label} augmente d'une unité, {y_label} augmente en moyenne de "
        f"**{fr(s.slope)} unités** (IC 95 % : {fr(s.slope_ci[0])} à {fr(s.slope_ci[1])}). "
        f"Elle est **{sig}** (p {fmt_p(s.p_slope)}).",
        f"**Ordonnée à l'origine** : {fr(s.intercept, 1)} (p {fmt_p(s.p_intercept)}). "
        f"C'est la valeur de {y_label} prédite pour {x_label} = 0 : elle n'a pas de sens biologique "
        "(aucun animal n'a une masse nulle), elle sert seulement à placer la droite.",
        f"**R² = {fr(s.r2, 3)}** : le modèle explique **{fr(s.r2 * 100, 1)} %** de la variabilité de {y_label}. "
        f"Erreur typique de prédiction (RMSE) : {fr(s.rmse, 0)}.",
    ]
    checks = [
        ("Normalité des résidus (Shapiro-Wilk)", s.residuals_normal, s.shapiro_p),
        ("Variance constante (Breusch-Pagan)", s.homoscedastic, s.breusch_pagan_p),
    ]
    for name, ok, p in checks:
        verdict = "✅ vérifiée" if ok else "❌ non vérifiée"
        lines.append(f"**{name}** : {verdict} (p = {fmt_p(p)}).")
    dw_verdict = "✅ pas d'autocorrélation" if s.independent else "⚠️ autocorrélation possible"
    lines.append(f"**Indépendance (Durbin-Watson = {fr(s.durbin_watson)})** : {dw_verdict}.")
    return lines


def compare_models(s1: ModelSummary, s2: ModelSummary) -> pd.DataFrame:
    """Tableau comparatif modèle 1 / modèle 2."""
    rows = [
        ("Nombre d'espèces", s1.n, s2.n),
        ("Ordonnée à l'origine", round(s1.intercept, 2), round(s2.intercept, 2)),
        ("Pente", round(s1.slope, 3), round(s2.slope, 3)),
        ("R²", round(s1.r2, 4), round(s2.r2, 4)),
        ("R² ajusté", round(s1.r2_adj, 4), round(s2.r2_adj, 4)),
        ("RMSE", round(s1.rmse, 1), round(s2.rmse, 1)),
        ("Shapiro-Wilk (p)", round(s1.shapiro_p, 4), round(s2.shapiro_p, 4)),
        ("Breusch-Pagan (p)", round(s1.breusch_pagan_p, 4), round(s2.breusch_pagan_p, 4)),
        ("Durbin-Watson", round(s1.durbin_watson, 3), round(s2.durbin_watson, 3)),
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Modèle 1 (toutes les espèces)", f"Modèle 2 (sans {ATYPICAL_SPECIES})"])
