"""
TalBats — application Streamlit.
Relation entre la masse corporelle (BOW) et la masse du cerveau (BRW) chez 29 espèces de chauves-souris.

Lancer :  streamlit run app.py
Les calculs sont dans src/analysis.py ; ce fichier ne fait que l'affichage.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analysis as an
from src.analysis import fr

# ─────────────────────────────────────────────────────────────
# Configuration et style
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalBats · Régression linéaire", page_icon="🦇", layout="wide")

BLUE, GREEN, RED, GREY = "#4f6bed", "#1f9d6b", "#e5484d", "#8b8fa3"
PLOT_LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13))

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1200px; }
      .hero { padding: 1.6rem 1.8rem; border-radius: 16px; color: #fff;
              background: linear-gradient(120deg, #1d1b4b 0%, #4f3bbd 60%, #7b5cff 100%); }
      .hero h1 { margin: 0 0 .4rem; font-size: 2.2rem; }
      .hero p { margin: 0; opacity: .92; font-size: 1.05rem; }
      .card { border: 1px solid #e6e6ef; border-radius: 12px; padding: 1rem 1.2rem; background: #fafaff; }
      .verdict { border-left: 4px solid #4f3bbd; background: #f4f2ff; padding: .8rem 1rem; border-radius: 0 8px 8px 0; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Données (mises en cache : le calcul n'est fait qu'une fois)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def get_data(uploaded_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_bytes is None:
        return an.load_data()
    from io import BytesIO

    return an.load_data(BytesIO(uploaded_bytes))


with st.sidebar:
    st.markdown("## 🦇 TalBats")
    st.caption("Machine Learning I · TP Régression")
    uploaded = st.file_uploader("Utiliser un autre fichier tabBats.txt", type=["txt", "csv"])
    st.caption("Par défaut, l'application charge `data/tabBats.txt`.")
    st.divider()
    st.markdown(f"Seuil de significativité : **α = {fr(an.ALPHA)}**")
    st.markdown("**Auteur** : Elie Samson Kombiwo")
    st.markdown("[Code source sur GitHub](https://github.com/gitsam1path/talbats)")

try:
    df = get_data(uploaded.getvalue() if uploaded else None)
except Exception as err:  # fichier mal formé
    st.error(f"Impossible de lire le fichier : {err}")
    st.stop()

tab = an.regression_table(df)
tab2 = an.without_species(tab)
model1, model2 = an.fit_ols(tab), an.fit_ols(tab2)
s1, s2 = an.summarize(model1), an.summarize(model2)
influence = an.influence_table(model1, tab)
loglog = an.fit_loglog(tab)
s_log = an.summarize(loglog)
r_pearson, p_pearson = an.correlation_test(tab)
atypical = tab[tab["Species"] == an.ATYPICAL_SPECIES].iloc[0]


# ─────────────────────────────────────────────────────────────
# Fonctions de graphiques (Plotly : survol = nom de l'espèce)
# ─────────────────────────────────────────────────────────────
def scatter(data: pd.DataFrame, title: str, highlight: str | None = an.ATYPICAL_SPECIES, log: bool = False):
    data = data.assign(Groupe=np.where(data["Species"] == highlight, highlight or "", "Autres espèces"))
    fig = px.scatter(
        data, x="BOW", y="BRW", color="Groupe", hover_name="Species",
        color_discrete_map={"Autres espèces": BLUE, an.ATYPICAL_SPECIES: RED},
        log_x=log, log_y=log, title=title,
        labels={"BOW": "BOW — masse corporelle", "BRW": "BRW — masse du cerveau"},
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.update_layout(**PLOT_LAYOUT, legend_title_text="", legend=dict(orientation="h", y=-0.2))
    return fig


def add_line(fig, model, x_min, x_max, name, color, dash="solid", log=False):
    xs = np.logspace(np.log10(x_min), np.log10(x_max), 200) if log else np.linspace(x_min, x_max, 200)
    b0, b1 = model.params.to_numpy()
    ys = 10 ** (b0 + b1 * np.log10(xs)) if log else b0 + b1 * xs
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, line=dict(color=color, width=3, dash=dash)))
    return fig


def residual_panel(model, data: pd.DataFrame, title_suffix: str):
    """Les trois graphiques classiques de diagnostic des résidus."""
    resid = model.resid.to_numpy()
    fitted = model.fittedvalues.to_numpy()
    c1, c2, c3 = st.columns(3)

    f1 = px.scatter(x=fitted, y=resid, hover_name=data["Species"], title="Résidus vs valeurs ajustées",
                    labels={"x": "Valeurs ajustées", "y": "Résidus"})
    f1.add_hline(y=0, line_dash="dash", line_color=RED)
    f1.update_traces(marker=dict(size=9, color=BLUE))
    f1.update_layout(**PLOT_LAYOUT, height=330)
    c1.plotly_chart(f1, use_container_width=True, key=f"rf{title_suffix}")

    th, ob = an.qq_points(resid)
    f2 = go.Figure(go.Scatter(x=th, y=ob, mode="markers", marker=dict(size=9, color=BLUE), name="Résidus"))
    lim = [min(th.min(), ob.min()), max(th.max(), ob.max())]
    f2.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=RED, dash="dash"), name="Loi normale"))
    f2.update_layout(**PLOT_LAYOUT, height=330, title="QQ-plot (normalité)", showlegend=False,
                     xaxis_title="Quantiles théoriques", yaxis_title="Quantiles observés")
    c2.plotly_chart(f2, use_container_width=True, key=f"qq{title_suffix}")

    f3 = px.histogram(x=resid, nbins=12, title="Distribution des résidus", labels={"x": "Résidus"})
    f3.update_traces(marker_color=BLUE)
    f3.update_layout(**PLOT_LAYOUT, height=330, yaxis_title="Nombre d'espèces")
    c3.plotly_chart(f3, use_container_width=True, key=f"hi{title_suffix}")


def show_comments(lines):
    for line in lines:
        st.markdown(f"- {line}")


# ─────────────────────────────────────────────────────────────
# En-tête
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1>🦇 TalBats — le cerveau des chauves-souris</h1>
      <p>Les grandes chauves-souris ont-elles un plus gros cerveau ? Et de combien ?
         Une étude par régression linéaire, de l'exploration des données au diagnostic du modèle.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Espèces étudiées", len(tab))
k2.metric("Corrélation BOW / BRW", fr(r_pearson, 3))
k3.metric("R² sans l'espèce atypique", fr(s2.r2, 3), delta=f"+{fr(s2.r2 - s1.r2, 3)} vs toutes les espèces")
k4.metric("Exposant de la loi d'échelle", fr(s_log.slope, 2), help="Pente de la régression en log-log (partie D)")

tA, tB, tC, tD, tE = st.tabs(
    ["A · Exploration", "B · Régression simple", "C · Espèce atypique", "D · Pour aller plus loin", "✅ Conclusion"]
)

# ─────────────────────────────────────────────────────────────
# Partie A — exploration
# ─────────────────────────────────────────────────────────────
with tA:
    st.subheader("A1–A2 · Les données")
    st.markdown(
        f"Le fichier contient **{len(df)} espèces** et **{df.shape[1]} colonnes**. "
        "Toutes les espèces de ce fichier ont le régime alimentaire **1 (phytophage)**."
        if df["Diet"].nunique() == 1 else f"Le fichier contient **{len(df)} espèces** et **{df.shape[1]} colonnes**."
    )
    st.dataframe(df, use_container_width=True, hide_index=True, height=260)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Statistiques descriptives**")
        st.dataframe(df.drop(columns=["id", "Diet"]).describe().round(2), use_container_width=True)
    with c2:
        st.markdown("**Types et valeurs manquantes**")
        info = pd.DataFrame({"Type": df.dtypes.astype(str), "Valeurs manquantes": df.isna().sum()})
        st.dataframe(info, use_container_width=True)
    st.info(
        f"**À retenir** : les masses sont très dispersées. BOW va de {fr(df.BOW.min(), 1)} à {fr(df.BOW.max(), 0)}, "
        f"mais la médiane n'est que de {fr(df.BOW.median(), 1)} : la plupart des espèces sont petites, "
        "et quelques-unes sont beaucoup plus grandes. Aucune valeur manquante."
    )

    st.subheader("A3 · Quelles variables pour la régression ?")
    num = df.drop(columns=["id", "Diet"]).select_dtypes("number")
    corr = num.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Matrice de corrélation")
    fig.update_layout(**PLOT_LAYOUT, height=430)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, use_container_width=True)
    c2.markdown(
        f"""
        <div class="card">
        <b>Choix des variables</b><br><br>
        • <b>X = BOW</b> (masse corporelle) et <b>y = BRW</b> (masse du cerveau).<br>
        • Corrélation de Pearson : <b>r = {fr(r_pearson, 3)}</b>, p {an.fmt_p(p_pearson)} → relation linéaire
          très forte et significative.<br>
        • <b>Species</b> est conservée pour identifier les points atypiques.<br>
        • AUD, MOB, HIP sont des parties du cerveau : elles décrivent la même chose que BRW
          (MOB–HIP : r = {fr(corr.loc['MOB', 'HIP'], 3)}). On ne les utilise pas en régression simple.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Tableau retenu** (`Species`, `BOW`, `BRW`)")
    st.dataframe(tab, use_container_width=True, hide_index=True, height=200)

# ─────────────────────────────────────────────────────────────
# Partie B — première régression
# ─────────────────────────────────────────────────────────────
with tB:
    st.subheader("B1 · Le nuage de points")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(scatter(tab, "BRW en fonction de BOW (survolez un point)"), use_container_width=True)
    c2.markdown(
        f"""
        <div class="card">
        <b>Tendance observée</b><br><br>
        La masse du cerveau augmente avec la masse corporelle : la relation est <b>croissante et
        globalement linéaire</b>.<br><br>
        <b>Une espèce se détache nettement</b> : <i>{an.ATYPICAL_SPECIES}</i>
        (BOW = {fr(atypical.BOW, 0)}, BRW = {fr(atypical.BRW, 0)}), environ 4 fois plus lourde que la
        deuxième espèce la plus grande. Les autres espèces sont regroupées en bas à gauche.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("B2–B3 · Le modèle BRW ~ BOW (toutes les espèces)")
    st.markdown(
        f'<div class="verdict"><b>Équation</b> : BRW = {fr(s1.intercept)} + {fr(s1.slope)} × BOW'
        f" &nbsp;·&nbsp; R² = {fr(s1.r2, 3)} &nbsp;·&nbsp; n = {s1.n}</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    show_comments(an.comment_model(s1))

    st.markdown("**Analyse des résidus**")
    residual_panel(model1, tab, "m1")
    st.warning(
        "Le R² est élevé, **mais les hypothèses du modèle ne sont pas respectées** : les résidus ne suivent pas "
        "une loi normale et leur dispersion augmente avec la taille de l'espèce. Sur les graphiques, un point "
        f"isolé (*{an.ATYPICAL_SPECIES}*) s'écarte fortement des autres. Les p-values et les intervalles de "
        "confiance sont donc à interpréter avec prudence."
    )
    with st.expander("Voir le résumé complet statsmodels"):
        st.text(model1.summary().as_text())

    st.subheader("B4 · La droite de régression")
    fig = add_line(scatter(tab, "Droite de régression — modèle 1"), model1, tab.BOW.min(), tab.BOW.max(),
                   "Modèle 1", BLUE)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Partie C — retrait de l'espèce atypique
# ─────────────────────────────────────────────────────────────
with tC:
    st.subheader(f"C1–C2 · tab vs tab2 (sans *{an.ATYPICAL_SPECIES}*)")
    c1, c2 = st.columns(2)
    c1.plotly_chart(scatter(tab, f"tab — {len(tab)} espèces"), use_container_width=True)
    c2.plotly_chart(scatter(tab2, f"tab2 — {len(tab2)} espèces", highlight=None), use_container_width=True)
    st.markdown(
        "Sans cette espèce, l'axe horizontal se resserre et l'on **voit enfin la structure du nuage** : "
        "les 28 espèces restantes s'alignent très régulièrement."
    )

    st.subheader("C3 · Le modèle 2 et la comparaison")
    st.markdown(
        f'<div class="verdict"><b>Équation</b> : BRW = {fr(s2.intercept)} + {fr(s2.slope)} × BOW'
        f" &nbsp;·&nbsp; R² = {fr(s2.r2, 3)} &nbsp;·&nbsp; n = {s2.n}</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    show_comments(an.comment_model(s2))
    st.markdown("**Comparaison des deux modèles**")
    st.dataframe(an.compare_models(s1, s2), use_container_width=True, hide_index=True)
    st.markdown("**Résidus du modèle 2**")
    residual_panel(model2, tab2, "m2")

    st.subheader("C4 · Superposition des deux droites")
    fig = scatter(tab, "Effet de l'espèce atypique sur la droite")
    add_line(fig, model1, tab.BOW.min(), tab.BOW.max(), "Modèle 1 (toutes les espèces)", BLUE, "dash")
    add_line(fig, model2, tab.BOW.min(), tab.BOW.max(), "Modèle 2 (sans l'espèce atypique)", GREEN)
    st.plotly_chart(fig, use_container_width=True)
    slope_change = (s2.slope - s1.slope) / s1.slope * 100
    st.success(
        f"**Une seule espèce sur {s1.n} change tout** : sans *{an.ATYPICAL_SPECIES}*, la pente passe de "
        f"{fr(s1.slope)} à **{fr(s2.slope)}** (+{fr(slope_change, 0)} %), le R² de {fr(s1.r2, 3)} à "
        f"**{fr(s2.r2, 3)}**, l'erreur typique (RMSE) est divisée par {fr(s1.rmse / s2.rmse, 1)}, et "
        "**les résidus deviennent normaux et de variance constante**. Cette espèce, très lourde et isolée, "
        "« tire » la droite vers elle et aplatit la pente : c'est un point **à fort effet de levier**. "
        "Le modèle 2 décrit beaucoup mieux la relation pour la grande majorité des espèces, mais il ne doit "
        f"pas être utilisé pour prédire au-delà de BOW ≈ {fr(tab2.BOW.max(), 0)}."
    )

# ─────────────────────────────────────────────────────────────
# Partie D — pour aller plus loin
# ─────────────────────────────────────────────────────────────
with tD:
    st.subheader("D1 · Mesurer objectivement l'influence de chaque espèce")
    st.markdown(
        "Retirer un point « à l'œil » se discute. La **distance de Cook** mesure de combien la droite bougerait "
        f"si l'on retirait chaque espèce. Règle usuelle : un point est influent au-delà de **4/n = {fr(4 / s1.n, 3)}**."
    )
    top = influence.head(8)
    fig = px.bar(top[::-1], x="distance_cook", y="Species", orientation="h", log_x=True,
                 color="influent", color_discrete_map={True: RED, False: GREY},
                 title="Distance de Cook — 8 espèces les plus influentes (échelle log)",
                 labels={"distance_cook": "Distance de Cook", "Species": "", "influent": "Influent"})
    fig.add_vline(x=4 / s1.n, line_dash="dash", line_color="black", annotation_text="seuil 4/n")
    fig.update_layout(**PLOT_LAYOUT, height=380)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, use_container_width=True)
    first, second = influence.iloc[0], influence.iloc[1]
    c2.markdown(
        f"""
        <div class="card">
        • <i>{first.Species}</i> : distance de Cook = <b>{fr(first.distance_cook, 1)}</b>, soit
          <b>{fr(first.distance_cook / (4 / s1.n), 0)} fois</b> le seuil, et un levier de {fr(first.levier, 2)}
          (sur 1 au maximum). Son retrait est donc <b>justifié statistiquement</b>.<br><br>
        • <i>{second.Species}</i> dépasse aussi le seuil ({fr(second.distance_cook, 2)}), mais de beaucoup moins :
          c'est la 2ᵉ plus grande espèce.<br><br>
        • Ce qui pose problème n'est pas une erreur de mesure, mais une <b>espèce réelle beaucoup plus grande</b>
          que les autres.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        top.round(3).rename(columns={"residu_studentise": "résidu studentisé", "distance_cook": "distance de Cook"}),
        use_container_width=True, hide_index=True,
    )

    st.subheader("D2 · Et si on gardait toutes les espèces ? La régression en log-log")
    st.markdown(
        "En biologie, la masse du cerveau suit souvent une **loi d'échelle** (allométrie) : "
        "BRW = a × BOWᵇ. En passant au logarithme, cette relation devient une droite : "
        "log(BRW) = log(a) + b × log(BOW). On peut alors **garder toutes les espèces**, y compris la plus grande."
    )
    c1, c2 = st.columns([3, 2])
    fig = scatter(tab, "Échelle logarithmique : toutes les espèces s'alignent", log=True)
    add_line(fig, loglog, tab.BOW.min(), tab.BOW.max(), "Modèle log-log", GREEN, log=True)
    c1.plotly_chart(fig, use_container_width=True)
    max_cook_log = an.influence_table(loglog, tab).iloc[0]
    c2.markdown(
        f"""
        <div class="card">
        <b>log₁₀(BRW) = {fr(s_log.intercept, 3)} + {fr(s_log.slope, 3)} × log₁₀(BOW)</b><br><br>
        • <b>R² = {fr(s_log.r2, 3)}</b> avec les <b>{s_log.n} espèces</b>, mieux que le modèle 2
          ({fr(s2.r2, 3)}) qui en écarte une.<br>
        • Résidus normaux (Shapiro p = {an.fmt_p(s_log.shapiro_p)}) et variance constante
          (Breusch-Pagan p = {an.fmt_p(s_log.breusch_pagan_p)}).<br>
        • Plus aucun point dominant : la plus grande distance de Cook tombe à {fr(max_cook_log.distance_cook, 2)}.<br><br>
        <b>Lecture biologique</b> : l'exposant b = {fr(s_log.slope, 2)}
        (IC 95 % : {fr(s_log.slope_ci[0], 2)} à {fr(s_log.slope_ci[1], 2)}) est inférieur à 1.
        Quand la masse corporelle est multipliée par 10, le cerveau n'est multiplié que par
        <b>{fr(10 ** s_log.slope, 1)}</b> : les grandes espèces ont un cerveau plus lourd, mais
        <b>proportionnellement plus petit</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    residual_panel(loglog, tab, "log")

    st.subheader("D3 · Prédire la masse du cerveau d'une espèce")
    bow = st.slider("Masse corporelle (BOW)", float(tab.BOW.min()), float(tab.BOW.max()), 50.0, step=1.0)
    p1 = s1.intercept + s1.slope * bow
    p2 = s2.intercept + s2.slope * bow
    p_log = 10 ** (s_log.intercept + s_log.slope * np.log10(bow))
    m1c, m2c, m3c = st.columns(3)
    m1c.metric("Modèle 1 (toutes les espèces)", fr(p1, 0))
    m2c.metric("Modèle 2 (sans l'espèce atypique)", fr(p2, 0))
    m3c.metric("Modèle log-log", fr(p_log, 0))
    if bow > tab2.BOW.max():
        st.caption(f"⚠️ Au-delà de BOW = {fr(tab2.BOW.max(), 0)}, le modèle 2 extrapole : il n'a vu aucune espèce aussi grande.")

# ─────────────────────────────────────────────────────────────
# Conclusion
# ─────────────────────────────────────────────────────────────
with tE:
    st.subheader("Ce qu'il faut retenir")
    st.markdown(
        f"""
1. **La relation existe et elle est forte** : r = {fr(r_pearson, 3)}. Plus une chauve-souris est lourde,
   plus son cerveau l'est.
2. **Un bon R² ne suffit pas.** Le modèle 1 explique {fr(s1.r2 * 100, 1)} % de la variance, mais ses
   résidus violent les hypothèses de la régression : on ne peut pas se fier à ses tests.
3. **Une seule espèce peut fausser un modèle.** *{an.ATYPICAL_SPECIES}* a une distance de Cook
   {fr(influence.iloc[0].distance_cook / (4 / s1.n), 0)} fois au-dessus du seuil. Sans elle, la pente passe de
   {fr(s1.slope)} à {fr(s2.slope)} et toutes les hypothèses sont respectées.
4. **Le bon modèle est en log-log.** Il garde les {s_log.n} espèces, atteint R² = {fr(s_log.r2, 3)}, respecte les
   hypothèses et a un sens biologique : cerveau ∝ masse^{fr(s_log.slope, 2)}.
        """
    )
    st.markdown("**Résumé des trois modèles**")
    recap = pd.DataFrame(
        [
            ("Modèle 1 — linéaire, 29 espèces", s1.n, fr(s1.r2, 3), "❌", "❌", "Pente aplatie par un point très influent"),
            ("Modèle 2 — linéaire, sans l'espèce atypique", s2.n, fr(s2.r2, 3), "✅" if s2.residuals_normal else "❌",
             "✅" if s2.homoscedastic else "❌", "Bon modèle, mais seulement pour les espèces de taille courante"),
            ("Modèle 3 — log-log, 29 espèces", s_log.n, fr(s_log.r2, 3), "✅" if s_log.residuals_normal else "❌",
             "✅" if s_log.homoscedastic else "❌", "Recommandé : toutes les espèces, sens biologique"),
        ],
        columns=["Modèle", "Espèces", "R²", "Résidus normaux", "Variance constante", "Verdict"],
    )
    st.dataframe(recap, use_container_width=True, hide_index=True)
