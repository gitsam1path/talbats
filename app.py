import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Régression Bats", layout="wide", page_icon="🦇")
st.title("🦇 TP Régression linéaire - Données de chauves-souris")

# ---------------------------
# Import des données
# ---------------------------
uploaded_file = st.file_uploader("📂 Importer tabBats.txt", type=["txt", "csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, quotechar='"')

    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Exploration",
        "📊 Corrélation",
        "📈 Régression modèle 1",
        "⚖️ Comparaison modèle 1 vs 2"
    ])

    # ---------------------------
    # Exploration
    # ---------------------------
    with tab1:
        st.subheader("👀 Aperçu des données")
        st.dataframe(df.head())
        st.write(f"Shape : {df.shape}")
        st.write("Colonnes :", list(df.columns))

        st.write("📈 Types de colonnes")
        st.write(df.dtypes)

        st.write("📉 Statistiques descriptives")
        st.write(df.describe())

        st.write("❓ Valeurs manquantes")
        st.write(df.isnull().sum())

    # ---------------------------
    # Corrélation
    # ---------------------------
    with tab2:
        st.subheader("📊 Matrice de corrélation")
        df_num = df.select_dtypes(include="number")
        corr = df_num.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, mask=mask, fmt=".3f", ax=ax)
        st.pyplot(fig)

        st.subheader("🧪 Test Pearson BOW vs BRW")
        corr_coef, p_value = pearsonr(df['BOW'], df['BRW'])
        st.write(f"r = {corr_coef:.4f}, p = {p_value:.6f}")
        st.write("👉 Corrélation **significative**" if p_value < 0.05 else "👉 Corrélation **non significative**")

    # ---------------------------
    # Régression modèle 1
    # ---------------------------
    with tab3:
        st.subheader("📈 Régression linéaire - Modèle 1 (toutes données)")
        df_reg = df[['Species', 'BOW', 'BRW']].copy()
        X = sm.add_constant(df_reg['BOW'])
        y = df_reg['BRW']
        model1 = sm.OLS(y, X).fit()
        st.write(model1.summary())

        intercept, slope = model1.params
        st.markdown(f"""
        - **Equation** : BRW = {intercept:.2f} + {slope:.4f} × BOW  
        - R² = {model1.rsquared:.4f}, R² ajusté = {model1.rsquared_adj:.4f}
        """)

        # Résidus
        residuals = model1.resid
        fitted = model1.fittedvalues

        st.subheader("🔍 Analyse des résidus")
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        st.write(f"Test Shapiro-Wilk p={shapiro_p:.4f} → {'Normaux' if shapiro_p>0.05 else 'Non normaux'}")

        try:
            dw_stat = sm.stats.stattools.durbin_watson(residuals)
            st.write(f"Durbin-Watson = {dw_stat:.4f}")
        except:
            st.write("⚠️ Test Durbin-Watson indisponible")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # 1. Résidus vs valeurs ajustées
        axes[0,0].scatter(fitted, residuals)
        axes[0,0].axhline(0, color="red", linestyle="--")
        axes[0,0].set_title("Résidus vs valeurs ajustées")
        # 2. QQ plot
        sm.qqplot(residuals, line="45", ax=axes[0,1])
        axes[0,1].set_title("QQ-plot des résidus")
        # 3. Histogramme
        axes[1,0].hist(residuals, bins=15, edgecolor="black")
        axes[1,0].set_title("Histogramme des résidus")
        # 4. Résidus vs ordre
        axes[1,1].plot(residuals, marker="o")
        axes[1,1].axhline(0, color="red", linestyle="--")
        axes[1,1].set_title("Résidus vs ordre d’observation")
        st.pyplot(fig)

        # Droite de régression
        st.subheader("📉 Tracé de la droite de régression")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.scatter(df_reg['BOW'], df_reg['BRW'], c="steelblue", edgecolors="black")
        x_range = np.linspace(df_reg['BOW'].min(), df_reg['BOW'].max(), 100)
        y_pred = intercept + slope * x_range
        ax.plot(x_range, y_pred, "r-", label="Régression")
        ax.set_xlabel("BOW (masse corporelle)")
        ax.set_ylabel("BRW (masse cerveau)")
        ax.legend()
        st.pyplot(fig)

    # ---------------------------
    # Comparaison modèle 1 vs modèle 2
    # ---------------------------
    with tab4:
        st.subheader("⚖️ Comparaison avec retrait d’espèce atypique")
        # Retrait Pteropus vampyrus si présent
        mask = df_reg['Species'].str.contains("Pteropus vampyrus", case=False, na=False)
        if mask.any():
            tab2 = df_reg[~mask].copy()
        else:
            tab2 = df_reg.drop(df_reg['BOW'].idxmax())  # enlève l’outlier principal

        X2 = sm.add_constant(tab2['BOW'])
        y2 = tab2['BRW']
        model2 = sm.OLS(y2, X2).fit()

        st.write("📊 Résumé modèle 2 (sans outlier)")
        st.write(model2.summary())

        # Comparaison
        st.markdown(f"""
        - **Intercept** : Modèle1={model1.params[0]:.4f}, Modèle2={model2.params[0]:.4f}  
        - **Pente** : Modèle1={model1.params[1]:.4f}, Modèle2={model2.params[1]:.4f}  
        - **R²** : Modèle1={model1.rsquared:.4f}, Modèle2={model2.rsquared:.4f}  
        """)

        # Superposition
        st.subheader("📉 Superposition des droites")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.scatter(df_reg['BOW'], df_reg['BRW'], c="blue", label="Données complètes")
        ax.scatter(tab2['BOW'], tab2['BRW'], c="green", label="Sans outlier")
        x_range = np.linspace(df_reg['BOW'].min(), df_reg['BOW'].max(), 100)
        ax.plot(x_range, model1.params[0] + model1.params[1]*x_range, "b--", label="Modèle 1")
        ax.plot(x_range, model2.params[0] + model2.params[1]*x_range, "g-", label="Modèle 2")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("➡️ Importer le fichier pour lancer l’analyse")
