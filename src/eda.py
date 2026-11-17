from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/happiness_clean.csv")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

FEATS = ["gdp", "social_support", "life_expectancy",
         "freedom", "corruption", "generosity"]
YEARS = sorted(df["year"].unique())
# Paleta Okabe-Ito (apta para daltonismo) + marcadores distintos por año
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "P"]


#estadísticos descriptivos

desc = df[FEATS + ["score"]].describe().T.round(3)
desc.to_csv("reports/descriptive_stats.csv")
print("=== Descriptivos ===")
print(desc)

print("\n=== Filas con nulos ===")
print(df[df.isna().any(axis=1)])

print("\n=== Cobertura: registros por año ===")
print(df.groupby("year").size())


#Correlaciones
corr = df[FEATS + ["score"]].corr()
print("\n=== Correlación con score (global) ===")
print(corr["score"].drop("score").sort_values(ascending=False).round(3))

print("\n=== Pares de variables con |corr| > 0.6 (posible multicolinealidad) ===")
for i, a in enumerate(FEATS):
    for b in FEATS[i + 1:]:
        if abs(corr.loc[a, b]) > 0.6:
            print(f"  {a} - {b}: {corr.loc[a, b]:.3f}")

by_year = pd.DataFrame({
    y: df[df["year"] == y][FEATS].corrwith(df[df["year"] == y]["score"])
    for y in YEARS
}).round(3)
by_year.to_csv("reports/corr_with_score_by_year.csv")
print("\n=== Correlación con score, por año ===")
print(by_year)

means = df.groupby("year")[FEATS + ["score"]].mean().round(3)
means.to_csv("reports/means_by_year.csv")


#Figuras

#distribución del score
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(df["score"], bins=25, color=OKABE[0], edgecolor="white")
axes[0].set_title("Distribución del score")
axes[0].set_xlabel("score")
axes[0].set_ylabel("frecuencia")
axes[1].boxplot([df[df["year"] == y]["score"] for y in YEARS])
axes[1].set_xticks(range(1, len(YEARS) + 1), YEARS)
axes[1].set_title("Score por año")
axes[1].set_ylabel("score")
fig.tight_layout()
fig.savefig(FIG / "01_distribucion_score.png", dpi=150)
plt.close(fig)

#matriz de correlación
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="cividis", vmin=-1, vmax=1)
labels = list(corr.columns)
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.4 else "black", fontsize=9)
fig.colorbar(im, ax=ax)
ax.set_title("Matriz de correlación")
fig.tight_layout()
fig.savefig(FIG / "02_correlaciones.png", dpi=150)
plt.close(fig)

#cada variable contra el score (color + marcador por año)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.ravel(), FEATS):
    for y, m, c in zip(YEARS, MARKERS, OKABE):
        g = df[df["year"] == y]
        ax.scatter(g[col], g["score"], s=18, alpha=0.7,
                   marker=m, color=c, label=str(y))
    d = df.dropna(subset=[col])
    k, b = np.polyfit(d[col], d["score"], 1)
    xs = np.linspace(d[col].min(), d[col].max(), 50)
    ax.plot(xs, k * xs + b, color="black", lw=1.5, ls="--")
    ax.set_xlabel(col)
    ax.set_ylabel("score")
    ax.set_title(f"{col} (r = {d[col].corr(d['score']):.2f})")
axes[0, 0].legend(title="Año", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "03_variables_vs_score.png", dpi=150)
plt.close(fig)

#media de cada variable por año (cambios de escala entre ediciones)
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for ax, col in zip(axes.ravel(), FEATS):
    ax.plot(means.index, means[col], marker="o", color=OKABE[0])
    ax.set_title(col)
    ax.set_xticks(YEARS)
fig.suptitle("Media por año de cada variable")
fig.tight_layout()
fig.savefig(FIG / "04_medias_por_ano.png", dpi=150)
plt.close(fig)

#outliers
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.boxplot([df[c].dropna() for c in FEATS])
ax.set_xticks(range(1, len(FEATS) + 1), FEATS, rotation=20)
ax.set_title("Dispersión y outliers por variable")
fig.tight_layout()
fig.savefig(FIG / "05_outliers.png", dpi=150)
plt.close(fig)

print("\nFiguras guardadas en", FIG)