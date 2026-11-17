import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GroupShuffleSplit, KFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "score"
BASE = ["gdp", "social_support", "life_expectancy",
        "freedom", "corruption", "generosity"]
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "P"]

Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/happiness_clean.csv")
YEARS = sorted(df["year"].unique())

#variables a comparar
FEATURE_SETS = {
    "base6": (BASE, False),
    "sin_generosity": ([c for c in BASE if c != "generosity"], False),
    "base6+year": (BASE, True),
}

MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
}


def input_cols(num_cols, use_year):
    return num_cols + (["year"] if use_year else [])


def build(model, num_cols, use_year):
    """Pipeline completo: el imputer y el scaler se ajustan SOLO con train."""
    steps = [("num",
              Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler())]),
              num_cols)]
    if use_year:
        steps.append(("year", OneHotEncoder(handle_unknown="ignore"), ["year"]))
    return Pipeline([("prep", ColumnTransformer(steps)), ("model", model)])


def evaluate(y_true, y_pred, n_features):
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "R2": float(r2), "R2_ajustado": float(adj)}



#Split train / test

train, test = train_test_split(df, test_size=0.2, random_state=SEED)
print(f"Train: {len(train)} filas | Test: {len(test)} filas")
test.to_csv("data/processed/test_split.csv", index=False)   # lo usará el productor


# comparación con validación cruzada (solo sobre train)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for fs_name, (num_cols, use_year) in FEATURE_SETS.items():
    cols = input_cols(num_cols, use_year)
    for m_name, model in MODELS.items():
        pipe = build(clone(model), num_cols, use_year)
        s = -cross_val_score(pipe, train[cols], train[TARGET], cv=kf,
                             scoring="neg_root_mean_squared_error")
        rows.append({"features": fs_name, "model": m_name,
                     "cv_rmse_mean": s.mean(), "cv_rmse_std": s.std()})
comp = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
comp.to_csv("reports/model_comparison.csv", index=False)
print("\n=== Comparación (CV 5-fold sobre train, RMSE) ===")
print(comp.round(4).to_string(index=False))

best = comp.iloc[0]
num_cols, use_year = FEATURE_SETS[best["features"]]
cols = input_cols(num_cols, use_year)
print(f"\nSeleccionado: {best['model']} con '{best['features']}'")

#Entrenamiento final y evaluación en test

pipe = build(clone(MODELS[best["model"]]), num_cols, use_year)
pipe.fit(train[cols], train[TARGET])
X_test, y_test = test[cols], test[TARGET].to_numpy()
pred = pipe.predict(X_test)
metrics = evaluate(y_test, pred, len(cols))

base = DummyRegressor(strategy="mean").fit(train[cols], train[TARGET])
metrics_base = evaluate(y_test, base.predict(X_test), len(cols))

print("\n=== Test ===")
print("Modelo   :", {k: round(v, 4) for k, v in metrics.items()})
print("Baseline :", {k: round(v, 4) for k, v in metrics_base.items()},
      "(predecir siempre la media)")


#Chequeos

#emporal: entrenar 2015-2017 y evaluar 2018-2019
tr_t, te_t = df[df["year"] <= 2017], df[df["year"] >= 2018]
p_t = clone(pipe).fit(tr_t[cols], tr_t[TARGET])
temporal = evaluate(te_t[TARGET].to_numpy(), p_t.predict(te_t[cols]), len(cols))

#Agrupado por país: ningún país está a la vez en train y en test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
i_tr, i_te = next(gss.split(df, groups=df["country"]))
tr_g, te_g = df.iloc[i_tr], df.iloc[i_te]
p_g = clone(pipe).fit(tr_g[cols], tr_g[TARGET])
agrupado = evaluate(te_g[TARGET].to_numpy(), p_g.predict(te_g[cols]), len(cols))

print("\n=== Robustez ===")
print("Temporal (train 2015-17 / test 2018-19):",
      {k: round(v, 4) for k, v in temporal.items()})
print("Agrupado por país                      :",
      {k: round(v, 4) for k, v in agrupado.items()})


FIG = Path("reports/figures")

fig, ax = plt.subplots(figsize=(6.5, 6))
for y, m, c in zip(YEARS, MARKERS, OKABE):
    mask = test["year"].to_numpy() == y
    ax.scatter(y_test[mask], pred[mask], marker=m, color=c,
               s=30, alpha=0.8, label=str(y))
lims = [min(y_test.min(), pred.min()) - 0.1, max(y_test.max(), pred.max()) + 0.1]
ax.plot(lims, lims, "k--", lw=1, label="predicción perfecta")
ax.set_xlabel("score real")
ax.set_ylabel("score predicho")
ax.set_title(f"Real vs predicho ({best['model']}, R² = {metrics['R2']:.2f})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "06_real_vs_predicho.png", dpi=150)
plt.close(fig)

res = y_test - pred
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(res, bins=25, color=OKABE[0], edgecolor="white")
axes[0].axvline(0, color="black", ls="--")
axes[0].set_title("Distribución de residuos")
axes[0].set_xlabel("real - predicho")
axes[1].scatter(pred, res, color=OKABE[1], s=22, alpha=0.8)
axes[1].axhline(0, color="black", ls="--")
axes[1].set_title("Residuos vs predicho")
axes[1].set_xlabel("score predicho")
axes[1].set_ylabel("residuo")
fig.tight_layout()
fig.savefig(FIG / "07_residuos.png", dpi=150)
plt.close(fig)

imp = permutation_importance(pipe, X_test, y_test, n_repeats=20,
                             random_state=SEED,
                             scoring="neg_root_mean_squared_error")
imp_df = (pd.DataFrame({"variable": cols, "aumento_rmse": imp.importances_mean})
          .sort_values("aumento_rmse"))
imp_df.to_csv("reports/feature_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp_df["variable"], imp_df["aumento_rmse"], color=OKABE[2], hatch="//",
        edgecolor="black")
ax.set_xlabel("aumento del RMSE al desordenar la variable")
ax.set_title("Importancia por permutación (test)")
fig.tight_layout()
fig.savefig(FIG / "08_importancia.png", dpi=150)
plt.close(fig)


#Artefacto serializado + métricas

joblib.dump({"pipeline": pipe, "features": cols, "target": TARGET,
             "model_name": best["model"], "feature_set": best["features"],
             "metrics": metrics, "sklearn_version": sklearn.__version__},
            "models/happiness_model.joblib")
with open("reports/metrics.json", "w", encoding="utf-8") as f:
    json.dump({"model": best["model"], "feature_set": best["features"],
               "features": cols, "test": metrics, "baseline": metrics_base,
               "temporal": temporal, "agrupado_por_pais": agrupado},
              f, indent=2, ensure_ascii=False)
print("\nGuardado: models/happiness_model.joblib, reports/metrics.json")