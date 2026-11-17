import json
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/records.jsonl")
OUT = Path("data/processed/happiness_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

RENAME = {
    "Country": "country", "Country or region": "country",
    "Happiness Score": "score", "Happiness.Score": "score", "Score": "score",
    "Economy (GDP per Capita)": "gdp", "Economy..GDP.per.Capita.": "gdp",
    "GDP per capita": "gdp",
    "Family": "social_support", "Social support": "social_support",
    "Health (Life Expectancy)": "life_expectancy",
    "Health..Life.Expectancy.": "life_expectancy",
    "Healthy life expectancy": "life_expectancy",
    "Freedom": "freedom", "Freedom to make life choices": "freedom",
    "Trust (Government Corruption)": "corruption",
    "Trust..Government.Corruption.": "corruption",
    "Perceptions of corruption": "corruption",
    "Generosity": "generosity",
}
NUM = ["score", "gdp", "social_support", "life_expectancy",
       "freedom", "corruption", "generosity"]

COUNTRY_FIX = {
    "Somaliland region": "Somaliland Region",
    "Hong Kong S.A.R., China": "Hong Kong",
    "Taiwan Province of China": "Taiwan",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "North Cyprus": "Northern Cyprus",
    "Macedonia": "North Macedonia",
}

#Cargar y renombrar (las columnas no mapeadas se descartan a propósito)
records = []
with open(RAW, encoding="utf-8") as f:
    for line in f:
        v = json.loads(line)["value"]
        rec = {RENAME[k]: val for k, val in v["data"].items() if k in RENAME}
        rec["year"] = v["metadata"]["year"]
        rec["record_id"] = v["metadata"]["record_id"]
        records.append(rec)
df = pd.DataFrame(records)
print("Forma inicial:", df.shape)

#Normalizar nombres de países
df["country"] = df["country"].str.strip().replace(COUNTRY_FIX)


for c in NUM:
    antes = df[c].isna().sum()
    df[c] = pd.to_numeric(df[c], errors="coerce")
    nuevos = df[c].isna().sum() - antes
    if nuevos:
        print(f"[tipos] {c}: {nuevos} valores no numéricos convertidos a NaN")

#Calidad
print("\nNulos por año:")
print(df[NUM].isna().groupby(df["year"]).sum())

print("\nDuplicados (country, year):", df.duplicated(["country", "year"]).sum())

print("\nRangos:")
print(df[NUM].describe().T[["min", "max", "mean"]])

print("\nOutliers por IQR (solo se reportan, no se eliminan):")
for c in NUM:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"  {c}: {n}")

print("\nPaíses según en cuántos años aparecen:")
print(df.groupby("country")["year"].nunique().value_counts().sort_index())
print("Total de países distintos:", df["country"].nunique())

print("\nMedias por año (para detectar cambios de escala entre ediciones):")
print(df.groupby("year")[NUM].mean().round(3))

print("\nsocial_support por año (media, desv., correlación con score):")
for y, g in df.groupby("year"):
    print(y, round(g["social_support"].mean(), 3),
          round(g["social_support"].std(), 3),
          round(g["social_support"].corr(g["score"]), 3))

df = df.dropna(subset=["score"])
df = df.drop_duplicates(["country", "year"], keep="last")
df = df[["country", "year"] + NUM].sort_values(["year", "country"])
df.to_csv(OUT, index=False)
print("\nGuardado:", OUT, df.shape)