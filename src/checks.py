import pandas as pd

df = pd.read_csv("data/processed/happiness_clean.csv")
cols = ["gdp", "social_support", "life_expectancy", "freedom", "corruption", "generosity"]

print("=== 1. Filas con algún valor en 0 ===")
ceros = df[(df[cols] == 0).any(axis=1)]
print(ceros[["country", "year", "score"] + cols].to_string(index=False))

print("\n=== 2. Países que no están en los 5 años ===")
anios = df.groupby("country")["year"].agg(["nunique", "min", "max"])
print(anios[anios["nunique"] < 5].sort_values("min").to_string())

print("\n=== 3. social_support por año ===")
print(df.groupby("year")["social_support"].agg(["mean", "std", "min", "max"]))
print("\nCorrelación social_support vs score, por año:")
for y, g in df.groupby("year"):
    print(y, round(g["social_support"].corr(g["score"]), 3))