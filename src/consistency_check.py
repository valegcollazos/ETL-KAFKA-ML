import pandas as pd

d = pd.read_csv("data/processed/happiness_clean.csv").sort_values(["country", "year"])
cols = ["gdp", "social_support", "life_expectancy",
        "freedom", "corruption", "generosity"]

cambios = d.groupby("country")[cols].diff().abs().dropna(how="all")
out = d.loc[cambios.index, ["country", "year"]].copy()
out["variable"] = cambios.idxmax(axis=1)
out["salto"] = cambios.max(axis=1)

print("Los 10 mayores saltos entre años consecutivos de un mismo país:")
print(out.nlargest(10, "salto").round(2).to_string(index=False))

dif = d.groupby("country")[cols].diff()
print("\nCambio mediano respecto al año anterior (mismo país):")
print(dif.groupby(d["year"]).median().round(3))