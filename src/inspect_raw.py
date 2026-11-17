import json
from collections import Counter, defaultdict

rows = [json.loads(line) for line in open("data/raw/records.jsonl", encoding="utf-8")]
print("Total:", len(rows))

anios = Counter(r["value"]["metadata"]["year"] for r in rows)
print("Registros por año:", dict(sorted(anios.items())))

ids = [r["value"]["metadata"]["record_id"] for r in rows]
print("record_id únicos:", len(set(ids)), "de", len(ids))

cols = defaultdict(set)
for r in rows:
    cols[r["value"]["metadata"]["year"]].update(r["value"]["data"].keys())

comunes = set.intersection(*cols.values())
print("\nColumnas comunes a todos los años:", sorted(comunes))
for y in sorted(cols):
    print(f"\n{y} - columnas no comunes:", sorted(cols[y] - comunes))