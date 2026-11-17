import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer

parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=float, default=1.0,
                    help="segundos entre mensajes")
parser.add_argument("--limit", type=int, default=0,
                    help="enviar solo los primeros N (0 = todos)")
args = parser.parse_args()

TOPIC = "happiness.stream"
producer = Producer({"bootstrap.servers": "localhost:9092"})

df = pd.read_csv("data/processed/test_split.csv")

records = json.loads(df.to_json(orient="records"))
if args.limit:
    records = records[:args.limit]


def on_delivery(err, msg):
    if err:
        print("Error de entrega:", err)


for i, rec in enumerate(records, 1):
    key = f"{rec['country']}-{rec['year']}"
    producer.produce(TOPIC, key=key, value=json.dumps(rec),
                     callback=on_delivery)
    producer.poll(0)
    print(f"[{i}/{len(records)}] enviado {key}")
    time.sleep(args.delay)

producer.flush()
print("Listo.")