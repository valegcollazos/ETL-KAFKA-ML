import json, os, time
import certifi
from confluent_kafka import Consumer
from dotenv import load_dotenv

load_dotenv()

conf = {
    "bootstrap.servers": "kafka-classroom-2026.westus.cloudapp.azure.com:9092",
    "security.protocol": "SASL_SSL",
    "sasl.mechanism": "SCRAM-SHA-512",
    "sasl.username": "classroom-students",
    "sasl.password": os.environ["KAFKA_PASSWORD"],
    "ssl.ca.location": certifi.where(),
    "group.id": os.environ["GROUP_ID"],
    "auto.offset.reset": "earliest",
}

consumer = Consumer(conf)
consumer.subscribe(["etl.records"])

IDLE_SECONDS = 20          # si no llega nada en 20 s, asumimos que ya leímos todo
count, last = 0, time.time()

with open("data/raw/records_tmp.jsonl", "w", encoding="utf-8") as f:
    try:
        while time.time() - last < IDLE_SECONDS:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print("Error:", msg.error())
                continue
            last = time.time()
            raw = msg.value().decode("utf-8")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            meta = {
                "partition": msg.partition(),
                "offset": msg.offset(),
                "timestamp": msg.timestamp()[1],
                "key": msg.key().decode() if msg.key() else None,
                "headers": {k: v.decode() for k, v in (msg.headers() or [])},
            }
            f.write(json.dumps({"meta": meta, "value": value}) + "\n")
            count += 1
            if count == 1:
                print("Primer mensaje:", json.dumps(value, indent=2)[:800])
    finally:
        consumer.close()

print(f"Mensajes consumidos: {count}")

import os
if count > 0:
    os.replace("data/raw/records_tmp.jsonl", "data/raw/records.jsonl")
else:
    print("No llegaron mensajes: records.jsonl no se tocó.")