import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import certifi
from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

load_dotenv()

TOPIC = "etl.records"
OUT = Path("data/raw/records.jsonl")
IDLE_SECONDS = 20       
MAX_WAIT_FIRST = 60     


def build_consumer(group_id):
    return Consumer({
        "bootstrap.servers": "kafka-classroom-2026.westus.cloudapp.azure.com:9092",
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": "classroom-students",
        "sasl.password": os.environ["KAFKA_PASSWORD"],
        "ssl.ca.location": certifi.where(),
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,   # se confirma al final, después de guardar
    })


def consume(consumer, idle_seconds=IDLE_SECONDS, max_wait_first=MAX_WAIT_FIRST):
    """Consume el tópico y acumula los registros EN MEMORIA.

    Estado en memoria: dict {record_id: {"meta": ..., "value": ...}}.
    Al estar indexado por record_id, un mensaje repetido no duplica el registro.
    """
    state = {}
    stats = {"received": 0, "duplicates": 0, "invalid": 0, "errors": 0}
    last = time.time()

    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            limit = idle_seconds if stats["received"] else max_wait_first
            if time.time() - last > limit:
                break
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            stats["errors"] += 1
            print("Error de Kafka:", msg.error())
            continue

        last = time.time()
        stats["received"] += 1

        
        try:
            value = json.loads(msg.value().decode("utf-8"))
            if not (isinstance(value.get("metadata"), dict)
                    and isinstance(value.get("data"), dict)):
                raise ValueError("faltan 'metadata' o 'data'")
        except Exception as e:
            stats["invalid"] += 1
            print(f"Mensaje inválido (offset {msg.offset()}): {e}")
            continue

        meta = {
            "partition": msg.partition(),
            "offset": msg.offset(),
            "timestamp": msg.timestamp()[1],
            "key": msg.key().decode() if msg.key() else None,
            "headers": {k: v.decode() for k, v in (msg.headers() or [])},
        }

        # 2. Acumular en el estado (sin duplicados)
        record_id = value["metadata"].get("record_id") or f"{meta['partition']}-{meta['offset']}"
        if record_id in state:
            stats["duplicates"] += 1
            continue
        state[record_id] = {"meta": meta, "value": value}

        if len(state) == 1:
            print("Primer mensaje:", json.dumps(value, indent=2)[:800])

    return state, stats


def verify(state, stats):
    """Verifica el estado construido en memoria."""
    print("\n=== Verificación del estado en memoria ===")
    print(f"Mensajes recibidos        : {stats['received']}")
    print(f"Registros únicos (estado) : {len(state)}")
    print(f"Duplicados ignorados      : {stats['duplicates']}")
    print(f"Mensajes inválidos        : {stats['invalid']}")
    por_anio = Counter(r["value"]["metadata"].get("year") for r in state.values())
    print("Registros por año         :", dict(sorted(por_anio.items(), key=lambda x: str(x[0]))))
    ok = stats["received"] == len(state) + stats["duplicates"] + stats["invalid"]
    print("Consistencia (recibidos = únicos + duplicados + inválidos):",
          "OK" if ok else "REVISAR")
    return ok


def save(state, out=OUT):
    """Persiste el estado en JSONL (escritura atómica: no pisa el archivo si algo falla)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in state.values():
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, out)
    with open(out, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="usa un group.id nuevo para releer el tópico desde el inicio")
    args = ap.parse_args()

    group_id = os.environ["GROUP_ID"]
    if args.fresh:
        group_id = f"{group_id}-{int(time.time())}"

    consumer = build_consumer(group_id)
    consumer.subscribe([TOPIC])
    print(f"Consumiendo '{TOPIC}' con group.id='{group_id}'...")

    try:
        state, stats = consume(consumer)
        verify(state, stats)

        if not state:
            print("\nNo llegaron mensajes: records.jsonl no se modificó. "
                  "Usa --fresh (o cambia GROUP_ID) para releer el tópico.")
            return

        en_disco = save(state)
        print(f"\nGuardado en {OUT}: {en_disco} líneas "
              f"({'coincide' if en_disco == len(state) else 'NO coincide'} con el estado en memoria)")

        try:
            consumer.commit(asynchronous=False)
            print("Offsets confirmados en Kafka.")
        except KafkaException as e:
            print("Aviso: no se pudieron confirmar los offsets:", e)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()