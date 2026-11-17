import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from confluent_kafka import Consumer, KafkaError

from db import get_conn, init_db, insert_prediction

TOPIC = "happiness.stream"
MODEL_PATH = "models/happiness_model.joblib"
RAW_COLS = ["gdp", "social_support", "life_expectancy",
            "freedom", "corruption", "generosity"]


def predict_record(rec, artifact, error_threshold, low_threshold):
    """Predice un evento y decide si dispara alerta."""
    X = pd.DataFrame([rec]).reindex(columns=artifact["features"])
    X = X.apply(pd.to_numeric, errors="coerce")      # None -> NaN (lo imputa el pipeline)
    predicted = float(artifact["pipeline"].predict(X)[0])

    actual = rec.get("score")
    abs_error = abs(actual - predicted) if actual is not None else None

    reasons = []
    if abs_error is not None and abs_error >= error_threshold:
        reasons.append("error_alto")
    if predicted < low_threshold:
        reasons.append("bienestar_bajo")
    return predicted, abs_error, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-threshold", type=float, default=0.8,
                    help="alerta si |real - predicho| >= este valor")
    ap.add_argument("--low-threshold", type=float, default=4.0,
                    help="alerta si el score predicho es menor a este valor")
    ap.add_argument("--group-id", default="happiness-inference")
    args = ap.parse_args()

    artifact = joblib.load(MODEL_PATH)
    print(f"Modelo cargado: {artifact['model_name']} | features: {artifact['features']}")

    conn = get_conn()
    init_db(conn)

    consumer = Consumer({
        "bootstrap.servers": "localhost:9092",
        "group.id": args.group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,   # confirmamos solo después de guardar en la BD
    })
    consumer.subscribe([TOPIC])
    print(f"Escuchando '{TOPIC}'... (Ctrl+C para detener)")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print("Error de Kafka:", msg.error())
                continue

            try:
                rec = json.loads(msg.value().decode("utf-8"))
                predicted, abs_error, reasons = predict_record(
                    rec, artifact, args.error_threshold, args.low_threshold)
            except Exception as e:
                print(f"Mensaje inválido (offset {msg.offset()}): {e}")
                consumer.commit(message=msg, asynchronous=False)
                continue

            row = {
                "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "country": rec.get("country"),
                "year": rec.get("year"),
                **{c: rec.get(c) for c in RAW_COLS},
                "actual": rec.get("score"),
                "predicted": predicted,
                "abs_error": abs_error,
                "alert": int(bool(reasons)),
                "alert_reason": ",".join(reasons) or None,
                "kafka_partition": msg.partition(),
                "kafka_offset": msg.offset(),
            }
            insert_prediction(conn, row)
            consumer.commit(message=msg, asynchronous=False)

            real_txt = f"{rec['score']:.2f}" if rec.get("score") is not None else "-"
            err_txt = f"{abs_error:.2f}" if abs_error is not None else "-"
            flag = f"  ALERTA: {','.join(reasons)}" if reasons else ""
            print(f"{rec.get('country')} {rec.get('year')} | real={real_txt} "
                  f"pred={predicted:.2f} err={err_txt}{flag}")
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    main()