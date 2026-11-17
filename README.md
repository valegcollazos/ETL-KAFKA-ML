# Predicción del Happiness Score con Apache Kafka + Machine Learning

Pipeline ETL con **streaming de eventos (Kafka)**, **regresión con scikit-learn** y un **dashboard operativo en Streamlit** para predecir el *Happiness Score* de países a partir de los datos del [World Happiness Report](https://www.worldhappiness.report/data-sharing/) (2015-2019).

https://github.com/user-attachments/assets/2b37e333-76aa-4072-9cc2-af86dc563c68

*Dashboard actualizándose en tiempo real: el consumidor predice cada evento de Kafka, lo guarda en SQLite y Streamlit muestra métricas, gráficos y alertas.*

---

## Tabla de contenido

1. [Resumen](#1-resumen)
2. [Arquitectura](#2-arquitectura)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Cómo reproducirlo](#4-cómo-reproducirlo)
5. [Extracción desde Kafka](#5-extracción-desde-kafka)
6. [Preparación e integración de los datos](#6-preparación-e-integración-de-los-datos)
7. [Análisis exploratorio (EDA)](#7-análisis-exploratorio-eda)
8. [Modelado y evaluación](#8-modelado-y-evaluación)
9. [Streaming: productor, consumidor y base de datos](#9-streaming-productor-consumidor-y-base-de-datos)
10. [Dashboard en Streamlit](#10-dashboard-en-streamlit)
11. [Decisiones de diseño](#11-decisiones-de-diseño)
12. [Limitaciones](#12-limitaciones)
13. [Posibles mejoras](#13-posibles-mejoras)
14. [Solución de problemas](#14-solución-de-problemas)

---

## 1. Resumen

| Aspecto | Detalle |
|---|---|
| Fuente de datos | Tópico Kafka `etl.records` (5 CSV anuales, 2015-2019, publicados por el profesor) |
| Registros consumidos | **782** (2015: 158 · 2016: 157 · 2017: 155 · 2018: 156 · 2019: 156) |
| Objetivo | `score` (Happiness Score), rango 2.69 – 7.77, media 5.38 |
| Modelo final | `GradientBoostingRegressor` + variable `year` (pipeline con imputación y escalado) |
| Resultado en test | **MAE 0.405 · RMSE 0.491 · R² 0.808** (baseline: RMSE 1.124 · R² ≈ 0) |
| Streaming | Kafka local (Docker) → consumidor con el modelo serializado → SQLite → Streamlit |
| Verificación de punta a punta | 157 eventos procesados; el MAE en streaming (0.4045) coincide con el MAE de test |

---

## 2. Arquitectura

El proyecto tiene dos etapas, siguiendo el diagrama de bloques del enunciado.

```
ETAPA 1 · Offline (análisis y entrenamiento)

  Broker Kafka del curso                                       
  tópico etl.records  ──►  consume_raw.py ──► data/raw/records.jsonl
  (SASL_SSL)                                      │
                                                  ▼
                                           prepare.py ──► data/processed/happiness_clean.csv
                                                  │
                                      ┌───────────┴───────────┐
                                      ▼                       ▼
                                   eda.py               train.py
                           consistency_check.py             │
                                   (reports/)               ├─► models/happiness_model.joblib  (artefacto)
                                                            ├─► data/processed/test_split.csv
                                                            └─► reports/ (métricas y figuras)

ETAPA 2 · Streaming (inferencia en tiempo real)

  test_split.csv ──► producer.py ──► Kafka local (Docker)  ──► inference_consumer.py ──► SQLite ──► Streamlit
                                     tópico happiness.stream    (carga el .joblib,                  (dashboard
                                                                 predice, genera alertas)            en vivo)
```

---

## 3. Estructura del repositorio

```
.
├── .env.example                 # plantilla de variables de entorno (sin secretos)
├── .gitignore
├── README.md
├── docker-compose.yml           # Kafka local (modo KRaft, un solo nodo)
├── requirements.txt
├── dashboard/
│   ├── app.py                   # interfaz Streamlit
│   └── queries.py               # consultas a SQLite (sin dependencia de Streamlit)
├── data/
│   ├── raw/                     # records.jsonl (ignorado por Git)
│   ├── processed/
│   │   ├── happiness_clean.csv  # dataset integrado y limpio
│   │   └── test_split.csv       # conjunto de prueba que reproduce el productor
│   └── predictions.db           # salida del consumidor (ignorado por Git)
├── models/
│   └── happiness_model.joblib   # artefacto: pipeline completo + metadatos
├── reports/
│   ├── figures/                 # 01 a 08 (.png)
│   ├── descriptive_stats.csv
│   ├── corr_with_score_by_year.csv
│   ├── means_by_year.csv
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── metrics.json
└── src/
    ├── consume_raw.py           # consumidor Kafka (broker del curso)
    ├── inspect_raw.py           # estructura de mensajes y columnas por año
    ├── prepare.py               # armonización, limpieza y controles de calidad
    ├── eda.py                   # estadísticos, correlaciones y figuras
    ├── consistency_check.py     # saltos y desplazamientos entre años
    ├── train.py                 # comparación de modelos, evaluación y artefacto
    ├── producer.py              # simulador de eventos (Kafka local)
    ├── db.py                    # esquema y escritura en SQLite
    └── inference_consumer.py    # consumidor con inferencia y alertas
```

---

## 4. Cómo reproducirlo

### Requisitos

- Python 3.14 (desarrollado y probado con esta versión, en Windows).
- Docker Desktop (solo para la etapa de streaming).
- Contraseña del broker del curso (la entrega el profesor).

### Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # luego edita .env con tu contraseña y tu group.id
```

> **Nunca subas `.env` a GitHub.** Está incluido en `.gitignore`. Verifícalo con `git status`.

### Etapa 1 · Análisis y entrenamiento

```powershell
python src\consume_raw.py          # 1. consume etl.records -> data/raw/records.jsonl
python src\inspect_raw.py          # 2. estructura de mensajes y columnas por año
python src\prepare.py              # 3. armoniza y limpia -> data/processed/happiness_clean.csv
python src\eda.py                  # 4. estadísticos y figuras -> reports/
python src\consistency_check.py    # 5. consistencia entre años
python src\train.py                # 6. entrena, evalúa y exporta models/happiness_model.joblib
```

### Etapa 2 · Streaming y dashboard

```powershell
# Una vez: levantar Kafka local y crear el tópico
docker compose up -d
docker exec -it kafka-local /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic happiness.stream --partitions 1 --replication-factor 1

# Terminal A: dashboard
streamlit run dashboard/app.py

# Terminal B: consumidor con inferencia (arrancarlo ANTES que el productor)
python src\inference_consumer.py

# Terminal C: productor (ejecutarlo UNA sola vez; ver sección 14)
python src\producer.py --delay 1
```

Verificación de la base de datos (debe dar `(157, 157, 0.4045)`):

```powershell
python -c "import sqlite3; c = sqlite3.connect('data/predictions.db'); print(c.execute('select count(*), count(distinct country || year), round(avg(abs_error),4) from predictions').fetchone())"
```

---

## 5. Extracción desde Kafka

**Conexión (broker del curso).** `SASL_SSL` con `SCRAM-SHA-512`, usuario `classroom-students`, tópico `etl.records`. La contraseña se lee desde `.env` y el `group.id` es el nombre completo en minúsculas separado por guiones. El certificado se toma de `certifi`.

**Consumo.** `src/consume_raw.py` usa `confluent-kafka`, `auto.offset.reset=earliest` y termina cuando pasan 20 segundos sin mensajes nuevos. Cada mensaje se guarda en `data/raw/records.jsonl` junto con sus metadatos (partición, offset, timestamp, clave y cabeceras).

**Estructura de los mensajes** (revisada con `inspect_raw.py`):

```json
{
  "metadata": {"source_file": "2015.csv", "year": 2015, "record_id": "2015-3"},
  "data": {"Country": "Denmark", "Region": "Western Europe", "Happiness Score": 7.527, "...": "..."}
}
```

El año viene explícito en `metadata.year`, y `data` conserva las columnas originales de cada CSV.

**Verificación del estado construido.**

- 782 mensajes consumidos y 782 `record_id` únicos (sin duplicados ni pérdidas).
- Distribución por año: 2015: 158 · 2016: 157 · 2017: 155 · 2018: 156 · 2019: 156.

---

## 6. Preparación e integración de los datos

### Columnas comunes y únicas

Los cinco CSV **no comparten esquema**: la única columna que se llama igual en todos los años es `Generosity`.

| Variable final | 2015-2016 | 2017 | 2018-2019 |
|---|---|---|---|
| `country` | Country | Country | Country or region |
| `score` | Happiness Score | Happiness.Score | Score |
| `gdp` | Economy (GDP per Capita) | Economy..GDP.per.Capita. | GDP per capita |
| `social_support` | Family | Family | Social support |
| `life_expectancy` | Health (Life Expectancy) | Health..Life.Expectancy. | Healthy life expectancy |
| `freedom` | Freedom | Freedom | Freedom to make life choices |
| `corruption` | Trust (Government Corruption) | Trust..Government.Corruption. | Perceptions of corruption |
| `generosity` | Generosity | Generosity | Generosity |

Las variables del dataset son *contribuciones* al score (coeficiente × distancia a Dystopia, el país imaginario con los valores más bajos observados), no mediciones crudas. Por eso el mínimo de cada variable es 0.

**Columnas descartadas y por qué**

| Columna | Motivo |
|---|---|
| `Happiness Rank`, `Overall rank` | Es el propio score expresado como posición (fuga de información). |
| `Dystopia Residual` | El score es la suma de las seis contribuciones, el valor de Dystopia (constante) y el residuo de cada país; usarlo permitiría predecir el objetivo casi sin error. Tampoco existe en 2018-2019. |
| `Standard Error`, `Whisker.high/low`, `Lower/Upper Confidence Interval` | Se derivan del score y solo existen en algunos años. |
| `Region` | Solo existe en 2015-2016. |

### Integración de los años

Se apilaron los cinco años en formato largo (una fila por país-año, con la columna `year`). Resultado: `data/processed/happiness_clean.csv`, de 782 filas × 9 columnas.

- *Justificación:* es la forma más simple de conservar toda la información y permite que `year` actúe como variable.
- *Limitación:* un mismo país aparece varias veces, lo que afecta cómo debe interpretarse el split train/test (ver [sección 8](#robustez)).

### Calidad de los datos

| Control | Resultado | Tratamiento |
|---|---|---|
| Nulos | 1 en total: `corruption` de Emiratos Árabes Unidos 2018 | Se imputa con la mediana **dentro del pipeline** (solo con datos de entrenamiento) |
| Duplicados `(country, year)` | 0 | — |
| Ceros | Hay filas con un 0 en alguna variable (p. ej. Central African Republic, Somalia, Greece en `generosity`) | Se conservan como valores válidos (ver [limitaciones](#12-limitaciones)) |
| Outliers por IQR | `social_support`: 10 · `corruption`: 67 · `generosity`: 17 · el resto, 0 | Solo se reportan, **no se eliminan** (son colas de distribuciones sesgadas, no errores evidentes) |
| Nombres de país inconsistentes | 6 casos | Se unifican con un diccionario (abajo) |

**Unificación de nombres de países**

| En los datos | Se unifica a |
|---|---|
| `Somaliland region` | `Somaliland Region` |
| `Hong Kong S.A.R., China` | `Hong Kong` |
| `Taiwan Province of China` | `Taiwan` |
| `Trinidad & Tobago` | `Trinidad and Tobago` |
| `North Cyprus` | `Northern Cyprus` |
| `Macedonia` | `North Macedonia` |

Tras la unificación hay **164 países distintos**: 146 aparecen en los cinco años y 18 en menos (cobertura desigual por año).

---

## 7. Análisis exploratorio (EDA)

Figuras en `reports/figures/` y tablas en `reports/`.

### Hallazgos principales

1. **El score es estable entre años.** Medias de 5.376, 5.382, 5.354, 5.376 y 5.407 (2015-2019); sin outliers. Mezclar los años es razonable para el objetivo.
2. **Relación con el score** (correlación de Pearson global): `gdp` 0.79 · `life_expectancy` 0.74 · `social_support` 0.65 · `freedom` 0.55 · `corruption` 0.40 · `generosity` 0.14. La relación de cada variable con el score es estable entre años, salvo `generosity`, que baja de 0.18 (2015) a 0.08 (2019).
3. **Multicolinealidad.** El único par con correlación alta (> 0.6) es `gdp`–`life_expectancy` (0.78).
4. **`corruption` no es lineal** (casi plana hasta ~0.1 y luego sube) y está sesgada a la derecha, lo que favorece a los modelos de árboles.
5. **Cambios de escala entre ediciones.** Las medias de varias variables se desplazan entre años mientras el score no varía:

| Año | `gdp` | `social_support` | `life_expectancy` | `freedom` | `corruption` | `generosity` |
|---|---|---|---|---|---|---|
| 2015 | 0.846 | 0.991 | 0.630 | 0.429 | 0.143 | 0.237 |
| 2016 | 0.954 | **0.794** | 0.558 | 0.371 | 0.138 | 0.243 |
| 2017 | 0.985 | **1.189** | 0.551 | 0.409 | 0.123 | 0.247 |
| 2018 | 0.891 | 1.213 | 0.597 | 0.455 | 0.112 | **0.181** |
| 2019 | 0.905 | 1.209 | **0.725** | 0.393 | 0.111 | 0.185 |

### Consistencia entre años (`consistency_check.py`)

Cambio **mediano** de cada variable respecto al año anterior, por país:

| Año | `gdp` | `social_support` | `life_expectancy` | `freedom` | `corruption` | `generosity` |
|---|---|---|---|---|---|---|
| 2016 | +0.104 | **−0.195** | −0.085 | −0.068 | −0.006 | +0.002 |
| 2017 | +0.038 | **+0.398** | −0.001 | +0.039 | −0.010 | +0.004 |
| 2018 | **−0.103** | +0.032 | +0.046 | +0.050 | −0.007 | **−0.056** |
| 2019 | +0.027 | −0.004 | **+0.131** | −0.074 | −0.004 | +0.004 |

- Ocho de los diez mayores saltos interanuales por país son de `social_support` en 2017 (≈ 0.5), en países muy distintos (Nigeria, Tajikistán, Bulgaria, Egipto, Montenegro, México, Líbano, Bangladesh). Un patrón así afecta a casi todos los países a la vez y apunta a un **cambio sistemático de escala o metodología**, no a errores de captura aislados.
- `corruption` es la variable más estable entre ediciones.
- Como el score medio no cambia mientras sus componentes sí, se interpreta como un cambio de definición o de método entre ediciones. **Es una hipótesis**: no se verificó con la documentación metodológica de cada edición.
- *Consecuencia en el modelo:* se incorpora `year` como variable para absorber estos desplazamientos.

### Un registro anómalo aislado: Emiratos Árabes Unidos 2018

| Año | `score` | `gdp` | `social_support` | `freedom` | `corruption` |
|---|---|---|---|---|---|
| 2015 | 6.901 | 1.427 | 1.126 | 0.642 | 0.386 |
| 2016 | 6.573 | 1.574 | 0.871 | 0.562 | 0.356 |
| 2017 | 6.648 | 1.626 | 1.266 | 0.608 | 0.324 |
| **2018** | **6.774** | **2.096** | **0.776** | **0.284** | **NaN** |
| 2019 | 6.825 | 1.503 | 1.310 | 0.598 | 0.182 |

El score de 2018 es normal, pero sus variables de entrada se salen de la serie del propio país (`gdp` es el valor máximo de todo el dataset, `freedom` y `social_support` son atípicamente bajos y `corruption` falta). A diferencia del cambio de 2017, este es un **registro puntual anómalo**. Se conserva y se documenta; eliminarlo habría mejorado artificialmente la métrica de test, ya que cayó en esa partición.

---

## 8. Modelado y evaluación

### Diseño

- **Objetivo:** `score`. **Predictores:** `gdp`, `social_support`, `life_expectancy`, `freedom`, `corruption`, `generosity` y, según el conjunto, `year` (codificada como categoría).
- **No se usa `country` como variable.** Habría convertido el modelo en una tabla de consulta por país en lugar de predecir a partir de indicadores.
- **Split:** 80 % entrenamiento (625 filas) / 20 % prueba (157 filas), con semilla fija (`random_state=42`). El test no se toca hasta la evaluación final.
- **Sin fuga de información.** Todo el preprocesamiento (imputación por mediana, escalado y codificación de `year`) está dentro de un `Pipeline` de scikit-learn y se ajusta **solo con los datos de entrenamiento**. En la validación cruzada se reajusta dentro de cada fold.
- **Comparación:** 4 modelos × 3 conjuntos de variables, con validación cruzada de 5 folds **solo sobre el entrenamiento**.
  - Modelos: `LinearRegression`, `Ridge(alpha=1)`, `RandomForest(300 árboles, min_samples_leaf=3)`, `GradientBoosting` (parámetros por defecto).
  - Conjuntos: `base6` (las seis variables), `sin_generosity` y `base6+year`.

### Resultados de la validación cruzada (RMSE, menor es mejor)

| Variables | Modelo | RMSE medio | Desv. |
|---|---|---|---|
| **base6+year** | **GradientBoosting** | **0.5061** | 0.0307 |
| base6+year | RandomForest | 0.5173 | 0.0436 |
| base6 | RandomForest | 0.5215 | 0.0427 |
| base6 | GradientBoosting | 0.5241 | 0.0331 |
| sin_generosity | GradientBoosting | 0.5295 | 0.0339 |
| sin_generosity | RandomForest | 0.5315 | 0.0417 |
| base6+year | Ridge | 0.5316 | 0.0398 |
| base6+year | LinearRegression | 0.5317 | 0.0400 |
| base6 | Ridge | 0.5593 | 0.0310 |
| base6 | LinearRegression | 0.5593 | 0.0310 |
| sin_generosity | Ridge | 0.5602 | 0.0342 |
| sin_generosity | LinearRegression | 0.5602 | 0.0343 |

**Lectura**

- **`year` mejora todos los modelos** (p. ej. regresión lineal: 0.559 → 0.532), lo que respalda la hipótesis de cambio de escala entre ediciones.
- **`generosity` aporta poco, pero quitarla empeora** ligeramente cada modelo; se conserva.
- **Los modelos de árboles superan a los lineales**, coherente con la no linealidad de `corruption`.
- **Matiz:** la diferencia entre GradientBoosting y RandomForest (0.506 vs 0.517) es menor que la desviación entre folds, así que no están claramente separados. Se eligió el de menor RMSE medio. Una alternativa defendible es la regresión lineal con `year` (RMSE 0.532): pierde ~0.03 de RMSE pero es totalmente interpretable.

### Evaluación final en el conjunto de prueba

| Métrica | Modelo (GradientBoosting + year) | Baseline (predecir la media) |
|---|---|---|
| MAE | **0.4045** | 0.9455 |
| MSE | 0.2406 | 1.2631 |
| RMSE | **0.4905** | 1.1239 |
| R² | **0.8083** | −0.0065 |
| R² ajustado | 0.7993 | −0.0538 |

El RMSE de test (0.49) es consistente con el de validación cruzada (0.506): no hay señales de sobreajuste. Un error típico de ~0.4 puntos sobre una escala cuya desviación estándar es 1.13.

### Robustez

| Evaluación | MAE | RMSE | R² | R² ajustado |
|---|---|---|---|---|
| Split aleatorio (principal) | 0.4045 | 0.4905 | 0.8083 | 0.7993 |
| **Agrupado por país** (ningún país en train y test a la vez) | 0.4438 | 0.5638 | 0.7935 | 0.7837 |
| **Temporal** (entrena 2015-2017, evalúa 2018-2019) | 0.4494 | 0.5562 | 0.7502 | 0.7445 |

- El split aleatorio es algo **optimista**: un mismo país aparece en varios años y parte del test se parece al entrenamiento. Para países nuevos, lo esperable es un RMSE cercano a 0.56.
- En la prueba temporal, los años 2018-2019 no existen en el entrenamiento y el codificador de `year` los ignora; es la prueba más exigente.
- Las tres evaluaciones usan muestras de prueba distintas, por lo que la comparación es aproximada.

### Importancia de las variables

Importancia por permutación sobre el test (valores aproximados, leídos de `reports/figures/08_importancia.png`; los exactos están en `reports/feature_importance.csv`): `gdp` ≈ 0.27 · `life_expectancy` ≈ 0.25 · `social_support` ≈ 0.12 · `freedom` ≈ 0.10 · `corruption` ≈ 0.04 · `year` ≈ 0.025 · `generosity` ≈ 0.02 (aumento del RMSE al desordenar cada variable). Como `gdp` y `life_expectancy` están muy correlacionadas (0.78), esta técnica puede repartir su importancia entre ambas; no conviene afirmar que una importa más que la otra.

### Análisis de errores

**Los 10 mayores errores del test** (error = real − predicho):

| País | Año | Real | Predicho | Error |
|---|---|---|---|---|
| United Arab Emirates | 2018 | 6.77 | 5.37 | +1.40 |
| Hong Kong | 2016 | 5.46 | 6.65 | −1.19 |
| Hong Kong | 2018 | 5.43 | 6.55 | −1.12 |
| Sri Lanka | 2017 | 4.44 | 5.55 | −1.11 |
| Costa Rica | 2015 | 7.23 | 6.12 | +1.10 |
| Botswana | 2017 | 3.77 | 4.81 | −1.05 |
| Moldova | 2016 | 5.90 | 4.86 | +1.03 |
| United States | 2017 | 6.99 | 5.99 | +1.00 |
| Suriname | 2016 | 6.27 | 5.29 | +0.98 |
| El Salvador | 2016 | 6.07 | 5.11 | +0.96 |

- Hay países que el modelo **sobrestima** (Hong Kong, Sri Lanka, Botswana) y otros que **subestima** (Costa Rica, El Salvador, Moldova). Hipótesis: factores culturales o regionales no capturados por las seis variables.
- Hong Kong falla de forma parecida en dos años distintos, aunque sus otros años estén en el entrenamiento: es lo esperado, porque `country` no es una variable del modelo.
- El mayor error (Emiratos Árabes Unidos 2018) se explica en buena parte por sus entradas anómalas y el nulo imputado (sección 7).
- **Gráficos:** el de real vs predicho muestra que el modelo **comprime los extremos** (sobrestima scores bajos y subestima scores altos). Los residuos se centran en 0, sin embudo claro, con una cola algo más larga hacia valores positivos.

### Artefacto serializado

`models/happiness_model.joblib` guarda un diccionario con el **pipeline completo** (preprocesamiento + modelo), la lista de variables, el nombre del modelo, las métricas y la versión de scikit-learn. El consumidor de streaming no repite el preprocesamiento: arma el DataFrame y llama a `predict`.

---

## 9. Streaming: productor, consumidor y base de datos

### Kafka local

`docker-compose.yml` levanta un broker Apache Kafka 3.9 de un solo nodo (modo KRaft, sin ZooKeeper) en `localhost:9092`. Se usa un broker propio para **no escribir en el broker compartido del curso**.

### Productor (`src/producer.py`)

Lee `data/processed/test_split.csv` (los 157 registros que el modelo **nunca vio**) y los publica de uno en uno en el tópico `happiness.stream`, con clave `país-año` y valor JSON. Los `NaN` se serializan como `null`.

```powershell
python src\producer.py --delay 1        # un mensaje por segundo
python src\producer.py --limit 5        # solo los primeros 5
```

### Consumidor de inferencia (`src/inference_consumer.py`)

1. Carga el artefacto `.joblib` una sola vez, al arrancar.
2. Escucha `happiness.stream` y, por cada evento, arma el DataFrame de entrada, predice y calcula el error.
3. Evalúa las reglas de alerta y guarda el resultado en SQLite.
4. **Confirma el offset a Kafka solo después de guardar en la base de datos.**

**Reglas de alerta** (ajustables por argumento):

| Alerta | Condición | Por defecto |
|---|---|---|
| `error_alto` | \|real − predicho\| ≥ umbral | 0.8 (≈ 1.6 × RMSE) |
| `bienestar_bajo` | score predicho < umbral | 4.0 |

### Base de datos

SQLite en `data/predictions.db`, modo WAL (permite que el dashboard lea mientras el consumidor escribe). Tabla `predictions`: `received_at`, `country`, `year`, las seis variables, `actual`, `predicted`, `abs_error`, `alert`, `alert_reason`, `kafka_partition` y `kafka_offset`, con restricción `UNIQUE (kafka_partition, kafka_offset)`.

### Garantías

- **Sin pérdidas:** el offset se confirma después de guardar; si el consumidor se cae a mitad, Kafka reentrega el mensaje.
- **Sin duplicados por reentrega:** la restricción `UNIQUE` con `INSERT OR IGNORE` hace que reprocesar el mismo mensaje no cree filas nuevas.
- **Validación de punta a punta:** tras procesar el tópico completo, la base de datos contiene 157 filas (157 combinaciones país-año distintas) y el promedio de `abs_error` es 0.4045, **idéntico al MAE de test** obtenido en el entrenamiento. Es decir, la inferencia en streaming reproduce exactamente las predicciones del entrenamiento.

---

## 10. Dashboard en Streamlit

```powershell
streamlit run dashboard/app.py
```

Un reporte de **una sola página**, de arriba hacia abajo:

1. **Desempeño del modelo en test** (MAE, RMSE, R², R² ajustado), con el baseline como referencia.
2. **Indicadores en vivo:** eventos procesados, MAE acumulado (con su diferencia respecto al MAE de test), número de alertas y último evento.
3. **Real vs predicho** de los últimos 40 eventos, con las alertas marcadas con un rombo.
4. **Error absoluto por evento**, con la línea del umbral de alerta.
5. **Últimas alertas** en una tabla, con el motivo escrito.
6. *(Plegado)* **Variables más influyentes.**

### Optimización de carga

- `@st.cache_data` carga una sola vez los datos estáticos (métricas e importancia de variables).
- `@st.fragment(run_every=2)` aísla el panel en vivo: solo ese bloque se vuelve a ejecutar cada 2 segundos, sin recargar el resto de la aplicación.

### Accesibilidad

- Paleta **Okabe-Ito**, diseñada para ser distinguible con distintos tipos de daltonismo.
- Ninguna información depende solo del color: las series se distinguen también por **forma** (círculo/cuadrado) y **trazo** (continuo/discontinuo); las alertas usan un **rombo**; el umbral tiene una **línea con etiqueta de texto**; y el motivo de cada alerta aparece **escrito** en la tabla.

---

## 11. Decisiones de diseño

| Decisión | Justificación |
|---|---|
| Formato largo (años apilados) | Conserva toda la información y permite usar `year` como variable. |
| Descartar `Rank`, `Dystopia Residual`, errores e intervalos | Evitar fuga de información: se derivan del propio score. |
| No usar `country` como variable | Evitar un modelo de consulta por país; se busca predecir desde indicadores. |
| `year` como variable categórica | Absorber los cambios de escala entre ediciones; mejora la validación cruzada. |
| Imputación por mediana dentro del `Pipeline` | Se ajusta solo con entrenamiento (sin fuga) y viaja con el artefacto. |
| No eliminar outliers ni ceros | Son valores plausibles de distribuciones sesgadas; con ~780 filas se perdería información. |
| No eliminar Emiratos Árabes Unidos 2018 | Quitarlo por empeorar la métrica de test sería inflar los resultados; se documenta como anomalía. |
| Comparar modelos con CV solo sobre entrenamiento | El test se usa una única vez, al final. |
| Pruebas de robustez (por país y temporal) | El split aleatorio es optimista cuando un país se repite en varios años. |
| Kafka local para el streaming | No escribir en el broker compartido del curso. |
| El productor reproduce el **test split** | Los eventos simulados son datos que el modelo no vio. |
| Offset confirmado después de guardar + `UNIQUE` | Entrega al menos una vez y escritura idempotente por mensaje. |
| SQLite en modo WAL | Lectura concurrente del dashboard sin bloquear al consumidor. |
| Persistir el crudo en `records.jsonl` | Desacopla el análisis del broker y permite repetirlo sin volver a consumir. |

---

## 12. Limitaciones

1. **Cambios de escala entre ediciones.** Se observan desplazamientos sistemáticos en varias variables. Una posible causa es que las contribuciones se recalculan en cada edición con un modelo que ha evolucionado (según el FAQ) y revisiones de datos como el reescalado del PIB (apéndice 2019). No se verificó que esto explique los saltos concretos de 2016-2019.
2. **Equivalencia `Family` ≈ `Social support`.** Se define social support como el promedio nacional de respuestas sí/no a si la persona tiene a alguien con quien contar en un problema. No se encontró una nota oficial que confirme que Family (2015-2017) sea la misma pregunta; se asume por el nombre y porque su relación con el score es similar (correlación 0.74-0.78), aunque su nivel cambia.
3. **Ceros como valores válidos.** Son consistentes con la definición: un país con el valor más bajo observado en una variable tiene contribución 0.
4. **Un mismo país aparece en varios años.** El split aleatorio es optimista; el RMSE esperado para países nuevos ronda 0.56.
5. **Años no vistos.** El modelo no puede aprender el desplazamiento de un año ausente del entrenamiento (el codificador de `year` lo ignora); afecta a la prueba temporal (R² 0.75) y a cualquier predicción futura.
6. **Imputación por mediana global.** Para Emiratos Árabes Unidos 2018 la mediana (~0.09) queda lejos de los valores del país en otros años (0.18-0.39).
7. **Muestra pequeña.** 782 filas y un test de 157: las métricas tienen incertidumbre, y la diferencia entre los mejores modelos queda dentro de la desviación entre folds.
8. **Compresión de extremos.** El modelo sobrestima scores bajos y subestima scores altos.
9. **`Region` excluida.** Solo existe en 2015-2016; no se reconstruyó para los demás años.
10. **El streaming es una simulación.** El productor reproduce datos de prueba ya conocidos, no datos nuevos.
11. **Idempotencia por mensaje, no por contenido.** Si el productor se ejecuta dos veces, el tópico recibe los eventos dos veces y el consumidor los procesa ambas (probado: 225 filas con 157 combinaciones país-año distintas).
12. **Umbrales de alerta fijos** (0.8 y 4.0), elegidos de forma heurística. El dashboard repite el umbral de error como constante (`ERROR_THRESHOLD`), que debe coincidir con el del consumidor.
13. **Compatibilidad del artefacto.** Un archivo `.joblib` depende de la versión de scikit-learn con la que se creó (se guarda en el propio artefacto).
14. **Despliegue local.** Kafka, SQLite y Streamlit corren en una sola máquina; no hay autenticación ni alta disponibilidad.

---

## 13. Posibles mejoras

- **Ajuste de hiperparámetros** del `GradientBoosting` (`RandomizedSearchCV`); hoy usa los valores por defecto.
- **Estandarizar cada variable dentro de su año**, para que el modelo vea posiciones relativas y no niveles absolutos. Debe hacerse con estadísticas calculadas solo con entrenamiento, o declarar la fuga.
- **Reconstruir `Region`** con un mapeo país→región para todos los años; podría explicar los errores sistemáticos por país.
- **Mitigar la compresión de extremos** (calibración de predicciones o modelos que la reduzcan).
- **Imputación más informada** (por ejemplo, con los valores del mismo país en otros años), cuidando no introducir fuga hacia el test.
- **Contrastar las hipótesis** de las limitaciones 1-3 con la documentación metodológica del World Happiness Report.
- **Idempotencia por contenido** en el consumidor (clave país-año) o una guarda en el productor.
- **Monitoreo de deriva** en el dashboard (comparar el error acumulado con el esperado).
- **Base de datos de producción** (PostgreSQL/MySQL) y despliegue en contenedores.

---

## 14. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError` al ejecutar un script | El entorno virtual no está activo | `.venv\Scripts\Activate.ps1` (debe aparecer `(.venv)`) |
| `KeyError: 'KAFKA_PASSWORD'` | No se lee el `.env` | Ejecutar desde la raíz del proyecto y revisar que exista `.env` |
| Error de autenticación SASL | Contraseña mal copiada | Revisar `.env` (sin comillas ni espacios alrededor del `=`) |
| `Mensajes consumidos: 0` | El `group.id` ya consumió todo el tópico | Cambiar `GROUP_ID` en `.env` (por ejemplo, agregar un sufijo) |
| Timeout o conexión rechazada al broker del curso | Red que bloquea el puerto 9092 | Probar con otra red o con datos móviles |
| El consumidor muestra "tópico no disponible" | Arrancó antes de que existiera `happiness.stream` | Crear el tópico antes (sección 4); el aviso es inofensivo |
| La base de datos tiene más de 157 filas | El productor se ejecutó más de una vez | Reinicio limpio (abajo) |
| El dashboard dice "Esperando eventos" con datos en la base | Se ejecutó desde otra carpeta | Lanzarlo desde la raíz del proyecto |

**Reinicio limpio del streaming**

```powershell
# 1. Detener el consumidor (Ctrl+C) y borrar la base de datos (incluye archivos WAL)
Remove-Item data\predictions.db*

# 2. Borrar y volver a crear el tópico
docker exec -it kafka-local /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic happiness.stream
docker exec -it kafka-local /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic happiness.stream --partitions 1 --replication-factor 1

# 3. Consumidor con un grupo nuevo, y luego el productor UNA vez
python src\inference_consumer.py --group-id happiness-inference-v2
python src\producer.py --delay 0.3
```

## Referencias

- World Happiness Report, *Data sharing*: https://www.worldhappiness.report/data-sharing/
- Paleta Okabe-Ito (Okabe & Ito, 2008), diseñada para ser legible con daltonismo.
