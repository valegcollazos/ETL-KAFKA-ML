import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ERROR_THRESHOLD = 0.8      # mismo valor que usa el consumidor (--error-threshold)
REFRESH_SECONDS = 2
WINDOW = 40                # eventos visibles en los gráficos

# Paleta Okabe-Ito (apta para daltonismo)
BLUE, ORANGE, VERMILLION, GREEN = "#0072B2", "#E69F00", "#D55E00", "#009E73"
REASONS = {"error_alto": "Error alto",
           "bienestar_bajo": "Bienestar bajo predicho"}

st.set_page_config(page_title="Happiness Score en tiempo real", layout="wide")



#Datos ESTÁTICOS

@st.cache_data
def load_static():
    metrics = json.loads(
        (ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    importance = pd.read_csv(ROOT / "reports" / "feature_importance.csv")
    return metrics, importance


try:
    metrics, importance = load_static()
except FileNotFoundError:
    st.error("Faltan reports/metrics.json o reports/feature_importance.csv. "
             "Ejecuta primero: python src/train.py")
    st.stop()

test = metrics["test"]


st.title("Predicción del Happiness Score en tiempo real")
st.caption("Kafka → modelo → SQLite → dashboard. Paleta apta para daltonismo: "
           "cada serie se distingue también por forma y tipo de trazo.")

st.subheader(f"Desempeño del modelo en test · {metrics['model']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{test['MAE']:.3f}")
c2.metric("RMSE", f"{test['RMSE']:.3f}",
          f"baseline {metrics['baseline']['RMSE']:.3f}", delta_color="off")
c3.metric("R²", f"{test['R2']:.3f}")
c4.metric("R² ajustado", f"{test['R2_ajustado']:.3f}")



#gráficos

def chart_scores(df):
    """Real vs predicho; las alertas se marcan con un rombo hueco."""
    long = df.melt(id_vars=["evento", "country", "year"],
                   value_vars=["actual", "predicted"],
                   var_name="serie", value_name="score")
    long["serie"] = long["serie"].map({"actual": "Real", "predicted": "Predicho"})
    domain = ["Real", "Predicho"]

    base = alt.Chart(long).encode(
        x=alt.X("evento:Q", title="Evento (orden de llegada)"),
        y=alt.Y("score:Q", title="Happiness Score", scale=alt.Scale(zero=False)),
        color=alt.Color("serie:N", title="Serie",
                        scale=alt.Scale(domain=domain, range=[BLUE, ORANGE])),
        tooltip=["country", "year", "serie", alt.Tooltip("score:Q", format=".2f")],
    )
    lines = base.mark_line().encode(
        strokeDash=alt.StrokeDash("serie:N", title="Serie",
                                  scale=alt.Scale(domain=domain,
                                                  range=[[1, 0], [6, 4]])))
    points = base.mark_point(filled=True, size=55).encode(
        shape=alt.Shape("serie:N", title="Serie",
                        scale=alt.Scale(domain=domain,
                                        range=["circle", "square"])))
    layers = lines + points

    alerts = df[df["alert"] == 1]
    if not alerts.empty:
        ring = alt.Chart(alerts).mark_point(
            shape="diamond", size=230, filled=False,
            color=VERMILLION, strokeWidth=2.5).encode(
            x="evento:Q", y="predicted:Q",
            tooltip=["country", "year",
                     alt.Tooltip("predicted:Q", format=".2f"), "alert_reason"])
        layers = layers + ring
    return layers.properties(height=300, width="container")


def chart_errors(df):
    """Error absoluto por evento, con la línea del umbral de alerta."""
    d = df.dropna(subset=["abs_error"]).copy()
    d["Estado"] = d["abs_error"].ge(ERROR_THRESHOLD).map(
        {True: "Sobre el umbral", False: "Dentro del umbral"})
    bars = alt.Chart(d).mark_bar().encode(
        x=alt.X("evento:O", title="Evento (orden de llegada)",
                axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("abs_error:Q", title="Error absoluto"),
        color=alt.Color("Estado:N", title=None,
                        scale=alt.Scale(
                            domain=["Dentro del umbral", "Sobre el umbral"],
                            range=[BLUE, VERMILLION])),
        tooltip=["country", "year",
                 alt.Tooltip("abs_error:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"umbral": [ERROR_THRESHOLD]})).mark_rule(
        strokeDash=[6, 4], color="black", strokeWidth=1.5).encode(y="umbral:Q")
    label = alt.Chart(pd.DataFrame(
        {"umbral": [ERROR_THRESHOLD], "texto": [f"umbral {ERROR_THRESHOLD}"]})
    ).mark_text(align="left", dx=4, dy=-6, color="black").encode(
        y="umbral:Q", text="texto:N")
    return (bars + rule + label).properties(height=300, width="container")



#Panel EN VIVO: solo este fragmento se vuelve a ejecutar cada 2s

@st.fragment(run_every=REFRESH_SECONDS)
def live_panel():
    summary = queries.load_summary()
    st.subheader("Predicciones en vivo")
    if summary["total"] == 0:
        st.info("Esperando eventos. Inicia el consumidor "
                "(src/inference_consumer.py) y luego el productor "
                "(src/producer.py).")
        return

    recent = queries.load_recent(WINDOW)
    total = summary["total"]
    recent["evento"] = range(total - len(recent) + 1, total + 1)
    last = recent.iloc[-1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Eventos procesados", f"{total}")
    k2.metric("MAE acumulado", f"{summary['mae']:.3f}",
              f"{summary['mae'] - test['MAE']:+.3f} vs test",
              delta_color="inverse")
    k3.metric("Alertas", f"{summary['alerts']}",
              f"{summary['alerts'] / total:.0%} de los eventos",
              delta_color="off")
    k4.metric("Último evento", f"{last['country']} {int(last['year'])}")
    k4.caption(f"Predicho {last['predicted']:.2f} · "
               f"real {last['actual']:.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Real vs predicho** (últimos {len(recent)} eventos)")
        st.altair_chart(chart_scores(recent))
        st.caption("◇ Rombo naranja-rojizo = evento con alerta.")
    with right:
        st.markdown("**Error absoluto por evento**")
        st.altair_chart(chart_errors(recent))
        st.caption(f"Alerta de error alto: |real − predicho| ≥ "
                   f"{ERROR_THRESHOLD}.")

    st.markdown(f"**Últimas alertas** · {summary['error_alto']} por error alto, "
                f"{summary['bienestar_bajo']} por bienestar bajo predicho")
    alerts = queries.load_alerts(8)
    if alerts.empty:
        st.success("Sin alertas por ahora.")
    else:
        view = pd.DataFrame({
            "País": alerts["country"],
            "Año": alerts["year"],
            "Real": alerts["actual"].round(2),
            "Predicho": alerts["predicted"].round(2),
            "Error": alerts["abs_error"].round(2),
            "Motivo": alerts["alert_reason"].fillna("").apply(
                lambda s: " + ".join(REASONS.get(r, r) for r in s.split(","))),
        })
        st.dataframe(view, hide_index=True)


live_panel()


# Importancia de variables (estático)

with st.expander("Variables más influyentes (importancia por permutación, test)"):
    imp = importance.sort_values("aumento_rmse", ascending=False)
    chart = alt.Chart(imp).mark_bar(color=GREEN).encode(
        y=alt.Y("variable:N", sort="-x", title=None),
        x=alt.X("aumento_rmse:Q", title="Aumento del RMSE al desordenar la variable"),
        tooltip=["variable", alt.Tooltip("aumento_rmse:Q", format=".3f")])
    labels = chart.mark_text(align="left", dx=4).encode(
        text=alt.Text("aumento_rmse:Q", format=".3f"))
    st.altair_chart((chart + labels).properties(height=250, width="container"))