# -*- coding: utf-8 -*-
"""
IDC Dashboard (Streamlit)
- Gráfico con series por departamento (hist+forecast en línea continua)
- Promedio nacional continuo + capa punteada en años de forecast
- Bandas IC95 opacas con borde punteado
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import forecast_por_departamento


# ========================
# Configuración de página
# ========================
st.set_page_config(page_title="IDC Dashboard", layout="wide")

assets = Path("assets")
logo_utp = assets / "Logo_UTP.png"
logo_crc = assets / "Logo_CRC.png"

# === ENCABEZADO ===
col1, col2 = st.columns([3, 1], gap="large")
with col1:
    st.markdown(
        "<h1 style='margin-top: 10px; margin-bottom: 0px; text-align:left;'>"
        "Tablero de seguimiento al IDC"
        "</h1>",
        unsafe_allow_html=True,
    )
with col2:
    logo_col1, logo_col2 = st.columns([1, 1], gap="small")
    with logo_col1:
        st.image(str(logo_utp), width="stretch")
    with logo_col2:
        st.image(str(logo_crc), width="stretch")

st.markdown("---")

# ========================
# Carga de datos
# ========================
ruta_excel = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vToapNGBE4m3q7hFtf3NHJdREeYIUtnIxHUTpdN_gykbClhZFDonm1ZG8n4jm85ag/"
    "pub?output=xlsx"
)
hojas = pd.read_excel(ruta_excel, sheet_name=None)

df_listado_indicadores = hojas["Listado_indicadores"]
df_IDC = hojas["Valores_IDC"]
df_indicadores_normalizados = hojas["Valores_Indicadores_Normalizado"]
df_analisis_sensibilidad = hojas["Sensibilidad"]
del hojas, ruta_excel

# ========================
# Forecast por dpto
# ========================
out_df, modelos = forecast_por_departamento(df_IDC, horizonte=2)

# ========================
# Secciones C + E y D
# ========================
left, right = st.columns([2, 1], gap="large")  # 67% / 33%

# --------- C: Evolución y pronóstico del IDC ----------
# Limpieza básica
tmp = (
    out_df.copy()
    .dropna(subset=["Departamento", "Año"])
    .sort_values(["Departamento", "Año"])
)
tmp["Año"] = tmp["Año"].astype(int)

# Promedio nacional (sobre TODOS los departamentos)
prom = (
    tmp.groupby(["Año", "Tipo"], as_index=False)["IDC"]
    .mean()
    .rename(columns={"IDC": "IDC_prom"})
)

# Lista de interés (puede venir de otra parte; si no, dejamos una por defecto)
try:
    departamentos_pares  # noqa: F821  # type: ignore[name-defined]
except NameError:
    departamentos_pares = ["Quindío", "Caldas", "Risaralda"]


# ========================
# Figuras
# ========================
def _hex_to_rgb_tuple(hexcolor: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (r,g,b). Si falla, usa gris."""
    if isinstance(hexcolor, str) and hexcolor.startswith("#") and len(hexcolor) == 7:
        return tuple(int(hexcolor[i : i + 2], 16) for i in (1, 3, 5))
    return (80, 80, 80)


def build_idc_figure(tmp: pd.DataFrame, prom: pd.DataFrame, dptos_sel: List[str]) -> go.Figure:
    """
    Construye la figura IDC mostrando sólo `dptos_sel` y el promedio nacional.
    - Línea continua por dpto (hist+forecast) + capa punteada en forecast.
    - Promedio nacional continuo + capa punteada en forecast.
    - Bandas IC95 con baja opacidad y borde punteado.
    """
    fig = go.Figure()

    # Paleta de alto contraste
    palette = px.colors.qualitative.D3 + px.colors.qualitative.Bold + px.colors.qualitative.Dark24

    # --- Departamentos filtrados ---
    deptos_validos = [d for d in dptos_sel if d in tmp["Departamento"].unique()]
    for i, dept in enumerate(deptos_validos):
        color = palette[i % len(palette)]
        r, g, b = _hex_to_rgb_tuple(color)
        sub = tmp.loc[tmp["Departamento"] == dept].sort_values("Año")

        # --- Bandas IC95 (debajo) ---
        fc = sub[sub["Tipo"] == "forecast"]
        if not fc.empty and fc["Lo95"].notna().all() and fc["Hi95"].notna().all():
            # polígono relleno
            fig.add_trace(
                go.Scatter(
                    x=list(fc["Año"]) + list(fc["Año"][::-1]),
                    y=list(fc["Hi95"]) + list(fc["Lo95"][::-1]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.12)",
                    line=dict(width=0),
                    legendgroup=dept,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # bordes punteados para distinguir bandas solapadas
            fig.add_trace(
                go.Scatter(
                    x=fc["Año"],
                    y=fc["Hi95"],
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.55)", width=1, dash="dot"),
                    legendgroup=dept,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fc["Año"],
                    y=fc["Lo95"],
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.55)", width=1, dash="dot"),
                    legendgroup=dept,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # --- Línea continua (hist + forecast) ---
        fig.add_trace(
            go.Scatter(
                x=sub["Año"],
                y=sub["IDC"],
                mode="lines+markers",
                name=dept,
                legendgroup=dept,
                marker=dict(symbol="circle", size=7),
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{dept}</b><br>Año=%{{x}}<br>IDC=%{{y:.3f}}<extra></extra>",
                showlegend=True,
            )
        )

        # --- Capa punteada (sólo sobre forecast) ---
        if not fc.empty:
            fig.add_trace(
                go.Scatter(
                    x=fc["Año"],
                    y=fc["IDC"],
                    mode="lines+markers",
                    marker=dict(symbol="circle-open", size=8),
                    line=dict(color=color, width=2.5, dash="dash"),
                    legendgroup=dept,
                    hovertemplate=f"<b>{dept} (forecast)</b><br>Año=%{{x}}<br>IDC=%{{y:.3f}}<extra></extra>",
                    showlegend=False,
                )
            )

    # --- Promedio nacional continuo (una sola serie) ---
    prom_w = prom.pivot(index="Año", columns="Tipo", values="IDC_prom").sort_index()
    prom_w["IDC_line"] = np.where(prom_w.get("hist").notna(), prom_w["hist"], prom_w.get("forecast"))
    anos_fc = prom.loc[prom["Tipo"] == "forecast", "Año"].sort_values()

    # Traza continua negra (sin cortes)
    fig.add_trace(
        go.Scatter(
            x=prom_w.index.astype(int),
            y=prom_w["IDC_line"],
            mode="lines+markers",
            name="Promedio IDC",
            legendgroup="Promedio",
            marker=dict(symbol="circle", size=8),
            line=dict(color="black", width=3.2),
            hovertemplate="<b>Promedio</b><br>Año=%{x}<br>IDC=%{y:.3f}<extra></extra>",
            showlegend=True,
        )
    )
    # Capa punteada sobre los años forecast
    if len(anos_fc) > 0:
        y_fc = prom_w.loc[prom_w.index.isin(anos_fc), "IDC_line"]
        fig.add_trace(
            go.Scatter(
                x=anos_fc.astype(int),
                y=y_fc,
                mode="lines+markers",
                name="Promedio IDC (forecast)",
                legendgroup="Promedio",
                marker=dict(symbol="circle-open", size=9),
                line=dict(color="black", width=3.2, dash="dash"),
                hovertemplate="<b>Promedio (forecast)</b><br>Año=%{x}<br>IDC=%{y:.3f}<extra></extra>",
                showlegend=True,
            )
        )

    # --- Layout ---
    fig.update_layout(
        title="Evolución y pronóstico del IDC",
        xaxis_title="Año",
        yaxis_title="IDC",
        legend_title="Series",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
    )
    # fig.update_xaxes(type="category")
    # return fig
    fig.update_xaxes(type="linear", tickmode="linear", dtick=1)
    return fig


with left:
    st.subheader("Evolución y pronóstico del IDC")
    fig_plot = build_idc_figure(tmp, prom, departamentos_pares)

    # Orden cronológico en el eje X (numérico)
    st.plotly_chart(fig_plot, use_container_width=True)

    # --------- E: Lista dinámica ----------
    st.markdown("### Departamentos pares de Risaralda")
    if departamentos_pares:
        st.markdown("\n".join(f"- {d}" for d in departamentos_pares))
    else:
        st.markdown("_(sin departamentos seleccionados)_")

# --------- D: Top 10 Indicadores (derecha) ----------
top10_sensibilidad = df_analisis_sensibilidad.sort_values("Rank_Promedio").head(10)
tabla_final = (
    top10_sensibilidad.merge(
        df_listado_indicadores[["Indicador", "Nombre"]],
        how="left",
        left_on="Variable",
        right_on="Indicador",
    )[["Indicador", "Nombre", "Mu_star", "ST", "Rank_Promedio"]]
)
tabla_final.drop(["Mu_star", "ST", "Rank_Promedio"], axis=1, inplace=True)

with right:
    st.subheader("Top 10 Indicadores con mayor influencia sobre el IDC")
    sty = (
        tabla_final.style.hide(axis="index")
        .set_properties(**{"text-align": "left"})
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#1f77b4"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                    ],
                },
                {"selector": "tbody td", "props": [("border", "0px")]},
            ]
        )
        .apply(
            lambda s: ["background-color: #f2f2f2" if i % 2 == 0 else "" for i in range(len(s))],
            axis=1,
        )
    )
    st.table(sty)
