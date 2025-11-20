# -*- coding: utf-8 -*-
"""
IDC Dashboard (Streamlit)
- Gráfico con series por departamento (hist+forecast en línea continua)
- Promedio nacional continuo + capa punteada en años de forecast
- Bandas IC95 opacas con borde punteado
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import forecast_por_departamento, guardar_pronosticos_en_excel, forecast_one_series
from src.scenary_simulator import calcular_variable
from src.clustering import render_clustering
import base64
import math


# ========================
# Configuración de página
# ========================
st.set_page_config(page_title="IDC Dashboard", layout="wide")


st.markdown("""
    <style>
        /* Aumentar tamaño de los títulos de pestañas en Streamlit 1.51+ */
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p {
            font-size: 1.35rem !important;
            font-weight: 600 !important;
            margin-top: 0.2rem;
        }
    </style>
""", unsafe_allow_html=True)

assets = Path("assets")
logo_utp = assets / "Logo_UTP.png"
logo_crc = assets / "Logo_CRC.png"
logo_sand = assets / "Logo_SAND.jpeg"

def _img_to_base64(path: Path) -> str:
    """Convierte una imagen en base64 para incrustarla en HTML."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
    logo_col1, logo_col2, logo_col3 = st.columns([1.3, 1, 1], gap="small")

    # ---- Logo CRC con altura fija ----
    with logo_col1:
        crc_b64 = _img_to_base64(logo_crc)
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; align-items:center; height:100%;">
                <img src="data:image/png;base64,{crc_b64}" style="height:90px; width:auto; object-fit:contain;" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Otros logos normales (se adaptan a la columna) ----
    with logo_col2:
        st.image(str(logo_utp), width='content')
    with logo_col3:
        st.image(str(logo_sand), width='content')

st.markdown("---")

# ========================
# Carga de datos
# ========================
# ruta_excel = (
#     "https://docs.google.com/spreadsheets/d/e/"
#     "2PACX-1vToapNGBE4m3q7hFtf3NHJdREeYIUtnIxHUTpdN_gykbClhZFDonm1ZG8n4jm85ag/"
#     "pub?output=xlsx"
# )

ruta_excel = Path("data") / "Base_IDC_web_2024-2_dashboard.xlsx"
hojas = pd.read_excel(ruta_excel, sheet_name=None)


df_listado_indicadores = hojas["Listado_indicadores"]
df_IDC = hojas["Valores_IDC"]
df_indicadores_normalizados = hojas["Valores_Indicadores_Normalizado"]
df_analisis_sensibilidad = hojas["Sensibilidad"]


if "Pronostico_IDC" in hojas:
    # 1. YA EXISTE → lo usamos directamente (no calculamos nada)
    out_df = hojas["Pronostico_IDC"].copy()
    modelos = {}   # Los modelos no se usan en la app, así que no hace falta guardarlos

else:
    # 2. NO EXISTE → calculamos y guardamos
    out_df, modelos = forecast_por_departamento(df_IDC, horizonte=2)

    guardar_pronosticos_en_excel(
        out_df=out_df,
        ruta_excel=ruta_excel,
        sheet_name="Pronostico_IDC",
    )

del hojas, ruta_excel

# ========================
# Forecast por dpto
# ========================
# out_df, modelos = forecast_por_departamento(df_IDC, horizonte=2)



tab_idc, tab_clustering, tab_simulador, tab_info = st.tabs(["IDC", "Clustering", "Simulador", "Información"])


# ========================
# Pestaña Clustering
# ========================
with tab_clustering:
    render_clustering(df_indicadores_normalizados, df_IDC)


# ========================
# Pestaña IDC
# ========================
with tab_idc:
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
    # prom = (
    #     tmp.groupby(["Año", "Tipo"], as_index=False)["IDC"]
    #     .mean()
    #     .rename(columns={"IDC": "IDC_prom"})
    # )

    # ========================
    # Serie nacional (histórica) y forecast ETS con IC95
    # ========================

    # Usamos sólo datos históricos para construir la serie nacional
    tmp_hist_nac = out_df[out_df["Tipo"] == "hist"].copy()

    serie_nac = (
        tmp_hist_nac
        .groupby("Año", as_index=False)["IDC"]
        .mean()
        .sort_values("Año")
    )

    # Serie anual con índice de fin de año (igual que en forecasting.py)
    idx_nac = pd.to_datetime(serie_nac["Año"].astype(int).astype(str) + "-12-31")
    s_nac = pd.Series(serie_nac["IDC"].values, index=idx_nac).asfreq("YE-DEC")

    # Horizonte igual al de los deptos (2 años)
    fh_nac = 2
    yhat_nac, lo_nac, hi_nac, info_nac = forecast_one_series(s_nac, fh=fh_nac)

    # Construimos DataFrame con hist + forecast nacional
    reg_prom = []

    # histórico
    for t, val in s_nac.items():
        reg_prom.append(
            {
                "Año": t.year,
                "Tipo": "hist",
                "IDC_prom": float(val) if pd.notna(val) else np.nan,
                "Lo95_prom": np.nan,
                "Hi95_prom": np.nan,
            }
        )

    # forecast
    if len(yhat_nac) > 0:
        last_year_nac = s_nac.index[-1].year
        future_years_nac = [last_year_nac + i for i in range(1, len(yhat_nac) + 1)]
        for i, ypred in enumerate(yhat_nac):
            reg_prom.append(
                {
                    "Año": future_years_nac[i],
                    "Tipo": "forecast",
                    "IDC_prom": float(ypred),
                    "Lo95_prom": float(lo_nac[i]) if pd.notna(lo_nac[i]) else np.nan,
                    "Hi95_prom": float(hi_nac[i]) if pd.notna(hi_nac[i]) else np.nan,
                }
            )

    prom = pd.DataFrame(reg_prom)

    departamentos_pares = st.session_state.get('departamentos_pares', [])
    departamentos_pares2 = departamentos_pares.copy()

    departamentos_pares = [
    d for d in departamentos_pares 
    if d.strip().upper() != "RISARALDA"
    ]

    # Lista de interés (puede venir de otra parte; si no, dejamos una por defecto)
    # try:
    #     departamentos_pares  # noqa: F821  # type: ignore[name-defined]
    # except NameError:
    #     departamentos_pares = ["Quindío", "Caldas", "Risaralda"]


    # ========================
    # Figuras
    # ========================
    def _hex_to_rgb_tuple(hexcolor: str) -> tuple[int, int, int]:
        """'#RRGGBB' -> (r,g,b). Si falla, usa gris."""
        if isinstance(hexcolor, str) and hexcolor.startswith("#") and len(hexcolor) == 7:
            return tuple(int(hexcolor[i : i + 2], 16) for i in (1, 3, 5))
        return (80, 80, 80)


    def _hex_to_rgb_tuple(hexcolor: str) -> Tuple[int, int, int]:
        if isinstance(hexcolor, str) and hexcolor.startswith("#") and len(hexcolor) == 7:
            return tuple(int(hexcolor[i:i+2], 16) for i in (1, 3, 5))
        return (80, 80, 80)

    def _idc_en_anio(df_depto: pd.DataFrame, anio: int) -> float:
        """Devuelve IDC del año pedido; prioriza hist si existe, si no usa forecast. NaN si no hay."""
        fila = df_depto[df_depto["Año"].eq(anio)]
        if fila.empty:
            return np.nan
        # si en ese año hay ambos tipos, prioriza hist
        if (fila["Tipo"] == "hist").any():
            return float(fila.loc[fila["Tipo"] == "hist", "IDC"].iloc[0])
        return float(fila["IDC"].iloc[0])

    def build_idc_figure(tmp: pd.DataFrame, prom: pd.DataFrame, dptos_sel: List[str]) -> go.Figure:
        """
        Construye la figura IDC mostrando sólo `dptos_sel` y el promedio nacional.
        - Histórico: línea continua.
        - Forecast: sólo línea punteada.
        - Conector punteado entre último histórico y primer forecast.
        - Bandas IC95 en forecast extendidas desde el último histórico.
        - Hover unificado ordenado descendentemente por IDC para cada año,
        con el color de cada serie.
        """
        fig = go.Figure()

        # Paleta de alto contraste
        palette = px.colors.qualitative.D3 + px.colors.qualitative.Bold + px.colors.qualitative.Dark24

        # Mapa nombre de serie -> color
        color_map: dict[str, str] = {}

        # ============================
        # 1. Departamentos filtrados
        # ============================
        deptos_validos = [d for d in dptos_sel if d in tmp["Departamento"].unique()]

        for i, dept in enumerate(deptos_validos):
            color = palette[i % len(palette)]
            color_map[dept] = color

            r, g, b = _hex_to_rgb_tuple(color)

            sub = tmp.loc[tmp["Departamento"] == dept].sort_values("Año")
            sub_hist = sub[sub["Tipo"] == "hist"]
            sub_fc = sub[sub["Tipo"] == "forecast"]

            # Valores para conectar hist -> forecast
            last_hist_year = None
            last_hist_idc = None
            first_fc_year = None
            first_fc_idc = None

            if not sub_hist.empty:
                last_row_hist = sub_hist.sort_values("Año").iloc[-1]
                last_hist_year = int(last_row_hist["Año"])
                last_hist_idc = float(last_row_hist["IDC"])

            if not sub_fc.empty:
                first_row_fc = sub_fc.sort_values("Año").iloc[0]
                first_fc_year = int(first_row_fc["Año"])
                first_fc_idc = float(first_row_fc["IDC"])

            # --- Bandas IC95 (forecast) ---
            fc = sub_fc
            if (
                not fc.empty
                and fc["Lo95"].notna().all()
                and fc["Hi95"].notna().all()
            ):
                # Extendemos la banda desde el último histórico, usando IDC_hist como "punto" inicial
                x_hi = list(fc["Año"])
                y_hi = list(fc["Hi95"])
                x_lo = list(fc["Año"])
                y_lo = list(fc["Lo95"])

                if last_hist_year is not None and last_hist_idc is not None:
                    x_poly = [last_hist_year] + x_hi + x_lo[::-1] + [last_hist_year]
                    y_poly = [last_hist_idc] + y_hi + y_lo[::-1] + [last_hist_idc]
                else:
                    x_poly = x_hi + x_lo[::-1]
                    y_poly = y_hi + y_lo[::-1]

                fig.add_trace(
                    go.Scatter(
                        x=x_poly,
                        y=y_poly,
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},0.12)",
                        line=dict(width=0),
                        legendgroup=dept,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # Bordes punteados sólo sobre los años de forecast
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

            # --- Línea continua SOLO histórico ---
            if not sub_hist.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sub_hist["Año"],
                        y=sub_hist["IDC"],
                        mode="lines+markers",
                        name=dept,
                        legendgroup=dept,
                        marker=dict(symbol="circle", size=7),
                        line=dict(color=color, width=2.5),
                        hoverinfo="skip",  # hover centralizado
                        showlegend=True,
                    )
                )

            # --- Línea punteada SOLO forecast ---
            if not sub_fc.empty:
                serie_fc_name = f"{dept} (forecast)"
                color_map[serie_fc_name] = color

                # forecast propiamente dicho
                fig.add_trace(
                    go.Scatter(
                        x=sub_fc["Año"],
                        y=sub_fc["IDC"],
                        mode="lines+markers",
                        name=serie_fc_name,
                        legendgroup=dept,
                        marker=dict(symbol="circle-open", size=8),
                        line=dict(color=color, width=2.5, dash="dash"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # --- Conector punteado entre último histórico y primer forecast ---
                if (last_hist_year is not None) and (first_fc_year is not None):
                    fig.add_trace(
                        go.Scatter(
                            x=[last_hist_year, first_fc_year],
                            y=[last_hist_idc, first_fc_idc],
                            mode="lines",
                            line=dict(color=color, width=2.5, dash="dash"),
                            legendgroup=dept,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

        # ===========================================
        # 2. Bandas IC95 del promedio nacional (forecast)
        # ===========================================
        fc_prom = (
            prom[
                (prom["Tipo"] == "forecast")
                & prom["Lo95_prom"].notna()
                & prom["Hi95_prom"].notna()
            ]
            .sort_values("Año")
        )

        prom_hist = prom[
            (prom["Tipo"] == "hist") & prom["IDC_prom"].notna()
        ].sort_values("Año")

        last_hist_year_prom = None
        last_hist_idc_prom = None
        if not prom_hist.empty:
            last_row_ph = prom_hist.iloc[-1]
            last_hist_year_prom = int(last_row_ph["Año"])
            last_hist_idc_prom = float(last_row_ph["IDC_prom"])

        if not fc_prom.empty:
            # Polígono relleno extendido desde el último histórico
            x_hi_p = list(fc_prom["Año"])
            y_hi_p = list(fc_prom["Hi95_prom"])
            x_lo_p = list(fc_prom["Año"])
            y_lo_p = list(fc_prom["Lo95_prom"])

            if last_hist_year_prom is not None and last_hist_idc_prom is not None:
                x_poly_p = [last_hist_year_prom] + x_hi_p + x_lo_p[::-1] + [last_hist_year_prom]
                y_poly_p = [last_hist_idc_prom] + y_hi_p + y_lo_p[::-1] + [last_hist_idc_prom]
            else:
                x_poly_p = x_hi_p + x_lo_p[::-1]
                y_poly_p = y_hi_p + y_lo_p[::-1]

            fig.add_trace(
                go.Scatter(
                    x=x_poly_p,
                    y=y_poly_p,
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.10)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Bordes punteados sólo sobre forecast
            fig.add_trace(
                go.Scatter(
                    x=fc_prom["Año"],
                    y=fc_prom["Hi95_prom"],
                    mode="lines",
                    line=dict(color="black", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fc_prom["Año"],
                    y=fc_prom["Lo95_prom"],
                    mode="lines",
                    line=dict(color="black", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # ===============================
        # 3. Promedio nacional (línea)
        # ===============================
        if not prom_hist.empty:
            color_map["Promedio IDC"] = "black"
            fig.add_trace(
                go.Scatter(
                    x=prom_hist["Año"],
                    y=prom_hist["IDC_prom"],
                    mode="lines+markers",
                    name="Promedio IDC",
                    legendgroup="Promedio",
                    marker=dict(symbol="circle", size=8),
                    line=dict(color="black", width=3.2),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        prom_fc = prom[
            (prom["Tipo"] == "forecast") & prom["IDC_prom"].notna()
        ].sort_values("Año")

        if not prom_fc.empty:
            color_map["Promedio IDC (forecast)"] = "black"
            # forecast propiamente dicho
            fig.add_trace(
                go.Scatter(
                    x=prom_fc["Año"],
                    y=prom_fc["IDC_prom"],
                    mode="lines+markers",
                    name="Promedio IDC (forecast)",
                    legendgroup="Promedio",
                    marker=dict(symbol="circle-open", size=9),
                    line=dict(color="black", width=3.2, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # conector punteado hist -> forecast
            if (last_hist_year_prom is not None) and (not prom_fc.empty):
                first_row_pfc = prom_fc.iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=[last_hist_year_prom, int(first_row_pfc["Año"])],
                        y=[last_hist_idc_prom, float(first_row_pfc["IDC_prom"])],
                        mode="lines",
                        line=dict(color="black", width=3.2, dash="dash"),
                        legendgroup="Promedio",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # ===============================
        # 4. Layout básico
        # ===============================
        fig.update_layout(
            title="",
            xaxis_title="Año",
            yaxis_title="IDC",
            legend_title="Series",
            hovermode="x unified",
            hoverdistance=50,
            hoverlabel=dict(namelength=-1),
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=10),
            height=670,
        )
        fig.update_xaxes(type="linear", tickmode="linear", dtick=1)

        # ================================================
        # 5. Hover ordenado por IDC (desc) CON COLORES
        # ================================================

        base_tmp = tmp[tmp["Departamento"].isin(deptos_validos)][
            ["Año", "Departamento", "IDC", "Tipo"]
        ].copy()

        base_prom = prom[["Año", "IDC_prom", "Tipo"]].copy()
        base_prom.rename(columns={"IDC_prom": "IDC"}, inplace=True)
        base_prom["Departamento"] = "Promedio IDC"

        base_all = pd.concat([base_tmp, base_prom], ignore_index=True)

        def nombre_serie(row) -> str:
            dpto = row["Departamento"]
            tipo = row["Tipo"]
            if dpto == "Promedio IDC":
                return "Promedio IDC (forecast)" if tipo == "forecast" else "Promedio IDC"
            else:
                return f"{dpto} (forecast)" if tipo == "forecast" else dpto

        base_all["Serie"] = base_all.apply(nombre_serie, axis=1)

        base_all_sorted = base_all.sort_values(["Año", "IDC"], ascending=[True, False])

        def build_block(grp: pd.DataFrame) -> str:
            filas = []
            for _, r in grp.iterrows():
                serie = r["Serie"]
                val = r["IDC"]
                color = color_map.get(serie, "black")
                filas.append(
                    f'<span style="color:{color};font-weight:bold;">● {serie}</span>: {val:.3f}'
                )
            return "<br>".join(filas)

        hover_text_dict = base_all_sorted.groupby("Año").apply(build_block).to_dict()

        anos_unicos = sorted(base_all["Año"].unique())
        custom_hover = [hover_text_dict.get(a, "") for a in anos_unicos]

        # Traza "fantasma" que centraliza el hover
        fig.add_trace(
            go.Scatter(
                x=anos_unicos,
                y=[0] * len(anos_unicos),          # valor numérico para que haya hover
                mode="markers",
                marker=dict(size=8, opacity=0),    # marcador invisible
                line=dict(width=0),
                showlegend=False,
                name="__hover_central__",
                customdata=custom_hover,
                hovertemplate="%{customdata}<extra></extra>",
                hoverlabel=dict(namelength=-1),
            )
)

        return fig

    
def round_down_to_half(x: float) -> float:
    """Redondea hacia abajo al múltiplo de 0.5 más cercano."""
    n = math.floor(x)
    frac = x - n
    if frac < 0.5:
        return n
    else:
        return n + 0.5

def round_up_to_half(x: float) -> float:
    """Redondea hacia arriba al múltiplo de 0.5 más cercano."""
    n = math.floor(x)
    frac = x - n
    if frac == 0:
        return n
    elif frac < 0.5:
        return n + 0.5
    else:
        return n + 1.0

with left:
    st.subheader("Evolución y pronóstico del IDC")
    
    vista_global = st.toggle(
        "Usar escala global (0–10)",
        value=True,
        help=(
            "Activa para ver el IDC en la escala completa 0–10. "
            "Desactiva para hacer zoom automático según los valores mostrados, "
            "incluyendo los intervalos de confianza del último año pronosticado."
        )
    )
    
    fig_plot = build_idc_figure(tmp, prom, departamentos_pares2)

    if vista_global:
        # Escala fija 0–10
        fig_plot.update_yaxes(range=[0, 10])
    else:
        # Zoom: calculamos el rango a partir de lo que realmente se está mostrando
        base_tmp = tmp[tmp["Departamento"].isin(departamentos_pares2)]

        vals_minmax = []

        # IDC de departamentos pares
        if not base_tmp.empty:
            vals_minmax.append(base_tmp["IDC"].min())
            vals_minmax.append(base_tmp["IDC"].max())

        # IDC del promedio
        if not prom.empty:
            vals_minmax.append(prom["IDC_prom"].min())
            vals_minmax.append(prom["IDC_prom"].max())

        # ---------- límite superior basado en IC del último año forecast ----------
        y_max_candidates = []

        if not prom.empty:
            fc_years = prom.loc[prom["Tipo"] == "forecast", "Año"]
            if not fc_years.empty:
                last_fc_year = fc_years.max()

                # Hi95 de departamentos pares en ese año
                tmp_fc_last = base_tmp[
                    (base_tmp["Tipo"] == "forecast") &
                    (base_tmp["Año"] == last_fc_year)
                ]
                if "Hi95" in tmp_fc_last.columns:
                    y_max_candidates.extend(tmp_fc_last["Hi95"].dropna().tolist())

                # Hi95 del promedio en ese año
                prom_fc_last = prom[
                    (prom["Tipo"] == "forecast") &
                    (prom["Año"] == last_fc_year)
                ]
                if "Hi95_prom" in prom_fc_last.columns:
                    y_max_candidates.extend(prom_fc_last["Hi95_prom"].dropna().tolist())

        # ---------- construir y_min / y_max crudos ----------
        if vals_minmax:
            raw_min = min(vals_minmax)
        else:
            raw_min = 0.0

        if y_max_candidates:
            raw_max = max(y_max_candidates)
        elif vals_minmax:
            raw_max = max(vals_minmax)
        else:
            raw_max = 10.0

        # ---------- redondeo a múltiplos de 0.5 ----------
        y_min = round_down_to_half(raw_min)
        y_max = round_up_to_half(raw_max)

        # opcional: pequeño margen visual
        margen = 0.0  # si quieres puedes poner 0.1
        fig_plot.update_yaxes(range=[y_min - margen, y_max + margen])

    st.plotly_chart(fig_plot, width="stretch")

        

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

        # --------- E: Lista dinámica ----------
        st.markdown("### Departamentos pares de Risaralda")
        if departamentos_pares:
            st.markdown("".join(f"{d}, " for d in departamentos_pares))
        else:
            st.markdown("_(sin departamentos seleccionados)_")




# ========================
# Pestaña Simulacion
# ========================
with tab_simulador:
    st.subheader("Simulador de indicadores importantes")
    st.write("Ajusta los valores de las variables independientes y presiona **Recalcular resultados** para ver el nuevo valor.")

    # from src.scenary_simulator import calcular_variable

    valores_iniciales = {
        "INS-2-1": {"Ingresos_tributarios": 200, "Ingresos_no_tributarios": 50, "Transferencias": 30, "Ingresos_totales": 400},
        "NEG-2-2": {"Sociedades_empresariales": 1000, "Poblacion": 500000},
        "TIC-1-3": {"Hogares_con_computador": 60000, "Total_hogares": 100000},
        "TIC-1-1": {"Accesos_fijos_internet": 80000, "Poblacion": 500000},
        "EDU-1-3": {"Estudiantes_matriculados": 30000, "Poblacion_en_edad": 40000},
        "INN-2-4": {"Registro_marca_t2": 10, "Registro_marca_t1": 12, "Registro_marca_t": 15, "Poblacion": 500000},
        "SAL-3-3": {"Especializacion": 200, "Poblacion": 500000},
        "SAL-1-3": {"Nacidos_con_controles": 4800, "Total_nacimientos": 5000},
        "TIC-1-4": {"Poblacion_usa_internet": 450000, "Total_poblacion": 500000},
        "EDU-2-1": {"P_Escritura": 60, "P_Lectura": 65, "P_Razonamiento": 70},
    }

    for _, row in tabla_final.iterrows():
        codigo = row["Indicador"]
        nombre = row["Nombre"]
        valores_vars = valores_iniciales.get(codigo, {})

        with st.expander(f"{nombre} ({codigo})", expanded=False):
            st.caption("Introduce nuevos valores y luego presiona el botón para actualizar el cálculo.")

            # Inputs con session_state
            for var, val in valores_vars.items():
                st.number_input(
                    f"{var}",
                    value=float(val),
                    key=f"{codigo}_{var}",
                )

            # Botón de recalcular
            if st.button(f"Recalcular {codigo}"):
                valores = {var: st.session_state[f"{codigo}_{var}"] for var in valores_vars}
                resultado = calcular_variable(codigo, valores)
                if resultado is not None:
                    st.success(f"**Nuevo valor calculado:** {resultado:.4f}")


    # for _, row in tabla_final.iterrows():
    #     codigo = row["Indicador"]
    #     nombre = row["Nombre"]

    #     with st.expander(f"{nombre} ({codigo})", expanded=False):
    #         st.caption("Ajusta los valores para observar el resultado recalculado.")
    #         valores = {}
    #         for var, val in valores_iniciales.get(codigo, {}).items():
    #             valores[var] = st.number_input(f"{var}", value=float(val), key=f"{codigo}_{var}")

    #         if valores:
    #             resultado = calcular_variable(codigo, valores)
    #             if resultado is not None:
    #                 st.success(f"**Nuevo valor calculado:** {resultado:.4f}")

# ========================
# Pestaña Información
# ========================
with tab_info:
    st.markdown('''
    Este prototipo funcional de tablero (dashboard) en nivel TRL3 fue desarrollado por integrantes del Semillero de Investigación en Análisis de Datos (SAND), adscrito al Grupo de Investigación en Análisis de Datos y Sociología Computacional (GIADSc) en el marco del reto 'Asistente digital para el análisis y simulación de indicadores del IDC', de la convocatoria interna 'Convocatoria para financiar propuestas de solución a retos empresariales' de la Vicerrectoría de Investigaciones, Innovación y Extensión de la Universidad Tecnológica de Pereira en alianza con la Comisión Regional de Competitividad de Risaralda (CRC) en el semestre 2025-2. 
    ''')
    st.subheader("Equipo de trabajo:")
    st.markdown('''
    - **Cristian Camilo Galeano Largo (c.galeano@utp.edu.co):**\n
    Estudiante de último semestre de Ingeniería Física.\n
    *Perfil de LinkedIn:* \n
    - **Kevin Ossa Varela (kevin.ossa@utp.edu.co):**\n
    Estudiante de último semestre de Ingeniería de Sistemas.\n
    *Perfil de LinkedIn:* \n
    - **Ing. Sebastián Blandón Londoño (s.blandon@utp.edu.co):**\n
    Profesor de la Facultad de Ciencias Empresariales.\n
    *Perfil de LinkedIn:* https://www.linkedin.com/in/sebastian-blandon/
    - **P.hD. Julián David Echeverry Correa (jde@utp.edu.co):**\n
    Profesor de la Facultad de Ingenierías.\n
    *Perfil de LinkedIn:* https://www.linkedin.com/in/juliandecheverry/
    ''')
    st.subheader("**Semillero de Investigación en Análisis de Datos (SAND) | Grupo de Investigación en Análisis de Datos y Sociología Computacional (GIADSc) [2025]**")