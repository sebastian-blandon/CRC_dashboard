"""
IDC Dashboard (Streamlit)
- Gráfico con series por departamento (hist+forecast en línea continua)
- Promedio nacional continuo + capa punteada en años de forecast
- Bandas IC95 opacas con borde punteado
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import forecast_por_departamento, guardar_pronosticos_en_excel, forecast_one_series
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
df_factores = hojas["Valor_normalizadoAgrupado"]
df_pilares = hojas["Valor_normalizadoAgrupado_Pil"]


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



tab_idc, tab_factores, tab_clustering, tab_calculadora, tab_info = st.tabs(["IDC", "Factores", "Clustering", "Calculadora", "Información"])


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
        value=False,
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
# Pestaña Factores
# ========================
NOMBRES_FACTORES = {
    "Con_Habi": "Condiciones habilitantes",
    "Cap_Hum": "Capital humano",
    "Efi_Mer": "Eficiencia de los mercados",
    "Eco_Inno": "Ecosistema innovador",
}


def build_factores_figure(
    df_factores: pd.DataFrame,
    dpto_sel: str,
    factores: Optional[List[str]] = None
) -> go.Figure:
    """
    Gráfico de factores para un departamento:
    - Histórico: línea continua.
    - Forecast (2 años): línea punteada + banda IC95.
    - Hover unificado por año (x unified) con nombres extendidos.
    """
    if factores is None:
        factores = ["Con_Habi", "Cap_Hum", "Efi_Mer", "Eco_Inno"]

    # Filtrar departamento
    df_dep = df_factores[df_factores["Departamento"] == dpto_sel].copy()
    if df_dep.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Sin datos de factores para {dpto_sel}",
            template="plotly_white",
        )
        return fig

    df_dep["Año"] = df_dep["Año IDC"].astype(int)

    # Paleta y figura
    palette = px.colors.qualitative.D3 + px.colors.qualitative.Bold + px.colors.qualitative.Dark24
    fig = go.Figure()
    color_map: dict[str, str] = {}

    # Para construir el hover unificado
    registros_hover: list[dict] = []

    for i, factor_key in enumerate(factores):
        # Serie histórica del factor
        serie_hist = (
            df_dep[["Año", factor_key]]
            .dropna()
            .sort_values("Año")
        )
        if serie_hist.empty:
            continue

        # Nombre “bonito” del factor
        factor_name = NOMBRES_FACTORES.get(factor_key, factor_key)

        # Color
        color = palette[i % len(palette)]
        color_map[factor_name] = color
        color_map[f"{factor_name} (forecast)"] = color

        r, g, b = _hex_to_rgb_tuple(color)

        # =============================
        # 1. Serie temporal + forecast
        # =============================
        idx = pd.to_datetime(serie_hist["Año"].astype(int).astype(str) + "-12-31")
        s = pd.Series(serie_hist[factor_key].values, index=idx).asfreq("YE-DEC")

        fh = 2  # horizonte 2 años (p.ej. 2025, 2026)
        yhat, lo, hi, info = forecast_one_series(s, fh=fh)

        # Histórico
        registros = []
        for t, val in s.items():
            registros.append(
                {
                    "Año": t.year,
                    "Tipo": "hist",
                    "Valor": float(val) if pd.notna(val) else np.nan,
                    "Lo95": np.nan,
                    "Hi95": np.nan,
                    "FactorKey": factor_key,
                }
            )

        # Forecast
        if len(yhat) > 0:
            last_year = s.index[-1].year
            future_years = [last_year + j for j in range(1, len(yhat) + 1)]
            for j, ypred in enumerate(yhat):
                registros.append(
                    {
                        "Año": future_years[j],
                        "Tipo": "forecast",
                        "Valor": float(ypred),
                        "Lo95": float(lo[j]) if pd.notna(lo[j]) else np.nan,
                        "Hi95": float(hi[j]) if pd.notna(hi[j]) else np.nan,
                        "FactorKey": factor_key,
                    }
                )

        df_plot = pd.DataFrame(registros).sort_values("Año")

        # Acumular para hover global
        registros_hover.extend(df_plot.to_dict(orient="records"))

        sub_hist = df_plot[df_plot["Tipo"] == "hist"]
        sub_fc = df_plot[df_plot["Tipo"] == "forecast"]

        # Último histórico y primer forecast (para el conector)
        last_hist_year = None
        last_hist_val = None
        first_fc_year = None
        first_fc_val = None

        if not sub_hist.empty:
            last_row_hist = sub_hist.iloc[-1]
            last_hist_year = int(last_row_hist["Año"])
            last_hist_val = float(last_row_hist["Valor"])

        if not sub_fc.empty:
            first_row_fc = sub_fc.iloc[0]
            first_fc_year = int(first_row_fc["Año"])
            first_fc_val = float(first_row_fc["Valor"])

        # =============================
        # 2. Bandas IC95 del forecast
        # =============================
        fc = sub_fc.dropna(subset=["Lo95", "Hi95"])
        if not fc.empty:
            x_hi = list(fc["Año"])
            y_hi = list(fc["Hi95"])
            x_lo = list(fc["Año"])
            y_lo = list(fc["Lo95"])

            if (last_hist_year is not None) and (last_hist_val is not None):
                x_poly = [last_hist_year] + x_hi + x_lo[::-1] + [last_hist_year]
                y_poly = [last_hist_val] + y_hi + y_lo[::-1] + [last_hist_val]
            else:
                x_poly = x_hi + x_lo[::-1]
                y_poly = y_hi + y_lo[::-1]

            # Relleno
            fig.add_trace(
                go.Scatter(
                    x=x_poly,
                    y=y_poly,
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.12)",
                    line=dict(width=0),
                    legendgroup=factor_name,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Bordes punteados
            fig.add_trace(
                go.Scatter(
                    x=fc["Año"],
                    y=fc["Hi95"],
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.55)", width=1, dash="dot"),
                    legendgroup=factor_name,
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
                    legendgroup=factor_name,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # =============================
        # 3. Líneas hist / forecast
        # =============================
        # Histórico: línea continua
        if not sub_hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub_hist["Año"],
                    y=sub_hist["Valor"],
                    mode="lines+markers",
                    name=factor_name,
                    legendgroup=factor_name,
                    marker=dict(symbol="circle", size=7),
                    line=dict(color=color, width=2.5),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        # Forecast: línea punteada
        if not sub_fc.empty:
            serie_fc_name = f"{factor_name} (forecast)"
            fig.add_trace(
                go.Scatter(
                    x=sub_fc["Año"],
                    y=sub_fc["Valor"],
                    mode="lines+markers",
                    name=serie_fc_name,
                    legendgroup=factor_name,
                    marker=dict(symbol="circle-open", size=8),
                    line=dict(color=color, width=2.5, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Conector hist -> forecast
            if (last_hist_year is not None) and (first_fc_year is not None):
                fig.add_trace(
                    go.Scatter(
                        x=[last_hist_year, first_fc_year],
                        y=[last_hist_val, first_fc_val],
                        mode="lines",
                        line=dict(color=color, width=2.5, dash="dash"),
                        legendgroup=factor_name,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # =============================
    # 4. Hover unificado por año
    # =============================
    if registros_hover:
        base_all = pd.DataFrame(registros_hover)

        def nombre_serie_row(row) -> str:
            fname = NOMBRES_FACTORES.get(row["FactorKey"], row["FactorKey"])
            return f"{fname} (forecast)" if row["Tipo"] == "forecast" else fname

        base_all["Serie"] = base_all.apply(nombre_serie_row, axis=1)
        base_all_sorted = base_all.sort_values(["Año", "Valor"], ascending=[True, False])

        def build_block(grp: pd.DataFrame) -> str:
            filas = []
            for _, r in grp.iterrows():
                serie = r["Serie"]
                val = r["Valor"]
                color = color_map.get(serie, "black")
                filas.append(
                    f'<span style="color:{color};font-weight:bold;">● {serie}</span>: {val:.3f}'
                )
            return "<br>".join(filas)

        hover_text_dict = base_all_sorted.groupby("Año").apply(build_block).to_dict()
        anos_unicos = sorted(base_all["Año"].unique())
        custom_hover = [hover_text_dict.get(a, "") for a in anos_unicos]

        fig.add_trace(
            go.Scatter(
                x=anos_unicos,
                y=[0] * len(anos_unicos),
                mode="markers",
                marker=dict(size=8, opacity=0),
                line=dict(width=0),
                showlegend=False,
                name="__hover_factores__",
                customdata=custom_hover,
                hovertemplate="%{customdata}<extra></extra>",
                hoverlabel=dict(namelength=-1),
            )
        )

    # =============================
    # 5. Layout
    # =============================
    fig.update_layout(
        title=f"Evolución y pronóstico de factores — {dpto_sel}",
        xaxis_title="Año",
        yaxis_title="Valor",
        legend_title="Factores",
        hovermode="x unified",
        hoverdistance=50,
        hoverlabel=dict(namelength=-1),
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=670,
    )
    fig.update_xaxes(type="linear", tickmode="linear", dtick=1)

    return fig


with tab_factores:
    st.subheader("Evolución de factores del IDC")

    # Selector de Departamento
    dptos_factores = sorted(df_factores["Departamento"].dropna().unique())
    if "Risaralda" in dptos_factores:
        idx_default = dptos_factores.index("Risaralda")
    else:
        idx_default = 0

    dpto_fact_sel = st.selectbox(
        "Selecciona el departamento:",
        dptos_factores,
        index=idx_default,
        key="dpto_factores_sel",
    )

    # Selector de factores con nombres extendidos
    factores_disponibles = ["Con_Habi", "Cap_Hum", "Efi_Mer", "Eco_Inno"]

    factores_sel = st.multiselect(
        "Selecciona los factores a mostrar:",
        options=factores_disponibles,
        format_func=lambda x: NOMBRES_FACTORES.get(x, x),
        default=factores_disponibles,
        key="factores_sel",
    )

    if not factores_sel:
        st.info("Selecciona al menos un factor para visualizar el gráfico.")
    else:

        # ---------------------------
        # Switch de escala
        # ---------------------------
        vista_global_f = st.toggle(
            "Usar escala global (0–10)",
            value=False,
            help=(
                "Activa para ver los factores en la escala completa 0–10. "
                "Desactiva para aplicar zoom automático según los valores mostrados."
            )
        )

        # ---------------------------
        # Construir gráfica
        # ---------------------------
        fig_factores = build_factores_figure(df_factores, dpto_fact_sel, factores_sel)

        # ---------------------------
        # Ajuste de escala
        # ---------------------------
        if vista_global_f:
            # Escala fija 0–10
            fig_factores.update_yaxes(range=[0, 10])

        else:
            # Zoom basado en los datos seleccionados
            df_dep = df_factores[df_factores["Departamento"] == dpto_fact_sel].copy()
            df_dep["Año"] = df_dep["Año IDC"].astype(int)

            df_long = df_dep.melt(
                id_vars=["Año", "Departamento"],
                value_vars=factores_sel,
                var_name="Factor",
                value_name="Valor",
            ).dropna(subset=["Valor"])

            if not df_long.empty:
                raw_min = float(df_long["Valor"].min())
                raw_max = float(df_long["Valor"].max())
            else:
                raw_min, raw_max = 0.0, 10.0

            y_min = round_down_to_half(raw_min)
            y_max = round_up_to_half(raw_max)

            if y_min == y_max:
                y_min = max(0.0, y_min - 0.5)
                y_max = y_max + 0.5

            fig_factores.update_yaxes(range=[y_min, y_max])

        # Render final
        st.plotly_chart(fig_factores, width="stretch")


# ========================
# Pestaña Simulacion
# ========================
FACTORES_PILARES = OrderedDict({
    "Condiciones habilitantes": [
        ("P1", "1. Instituciones"),
        ("P2", "2. Infraestructura"),
        ("P3", "3. Adopción TIC"),
        ("P4", "4. Sostenibilidad ambiental"),
    ],
    "Capital humano": [
        ("P5", "5. Salud"),
        ("P6", "6. Educación básica y media"),
        ("P7", "7. Educación superior y formación para el trabajo"),
    ],
    "Eficiencia de los mercados": [
        ("P8", "8. Entorno para los negocios"),
        ("P9", "9. Mercado laboral"),
        ("P10", "10. Sistema financiero"),
        ("P11", "11. Tamaño del mercado"),
    ],
    "Ecosistema innovador": [
        ("P12", "12. Sofisticación y diversificación"),
        ("P13", "13. Innovación"),
    ],
})


MAPA_PILAR_COL = {
    "1. Instituciones": "INS",
    "2. Infraestructura": "INF",
    "3. Adopción TIC": "TIC",
    "4. Sostenibilidad ambiental": "AMB",   # uso el texto de FACTORES_PILARES
    "5. Salud": "SAL",
    "6. Educación básica y media": "EDU",
    "7. Educación superior y formación para el trabajo": "EDS",
    "8. Entorno para los negocios": "NEG",
    "9. Mercado laboral": "LAB",
    "10. Sistema financiero": "FIN",
    "11. Tamaño del mercado": "TAM",
    "12. Sofisticación y diversificación": "SOF",
    "13. Innovación": "INN",
}

PILAR_COL_MAP = {
    "P1": "INS",
    "P2": "INF",
    "P3": "TIC",
    "P4": "AMB",
    "P5": "SAL",
    "P6": "EDU",
    "P7": "EDS",
    "P8": "NEG",
    "P9": "LAB",
    "P10": "FIN",
    "P11": "TAM",
    "P12": "SOF",
    "P13": "INN",
}




def chip_promedio(label: str, value: float | None) -> None:
    """
    Renderiza un 'chip' tipo botón con el texto Promedio y el valor.
    Si value es None, muestra '--'.
    """
    if value is None:
        txt_val = "--"
    else:
        txt_val = f"{value:.2f}"

    st.markdown(
        f"""
        <div style="
            text-align:center;
            background-color:#008b5a;
            color:white;
            padding:0.25rem 0.75rem;
            border-radius:999px;
            font-weight:bold;
            border:1px solid #006f46;
            font-size:0.9rem;
        ">
            {label}<br>{txt_val}
        </div>
        """,
        unsafe_allow_html=True,
    )

ANIO_COL_PILARES = "Año IDC"

with tab_calculadora:
    st.subheader("Calculadora rápida del IDC (0–10)")

    # ---------------------------------
    # Selección de departamento origen
    # ---------------------------------
    dptos_calc = sorted(df_pilares["Departamento"].dropna().unique())
    if "Risaralda" in dptos_calc:
        idx_default_calc = dptos_calc.index("Risaralda")
    else:
        idx_default_calc = 0

    depto_calc = st.selectbox(
        "Selecciona el departamento para precargar los valores:",
        dptos_calc,
        index=idx_default_calc,
        key="calc_depto_sel",
    )

    # ----------------------------------
    # 2) Precarga de valores por cambio de departamento
    # ----------------------------------
    prev_depto = st.session_state.get("calc_depto_prev", None)

    if prev_depto != depto_calc:
        st.session_state["calc_depto_prev"] = depto_calc

        df_dep_pilares = df_pilares[df_pilares["Departamento"] == depto_calc].copy()

        # Reset de promedios de factores
        for i, _ in enumerate(FACTORES_PILARES.items(), start=1):
            st.session_state[f"prom_factor_{i}"] = None

        # Si hay datos para el departamento, tomamos el último año
        fila_ult = None
        if not df_dep_pilares.empty and ANIO_COL_PILARES in df_dep_pilares.columns:
            anio_max = df_dep_pilares[ANIO_COL_PILARES].max()
            fila_ult = (
                df_dep_pilares[df_dep_pilares[ANIO_COL_PILARES] == anio_max]
                .sort_values(ANIO_COL_PILARES)
                .iloc[0]
            )

        # Precargar pilares (valor base) y limpiar porcentaje
        for _, pilares in FACTORES_PILARES.items():
            for codigo, _nombre_pilar in pilares:
                key_pilar = f"pilar_{codigo}"
                key_pct = f"pilar_pct_{codigo}"

                valor_texto = ""
                if fila_ult is not None:
                    col_name = PILAR_COL_MAP.get(codigo)
                    if col_name and col_name in fila_ult.index and pd.notna(fila_ult[col_name]):
                        valor_texto = str(float(fila_ult[col_name]))

                st.session_state[key_pilar] = valor_texto
                st.session_state[key_pct] = ""  # 0% por defecto




    col_left, col_right = st.columns([2, 1], gap="large")

    # -----------------------------
    # COLUMNA IZQUIERDA: FACTORES
    # -----------------------------
    with col_left:
        for i, (nombre_factor, pilares) in enumerate(FACTORES_PILARES.items(), start=1):
            key_factor = f"prom_factor_{i}"
            if key_factor not in st.session_state:
                st.session_state[key_factor] = None

            with st.form(f"form_factor_{i}"):

                # Encabezado (título + chip) en dos columnas
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Factor {i}. {nombre_factor}**")

                # --- Inputs de los pilares (ya vienen precargados en session_state) ---
                # for codigo, nombre_pilar in pilares:
                #     st.text_input(
                #         nombre_pilar,
                #         placeholder="Ingresa un valor (0–10)",
                #         key=f"pilar_{codigo}",
                #     )
                for codigo, nombre_pilar in pilares:
                    col_val, col_pct = st.columns([3, 1])
                    with col_val:
                        st.text_input(
                            nombre_pilar,
                            placeholder="Ingresa un valor (0–10)",
                            key=f"pilar_{codigo}",
                        )
                    with col_pct:
                        st.text_input(
                            "Cambio %",
                            placeholder="0",
                            key=f"pilar_pct_{codigo}",
                        )
                # Botón que dispara la validación y el cálculo
                submitted = st.form_submit_button("Calcular promedio del factor")

                # ---- Lógica de validación y cálculo ----
                prom_factor = st.session_state[key_factor]  # valor previo por defecto

                if submitted:
                    valores_factor: list[float] = []
                    errores_factor: list[str] = []

                    for codigo, nombre_pilar in pilares:
                        # txt = str(st.session_state.get(f"pilar_{codigo}", "")).strip()

                        # if txt == "":
                        #     errores_factor.append(f"Falta valor en {nombre_pilar}.")
                        #     continue

                        # try:
                        #     val = float(txt.replace(",", "."))
                        #     if not (0.0 <= val <= 10.0):
                        #         errores_factor.append(
                        #             f"El valor de {nombre_pilar} debe estar entre 0 y 10."
                        #         )
                        #     else:
                        #         valores_factor.append(val)
                        # except ValueError:
                        #     errores_factor.append(
                        #         f"El valor de {nombre_pilar} debe ser numérico."
                        #     )
                        txt_val = str(st.session_state.get(f"pilar_{codigo}", "")).strip()
                        txt_pct = str(st.session_state.get(f"pilar_pct_{codigo}", "")).strip()

                        if txt_val == "":
                            errores_factor.append(f"Falta valor en {nombre_pilar}.")
                            continue

                        try:
                            val_base = float(txt_val.replace(",", "."))
                        except ValueError:
                            errores_factor.append(f"El valor de {nombre_pilar} debe ser numérico.")
                            continue

                        if txt_pct == "":
                            pct = 0.0
                        else:
                            try:
                                pct = float(txt_pct.replace(",", "."))
                            except ValueError:
                                errores_factor.append(
                                    f"El cambio % de {nombre_pilar} debe ser numérico."
                                )
                                continue

                        val_ajustado = val_base * (1.0 + pct / 100.0)

                        if not (0.0 <= val_ajustado <= 10.0):
                            errores_factor.append(
                                f"El valor ajustado de {nombre_pilar} ({val_ajustado:.2f}) debe estar entre 0 y 10."
                            )
                        else:
                            valores_factor.append(val_ajustado)

                    if errores_factor or len(valores_factor) < len(pilares):
                        prom_factor = None
                        st.session_state[key_factor] = None

                        for msg in errores_factor:
                            st.error(msg)
                        st.info(
                            "Debes diligenciar correctamente todos los pilares de este "
                            "factor para poder calcular su promedio."
                        )
                    else:
                        prom_factor = sum(valores_factor) / len(valores_factor)
                        st.session_state[key_factor] = prom_factor
                        st.success(f"Promedio del factor calculado: {prom_factor:.2f}")

                # --- Ahora sí dibujamos el chip con el valor actual ---
                with c2:
                    chip_promedio("Promedio", prom_factor)

            st.markdown("&nbsp;")  # espacio entre factores

    # -----------------------------
    # COLUMNA DERECHA: IDC calculado
    # -----------------------------
    with col_right:
        st.markdown("### ")

        promedios_factores: list[float | None] = []
        for i, _ in enumerate(FACTORES_PILARES.keys(), start=1):
            key_factor = f"prom_factor_{i}"
            promedios_factores.append(st.session_state.get(key_factor, None))

        # IDC sólo cuando TODOS los factores tienen promedio
        if all(p is not None for p in promedios_factores):
            idc_calc = sum(promedios_factores) / len(promedios_factores)

            st.markdown(
                """
                <div style="font-size:2.5rem; font-weight:bold; margin-top:1rem;">
                    IDC calculado
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="
                    font-size:4rem;
                    font-weight:bold;
                    color:#008b5a;
                    margin-top:0rem;
                ">
                    {idc_calc:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="font-size:1.2rem; font-weight:bold; margin-top:1rem;">
                    IDC calculado
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="
                    font-size:4rem;
                    font-weight:bold;
                    color:#cccccc;
                    margin-top:0.5rem;
                ">
                    --
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.warning(
                "Para calcular el IDC primero debes calcular el promedio de todos "
                "los factores usando el botón de cada uno."
            )



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
    *Perfil de LinkedIn:* https://www.linkedin.com/in/cristian-camilo-galeano-largo-67ab8b340/\n
    - **Kevin Ossa Varela (kevin.ossa@utp.edu.co):**\n
    Estudiante de último semestre de Ingeniería de Sistemas y Computación.\n
    *Perfil de LinkedIn:* https://www.linkedin.com/in/kevin-ossa-varela-886aa7283/\n
    - **Ing. Sebastián Blandón Londoño (s.blandon@utp.edu.co):**\n
    Profesor de la Facultad de Ciencias Empresariales.\n
    *Perfil de LinkedIn:* https://www.linkedin.com/in/sebastian-blandon/
    - **Ph.D. Julián David Echeverry Correa (jde@utp.edu.co):**\n
    Profesor de la Facultad de Ingenierías.\n
    *Perfil de LinkedIn:* https://www.linkedin.com/in/juliandecheverry/
    ''')
    st.subheader("**Semillero de Investigación en Análisis de Datos (SAND) | Grupo de Investigación en Análisis de Datos y Sociología Computacional (GIADSc) [2025]**")