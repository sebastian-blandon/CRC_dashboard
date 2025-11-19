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
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import forecast_por_departamento, guardar_pronosticos_en_excel, forecast_one_series
from src.scenary_simulator import calcular_variable
from src.clustering import render_clustering
import base64


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



tab_idc, tab_clustering, tab_simulador = st.tabs(["IDC", "Clustering", "Simulador"])


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

        # --- Bandas IC95 del promedio nacional (forecast) ---
        fc_prom = (
            prom[
                (prom["Tipo"] == "forecast")
                & prom["Lo95_prom"].notna()
                & prom["Hi95_prom"].notna()
            ]
            .sort_values("Año")
        )

        if not fc_prom.empty:
            # Polígono relleno
            fig.add_trace(
                go.Scatter(
                    x=list(fc_prom["Año"]) + list(fc_prom["Año"][::-1]),
                    y=list(fc_prom["Hi95_prom"]) + list(fc_prom["Lo95_prom"][::-1]),
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.10)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Bordes punteados
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
        # fig.update_xaxes(type="category")
        # return fig
        fig.update_xaxes(type="linear", tickmode="linear", dtick=1)
        return fig


    with left:
        st.subheader("Evolución y pronóstico del IDC")
        fig_plot = build_idc_figure(tmp, prom, departamentos_pares2)

        # Orden cronológico en el eje X (numérico)
        st.plotly_chart(fig_plot, width='stretch')

        

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