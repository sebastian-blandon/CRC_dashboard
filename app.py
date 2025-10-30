import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from src.forecasting import forecast_por_departamento
import plotly.graph_objects as go
import plotly.express as px

# Configuración de página
st.set_page_config(page_title="IDC Dashboard", layout="wide")

# Rutas de los logos
assets = Path("assets")
logo_utp = assets / "Logo_UTP.png"
logo_crc = assets / "Logo_CRC.png"

# === ENCABEZADO ===
col1, col2 = st.columns([3, 1], gap="large")

# --- Columna izquierda: título ---
with col1:
    st.markdown(
        "<h1 style='margin-top: 10px; margin-bottom: 0px; text-align:left;'>"
        "Tablero de seguimiento al IDC"
        "</h1>",
        unsafe_allow_html=True
    )

# --- Columna derecha: logos alineados a la derecha ---
with col2:
    logo_col1, logo_col2 = st.columns([1, 1], gap="small")
    with logo_col1:
        st.image(str(logo_utp), width='stretch')
    with logo_col2:
        st.image(str(logo_crc), width='stretch')

st.markdown("---")


# Carga de datos
ruta_excel = "https://docs.google.com/spreadsheets/d/e/2PACX-1vToapNGBE4m3q7hFtf3NHJdREeYIUtnIxHUTpdN_gykbClhZFDonm1ZG8n4jm85ag/pub?output=xlsx"

hojas = pd.read_excel(
    ruta_excel,
    sheet_name=None  # lee todas las hojas
)

df_listado_indicadores = hojas["Listado_indicadores"]
df_IDC = hojas["Valores_IDC"]
df_indicadores_normalizados = hojas["Valores_Indicadores_Normalizado"]
df_analisis_sensibilidad = hojas["Sensibilidad"]

del hojas, ruta_excel

# Seccion C 
out_df, modelos = forecast_por_departamento(df_IDC, horizonte=2)

# ============================================================
#      Secciones C (gráfico) + E (listado) y D (tabla)
# ============================================================

left, right = st.columns([2, 1], gap="large")   # 67% / 33% como en el mockup

# --------- C: Evolución y pronóstico del IDC (izquierda/arriba) ----------

# --- Limpieza básica ---
tmp = out_df.copy()
tmp = tmp.dropna(subset=["Departamento", "Año"]).sort_values(["Departamento", "Año"])
tmp["Año"] = tmp["Año"].astype(int)

# --- Serie de Promedio IDC (hist y forecast por separado) ---
prom = (
    tmp.groupby(["Año", "Tipo"], as_index=False)["IDC"]
       .mean()
       .rename(columns={"IDC": "IDC_prom"})
)

# --- Preparar lienzo Plotly ---
fig = go.Figure()
palette = px.colors.qualitative.Set3 + px.colors.qualitative.Set2 + px.colors.qualitative.Set1
departamentos = tmp["Departamento"].unique().tolist()

# --- Trazas por departamento ---
for i, dept in enumerate(departamentos):
    color = palette[i % len(palette)]
    sub = tmp[tmp["Departamento"] == dept]

    # Histórico
    hist = sub[sub["Tipo"] == "hist"]
    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist["Año"], y=hist["IDC"],
                mode="lines+markers",
                name=dept,
                legendgroup=dept,
                marker=dict(symbol="circle", size=6),
                line=dict(color=color, width=2, dash="solid"),
                hovertemplate=f"<b>{dept}</b><br>Año=%{{x}}<br>IDC=%{{y:.3f}}<extra></extra>",
                showlegend=True
            )
        )

    # Pronóstico (línea punteada + círculo abierto)
    fc = sub[sub["Tipo"] == "forecast"].sort_values("Año")
    if not fc.empty:
        # Banda de error (Lo95-Hi95) si existen ambas
        if fc["Lo95"].notna().all() and fc["Hi95"].notna().all():
            # Traza inferior (invisible)
            fig.add_trace(
                go.Scatter(
                    x=fc["Año"], y=fc["Lo95"],
                    line=dict(color=color, width=0),
                    legendgroup=dept,
                    hoverinfo="skip",
                    showlegend=False
                )
            )
            # Traza superior con relleno hacia la inferior
            fig.add_trace(
                go.Scatter(
                    x=fc["Año"], y=fc["Hi95"],
                    fill="tonexty",
                    fillcolor=f"rgba{tuple(int(color.strip('#')[j:j+2],16) for j in (0,2,4)) + (0.15,)}"
                              if color.startswith("#") else "rgba(0,0,0,0.15)",
                    line=dict(color=color, width=0),
                    legendgroup=dept,
                    name=f"{dept} (IC 95%)",
                    hovertemplate=f"<b>{dept}</b> IC95<br>Año=%{{x}}<br>Lo=%{{y:.3f}}<extra></extra>",
                    showlegend=False
                )
            )

        # Traza de pronóstico
        fig.add_trace(
            go.Scatter(
                x=fc["Año"], y=fc["IDC"],
                mode="lines+markers",
                name=f"{dept} (forecast)",
                legendgroup=dept,
                marker=dict(symbol="circle-open", size=7),
                line=dict(color=color, width=2, dash="dash"),
                hovertemplate=f"<b>{dept} (forecast)</b><br>Año=%{{x}}<br>IDC=%{{y:.3f}}<extra></extra>",
                showlegend=False  # evitamos duplicar en la leyenda; queda 1 por depto
            )
        )

# --- Promedio IDC (hist y forecast) ---
prom_hist = prom[prom["Tipo"] == "hist"]
prom_fc   = prom[prom["Tipo"] == "forecast"]

# Color y estilo del promedio
avg_color = "black"

if not prom_hist.empty:
    fig.add_trace(
        go.Scatter(
            x=prom_hist["Año"], y=prom_hist["IDC_prom"],
            mode="lines+markers",
            name="Promedio IDC",
            legendgroup="Promedio",
            marker=dict(symbol="circle", size=7),
            line=dict(color=avg_color, width=3, dash="solid"),
            hovertemplate="<b>Promedio (hist)</b><br>Año=%{x}<br>IDC=%{y:.3f}<extra></extra>",
            showlegend=True
        )
    )

if not prom_fc.empty:
    fig.add_trace(
        go.Scatter(
            x=prom_fc["Año"], y=prom_fc["IDC_prom"],
            mode="lines+markers",
            name="Promedio IDC (forecast)",
            legendgroup="Promedio",
            marker=dict(symbol="circle-open", size=8),
            line=dict(color=avg_color, width=3, dash="dash"),
            hovertemplate="<b>Promedio (forecast)</b><br>Año=%{x}<br>IDC=%{y:.3f}<extra></extra>",
            showlegend=True
        )
    )

# --- Layout del gráfico ---
fig.update_layout(
    title="Evolución y pronóstico del IDC",
    xaxis_title="Año",
    yaxis_title="IDC",
    legend_title="Series",
    hovermode="x unified",
    template="plotly_white",
    margin=dict(l=10, r=10, t=50, b=10),
    height=420
)

# Asegurar que el eje X muestre solo enteros
fig.update_xaxes(type="category")


with left:
    st.subheader("Evolución y pronóstico del IDC")

    # # Datos de ejemplo; reemplaza por tus series reales
    # anios = np.arange(2019, 2026)
    # df = pd.DataFrame({
    #     "Año": anios,
    #     "Dep1": [6.2, 7.0, 7.4, 7.2, 7.8, 7.5, 7.7],
    #     "Dep2": [1.0, 1.5, 2.1, 2.6, 3.2, 3.8, 4.1],
    #     "Dep3": [6.0, 5.4, 4.8, 5.5, 7.0, 4.8, 5.6],
    #     "Dep4": [6.5, 7.2, 7.5, 7.1, 7.4, 7.2, 7.8],
    # }).set_index("Año")

    # st.line_chart(df, width='stretch', height=360)
    st.plotly_chart(fig, width='stretch')

    # --------- E: Lista de departamentos (izquierda/abajo) ----------
    st.markdown("### Departamentos pares de Risaralda")
    st.markdown(
        """
- Departamento 1  
- Departamento 2  
- …  
- Departamento N
        """
    )

# --------- D: Top 11 Indicadores (derecha) ----------
top10_sensibilidad = df_analisis_sensibilidad.sort_values("Rank_Promedio").head(10)

tabla_final = (
    top10_sensibilidad.merge(
        df_listado_indicadores[["Indicador", "Nombre"]],
        how="left",
        left_on="Variable",
        right_on="Indicador"
    )[["Indicador", "Nombre", "Mu_star", "ST", "Rank_Promedio"]]
)

tabla_final.drop(["Mu_star", "ST", "Rank_Promedio"], axis=1, inplace=True)

with right:
    st.subheader("Top 10 Indicadores con mayor influencia sobre el IDC")

    # top11 = pd.DataFrame({
    #     "Indicador": [
    #         "EDS-2-4","INN-2-4","EDS-2-1","TIC-1-1","NEG-2-2",
    #         "TIC-1-3","INF-1-1","TIC-1-2","INN-2-1","INN-2-3","SAL-3-2"
    #     ],
    #     "Nombre": [
    #         "Dominio del inglés","Marcas","Puntaje pruebas Saber Pro",
    #         "Penetración de internet banda ancha fijo","Densidad empresarial",
    #         "Hogares con computador","Cobertura de acueducto","Ancho banda de internet",
    #         "Patentes","Diseños industriales","Médicos generales"
    #     ],
    # })

    

    # Estilo sin CSS externo: pandas Styler + st.table
    sty = (
        tabla_final.style
        .hide(axis="index")
        .set_properties(**{"text-align": "left"})
        .set_table_styles([
            {"selector": "thead th",
             "props": [("background-color", "#1f77b4"), ("color", "white"),
                       ("font-weight", "bold")]},
            {"selector": "tbody td", "props": [("border", "0px")]}
        ])
        .apply(lambda s: ["background-color: #f2f2f2" if i % 2 == 0 else ""
                          for i in range(len(s))], axis=1)
    )
    st.table(sty)  # st.dataframe no acepta Styler; st.table sí

# ==================== Notas de ajuste ====================
# - Ancho de columnas principales: cambia [2,1] para modificar 67%/33%.
# - Altura del gráfico: ajusta 'height' en st.line_chart.
# - Tamaño relativo de logos: se controla con el ancho de su columna 'col2'
#   y el valor use_container_width=True (se adaptan al espacio).



# ============================================================
#      Secciones C (gráfico) + E (listado) y D (tabla)
# ============================================================




