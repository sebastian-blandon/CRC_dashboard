import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.express as px
import os

def render_clustering(df, df_idc):
    st.subheader("Departamentos pares de Risaralda — Clustering PCA")

    # ====================================
    # 1. Filtro por año
    # ====================================
    anios_disponibles = sorted(df['Año'].unique())
    colA, colB = st.columns([1, 3])

    with colA:
        anio_seleccionado = st.selectbox(
            "Selecciona el año:",
            anios_disponibles,
            index=len(anios_disponibles)-1
        )

    df_anio = df[df['Año'] == anio_seleccionado].copy()
    if df_anio.empty:
        st.warning(f"No hay datos para el año {anio_seleccionado}")
        return
    
    indicadores = [
        'INS-2-1','NEG-2-2','TIC-1-3','TIC-1-1','EDU-1-3','INN-2-4',
        'SAL-3-3','SAL-1-3','TIC-1-4','EDU-2-1','EDS-2-1','FIN-1-4',
        'FIN-1-3','TAM-1-1','TIC-1-2','SOF-1-1'
    ]

    df_ind = df_anio[['Departamento'] + indicadores].copy()

    # ====================================
    # 2. PCA
    # ====================================
    X = df_ind[indicadores].values
    pca = PCA().fit(X)
    scores = pca.transform(X)

    df_scores = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(scores.shape[1])])
    df_scores['Departamento'] = df_ind['Departamento'].reset_index(drop=True)
    df_scores['Año'] = anio_seleccionado

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    st.markdown(f"**Varianza explicada:** PC1 = {var_pc1:.2f}% | PC2 = {var_pc2:.2f}%")

    # ====================================
    # 3. Distancia a Risaralda
    # ====================================
    df_scores['Departamento_norm'] = df_scores['Departamento'].str.upper().str.strip()

    if "RISARALDA" not in df_scores["Departamento_norm"].values:
        st.error("No se encontró Risaralda en los datos.")
        return

    ris_coords = df_scores.loc[df_scores['Departamento_norm']=="RISARALDA", ['PC1','PC2']].values[0]
    df_scores['Distancia_PC1_PC2'] = np.linalg.norm(
        df_scores[['PC1','PC2']] - ris_coords, axis=1
    )

    

    # ====================================
    # 4. Carpeta /data para guardar archivos
    # ====================================

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_dir = os.path.join(base_dir, "data", "clusters")
    os.makedirs(data_dir, exist_ok=True)

    save_path = os.path.join(data_dir, f"clusters_resultados_{anio_seleccionado}.csv")
    global_path = os.path.join(data_dir, "df_clusters_full.csv")

    # ====================================
    # 5. Clustering
    # ====================================
    if os.path.exists(save_path):
        df_scores = pd.read_csv(save_path)
        if 'Silhouette' not in df_scores.columns:
            df_scores['Silhouette'] = np.nan
            print("Columna 'Silhouette' no encontrada en el archivo. Se agregó con valor NaN.")
        st.info("Resultados cargados desde archivo local.")
    else:
        st.info("Calculando clústeres...")

        X_pca2 = df_scores[['PC1','PC2']].values
        k_range = range(9, 21)
        resultados = []

        def evaluar_modelos(X, k):
            modelos = {
                'K-Means': KMeans(n_clusters=k, random_state=42, n_init=10),
                'Jerárquico': AgglomerativeClustering(n_clusters=k),
                'Spectral': SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42),
                'GaussianMixture': GaussianMixture(n_components=k, random_state=42)
            }
            res = []
            for nombre, modelo in modelos.items():
                try:
                    labels = modelo.fit_predict(X)
                    sil = silhouette_score(X, labels)
                    res.append({'Modelo': nombre, 'K': k, 'Silhouette': sil})
                except:
                    pass
            return res

        for k in k_range:
            resultados.extend(evaluar_modelos(X_pca2, k))

        df_res = pd.DataFrame(resultados)
        best = df_res.loc[df_res['Silhouette'].idxmax()]

        best_model = best["Modelo"]
        best_k = int(best["K"])
        best_sil = float(best["Silhouette"])

        # Modelo final
        if best_model == 'K-Means':
            model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        elif best_model == 'Jerárquico':
            model = AgglomerativeClustering(n_clusters=best_k)
        elif best_model == 'Spectral':
            model = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', random_state=42)
        else:
            model = GaussianMixture(n_components=best_k, random_state=42)

        df_scores['Cluster'] = model.fit_predict(X_pca2)
        df_scores['Modelo'] = best_model
        df_scores['K'] = best_k
        df_scores['Silhouette'] = best_sil

        df_scores.to_csv(save_path, index=False)
        st.success(f"Modelo guardado en {save_path}")

        

        # Guardado global:
        if os.path.exists(global_path):
            df_all = pd.read_csv(global_path)
            df_all = pd.concat([df_all, df_scores], ignore_index=True)
        else:
            df_all = df_scores.copy()

        df_all.to_csv(global_path, index=False)

    # ====================================
    # 6. Visualización PCA
    # ====================================
    df_scores['Label'] = np.where(df_scores['Departamento_norm']=="RISARALDA","Risaralda","")
    

    fig = px.scatter(
        df_scores, x="PC1", y="PC2", color="Cluster",
        hover_data=['Departamento','Cluster'],
        text='Label', symbol='Label',
        title=f"Clustering PCA — Año {anio_seleccionado}",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, width='stretch')

    # ====================================
    # 7. Tabla final: Silhouette + IDC
    # ====================================
    st.markdown("### Departamentos más cercanos a Risaralda")

    # df_scores_ordenado = df_scores.sort_values("Distancia_PC1_PC2")
    # df_resultado = df_scores_ordenado[['Departamento','Silhouette','Año']].merge(
    #     df_idc[['Departamento','Año','IDC']],
    #     on=['Departamento','Año'],
    #     how='left'
    # )

    # st.dataframe(df_resultado[['Departamento','IDC','Silhouette']],
    #              use_container_width=True)
    
    df_scores_ordenado = df_scores.sort_values("Distancia_PC1_PC2")

    # === 7.1 Identificar el cluster de Risaralda ===
    cluster_risaralda = df_scores.loc[
        df_scores["Departamento_norm"] == "RISARALDA", "Cluster"
    ].iloc[0]

    # === 7.2 Filtrar SOLO los que están en ese cluster ===
    df_cluster = df_scores[df_scores["Cluster"] == cluster_risaralda]

    # === 7.3 Unir con IDC ===
    df_resultado = df_cluster[['Departamento','Silhouette','Año']].merge(
        df_idc[['Departamento','Año','IDC']],
        on=['Departamento','Año'],
        how='left'
    )

    # === 7.4 Mostrar tabla ===
    # st.dataframe(
    #     # df_resultado[['Departamento','IDC','Silhouette']].sort_values("IDC", ascending=False).reset_index(drop=True),
    #     df_resultado[['Departamento','IDC']].sort_values("IDC", ascending=False).reset_index(drop=True),
    #     width='content'
    # )

    df_tabla = (
        df_resultado[['Departamento', 'IDC']]
        .sort_values("IDC", ascending=False)
        .reset_index(drop=True)
    )

    # índice empezando en 1
    df_tabla.index = df_tabla.index + 1
    df_tabla.index.name = ""   # opcional: sin título de índice

    st.dataframe(
        df_tabla,
        width="content"
    )

    st.session_state['departamentos_pares'] = df_resultado['Departamento'].tolist()
    