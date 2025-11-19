from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters.results import HoltWintersResults
from pathlib import Path

# ------------------------------
# Config
# ------------------------------
Z_95 = 1.96  # para intervalos de confianza ~95%

@dataclass
class ForecastResult:
    modelo: str
    fitted: Optional[HoltWintersResults]
    fh: int
    resid_std: float

# ------------------------------
# Utilidades
# ------------------------------
def _to_yearly_index(df: pd.DataFrame, year_col: str) -> pd.Series:
    # último día del año para índice anual
    dt = pd.to_datetime(df[year_col].astype(int).astype(str) + "-12-31")
    s = pd.Series(df["IDC"].values, index=dt).sort_index()
    s = s.asfreq("YE-DEC")
    return s

def _fill_gaps_annual(s: pd.Series, method: str = "interpolate") -> pd.Series:
    s = s.asfreq("YE-DEC")   # <- y aquí
    if method == "interpolate":
        return s.interpolate(limit_direction="both")
    elif method == "ffill":
        return s.ffill()
    elif method == "bfill":
        return s.bfill()
    return s

def _naive_forecast(last_value: float, fh: int) -> np.ndarray:
    return np.repeat(last_value, fh)

# def _calc_pi(point_fcst: np.ndarray, resid_std: float) -> Tuple[np.ndarray, np.ndarray]:
#     if np.isnan(resid_std) or resid_std <= 0:
#         return np.full_like(point_fcst, np.nan), np.full_like(point_fcst, np.nan)
#     lo = point_fcst - Z_95 * resid_std
#     hi = point_fcst + Z_95 * resid_std
#     return lo, hi

def _calc_pi(point_fcst: np.ndarray, resid_std: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intervalos con varianza que crece en el horizonte:
      Var(e_h) ~ h * resid_std^2  =>  Std(e_h) ~ sqrt(h) * resid_std
    """
    if np.isnan(resid_std) or resid_std <= 0:
        return np.full_like(point_fcst, np.nan), np.full_like(point_fcst, np.nan)

    # h = 1, 2, ..., fh
    h = np.arange(1, len(point_fcst) + 1, dtype=float)
    half_width = Z_95 * resid_std * np.sqrt(h)  # ancho crece con h

    lo = point_fcst - half_width
    hi = point_fcst + half_width
    return lo, hi

# ------------------------------
# Modelado por serie
# ------------------------------
def _fit_ets_auto(y: pd.Series) -> Tuple[Optional[HoltWintersResults], str]:
    """
    Prueba configuraciones razonables de ETS (Holt-Winters) y elige la de menor AIC.
    Sin estacionalidad (anual); si tuvieras estacionalidad > 1 año, ajusta 'seasonal_periods'.
    """
    candidates = [
        dict(trend=None, damped_trend=False),
        dict(trend="add", damped_trend=False),
        dict(trend="add", damped_trend=True),
        dict(trend="mul", damped_trend=False),  # por si varía proporcionalmente
        dict(trend="mul", damped_trend=True),
    ]
    best_aic = np.inf
    best_fit = None
    best_name = "naive"
    for c in candidates:
        try:
            m = ExponentialSmoothing(
                y, trend=c["trend"], seasonal=None, damped_trend=c["damped_trend"], initialization_method="estimated"
            )
            fit = m.fit(optimized=True, use_brute=True)
            aic = getattr(fit, "aic", np.inf)
            if aic < best_aic:
                best_aic, best_fit = aic, fit
                t = c["trend"] if c["trend"] else "none"
                best_name = f"ETS(trend={t}, damped={c['damped_trend']})"
        except Exception:
            continue
    return best_fit, best_name

def forecast_one_series(y: pd.Series, fh: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ForecastResult]:
    """
    Pronostica una serie anual y devuelve:
      - yhat: pronóstico puntual (fh pasos)
      - lo, hi: intervalos ~95%
      - info del modelo
    Reglas:
      - Si len(y) < 3 => método ingenuo (última observación).
      - Si ETS falla => ingenuo.
    """
    assert fh in (1, 2), "Solo se admite horizonte 1 o 2 pasos."
    y = y.dropna()
    if len(y) == 0:
        return np.array([]), np.array([]), np.array([]), ForecastResult("empty", None, fh, np.nan)
    if len(y) < 3:
        yhat = _naive_forecast(y.iloc[-1], fh)
        lo, hi = np.full(fh, np.nan), np.full(fh, np.nan)
        return yhat, lo, hi, ForecastResult("naive(len<3)", None, fh, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit, name = _fit_ets_auto(y)

    if fit is None:
        yhat = _naive_forecast(y.iloc[-1], fh)
        lo, hi = np.full(fh, np.nan), np.full(fh, np.nan)
        return yhat, lo, hi, ForecastResult("naive(fallback)", None, fh, np.nan)

    # Pronóstico
    fcst = fit.forecast(fh).to_numpy()

    # Intervalos aproximados con desvío de residuos in-sample
    resid = y - fit.fittedvalues.reindex(y.index)
    resid_std = float(resid.std(ddof=1)) if resid.notna().sum() > 1 else np.nan
    lo, hi = _calc_pi(fcst, resid_std)

    return fcst, lo, hi, ForecastResult(name, fit, fh, resid_std)

# ------------------------------
# Pipeline sobre df_IDC
# ------------------------------
def forecast_por_departamento(
    df_IDC: pd.DataFrame,
    dept_col: str = "Departamento",
    year_col: str = "Año",
    value_col: str = "IDC",
    horizonte: int = 2,
    fill_gaps: str = "interpolate",  # "interpolate" | "ffill" | "bfill" | None
) -> Tuple[pd.DataFrame, Dict[str, ForecastResult]]:
    """
    Genera pronóstico automático por departamento. Devuelve:
      - DataFrame con histórico + pronóstico (columnas: Departamento, Año, IDC, Tipo, Lo95, Hi95)
      - dict con info de modelos por departamento
    """
    assert horizonte in (1, 2), "horizonte debe ser 1 o 2"

    modelos: Dict[str, ForecastResult] = {}
    registros: List[dict] = []

    # Iteramos por departamento
    for dept, g in df_IDC[[dept_col, year_col, value_col]].dropna(subset=[dept_col, year_col]).groupby(dept_col):
        g = g.sort_values(year_col)

        # Serie anual
        s = _to_yearly_index(g.rename(columns={year_col: "Año"}), "Año")
        if fill_gaps:
            s = _fill_gaps_annual(s, method=fill_gaps)

        # Guardar histórico
        for t, val in s.items():
            registros.append({
                dept_col: dept,
                "Año": t.year,
                value_col: float(val) if pd.notna(val) else np.nan,
                "Tipo": "hist",
                "Lo95": np.nan,
                "Hi95": np.nan,
            })

        # Pronóstico
        yhat, lo, hi, info = forecast_one_series(s, fh=horizonte)
        modelos[dept] = info

        # Índices futuros (años)
        if len(s) > 0 and len(yhat) > 0:
            last_year = s.index[-1].year
            future_years = [last_year + i for i in range(1, horizonte + 1)]
            for i, ypred in enumerate(yhat):
                registros.append({
                    dept_col: dept,
                    "Año": future_years[i],
                    value_col: float(ypred),
                    "Tipo": "forecast",
                    "Lo95": float(lo[i]) if pd.notna(lo[i]) else np.nan,
                    "Hi95": float(hi[i]) if pd.notna(hi[i]) else np.nan,
                })

    out = pd.DataFrame(registros)
    # Orden típico: por Departamento, luego Año
    out = out.sort_values([dept_col, "Año"], kind="mergesort").reset_index(drop=True)
    return out, modelos

def guardar_pronosticos_en_excel(
    out_df: pd.DataFrame,
    ruta_excel: str | Path,
    sheet_name: str = "Pronostico_IDC",
) -> None:
    ruta_excel = Path(ruta_excel)

    try:
        # Si el archivo existe, actualizamos y reemplazamos sólo esa hoja
        with pd.ExcelWriter(
            ruta_excel,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

    except FileNotFoundError:
        # Si no existe, creamos el archivo
        with pd.ExcelWriter(
            ruta_excel,
            engine="openpyxl",
            mode="w",
        ) as writer:
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

# ------------------------------
# Ejecución (ejemplo)
# ------------------------------
# df_IDC: columnas "Departamento", "Año", "IDC"
# out_df, modelos = forecast_por_departamento(df_IDC, horizonte=2)

# Ejemplos de quick-check (testing básico)
# assert set(out_df["Tipo"].unique()) <= {"hist", "forecast"}
# assert out_df.groupby("Departamento")["Tipo"].apply(lambda s: (s=="forecast").sum()).between(1,2).all()

# Ejemplo de uso (comentado)
# if __name__ == "__main__":
#     df_IDC = pd.read_excel(".../tu_archivo.xlsx")
#     out_df, modelos = forecast_por_departamento(df_IDC, horizonte=2)
#     print(out_df.head())