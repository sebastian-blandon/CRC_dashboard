import streamlit as st

def calcular_variable(codigo, valores):
    """Calcula el valor de una variable según su fórmula y los valores independientes."""
    try:
        if codigo == "INS-2-1":  # Autonomía fiscal
            return (valores["Ingresos_tributarios"] + valores["Ingresos_no_tributarios"] - valores["Transferencias"]) / valores["Ingresos_totales"]

        elif codigo == "NEG-2-2":  # Densidad empresarial
            return (valores["Sociedades_empresariales"] / valores["Poblacion"]) * 100000

        elif codigo == "TIC-1-3":  # Hogares con computador
            return (valores["Hogares_con_computador"] / valores["Total_hogares"])

        elif codigo == "TIC-1-1":  # Penetración de internet
            return (valores["Accesos_fijos_internet"] / valores["Poblacion"])

        elif codigo == "EDU-1-3":  # Cobertura neta secundaria
            return (valores["Estudiantes_matriculados"] / valores["Poblacion_en_edad"])

        elif codigo == "INN-2-4":  # Marcas
            return ((valores["Registro_marca_t2"] + valores["Registro_marca_t1"] + valores["Registro_marca_t"]) / 3) / valores["Poblacion"] * 1_000_000

        elif codigo == "SAL-3-3":  # Médicos especialistas
            return (valores["Especializacion"] / valores["Poblacion"]) * 10000

        elif codigo == "SAL-1-3":  # Controles prenatales
            return (valores["Nacidos_con_controles"] / valores["Total_nacimientos"]) * 100

        elif codigo == "TIC-1-4":  # Uso de Internet
            return (valores["Poblacion_usa_internet"] / valores["Total_poblacion"])

        elif codigo == "EDU-2-1":  # Saber 11
            return (valores["P_Escritura"] + valores["P_Lectura"] + valores["P_Razonamiento"]) / 3

    except KeyError as e:
        st.warning(f"Falta el valor: {e}")
    except ZeroDivisionError:
        st.error("No se puede dividir por cero.")
    return None
