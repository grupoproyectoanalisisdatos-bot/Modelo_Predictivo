import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Configuración de la interfaz
st.set_page_config(page_title="Predictor Climático - Railway", page_icon="🌡️")

st.title("🌦️ Predicción de Temperatura Máxima")
st.markdown("Esta herramienta utiliza un modelo entrenado con datos de MySQL en Railway.")

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('modelo_temperatura_railway.pkl')

modelo = load_model()

# Entradas del usuario
st.sidebar.header("Configuración de entrada")
temp_min = st.sidebar.number_input("Temperatura Mínima (°C)", value=15.0)
fecha = st.sidebar.date_input("Fecha de interés", datetime.now())

# Preparar los datos para el modelo (Mes y Día)
mes = fecha.month
dia = fecha.day

# El orden de las columnas debe ser igual al del entrenamiento: temp_min, Mes, Dia
input_df = pd.DataFrame([[temp_min, mes, dia]], columns=['temp_min', 'Mes', 'Dia'])

if st.button("Predecir Temperatura Máxima"):
    prediccion = modelo.predict(input_df)
    st.metric(label="Temperatura Máxima Estimada", value=f"{prediccion[0]:.2f} °C")
    
    if prediccion[0] > 25:
        st.warning("Se espera un día caluroso.")
    else:
        st.info("Condiciones climáticas moderadas.")
