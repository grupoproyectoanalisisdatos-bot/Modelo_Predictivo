import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Título de la herramienta
st.title("🌡️ Predictor de Temperatura Máxima")
st.write("Herramienta analítica para el sector agrícola.")

# Función para cargar el modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_temperatura_railway.pkl')

modelo = cargar_modelo()

# Entradas laterales
st.sidebar.header("Datos de Entrada")
fecha = st.sidebar.date_input("Selecciona la Fecha", datetime.now())
t_min = st.sidebar.number_input("Temperatura Mínima (°C)", value=15.0)

# Botón para predecir
if st.button("Realizar Predicción"):
    # Preparamos los datos igual que en el entrenamiento
    input_data = pd.DataFrame([[t_min, fecha.month, fecha.day]], 
                              columns=['temp_min', 'Mes', 'Dia'])
    
    prediccion = modelo.predict(input_data)
    
    st.metric(label="Temperatura Máxima Estimada", value=f"{prediccion[0]:.2f} °C")
