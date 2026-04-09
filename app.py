import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# 1. Configuración estética
st.set_page_config(page_title="Prediccion Clima Antioquia", layout="wide")

# Estilo personalizado con Markdown
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌡️ Monitor Predictivo de Temperatura")
st.subheader("Análisis basado en estaciones de IDEAM")

# 2. Carga del modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_temperatura_railway.pkl')

modelo = cargar_modelo()

# 3. Interfaz Lateral (Sidebar)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4052/4052984.png", width=100)
st.sidebar.header("Parámetros de Consulta")

# Agregamos el selector de Municipio
municipio = st.sidebar.selectbox(
    "Seleccione el Municipio:",
    ["Alejandría", "Urrao", "Cañasgordas", "Otros"]
)

fecha = st.sidebar.date_input("Fecha de Análisis", datetime.now())
t_min = st.sidebar.slider("Temperatura Mínima esperada (°C)", 0.0, 30.0, 15.0)

# 4. Cuerpo principal
col1, col2 = st.columns([2, 1])

with col1:
    st.info(f"Configuración seleccionada: **{municipio}** para el día **{fecha.strftime('%d/%m/%Y')}**")
    
    if st.button("Generar Predicción"):
        # Preparación de datos (Igual que en el entrenamiento)
        input_data = pd.DataFrame([[t_min, fecha.month, fecha.day]], 
                                  columns=['temp_min', 'Mes', 'Dia'])
        
        prediccion = modelo.predict(input_data)[0]
        
        # Mostrar resultado con un diseño llamativo
        st.metric(label="Temperatura Máxima Estimada", value=f"{prediccion:.2f} °C", delta=f"{prediccion - t_min:.2f} °C (Oscilación)")
        
        # Lógica de advertencia según el municipio
        if prediccion > 28:
            st.error(f"⚠️ Alerta de calor en {municipio}. Se recomienda monitoreo de riego.")
        else:
            st.success(f"✅ Condiciones estables para {municipio}.")

with col2:
    st.write("### Detalles Técnicos")
    st.write(f"**Estación:** {municipio}")
    st.write(f"**Variable predictora:** Mínima de {t_min}°C")
    st.write("**Modelo:** Random Forest Regressor")
