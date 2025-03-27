# app.py
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import os
import pywin32  


# Cargar modelo
modelo = joblib.load('modelo_logistico.pkl')
with open('modelo_logistico.pkl', 'rb') as f:
    modelo = joblib.load(f)

# Título
st.set_page_config(page_title="Predicción Fallo Cardíaco", page_icon="❤️")
st.title("Predicción de Evento de Muerte en Pacientes con Falla Cardíaca")
st.markdown("Modelo de regresión logística basado en dataset de Kaggle")

# Formulario de entrada
st.header("Ingrese los datos del paciente")

time = st.slider("Duración seguimiento (days)", min_value=0, max_value=300, value=100)
ejection_fraction = st.slider("Fracción de eyección (%)", min_value=10, max_value=80, value=40)
serum_creatinine = st.number_input("Creatinina sérica (mg/dL)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# Botón de predicción
if st.button("Predecir"):
    # Crear DataFrame con entrada
    entrada = pd.DataFrame([[time, ejection_fraction, serum_creatinine]],
                           columns=["time", "ejection_fraction", "serum_creatinine"])
    
    # Predecir
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]
    
    st.subheader("Resultado:")
    if pred == 1:
        st.error(f"⚠️ Riesgo de muerte detectado (probabilidad: {prob:.2%})")
    else:
        st.success(f"✅ Bajo riesgo de muerte (probabilidad: {prob:.2%})")


print(os.path.exists('modelo_logistico.pkl'))  # Debe devolver Trueprint(os.path.abspath('modelo_logistico.pkl'))  
# En tu script (Corazon.py):
import os
if os.name == 'nt':  # Solo en Windows
  