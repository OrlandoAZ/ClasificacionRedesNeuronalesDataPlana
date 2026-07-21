"""
App de Streamlit — Predicción de Churn (Red Neuronal MLP)
============================================================
Despliega el modelo entrenado en `Red_Neuronal_Clasificacion_Churn.ipynb`
sobre el dataset Churn Modelling (banco).

Archivos requeridos en el mismo directorio que este script:
    - modelo_MLP_churn.keras
    - preprocesamiento_churn.pkl

Ejecutar localmente:
    streamlit run app.py
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Churn — MLP",
    page_icon="🏦",
    layout="centered",
)

MODEL_PATH = "modelo_MLP_churn.keras"
PREPROC_PATH = "preprocesamiento_churn.pkl"


# ──────────────────────────────────────────────────────────────────────────
# CARGA DE MODELO Y PREPROCESAMIENTO (cacheado)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def cargar_preprocesamiento():
    with open(PREPROC_PATH, "rb") as f:
        return pickle.load(f)


try:
    model = cargar_modelo()
    preproc = cargar_preprocesamiento()
    scaler = preproc["scaler"]
    feature_columns = preproc["feature_columns"]  # num_cols + dummy_cols, en ese orden
    num_cols = preproc["num_cols"]
    dummy_cols = preproc["dummy_cols"]
except FileNotFoundError as e:
    st.error(
        "No se encontraron los archivos del modelo. Asegúrate de subir "
        f"`{MODEL_PATH}` y `{PREPROC_PATH}` al mismo directorio que `app.py`.\n\n"
        f"Detalle: {e}"
    )
    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# ENCABEZADO
# ──────────────────────────────────────────────────────────────────────────
st.title("🏦 Predicción de Churn de Clientes")
st.caption("Red neuronal MLP (Keras / TensorFlow) entrenada sobre el dataset Churn Modelling")

st.markdown(
    "Completa los datos del cliente para estimar la probabilidad de que "
    "abandone el banco (`Exited = 1`)."
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# FORMULARIO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────
with st.form("form_cliente"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "CreditScore (Min: 300 — Max: 900)",
            min_value=300, max_value=900, value=650, step=1,
        )
        age = st.number_input(
            "Edad — Age (Min: 18 — Max: 100)",
            min_value=18, max_value=100, value=35, step=1,
        )
        tenure = st.number_input(
            "Antigüedad — Tenure, años (Min: 0 — Max: 10)",
            min_value=0, max_value=10, value=5, step=1,
        )
        balance = st.number_input(
            "Balance en cuenta — Balance (Min: 0.00)",
            min_value=0.0, value=0.0, step=100.0, format="%.2f",
        )
        num_products = st.number_input(
            "Número de productos — NumOfProducts (Min: 1 — Max: 4)",
            min_value=1, max_value=4, value=1, step=1,
        )

    with col2:
        estimated_salary = st.number_input(
            "Salario estimado — EstimatedSalary (Min: 0.00)",
            min_value=0.0, value=50000.0, step=500.0, format="%.2f",
        )
        geography = st.selectbox("País (Geography)", options=["France", "Germany", "Spain"])
        gender = st.selectbox("Género (Gender)", options=["Female", "Male"])
        has_cr_card = st.radio("¿Tiene tarjeta de crédito? (HasCrCard)", options=["Sí", "No"], horizontal=True)
        is_active = st.radio("¿Es miembro activo? (IsActiveMember)", options=["Sí", "No"], horizontal=True)

    umbral = st.slider(
        "Umbral de decisión para clasificar como Churn",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    )

    submitted = st.form_submit_button("Predecir", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO DE LA ENTRADA — replica exactamente la lógica del notebook
# ──────────────────────────────────────────────────────────────────────────
def construir_vector_entrada(
    credit_score, age, tenure, balance, num_products,
    estimated_salary, geography, gender, has_cr_card, is_active,
):
    """Construye un DataFrame de una fila con el mismo esquema de columnas
    dummy usado en el entrenamiento (get_dummies con drop_first=True),
    y luego reordena/escala según feature_columns."""

    fila = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "EstimatedSalary": estimated_salary,
        # Dummies — se inicializan en 0 y se activan según corresponda
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "HasCrCard_1": 1 if has_cr_card == "Sí" else 0,
        "IsActiveMember_1": 1 if is_active == "Sí" else 0,
    }

    df = pd.DataFrame([fila])

    # Asegurar que todas las columnas esperadas existan (por si el pickle
    # difiere ligeramente, p. ej. categorías no vistas)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]  # reordenar exactamente como en el entrenamiento

    X = df.values.astype(np.float32)
    n_num = len(num_cols)

    X_num_sc = scaler.transform(X[:, :n_num])
    X_sc = np.hstack([X_num_sc, X[:, n_num:]]).astype(np.float32)

    return X_sc


# ──────────────────────────────────────────────────────────────────────────
# PREDICCIÓN Y RESULTADOS
# ──────────────────────────────────────────────────────────────────────────
if submitted:
    X_input = construir_vector_entrada(
        credit_score, age, tenure, balance, num_products,
        estimated_salary, geography, gender, has_cr_card, is_active,
    )

    prob_churn = float(model.predict(X_input, verbose=0).flatten()[0])
    pred_clase = int(prob_churn >= umbral)

    st.divider()
    st.subheader("Resultado")

    c1, c2 = st.columns(2)
    c1.metric("Probabilidad de Churn", f"{prob_churn:.1%}")
    c2.metric(
        "Predicción",
        "🔴 Churn (se va)" if pred_clase == 1 else "🟢 Retenido",
    )

    st.progress(min(max(prob_churn, 0.0), 1.0))

    if pred_clase == 1:
        st.warning(
            f"Con el umbral de {umbral:.0%}, el modelo estima que este cliente "
            "tiene alta probabilidad de abandonar el banco."
        )
    else:
        st.success(
            f"Con el umbral de {umbral:.0%}, el modelo estima que este cliente "
            "probablemente se mantendrá como cliente activo."
        )

    with st.expander("Ver vector de entrada procesado (debug)"):
        st.write("Orden de features:", feature_columns)
        st.write(X_input)

st.divider()
st.caption(
    "⚠️ Esta predicción se basa en un modelo entrenado con datos históricos "
    "y no debe usarse como única base para decisiones de negocio."
)


# streamlit run 03app.py

# Cuando conecte Github y Streamlit, ir a Settings: Python 3.11