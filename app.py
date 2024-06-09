import streamlit as st
import joblib
from sklearn import datasets


# Cargar el dataset de Iris para obtener los nombres de las clases
iris = datasets.load_iris()
target_names = iris.target_names

model = joblib.load('modelo_iris_svm.pkl')

# Título de la aplicación
st.title("Clasificación de Iris")

# Crear entradas en la aplicación para las características
sepal_length = st.number_input("Longitud del sépalo", min_value=0.0, value=5.0, step=0.1)
sepal_width = st.number_input("Ancho del sépalo", min_value=0.0, value=3.5, step=0.1)
petal_length = st.number_input("Longitud del pétalo", min_value=0.0, value=1.5, step=0.1)
petal_width = st.number_input("Ancho del pétalo", min_value=0.0, value=0.2, step=0.1)

# Botón para predecir la especie de Iris
if st.button("Predecir"):
    # Realizar la predicción
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_class = target_names[prediction[0]]
    
    # Mostrar el resultado
    st.write(f"La especie de Iris predicha es: {predicted_class}")

