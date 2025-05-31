import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import tensorflow as tf

# ---------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------
st.set_page_config(page_title="Análisis de Publicidad", layout="centered")
st.title("🧠 Sistema Inteligente de Evaluación de Publicidad Visual")

# ---------------------------
# FUNCIONES DE ANÁLISIS
# ---------------------------
def extraer_colores_principales(imagen, n_colores=3):
    pixels = imagen.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colores)
    kmeans.fit(pixels)
    colores = kmeans.cluster_centers_.astype(int)
    return [tuple(color) for color in colores]

def estimar_porcentaje_texto(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(gris, 100, 200)
    return np.sum(bordes > 0) / (imagen.shape[0] * imagen.shape[1])

def calcular_contraste(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    return np.std(gris)

def convertir_color_a_nombre(rgb):
    nombres = {
        "rojo": (255, 0, 0), "verde": (0, 128, 0), "azul": (0, 0, 255),
        "amarillo": (255, 255, 0), "naranja": (255, 165, 0), "marrón": (139, 69, 19),
        "negro": (0, 0, 0), "blanco": (255, 255, 255), "gris": (128, 128, 128),
        "violeta": (238, 130, 238), "rosado": (255, 192, 203), "beige": (245, 245, 220),
        "dorado": (255, 215, 0)
    }
    dist = lambda c1, c2: np.linalg.norm(np.array(c1) - np.array(c2))
    return [min(nombres, key=lambda nombre: dist(rgb_valor, nombres[nombre])) for rgb_valor in rgb]

criterios_por_rubro = {
    "tecnología y servicios": {
        "colores_permitidos": ["azul", "negro", "gris", "blanco"],
        "peso_colores": 3,
        "max_texto": 0.35,
        "peso_texto": 4,
        "min_contraste": 40,
        "peso_contraste": 3,
        "comentario_bueno": "Buen uso del espacio visual, colores tecnológicos sobrios y claridad informativa.",
        "comentario_malo": "Colores muy brillantes o desordenados, texto denso y elementos mal distribuidos."
    }
}

def evaluar_reglas(imagen, rubro):
    criterios = criterios_por_rubro[rubro]
    colores = extraer_colores_principales(imagen)
    nombres_colores = convertir_color_a_nombre(colores)
    porcentaje_texto = estimar_porcentaje_texto(imagen)
    contraste = calcular_contraste(imagen)

    puntaje = 0
    colores_validos = sum([1 for c in nombres_colores if c in criterios["colores_permitidos"]])
    puntaje += colores_validos * criterios["peso_colores"]

    if porcentaje_texto <= criterios["max_texto"]:
        puntaje += criterios["peso_texto"]

    if contraste >= criterios["min_contraste"]:
        puntaje += criterios["peso_contraste"]

    if puntaje >= 7:
        resultado = "✅ Publicidad Buena"
        comentario = criterios["comentario_bueno"]
    elif 4 <= puntaje < 7:
        resultado = "⚠️ Publicidad Regular"
        comentario = "Algunos elementos pueden mejorarse para mayor impacto visual."
    else:
        resultado = "🟥 Publicidad Mala"
        comentario = criterios["comentario_malo"]

    return resultado, nombres_colores, porcentaje_texto, contraste, comentario

# ---------------------------
# INTERFAZ STREAMLIT
# ---------------------------
archivo = st.file_uploader("Subí una imagen publicitaria", type=["png", "jpg", "jpeg"])
rubro = st.selectbox("Seleccioná el rubro de la imagen:", list(criterios_por_rubro.keys()))

if archivo:
    imagen_pil = Image.open(archivo).convert("RGB")
    imagen_np = np.array(imagen_pil)

    st.image(imagen_pil, caption="Imagen cargada", use_column_width=True)

    resultado, colores, texto, contraste, comentario = evaluar_reglas(imagen_np, rubro)

    st.markdown("---")
    st.subheader("🔍 Resultados del análisis")
    st.write(f"**Colores detectados:** {colores}")
    st.write(f"**Porcentaje estimado de texto:** {texto:.0%}")
    st.write(f"**Contraste estimado:** {contraste:.2f}")

    st.markdown("---")
    st.subheader("📌 Clasificación Final")
    st.write(f"{resultado}")
    st.info(comentario)

    st.markdown("---")
    st.caption("Desarrollado por el equipo IA Emprendimientos • Defensa Final")
