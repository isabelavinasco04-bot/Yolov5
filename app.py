import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from gtts import gTTS
import io
import platform

# 🌸 Configuración inicial
st.set_page_config(page_title="Reconocimiento de Gestos ✨", page_icon="💖", layout="centered")

# 💅 Estilos personalizados
st.markdown("""
    <style>
    .main {
        background-color: #ffeef8;
        color: #3d2a42;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3, h4 {
        text-align: center;
        color: #ff69b4;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #ff9fd2;
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #ff7cc7;
        transform: scale(1.05);
    }
    .result-box {
        background-color: #ffd6eb;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0px 4px 15px rgba(255, 105, 180, 0.3);
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Mostrar versión
st.write("🧩 Python version:", platform.python_version())

# 📦 Cargar modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 🩷 Encabezado
st.title("💖 Reconocimiento de Gestos en Tiempo Real 💅")
st.markdown("Haz tu gesto frente a la cámara y deja que la IA adivine lo que haces ✨")

# 📸 Imagen decorativa
st.image("OIG5.jpg", width=350, caption="💫 ¡Haz tu gesto frente a la cámara!")

# 🧭 Sidebar con info
with st.sidebar:
    st.header("🪞 Guía rápida")
    st.write("""
    1️⃣ Asegúrate de tener buena iluminación.  
    2️⃣ Coloca tu mano o gesto frente a la cámara.  
    3️⃣ Espera unos segundos a que la app detecte el movimiento.  
    4️⃣ ¡Disfruta del resultado con voz y color! 💕
    """)

# 📷 Captura de cámara
img_file_buffer = st.camera_input("📸 Toma una foto")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))
    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # 🔮 Predicción
    prediction = model.predict(data)
    labels = ["Izquierda 👈", "Arriba 👆", "Derecha 👉", "Abajo 👇"]

    # 🎯 Mostrar resultados
    max_idx = np.argmax(prediction[0])
    gesture = labels[max_idx]
    confidence = float(prediction[0][max_idx])

    st.markdown(f"""
    <div class="result-box">
        <h2>🎀 Gesto detectado:</h2>
        <h1>{gesture}</h1>
        <p style='font-size:18px;'>Confianza: {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

    # 🌈 Mostrar barras de probabilidad
    st.markdown("### ✨ Probabilidades")
    for i, label in enumerate(labels):
        st.progress(float(prediction[0][i]))
        st.write(f"**{label}:** {prediction[0][i]:.2%}")

    # 🔊 Generar audio del resultado
    try:
        tts_text = f"El gesto detectado es {gesture.replace('👈','izquierda').replace('👆','arriba').replace('👉','derecha').replace('👇','abajo')}"
        tts = gTTS(text=tts_text, lang='es')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning(f"No se pudo reproducir el audio: {e}")

