import streamlit as st
import numpy as np
import os
from keras.models import load_model
from PIL import Image, ImageOps

# ---------------------------------------------------
# Seitenkonfiguration
# ---------------------------------------------------
st.set_page_config(
    page_title="🌦️ Wetter Klassifikator",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Wetter Klassifikator")
st.markdown("Lade ein Bild hoch und finde heraus, ob du rausgehen solltest!")

# ---------------------------------------------------
# Modell sicher laden (Cloud-kompatibel)
# ---------------------------------------------------
@st.cache_resource
def load_teachable_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "keras_Model.h5")
    
    if not os.path.exists(model_path):
        st.error("❌ Modell-Datei 'keras_Model.h5' nicht gefunden!")
        st.stop()

    return load_model(model_path, compile=False)

model = load_teachable_model()

# ---------------------------------------------------
# Labels laden
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labels_path = os.path.join(BASE_DIR, "labels.txt")

if not os.path.exists(labels_path):
    st.error("❌ labels.txt nicht gefunden!")
    st.stop()

class_names = open(labels_path, "r", encoding="utf-8").readlines()

# ---------------------------------------------------
# Bild Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📷 Bild hochladen (JPG oder PNG)",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Vorhersage
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Teachable Machine Preprocessing
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = float(prediction[0][index])

    # Label bereinigen (z.B. "0 gutes Wetter")
    class_label = class_names[index][2:].strip()

    st.markdown("---")
    st.subheader("🔎 Vorhersage")

    # Wetter-Logik
    if "gut" in class_label.lower():
        st.success("🌞 **Raus gehen empfohlen!**")
    else:
        st.warning("🌧️ **Lieber drinnen bleiben!**")

    st.info(f"Confidence Score: **{confidence_score * 100:.2f}%**")

    # Wahrscheinlichkeiten anzeigen
    st.markdown("### 📊 Wahrscheinlichkeiten")
    for i, prob in enumerate(prediction[0]):
        label = class_names[i][2:].strip()
        st.write(f"{label}: {prob * 100:.2f}%")
        st.progress(float(prob))
