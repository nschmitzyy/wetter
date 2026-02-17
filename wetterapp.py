import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# ---------------------------------------------------
# Streamlit Seiteneinstellungen
# ---------------------------------------------------
st.set_page_config(
    page_title="Wetter Klassifikator",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Wetter Klassifikator")
st.markdown("Lade ein Bild hoch und finde heraus, ob du rausgehen solltest!")

# ---------------------------------------------------
# Modell laden (Caching für Performance)
# ---------------------------------------------------
@st.cache_resource
def load_teachable_model():
    model = load_model("keras_Model.h5", compile=False)
    return model

model = load_teachable_model()

# Labels laden
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

# ---------------------------------------------------
# Bild Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📷 Bild hochladen (JPG oder PNG)",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Vorhersage-Logik (Original Teachable Machine)
# ---------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Array mit korrekter Form erstellen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Bild zuschneiden & skalieren (wie im Originalcode)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Bild in NumPy Array umwandeln
    image_array = np.asarray(image)

    # Normalisieren
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # In Datenarray laden
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # Entfernt die führende Zahl aus labels.txt (z.B. "0 gutes Wetter")
    class_label = class_name[2:].strip()

    st.markdown("---")
    st.subheader("🔎 Vorhersage")

    # Wetter-Logik
    if "gut" in class_label.lower():
        st.success("🌞 **Raus gehen empfohlen!**")
    else:
        st.warning("🌧️ **Lieber drinnen bleiben!**")

    # Confidence Anzeige
    st.info(f"Confidence Score: **{confidence_score * 100:.2f}%**")

    # Optional: Wahrscheinlichkeiten anzeigen
    st.markdown("### 📊 Wahrscheinlichkeiten")
    for i, prob in enumerate(prediction[0]):
        label = class_names[i][2:].strip()
        st.write(f"{label}: {prob * 100:.2f}%")
        st.progress(float(prob))
