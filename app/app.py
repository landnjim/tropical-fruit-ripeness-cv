import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Maturité des Fruits Tropicaux", page_icon="🍌")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fruit_ripeness_model.keras')

model = load_model()

classes = ["mango_unripe", "mango_ripe", "banana_unripe", "banana_ripe",
           "pineapple_unripe", "pineapple_ripe"]

labels_fr = {
    "mango_unripe": "Mangue — pas mûre",
    "mango_ripe": "Mangue — mûre",
    "banana_unripe": "Banane — pas mûre",
    "banana_ripe": "Banane — mûre",
    "pineapple_unripe": "Ananas — pas mûr",
    "pineapple_ripe": "Ananas — mûr",
}

st.title("🍌 Estimation de la Maturité des Fruits Tropicaux")
st.write("Uploadez une photo de mangue, banane ou ananas pour estimer son stade de maturité.")

uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = classes[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    st.subheader(f"Résultat : {labels_fr[predicted_class]}")
    st.write(f"Confiance : {confidence:.1f}%")

    st.bar_chart({labels_fr[c]: float(p) for c, p in zip(classes, predictions)})
