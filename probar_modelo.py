import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Definir las etiquetas oficiales de CIFAR-10
CLASSES = [
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
]

# 2. Cargar el modelo desde la carpeta 'models' con su extensión .keras
try:
    # Ajustado a la ruta correcta según tu captura de pantalla
    model = tf.keras.models.load_model("models/modelo_red_neuronal.keras")
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(
        f"Error al cargar el modelo. Verifica que la ruta 'models/modelo_red_neuronal.keras' sea accesible: {e}"
    )
    exit()


def clasificar_imagen(ruta_imagen):
    try:
        # 3. Preprocesar la imagen a 32x32 píxeles
        img = image.load_img(ruta_imagen, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Mantenemos la normalización que confirmaste que usas
        img_array = img_array / 255.0

        # 4. Hacer la predicción (devuelve los "logits")
        predicciones_raw = model.predict(img_array)

        # --- CAMBIO AQUÍ: Convertimos los logits a porcentajes reales con Softmax ---
        probabilidades = tf.nn.softmax(predicciones_raw).numpy()

        # Obtener el índice con la probabilidad más alta usando el array corregido
        clase_predicha_idx = np.argmax(probabilidades)
        nombre_clase = CLASSES[clase_predicha_idx]

        # Extraemos la confianza real de la clase ganadora multiplicando por 100
        confianza = probabilidades[0][clase_predicha_idx] * 100

        print(f"\n--- Resultado de la Clasificación ---")
        print(f"Predicción: {nombre_clase.upper()}")
        print(f"Confianza: {confianza:.2f}%")

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")


# ---- EJECUCIÓN ----
# Pon aquí el nombre de la imagen de tu gatito.
# Recuerda guardarla también en la raíz junto a este script.
ruta_de_tu_imagen = "data/boat.jpeg"
clasificar_imagen(ruta_de_tu_imagen)