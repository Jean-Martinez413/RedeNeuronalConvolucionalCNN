import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os 


os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================================
# a. Cargar el conjunto de datos (CIFAR-10)
# ==========================================
print("Descargando y cargando el dataset CIFAR-10...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar los valores de los píxeles para que estén entre 0 y 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Nombres de las 10 clases del dataset
class_names = ['avion', 'auto', 'pajaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camion']

# ==========================================
# Construcción la red neuronal CNN
# ==========================================
print("Construyendo el modelo CNN...")
model = models.Sequential([
    # Primera capa convolucional y de agrupación
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Segunda capa convolucional y de agrupación
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Tercera capa convolucional
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Capas densas (Clasificación final)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10) # 10 salidas porque hay 10 tipos de imágenes
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Mostrar la estructura en consola
model.summary()

# ==========================================
# Realizar el proceso de aprendizaje (Entrenamiento)
# ==========================================
print("Iniciando el entrenamiento (proceso de aprendizaje)...")
# Usaremos 5 épocas (vueltas de entrenamiento) 
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

# ==========================================
# Realizar la validación del modelo
# ==========================================
print("\nEvaluando el modelo con datos de prueba...")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nPrecisión final del modelo en validación: {test_acc*100:.2f}%')

# Guardar una gráfica de la evolución del aprendizaje en tu carpeta del proyecto
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label = 'Precisión Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig("results/resultado_aprendizaje.png")
print("Gráfica de rendimiento guardada en 'results/resultado_aprendizaje.png'")

# --- Guardar el modelo entrenado ---
print("\nGuardando el modelo...")
model.save("models/modelo_red_neuronal.keras")
print("¡Modelo guardado con éxito en 'models/modelo_red_neuronal.keras'!")
