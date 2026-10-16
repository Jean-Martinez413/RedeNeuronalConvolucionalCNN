# Clasificador de Imágenes con CNN

## Descripción

Proyecto de clasificación de imágenes mediante una red neuronal convolucional (CNN), desarrollada con Python, TensorFlow y Keras.

El modelo se entrena con el dataset CIFAR-10 y permite clasificar imágenes en diez categorías, mostrando la clase predicha y su nivel de confianza.

## Características

- Entrenamiento de una red neuronal convolucional.
- Normalización y procesamiento de imágenes.
- Evaluación del modelo mediante métricas de precisión.
- Generación de una gráfica del aprendizaje.
- Guardado y reutilización del modelo entrenado.
- Selección de imágenes mediante una ventana gráfica.

## Tecnologías

- Python
- TensorFlow y Keras
- NumPy
- Matplotlib
- Pillow
- Tkinter

## Estructura del proyecto

```
redNeuronal/
├── data/
├── entornoRedNeuronal/
├── models/
├── results/
├── src/
├── .gitignore
├── main.py
├── probar_modelo.py
├── requirements.txt
└── README.md
```

## Dataset

Se utiliza CIFAR-10, un conjunto de 60.000 imágenes RGB de 32 × 32 píxeles, distribuidas en diez clases:

Avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.

El dataset contiene 50.000 imágenes de entrenamiento y 10.000 de prueba.

## Arquitectura y entrenamiento

La red neuronal incluye tres capas convolucionales, capas de agrupación MaxPooling y capas densas para la clasificación final.

- Optimizador: Adam.
- Función de pérdida: Sparse Categorical Crossentropy.
- Métrica: Accuracy.
- Épocas: 5.
- Salidas: 10 clases.

## Instalación y ejecución

1. Clonar el repositorio

git clone https://github.com/Jean-Martinez413/RedeNeuronalConvolucionalCNN.git

cd RedeNeuronalConvolucionalCNN

2. Crear un entorno virtual

python -m venv entornoRedNeuronal

3. Activar el entorno virtual (Windows)

PowerShell:

.\entornoRedNeuronal\Scripts\Activate.ps1

CMD:

entornoRedNeuronal\Scripts\activate.bat

4. Instalar las dependencias

python -m pip install -r requirements.txt

5. Entrenar el modelo

python main.py

6. Clasificar una imagen

python probar_modelo.py

El programa permitirá seleccionar una imagen y mostrará la categoría predicha junto con su nivel de confianza.


## Resultados

- Modelo entrenado: `models/modelo_red_neuronal.keras`
- Gráfica de aprendizaje: `results/resultado_aprendizaje.png`

La precisión obtenida dependerá del entrenamiento realizado y de las imágenes evaluadas.

**Nota:** el modelo está diseñado para las diez categorías de CIFAR-10. La confianza de una predicción no garantiza que sea correcta.

## Limitaciones

El modelo se entrena durante solo cinco épocas, por lo que existe margen para experimentar con una mayor cantidad de épocas.

Las imágenes originales de CIFAR-10 son pequeñas y tienen características específicas.

Las imágenes externas pueden presentar diferencias de resolución, iluminación, encuadre y fondo.

La confianza calculada mediante Softmax no garantiza que una predicción sea correcta.

El modelo solo puede asignar las imágenes a una de las diez categorías conocidas; no está diseñado para reconocer cualquier objeto posible.

## Mejoras futuras

Experimentar con más épocas y diferentes hiperparámetros.

Implementar aumento de datos (data augmentation).

Comparar distintas arquitecturas convolucionales.

Incorporar una matriz de confusión y métricas por categoría.

Mostrar las probabilidades correspondientes a las diez clases.

Desarrollar una interfaz gráfica para presentar las predicciones.

Separar correctamente los conjuntos de entrenamiento, validación y prueba.

Agregar pruebas automatizadas y mejorar la modularización del código.