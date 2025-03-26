import cv2
import matplotlib.pyplot as plt

# Cargar el clasificador pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ruta de la imagen
ruta_imagen = r"C:\Users\User\Downloads\fm.jpg"

# Cargar la imagen
imagen = cv2.imread(ruta_imagen)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
else:
    # Convertir a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    rostros = face_cascade.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos en los rostros detectados
    for (x, y, w, h) in rostros:
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Guardar imagen procesada
    cv2.imwrite('resultado.jpg', imagen)

    # Mostrar imagen usando matplotlib
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib
    plt.imshow(imagen_rgb)
    plt.axis("off")  # Ocultar ejes
    plt.title("Detección Facial")
    plt.show()
