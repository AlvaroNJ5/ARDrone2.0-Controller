import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import tempfile
from utils import *

from pyardrone import ARDrone
import logging





'''
Nota de funcionamiento:

- Cuando se ejecuta tardará un rato en aparecer una ventana con la imágen de la webcam. Inicialmente estará en estado de espera.
- Si se presiona la tecla 'a' (OJO: estando activa la ventana de la imagen) se inicia el proceso de aprender una cara:
    > En la terminal te pedirá que introduzcas el nombre de la persona a reconocer.
    > Una vez introducido, en la ventana con la imagen empezará una cuenta atrás de 5 segundos para que te puedas colocar bien.
    > Cuando se acaba la cuenta atrás se toma una foto de la cara y se guarda en la carpeta 'images_prueba'.
- Una vez aprendida la cara se pulsa la tecla 'r' para iniciar el reconocimiento (con la ventana de la imágen activa):
    > Saldrá un cuadro verde y el nombre de la persona reconocida si se reconoce.
    > Si no se reconoce saldrá un cuadro rojo sobre la cara de la persona.
- En cualquier momento se podrá aprender una nueva cara pulsando la tecla 'a' y al volver a pulsar 'r' se reconocerá la nueva persona.
- Si hay varias personas aprendidas, se puede seleccionar a quién se desea reconocer pulsando la tecla 'c'.
    > Te pedirá el nombre de a quien se desea reconocer en la terminal.
    > Si el nombre es correcto comenzará el reconocimiento directamente.
    > Si el nombre no es correcto, te lo pedirá de nuevo.
- Al ejecutar el código de nuevo, no se puede pulsar 'r' para reconocer directamente en caso de que ya se tenga a la persona 
aprendida de otras ejecuciones. Se deberá aprender de nuevo otra persona con la tecla 'a' o elegir a la persona que ya 
se tenga de las veces anteriores con la tecla 'c'.

- Pulsando la tecla 'q' se cierra la ventana con la imágen y se acaba la ejecución.

IMPORTANTE: Hay que llevar cuidado con las teclas. Cuando hay que pulsar las letras 'a','c','r' o 'q' se 
debe hacer con la ventana de la imagen de la webcam activa. Cuando haya que introducir un nombre 
en la terminal hay que asegurarse de seleccionar la terminal (ya que si por ejemplo, en la terminal te pide que pongas un 
nombre y teniendo la ventana de la imagen activa escribes alvaro, al pulsar la tecla 'a' se iniciará de nuevo el proceso 
de aprendizaje o directamente dará error, por lo que, antes de escribir hay que hacer click en la terminal).

'''

'''
NOTA: En esta versión del código he intentado adapatarlo para trabajar con la imágen del drone en vez 
de la webcam en función de lo que he visto en los ejemplos que hay en el repositorio pero no estoy seguro de que funcione
ya que sin el drone no lo puedo probar.

'''

w_target = 224
h_target = 224
a_target = w_target*h_target
pid1 = [0.2, 0.2, 0]     #kp, kd, ki
pid2 = [0.3, 0.3, 0]
pid3 = [0.3, 0.3, 0]
pError1 = 0
pError2 = 0
pError3 = 0

def dibujar_cuenta_atrás(frame: np.ndarray, countdown: int) -> None:
    'Función que dibuja la cuenta atrás en la imágen'
    # Obtenemos las dimensiones del frame
    height, width, _ = frame.shape

    # Calculamos las coordenadas x e y para poner la cuenta (centro de la imágen)
    text_size, _ = cv2.getTextSize(str(countdown), cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Dibujamos la cuenta atrás
    cv2.putText(frame, str(countdown), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

def guardar_imagen(frame: np.ndarray, path: str, name: str) -> str:
    'Función para guardar la imágen con la que se va a hacer el reconocimiento'
    global capture_image, start_time

    if start_time is None:
        start_time = time.time()

    elapsed_time = time.time() - start_time
    countdown_timer = 5
    if elapsed_time < countdown_timer:
        dibujar_cuenta_atrás(frame, int(countdown_timer - elapsed_time))
    else:
        if not capture_image:
            os.makedirs(path, exist_ok=True)
            faces = detector_caras.detectMultiScale(frame, 1.3, 5)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    img_path = os.path.join(path, f'{name}.jpg')
                    cv2.imwrite(img_path, frame[y:y+h, x:x+w])  # Guardar solo la región de la cara
                    capture_image = True  # Marcar que la imagen ha sido capturada
                    return img_path  # Devolver la ruta completa del archivo de imagen guardado


def preprocess_image(image_path: str, target_size=(w_target, h_target)) -> np.ndarray:
    'Función de preprocesamiento de imágen (ajustarlas al mismo tamaño antes de comparar)'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return None
    resized_image = cv2.resize(image, target_size)
    return resized_image

def image_comparator(file1: str, file2: str) -> float:
    'Comparación de las caras detectadas con la aprendida'
    # Preprocess images
    processed_images = [preprocess_image(filename) for filename in [file1, file2]]

    # Check if any image failed to load
    if any(image is None for image in processed_images):
        print("Error: Al menos una de las imágenes no se pudo cargar correctamente.")
        return None

    # Create options for Image Embedder
    base_options = python.BaseOptions(model_asset_path='embedder.tflite')
    l2_normalize = True
    quantize = True
    options = vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)

    # Create Image Embedder
    with vision.ImageEmbedder.create_from_options(options) as embedder:
        # Format images for MediaPipe
        first_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_images[0])
        second_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_images[1])
        first_embedding_result = embedder.embed(first_image)
        second_embedding_result = embedder.embed(second_image)

        # Calculate and print similarity
        similarity = vision.ImageEmbedder.cosine_similarity(
            first_embedding_result.embeddings[0],
            second_embedding_result.embeddings[0])
    return similarity

# Ajustes iniciales

drone = init_drone()

detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0) # Ya no hace falta, es la webcam
capture_image = False
start_time = None
nameID = None
estado = 'esperando'
mensaje_mostrado = False  # Flag para controlar si se ha mostrado el mensaje de inicio de reconocimiento

try:
    while True:
        frame = drone.frame
        if frame is None:  # Tengo dudas de que esto no de problemas en caso de conexión inestable (si da problemas habria que buscar una forma de manejo de errores)
            break
        if estado == 'esperando':
            cv2.putText(frame, 'Estado: Esperando', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif estado == 'aprendiendo':
            
            cv2.putText(frame, 'Estado: Aprendiendo', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if nameID is None:
                nameID = str(input("Introduce tu nombre: ")).lower()
                path = 'imagenes_prueba/' + nameID      
                if not os.path.exists(path):
                    os.makedirs(path)

            image_path = guardar_imagen(frame, path, nameID)
            if image_path is not None:
                print("Imagen guardada. Presione 'r' para iniciar el reconocimiento.")
                mensaje_mostrado = False

        elif estado == 'reconociendo':
            if not mensaje_mostrado:
                print("Iniciando reconocimiento...")
                mensaje_mostrado = True
                print('\n\n\nTAKEOFF STARTING')
                print('TAKEOFF STARTING')
                print('TAKEOFF STARTING')
                print('TAKEOFF STARTING\n\n\n')
                time.sleep(10)
                takeoff_control(drone)

            cv2.putText(frame, 'Estado: Reconociendo', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            faces = detector_caras.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Reinicializar la información de la cara si no se detectan caras en la imagen (en caso de que se quiera trabajar asi para el control, es decir, si no hay cara el área de la cara es 0 y la posicion centrada)
            cx, cy, area = 0, 0, 0

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w] 
                # Guardar la imagen capturada temporalmente
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img_file:
                    temp_img_path = temp_img_file.name
                    cv2.imwrite(temp_img_path, face_img)
                similarity = image_comparator(os.path.join(path, f'{nameID}.jpg'), temp_img_path)
                os.unlink(temp_img_path)  # Eliminar la imagen temporal
                if similarity is not None:
                    print(f"Similitud: {similarity}")
                if similarity is not None and similarity > 0.5:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Dibujar rectángulo verde si la similitud es alta
                    cv2.putText(frame, nameID, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) #Mostrar el nombre de la persona
                    # Para el control PID guardamos las variables útiles (centro de la cara (x e y) y área de la cara)
                    cx = x + w//2
                    cy = y + h//2
                    area = w*h
                    trackFace(drone, cx, w_target, pid1, pError1)
                    height_control(drone, cy, h_target, pid2, pError2)
                    #dist_control(drone, area, a_target, pid3, pError3)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Dibujar rectángulo rojo si la similitud es baja
                    drone.hover()

        elif estado == 'cambiar_objetivo':
            nuevo_objetivo = input('A quién deseas reconocer: ').strip().lower()
            if os.path.isfile(f'imagenes_prueba/{nuevo_objetivo}/{nuevo_objetivo}.jpg'):
                path = 'imagenes_prueba/'+nuevo_objetivo
                nameID = nuevo_objetivo
                print(f"Objetivo cambiado. Comienza el reconocimiento")
                estado = 'reconociendo'      
            else:
                print("La persona seleccionada no existe. Por favor, introduzca el nombre de nuevo.")
                
        

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            if estado != 'aprendiendo':
                estado = 'aprendiendo'
                start_time = None
                nameID = None
                capture_image = False
        elif key == ord('r'):
            estado = 'reconociendo'
        elif key == ord('c'):
            estado = 'cambiar_objetivo'

finally:
    drone.close()
    cv2.destroyAllWindows()
