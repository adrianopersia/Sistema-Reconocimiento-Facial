import cv2
import face_recognition as fr
import numpy as np
import os
from datetime import datetime
import csv
from fer import FER  # Librería para reconocimiento de emociones
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from ttkthemes import ThemedTk


# Ruta a la carpeta con imágenes de clientes
ruta_clientes = 'Clientes_Imagenes'
if not os.path.exists(ruta_clientes):
    os.makedirs(ruta_clientes)

imagenes = []
nombres_clientes = []

# Cargar las imágenes y los nombres de los clientes
lista_clientes = os.listdir(ruta_clientes)
nombres_unicos = set()  # Para almacenar nombres únicos

for nombre in lista_clientes:
    imagen_actual = cv2.imread(f'{ruta_clientes}/{nombre}')
    if imagen_actual is not None:
        imagenes.append(imagen_actual)

        # Extraer el nombre base (antes del guion bajo si existe)
        nombre_base = nombre.split('_')[0]
        
        # Agregar a la lista solo si no se ha agregado antes
        if nombre_base not in nombres_unicos:
            nombres_clientes.append(nombre_base)
            nombres_unicos.add(nombre_base)
    else:
        print(f"Advertencia: No se pudo cargar la imagen {nombre}.")

# Función para codificar las imágenes
def codificar_imagenes(imagenes):
    lista_codificaciones = []
    for img in imagenes:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        codificacion = fr.face_encodings(img_rgb)
        if len(codificacion) > 0:
            lista_codificaciones.append(codificacion[0])
        else:
            print("Advertencia: No se detectó ningún rostro en una de las imágenes.")
    return lista_codificaciones

# Registrar asistencia en el archivo CSV
def registrar_asistencia(nombre, emocion):
    with open('Asistencias.csv', 'a', newline='') as archivo:
        escritor = csv.writer(archivo)
        ahora = datetime.now()
        cadena_tiempo = ahora.strftime('%Y-%m-%d %H:%M:%S')
        escritor.writerow([nombre, cadena_tiempo, emocion])

# Inicializar el detector de emociones
detector_emociones = FER(mtcnn=True)

# Función para registrar asistencias
def registrar_asistencias():
    global imagenes, nombres_clientes, codificaciones_conocidas
    # Verificar si hay clientes registrados
    if not nombres_clientes:
        messagebox.showinfo("Información", "No hay clientes registrados todavía. Por favor registre un nuevo cliente.")
        return  # Salir de la función si no hay clientes

    # Codificar las imágenes de los clientes
    codificaciones_conocidas = codificar_imagenes(imagenes)
    print('Codificaciones completadas.')

    # Inicializar la captura de video
    cap = cv2.VideoCapture(0)

    # Variable de control para el reconocimiento exitoso
    reconocimiento_exitoso = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reducir tamaño del frame para acelerar el procesamiento
        frame_pequeño = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frame_rgb = cv2.cvtColor(frame_pequeño, cv2.COLOR_BGR2RGB)

        # Detectar rostros para reconocimiento facial
        loc_rostros = fr.face_locations(frame_rgb)
        cod_rostros = fr.face_encodings(frame_rgb, loc_rostros)

        # Detectar emociones en el frame original
        resultados_emociones = detector_emociones.detect_emotions(frame)

        for cod_rostro, loc_rostro in zip(cod_rostros, loc_rostros):
            coincidencias = fr.compare_faces(codificaciones_conocidas, cod_rostro)
            distancias = fr.face_distance(codificaciones_conocidas, cod_rostro)
            indice_mejor_coincidencia = np.argmin(distancias)

            # Escalar las coordenadas al tamaño original
            y1, x2, y2, x1 = loc_rostro
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            # Calcular el índice del cliente
            indice_cliente = indice_mejor_coincidencia // 3  # Dividimos entre 3 para obtener el índice del cliente

            # Comprobamos que el índice no se pase del tamaño de la lista
            if indice_cliente < len(nombres_clientes):
                # Si hay coincidencia, usamos el índice del cliente y su nombre correspondiente
                nombre = nombres_clientes[indice_cliente].upper()
            else:
                # Si no hay coincidencia, se asigna 'Desconocido'
                nombre = 'Desconocido'

            emocion_detectada = ''
            # Encontrar la emoción correspondiente al rostro detectado
            for resultado in resultados_emociones:
                (x, y, w, h) = resultado["box"]
                # Verificar si las coordenadas del rostro se solapan
                if x1 < x + w/2 < x2 and y1 < y + h/2 < y2:
                    emocion, puntuacion = max(resultado["emotions"].items(), key=lambda item: item[1])
                    emocion_detectada = emocion.capitalize()
                    break

            # Dibujar el rectángulo y la información
            color_rectangulo = (0, 255, 0) if nombre != 'Desconocido' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_rectangulo, 2)
            texto_mostrar = f'{nombre} {emocion_detectada}'
            cv2.putText(frame, texto_mostrar, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color_rectangulo, 2)

            # Registrar asistencia y cerrar el programa si es un cliente conocido
            if nombre != 'Desconocido':
                registrar_asistencia(nombre, emocion_detectada)
                # Mostrar mensaje en consola y cerrar el programa
                reconocimiento_exitoso = True
                break  # Salir del bucle for

        # Mostrar el frame en la ventana
        cv2.imshow('Sistema de Asistencias con Reconocimiento de Emociones', frame)

        # Verificar si se ha realizado un reconocimiento exitoso
        if reconocimiento_exitoso:
            # Esperar un momento antes de cerrar para que se vea el recuadro en la ventana
            cv2.waitKey(8000)  # Espera 5000 milisegundos (2 segundos)
            break  # Salir del bucle while

        # Permitir salir del programa presionando la tecla 'ESC'
        if cv2.waitKey(1) == 27:
            break


    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

    return reconocimiento_exitoso

def registrar_nuevo_cliente(nombre):
    global imagenes, nombres_clientes, codificaciones_conocidas

    codificaciones_conocidas = codificar_imagenes(imagenes)
    print('Codificaciones completadas.')

    if nombre == '':
        print("El nombre no puede estar vacío.")
        return "El nombre no puede estar vacío."

    if nombre in nombres_clientes:
        print("Ya existe un cliente registrado con este nombre.")
        return "Ya existe un cliente registrado con este nombre."

    print("Por favor, mire a la cámara. Se tomarán 3 fotos para registrar su rostro.")
    
    # Configuración de la cámara
    cap = cv2.VideoCapture(0)
    fotos_tomadas = 0
    imagenes_nuevas = []

    # Captura de imágenes
    while fotos_tomadas < 3:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            cap.release()
            cv2.destroyAllWindows()
            return "No se pudo acceder a la cámara."

        cv2.imshow('Captura de Rostro - Presione Espacio para Tomar Foto', frame)
        
        # Captura de imagen cuando se presiona la tecla espacio
        if cv2.waitKey(1) & 0xFF == ord(' '):
            imagenes_nuevas.append(frame)
            fotos_tomadas += 1

        # Cancelar captura con tecla ESC
        if cv2.waitKey(1) == 27:
            print("Registro cancelado por el usuario.")
            cap.release()
            cv2.destroyAllWindows()
            return "Registro cancelado por el usuario."

    cap.release()
    cv2.destroyAllWindows()

    if fotos_tomadas != 3:
        print("No se pudieron tomar las fotos necesarias.")
        return "No se pudieron tomar las fotos necesarias."

    # Procesamiento de las imágenes capturadas
    for idx, imagen in enumerate(imagenes_nuevas):
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificacion = fr.face_encodings(imagen_rgb)

        if len(codificacion) > 0:
            # Verificar si la codificación ya existe en los registros
            coincidencias = fr.compare_faces(codificaciones_conocidas, codificacion[0])
            if True in coincidencias:
                print("El usuario ya existe en los registros.")
                return "El usuario ya existe en los registros."
        else:
            print(f"Advertencia: No se detectó ningún rostro en la imagen {idx + 1}.")
            return f"Advertencia: No se detectó ningún rostro en la imagen {idx + 1}."

    # Guardar las imágenes solo si el usuario no existe
    for idx, imagen in enumerate(imagenes_nuevas):
        ruta_imagen = f'{ruta_clientes}/{nombre}_{idx + 1}.jpg'
        cv2.imwrite(ruta_imagen, imagen)
        print(f"Imagen {idx + 1} guardada en la carpeta Clientes_Imagenes.")

    # Agregar nueva información
    imagenes.extend(imagenes_nuevas)
    nombres_clientes.append(nombre)
    codificaciones_conocidas.append(codificacion[0])

    # Guardar el nombre del usuario en el archivo CSV
    with open("usuarios.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([nombre])

    print(f"Cliente {nombre} registrado exitosamente.")
    return f"Cliente {nombre} registrado exitosamente."



# Ejecutar el programa
if __name__ == "__main__":
    
    import tkinter as tk
    from tkinter import messagebox
    from interfaz_asistencia import SistemaAsistenciaGimnasio

    # Codificar las imágenes de los clientes al inicio
    codificaciones_conocidas = codificar_imagenes(imagenes)
    print('Codificaciones iniciales completadas.')
    # menu_principal()
    root = ThemedTk(theme="arc")
    app = SistemaAsistenciaGimnasio(root, nombres_clientes)
    root.mainloop()
