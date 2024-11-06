import cv2
import face_recognition as fr
import numpy as np
import os
from datetime import datetime
import csv
from fer import FER  # Librería para reconocimiento de emociones
import pandas as pd # Librería para manejo de datos

# Ruta a la carpeta con imágenes de clientes
ruta_clientes = 'Clientes_Imagenes'
if not os.path.exists(ruta_clientes):
    os.makedirs(ruta_clientes)

imagenes = []
nombres_clientes = []

# Cargar las imágenes y los nombres de los clientes
lista_clientes = os.listdir(ruta_clientes)
for nombre in lista_clientes:
    imagen_actual = cv2.imread(f'{ruta_clientes}/{nombre}')
    if imagen_actual is not None:
        imagenes.append(imagen_actual)
        nombres_clientes.append(os.path.splitext(nombre)[0])
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

            if coincidencias[indice_mejor_coincidencia]:
                nombre = nombres_clientes[indice_mejor_coincidencia].upper()
            else:
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
                print(f"El nombre de la persona reconocida coincide correctamente: {nombre}. Puede ingresar al gimnasio.")
                reconocimiento_exitoso = True
                break  # Salir del bucle for

        # Mostrar el frame en la ventana
        cv2.imshow('Sistema de Asistencias con Reconocimiento de Emociones', frame)

        # Verificar si se ha realizado un reconocimiento exitoso
        if reconocimiento_exitoso:
            # Esperar un momento antes de cerrar para que se vea el recuadro en la ventana
            cv2.waitKey(2000)  # Espera 2000 milisegundos (2 segundos)
            break  # Salir del bucle while

        # Permitir salir del programa presionando la tecla 'ESC'
        if cv2.waitKey(1) == 27:
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Función para registrar nuevos clientes
def registrar_nuevo_cliente():
    global imagenes, nombres_clientes, codificaciones_conocidas
    nombre = input("Ingrese el nombre completo del nuevo cliente: ").strip()
    if nombre == '':
        print("El nombre no puede estar vacío.")
        return

    if nombre in nombres_clientes:
        print("Este cliente ya está registrado.")
        return

    print("Por favor, mire a la cámara. Se tomarán 3 fotos para registrar su rostro.")

    cap = cv2.VideoCapture(0)
    fotos_tomadas = 0
    imagenes_nuevas = []

    while fotos_tomadas < 3:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        cv2.imshow('Captura de Rostro - Presione Espacio para Tomar Foto', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            ruta_imagen = f'{ruta_clientes}/{nombre}_{fotos_tomadas + 1}.jpg'
            cv2.imwrite(ruta_imagen, frame)
            print(f"Imagen {fotos_tomadas + 1} guardada.")
            imagenes_nuevas.append(frame)
            fotos_tomadas += 1

        if cv2.waitKey(1) == 27:
            print("Registro cancelado por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if fotos_tomadas == 3:
        # Añadir las nuevas imágenes a las codificaciones conocidas
        for idx, imagen in enumerate(imagenes_nuevas):
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            codificacion = fr.face_encodings(imagen_rgb)
            if len(codificacion) > 0:
                imagenes.append(imagen)
                nombres_clientes.append(nombre)
                codificaciones_conocidas.append(codificacion[0])
            else:
                print(f"Advertencia: No se detectó ningún rostro en la imagen {idx + 1}.")
        print(f"Cliente {nombre} registrado exitosamente.")
    else:
        print("No se pudieron tomar las fotos necesarias.")

# Función para ver el historial de asistencia de un cliente
def ver_historial_cliente():
    nombre_cliente = input("Ingrese el nombre del cliente: ").strip().upper()
    try:
        # Cargar el archivo CSV de asistencias
        df = pd.read_csv('Asistencias.csv', names=['Nombre', 'Fecha y Hora', 'Emocion'])
        # Filtrar las asistencias por el nombre del cliente
        historial = df[df['Nombre'] == nombre_cliente]
        if historial.empty:
            print(f"No se encontraron asistencias para el cliente: {nombre_cliente}.")
        else:
            print(f"\nHistorial de asistencias para {nombre_cliente}:")
            print(historial.to_string(index=False))
    except FileNotFoundError:
        print("Error: No se encontró el archivo de asistencias.")
    except Exception as e:
        print(f"Ocurrió un error al obtener el historial: {e}")

# Función para analizar la frecuencia de asistencia de un cliente
def analizar_frecuencia_asistencia():
    nombre_cliente = input("Ingrese el nombre del cliente para análisis de frecuencia: ").strip().upper()
    try:
        # Cargar el archivo CSV de asistencias
        df = pd.read_csv('Asistencias.csv', names=['Nombre', 'Fecha y Hora', 'Emocion'])
        # Filtrar las asistencias por el nombre del cliente
        historial = df[df['Nombre'] == nombre_cliente]
        if historial.empty:
            print(f"No se encontraron asistencias para el cliente: {nombre_cliente}.")
            return

        # Convertir la columna de fecha y hora a formato datetime
        historial['Fecha y Hora'] = pd.to_datetime(historial['Fecha y Hora'])

        # Calcular frecuencia semanal y mensual
        frecuencia_semanal = historial['Fecha y Hora'].dt.isocalendar().week.value_counts().mean()
        frecuencia_mensual = historial['Fecha y Hora'].dt.month.value_counts().mean()

        print(f"\nFrecuencia de asistencia para {nombre_cliente}:")
        print(f"Promedio semanal: {frecuencia_semanal:.2f} veces por semana.")
        print(f"Promedio mensual: {frecuencia_mensual:.2f} veces por mes.")
        
    except FileNotFoundError:
        print("Error: No se encontró el archivo de asistencias.")
    except Exception as e:
        print(f"Ocurrió un error al analizar la frecuencia: {e}")

# Menú principal
def menu_principal():
    while True:
        print("\n.:Bienvenido al Sistema de Asistencias:.")
        print("Seleccione una opción:")
        print("1. Registrar Asistencia")
        print("2. Registrar Nuevo Cliente")
        print("3. Ver Historial de Cliente")
        print("4. Analizar Frecuencia de Asistencia de Cliente")
        print("5. Salir")
        opcion = input("Opción (1/2/3/4/5): ").strip()
        if opcion == '1':
            registrar_asistencias()
        elif opcion == '2':
            registrar_nuevo_cliente()
        elif opcion == '3':
            ver_historial_cliente()
        elif opcion == '4':
            analizar_frecuencia_asistencia()
        elif opcion == '5':
            print("Gracias por utilizar el sistema.")
            break
        else:
            print("Opción inválida. Por favor, intente de nuevo.")

# Ejecutar el programa
if __name__ == "__main__":
    # Codificar las imágenes de los clientes al inicio
    codificaciones_conocidas = codificar_imagenes(imagenes)
    print('Codificaciones iniciales completadas.')
    menu_principal()
