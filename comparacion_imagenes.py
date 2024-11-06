import cv2
import face_recognition as fr

# Función para cargar y procesar una imagen
def cargar_y_procesar_imagen(ruta):
    imagen = fr.load_image_file(ruta)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    ubicaciones = fr.face_locations(imagen)
    codificaciones = fr.face_encodings(imagen, ubicaciones)
    return imagen, ubicaciones, codificaciones

# Cargar y procesar ambas imágenes 
imagen_control, loc_control, enc_control = cargar_y_procesar_imagen(r'C:\Users\matie\dev\sistema-de-reconocimiento-facial\Sistema-Reconocimiento-Facial\Cliente_Control.jpg')
# imagen_prueba, loc_prueba, enc_prueba = cargar_y_procesar_imagen(r'C:\Users\matie\dev\sistema-de-reconocimiento-facial\Sistema-Reconocimiento-Facial\Cliente_Prueba.jpg')
# imagen_prueba, loc_prueba, enc_prueba = cargar_y_procesar_imagen(r'C:\Users\matie\dev\sistema-de-reconocimiento-facial\Sistema-Reconocimiento-Facial\Clientes_Prueba.jpg')
imagen_prueba, loc_prueba, enc_prueba = cargar_y_procesar_imagen(r'C:\Users\matie\dev\sistema-de-reconocimiento-facial\Sistema-Reconocimiento-Facial\Clientes_PruebaG.jpg')

# Iterar sobre cada rostro en la imagen de control y lo compara con cada rostro en la imagen de prueba
umbral = 0.6  # Umbral de similitud
for i, (ubicacion_control, cod_control) in enumerate(zip(loc_control, enc_control)):
    for j, (ubicacion_prueba, cod_prueba) in enumerate(zip(loc_prueba, enc_prueba)):
        resultado = fr.compare_faces([cod_control], cod_prueba, tolerance=umbral)
        distancia = fr.face_distance([cod_control], cod_prueba)
        
        # Mostrar resultados
        print(f"Comparación rostro {i+1} de la imagen de control con rostro {j+1} de la imagen de prueba:")
        print(f"¿Es la misma persona? {'Sí' if resultado[0] else 'No'}")
        print(f"Distancia de similitud: {distancia[0]:.2f}")
        
        # Añadir resultados y rectángulos
        cv2.rectangle(imagen_control, (ubicacion_control[3], ubicacion_control[0]), (ubicacion_control[1], ubicacion_control[2]), (0, 255, 0), 2)
        cv2.rectangle(imagen_prueba, (ubicacion_prueba[3], ubicacion_prueba[0]), (ubicacion_prueba[1], ubicacion_prueba[2]), (0, 255, 0), 2)
        cv2.putText(imagen_prueba,
                    f"Rostro {i+1} - {('Similitud' if resultado[0] else 'Diferente')} ({distancia[0]:.2f})",
                    (ubicacion_prueba[3], ubicacion_prueba[0] - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 255, 0) if resultado[0] else (0, 0, 255),
                    1)

# Mostrar las img
cv2.imshow('Imagen de Control', imagen_control)
cv2.imshow('Imagen de Prueba', imagen_prueba)
cv2.waitKey(0)
