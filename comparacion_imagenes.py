import cv2
import face_recognition as fr

# Cargar las imágenes
imagen_control = fr.load_image_file('Cliente_Control.jpg')
imagen_prueba = fr.load_image_file('Cliente_Prueba.jpg')

# Convertir las imágenes de BGR a RGB
imagen_control = cv2.cvtColor(imagen_control, cv2.COLOR_BGR2RGB)
imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2RGB)

# Localizar los rostros en las imágenes
loc_control = fr.face_locations(imagen_control)[0]
loc_prueba = fr.face_locations(imagen_prueba)[0]

# Codificar las características faciales
enc_control = fr.face_encodings(imagen_control)[0]
enc_prueba = fr.face_encodings(imagen_prueba)[0]

# Comparar los rostros
resultado = fr.compare_faces([enc_control], enc_prueba)
distancia = fr.face_distance([enc_control], enc_prueba)

# Mostrar resultados
print(f"¿Es la misma persona? {resultado[0]}")
print(f"Distancia de similitud: {distancia[0]}")

# Mostrar las imágenes con rectángulos alrededor del rostro
cv2.rectangle(imagen_control, (loc_control[3], loc_control[0]), (loc_control[1], loc_control[2]), (0,255,0), 2)
cv2.rectangle(imagen_prueba, (loc_prueba[3], loc_prueba[0]), (loc_prueba[1], loc_prueba[2]), (0,255,0), 2)

cv2.imshow('Imagen de Control', imagen_control)
cv2.imshow('Imagen de Prueba', imagen_prueba)
cv2.waitKey(0)
