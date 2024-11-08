import tkinter as tk
from threading import Thread
import cv2
from sistema_asistencias import registrar_asistencias
from tkinter import ttk
import csv
from tkinter import messagebox
import os

class SistemaAsistenciaGimnasio:
    def __init__(self, root, nombres_clientes):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial - Gimnasio")
        self.root.geometry("500x500")
        self.nombres_clientes = nombres_clientes  # Guardar la lista de clientes

        if not os.path.exists("usuarios.csv"):
            with open("usuarios.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Nombre"])  # Escribe el encabezado en el archivo

        # Agregar un título en la ventana principal
        self.titulo = tk.Label(self.root, text="Sistema de Reconocimiento Facial", font=("Arial", 16, "bold"))
        self.titulo.pack(pady=20)

        self.label_mensaje = tk.Label(self.root, text="")
        self.label_mensaje.pack(pady=10)

        # Botones de la interfaz
        btn_asistencia = tk.Button(self.root, text="Registrar asistencia", command=self.registrar_asistencia_y_mensaje)
        btn_asistencia.pack(pady=10)
        
        btn_nuevo_usuario = tk.Button(self.root, text="Registrar un nuevo usuario", command=self.abrir_ventana_nuevo_usuario)
        btn_nuevo_usuario.pack(pady=10)

        btn_ver_asistencias = tk.Button(self.root, text="Ver registro de asistencias", command=self.ver_asistencias)
        btn_ver_asistencias.pack(pady=10)

        btn_ver_usuarios = tk.Button(self.root, text="Ver usuarios", command=self.ver_usuarios)
        btn_ver_usuarios.pack(pady=10)

        btn_cerrar = tk.Button(self.root, text="Cerrar", command=self.root.quit)
        btn_cerrar.pack(pady=10)
    
    def registrar_asistencia_y_mensaje(self):
        # Ejecuta el registro de asistencia y muestra un mensaje de confirmación
        usuario = registrar_asistencias()
        print(usuario)
        if usuario:
            mensaje = f"Asistencia registrada correctamente - Puede ingresar al gimnasio."
        else:
            mensaje = "Error al registrar la asistencia."

        # Actualiza el label en la ventana principal
        self.label_mensaje.config(text=mensaje)
    
    def abrir_ventana_nuevo_usuario(self):
        ventana = tk.Toplevel(self.root)
        ventana.title("Registrar nuevo usuario")
        ventana.geometry("300x200")
        
        label_nombre = tk.Label(ventana, text="Ingrese el nombre del nuevo usuario:")
        label_nombre.pack(pady=10)

        entry_nombre = tk.Entry(ventana)
        entry_nombre.pack(pady=5)

        def confirmar_registro():
            nombre = entry_nombre.get()
            from sistema_asistencias import registrar_nuevo_cliente
            resultado = registrar_nuevo_cliente(nombre)
            if resultado == "El usuario ya existe en los registros.":
                messagebox.showinfo("Información", resultado)  # Mensaje en pantalla principal
            else:
                messagebox.showinfo("Información", resultado)  # Mensaje de éxito o error específico
            ventana.destroy()

        btn_confirmar = tk.Button(ventana, text="Confirmar", command=confirmar_registro)
        btn_confirmar.pack(pady=10)

    def ver_asistencias(self):
        # Crear una nueva ventana para mostrar las asistencias
        ventana_asistencias = tk.Toplevel(self.root)
        ventana_asistencias.title("Registro de Asistencias")
        ventana_asistencias.geometry("600x400")

        # Crear un Treeview para mostrar el contenido del CSV en formato de tabla
        tree = ttk.Treeview(ventana_asistencias, columns=("Nombre", "Fecha", "Emoción"), show='headings')
        tree.heading("Nombre", text="Nombre")
        tree.heading("Fecha", text="Fecha")
        tree.heading("Emoción", text="Emoción")
        tree.pack(expand=True, fill='both')

        # Función para actualizar el Treeview con los datos del archivo CSV
        def actualizar_registro():
            # Limpiar el Treeview antes de volver a cargar los datos
            for item in tree.get_children():
                tree.delete(item)

            # Leer el archivo CSV y cargar los datos en el Treeview
            try:
                with open("Asistencias.csv", "r") as file:
                    reader = csv.reader(file)
                    # Verificar si el archivo está vacío
                    rows = list(reader)
                    if not rows:  # Si no hay filas, mostrar un mensaje y regresar
                        return
                    # Saltar el encabezado si existe
                    for row in rows[1:]:
                        tree.insert("", "end", values=row)
            except FileNotFoundError:
                tk.Label(ventana_asistencias, text="El archivo Asistencias.csv no se encontró.").pack()

        # Cargar los registros por primera vez cuando se abre la ventana
        actualizar_registro()

        # Botones para limpiar o actualizar el registro de asistencias
        btn_limpiar = tk.Button(ventana_asistencias, text="Limpiar Registro de Asistencias", command=self.limpiar_asistencias)
        btn_limpiar.pack(pady=5)

        btn_actualizar = tk.Button(ventana_asistencias, text="Actualizar registro", command=actualizar_registro)
        btn_actualizar.pack(pady=5)

    def limpiar_asistencias(self):
        # Función para limpiar el contenido del archivo Asistencias.csv
        respuesta = messagebox.askyesno("Confirmación", "¿Está seguro de que desea borrar el registro de asistencias?")
        if respuesta:
            with open("Asistencias.csv", "w") as file:
                file.truncate()  # Borrar el contenido del archivo
            messagebox.showinfo("Éxito", "El registro de asistencias ha sido borrado.")

    def ver_usuarios(self):
        ventana_usuarios = tk.Toplevel(self.root)
        ventana_usuarios.title("Lista de Usuarios")
        ventana_usuarios.geometry("300x400")

        # Etiqueta de título
        label = tk.Label(ventana_usuarios, text="Usuarios Registrados", font=("Arial", 14))
        label.pack(pady=10)

        # Crear una lista en la ventana
        lista_usuarios = tk.Listbox(ventana_usuarios, font=("Arial", 12), width=25, height=15)
        lista_usuarios.pack(pady=10)

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
        
        # Llenar la lista con los nombres de los clientes
        for nombre in nombres_clientes:
            lista_usuarios.insert(tk.END, nombre)
