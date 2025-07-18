import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import time


class LaneDetectionApp:
    def __init__(self, root):
        # Configuración principal
        self.root = root
        self.root.title("Detector de Carriles para Vehículos Autónomos")
        self.root.geometry("1200x700")

        # Tema de colores
        self.bg_color = "#2E3440"
        self.text_color = "#ECEFF4"
        self.accent_color = "#88C0D0"
        self.highlight_color = "#A3BE8C"
        self.warning_color = "#EBCB8B"
        self.panel_color = "#3B4252"

        # Variables de control
        self.video_path = ""
        self.cap = None
        self.current_frame = None
        self.processed_frame = None
        self.is_playing = False
        self.frame_count = 0
        self.current_frame_number = 0

        # Parámetros de procesamiento (predeterminados)
        self.canny_low = tk.IntVar(value=50)
        self.canny_high = tk.IntVar(value=150)
        self.roi_height_percent = tk.IntVar(value=40)
        self.roi_width_percent = tk.IntVar(value=90)
        self.hough_threshold = tk.IntVar(value=20)
        self.hough_min_line_length = tk.IntVar(value=20)
        self.hough_max_line_gap = tk.IntVar(value=300)

        # Configurar interfaz
        self.setup_ui()

    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = tk.Label(main_frame, text="Detector de Carriles para Vehículos Autónomos",
                               font=("Helvetica", 20, "bold"), fg=self.accent_color, bg=self.bg_color)
        title_label.pack(pady=(0, 20))

        # Panel superior - Controles principales
        control_frame = tk.Frame(main_frame, bg=self.panel_color, padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 20))

        # Botones de carga y control
        load_video_btn = tk.Button(control_frame, text="Cargar Video",
                                   command=self.load_video,
                                   bg=self.accent_color, fg=self.bg_color,
                                   font=("Helvetica", 10), padx=10, pady=5)
        load_video_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = tk.Button(control_frame, text="Reproducir",
                                  command=self.toggle_play,
                                  bg=self.accent_color, fg=self.bg_color,
                                  font=("Helvetica", 10), padx=10, pady=5)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        reset_btn = tk.Button(control_frame, text="Reiniciar",
                              command=self.reset_video,
                              bg=self.accent_color, fg=self.bg_color,
                              font=("Helvetica", 10), padx=10, pady=5)
        reset_btn.pack(side=tk.LEFT, padx=5)

        # Panel izquierdo - Visualización de video
        left_frame = tk.Frame(main_frame, bg=self.panel_color)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        video_label = tk.Label(left_frame, text="Video",
                               font=("Helvetica", 12, "bold"), fg=self.text_color, bg=self.panel_color)
        video_label.pack(pady=10)

        # Panel de video original
        self.video_panel = tk.Label(left_frame, bg=self.panel_color)
        self.video_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Barra de progreso de video
        self.video_progress = ttk.Progressbar(left_frame, orient="horizontal",
                                              length=100, mode="determinate")
        self.video_progress.pack(fill=tk.X, padx=10, pady=10)

        # Panel derecho - Visualización procesada y controles
        right_frame = tk.Frame(main_frame, bg=self.panel_color)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        result_label = tk.Label(right_frame, text="Detección de Carriles",
                                font=("Helvetica", 12, "bold"), fg=self.text_color, bg=self.panel_color)
        result_label.pack(pady=10)

        # Panel de visualización procesada
        self.result_panel = tk.Label(right_frame, bg=self.panel_color)
        self.result_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel de parámetros
        param_frame = tk.Frame(right_frame, bg=self.panel_color, padx=10, pady=10)
        param_frame.pack(fill=tk.X)

        # Título de parámetros
        param_title = tk.Label(param_frame, text="Parámetros de Procesamiento",
                               font=("Helvetica", 11, "bold"), fg=self.text_color, bg=self.panel_color)
        param_title.grid(row=0, column=0, columnspan=4, pady=(0, 10), sticky=tk.W)

        # Parámetros Canny
        tk.Label(param_frame, text="Canny Low:", fg=self.text_color, bg=self.panel_color).grid(row=1, column=0,
                                                                                               sticky=tk.W)
        tk.Scale(param_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.canny_low,
                 bg=self.panel_color, fg=self.text_color, troughcolor=self.bg_color,
                 highlightthickness=0, bd=0).grid(row=1, column=1)

        tk.Label(param_frame, text="Canny High:", fg=self.text_color, bg=self.panel_color).grid(row=1, column=2,
                                                                                                sticky=tk.W)
        tk.Scale(param_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.canny_high,
                 bg=self.panel_color, fg=self.text_color, troughcolor=self.bg_color,
                 highlightthickness=0, bd=0).grid(row=1, column=3)

        # Parámetros ROI
        tk.Label(param_frame, text="ROI Height %:", fg=self.text_color, bg=self.panel_color).grid(row=2, column=0,
                                                                                                  sticky=tk.W)
        tk.Scale(param_frame, from_=20, to=80, orient=tk.HORIZONTAL, variable=self.roi_height_percent,
                 bg=self.panel_color, fg=self.text_color, troughcolor=self.bg_color,
                 highlightthickness=0, bd=0).grid(row=2, column=1)

        tk.Label(param_frame, text="ROI Width %:", fg=self.text_color, bg=self.panel_color).grid(row=2, column=2,
                                                                                                 sticky=tk.W)
        tk.Scale(param_frame, from_=20, to=100, orient=tk.HORIZONTAL, variable=self.roi_width_percent,
                 bg=self.panel_color, fg=self.text_color, troughcolor=self.bg_color,
                 highlightthickness=0, bd=0).grid(row=2, column=3)

        # Parámetros Hough
        tk.Label(param_frame, text="Hough Threshold:", fg=self.text_color, bg=self.panel_color).grid(row=3, column=0,
                                                                                                     sticky=tk.W)
        tk.Scale(param_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.hough_threshold,
                 bg=self.panel_color, fg=self.text_color, troughcolor=self.bg_color,
                 highlightthickness=0, bd=0).grid(row=3, column=1)

        tk.Label(param_frame, text="Min Line Length:", fg=self.text_color, bg=self.panel_color).grid(row=3, column=2,
                                                                                                     sticky=tk.W)
        tk.Scale(param_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.hough_min_line_length,
                 bg=self.panel_color, fg=self.text_color, troughcolor=self.bg_color,
                 highlightthickness=0, bd=0).grid(row=3, column=3)

        tk.Label(param_frame, text="Max Line Gap:", fg=self.text_color, bg=self.panel_color).grid(row=4, column=0,
                                                                                                  sticky=tk.W)
        tk.Scale(param_frame, from_=1, to=500, orient=tk.HORIZONTAL, variable=self.hough_max_line_gap,
                 bg=self.panel_color, fg=self.text_color, troughcolor=self.bg_color,
                 highlightthickness=0, bd=0).grid(row=4, column=1)

        # Botón de aplicar
        apply_btn = tk.Button(param_frame, text="Aplicar Cambios",
                              command=self.apply_parameters,
                              bg=self.highlight_color, fg=self.bg_color,
                              font=("Helvetica", 10), padx=10, pady=5)
        apply_btn.grid(row=4, column=2, columnspan=2, pady=10)

        # Configurar grid
        for i in range(5):
            param_frame.grid_rowconfigure(i, weight=1)
        for i in range(4):
            param_frame.grid_columnconfigure(i, weight=1)

        # Barra de estado
        self.status_var = tk.StringVar(value="Listo para comenzar. Cargue un video.")
        status_bar = tk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W,
                              bg=self.panel_color, fg=self.text_color,
                              font=("Helvetica", 9), padx=10, pady=5)
        status_bar.pack(fill=tk.X, pady=(20, 0))

    def load_video(self):
        """Permite al usuario seleccionar un archivo de video"""
        video_path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Archivos de Video", "*.mp4 *.avi *.mov *.mkv")]
        )

        if not video_path:
            return

        # Cerrar video anterior si existe
        if self.cap is not None:
            self.cap.release()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.status_var.set(f"Error: No se pudo abrir el video {os.path.basename(video_path)}")
            return

        # Obtener información del video
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_number = 0

        # Actualizar interfaz
        self.status_var.set(f"Video cargado: {os.path.basename(video_path)}")
        self.play_btn.configure(text="Reproducir")
        self.is_playing = False

        # Mostrar primer frame
        self.show_frame()

    def show_frame(self):
        """Muestra el frame actual y su versión procesada"""
        if self.cap is None:
            return

        # Obtener frame
        ret, frame = self.cap.read()

        if not ret:
            self.status_var.set("Fin del video")
            self.is_playing = False
            self.play_btn.configure(text="Reproducir")
            return

        # Actualizar contador de frames
        self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Actualizar barra de progreso
        if self.frame_count > 0:
            progress = (self.current_frame_number / self.frame_count) * 100
            self.video_progress["value"] = progress

        # Guardar frame actual
        self.current_frame = frame

        # Procesar frame para detección de carriles
        self.processed_frame = self.detect_lanes(frame)

        # Mostrar frames en los paneles
        self.display_frame(frame, self.video_panel)
        self.display_frame(self.processed_frame, self.result_panel)

        # Si está en reproducción, programar siguiente frame
        if self.is_playing:
            self.root.after(30, self.show_frame)  # ~30ms para ~30fps

    def detect_lanes(self, frame):
        """Detección de carriles en el frame actual"""
        # Crear copia para dibujar resultados
        result = frame.copy()

        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Aplicar desenfoque gaussiano para reducir ruido
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detección de bordes con Canny
            edges = cv2.Canny(blur, self.canny_low.get(), self.canny_high.get())

            # Crear máscara para región de interés (ROI)
            height, width = edges.shape[:2]

            # Calcular ROI basado en porcentajes configurados
            roi_height = int(height * self.roi_height_percent.get() / 100)
            roi_width = int(width * self.roi_width_percent.get() / 100)

            roi_top = height - roi_height
            roi_left = (width - roi_width) // 2

            # Crear máscara
            mask = np.zeros_like(edges)

            # Definir región trapezoidal (se aproxima a la perspectiva de la carretera)
            polygon = np.array([
                [roi_left, height],  # Bottom left
                [roi_left + roi_width, height],  # Bottom right
                [roi_left + roi_width * 0.55, roi_top],  # Top right
                [roi_left + roi_width * 0.45, roi_top]  # Top left
            ], np.int32)

            cv2.fillPoly(mask, [polygon], 255)

            # Aplicar máscara
            masked_edges = cv2.bitwise_and(edges, mask)

            # Dibujar ROI en la imagen de resultado (para visualización)
            cv2.polylines(result, [polygon], True, (0, 255, 0), 2)

            # Aplicar transformada de Hough para detectar líneas
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold.get(),
                minLineLength=self.hough_min_line_length.get(),
                maxLineGap=self.hough_max_line_gap.get()
            )

            # Agrupar líneas por pendiente (izquierda / derecha)
            left_lines = []
            right_lines = []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    # Calcular pendiente
                    if x2 - x1 == 0:  # Evitar división por cero
                        continue

                    slope = (y2 - y1) / (x2 - x1)

                    # Filtrar líneas horizontales
                    if abs(slope) < 0.1:
                        continue

                    # Agrupar en izquierda (pendiente negativa) o derecha (pendiente positiva)
                    if slope < 0:
                        left_lines.append(line[0])
                    else:
                        right_lines.append(line[0])

            # Promediar y extender líneas
            def average_lines(lines):
                if len(lines) == 0:
                    return None

                avg_line = np.mean(lines, axis=0, dtype=np.int32)
                x1, y1, x2, y2 = avg_line

                # Calcular pendiente y ordenada al origen
                if x2 - x1 == 0:  # Evitar división por cero
                    return None

                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # Extender línea hasta los bordes de la ROI
                y_bottom = height
                y_top = roi_top

                x_bottom = int((y_bottom - intercept) / slope)
                x_top = int((y_top - intercept) / slope)

                return [x_top, y_top, x_bottom, y_bottom]

            # Promediar y dibujar líneas
            left_line = average_lines(left_lines)
            right_line = average_lines(right_lines)

            # Dibujar líneas de carril
            if left_line is not None:
                x1, y1, x2, y2 = left_line
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)

            if right_line is not None:
                x1, y1, x2, y2 = right_line
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Dibujar área de carril si ambas líneas están detectadas
            if left_line is not None and right_line is not None:
                lane_points = np.array([
                    [left_line[2], left_line[3]],  # Bottom left
                    [right_line[2], right_line[3]],  # Bottom right
                    [right_line[0], right_line[1]],  # Top right
                    [left_line[0], left_line[1]]  # Top left
                ], np.int32)

                # Crear overlay semi-transparente
                overlay = result.copy()
                cv2.fillPoly(overlay, [lane_points], (0, 255, 0))

                # Mezclar con transparencia
                alpha = 0.3
                result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

            # Mostrar estadísticas
            text = f"Líneas detectadas: I={len(left_lines)}, D={len(right_lines)}"
            cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        except Exception as e:
            # En caso de error, mostrar mensaje
            cv2.putText(result, f"Error: {str(e)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result

    def display_frame(self, frame, panel):
        """Muestra un frame en el panel especificado"""
        if frame is None:
            return

        # Convertir de BGR a RGB para mostrar en tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Obtener dimensiones reales del panel (usar valores fijos si aún no están disponibles)
        # Esto evita que el tamaño crezca descontroladamente
        panel_width = panel.winfo_width()
        panel_height = panel.winfo_height()

        # Si el panel aún no tiene dimensiones definidas, usar valores por defecto
        if panel_width < 50:  # Es probablemente 1 o 0 antes de que se render la interfaz
            panel_width = 500
        if panel_height < 50:
            panel_height = 400

        # Limitar el tamaño máximo para evitar el crecimiento desmedido
        panel_width = min(panel_width, 600)
        panel_height = min(panel_height, 500)

        # Calcular relación de aspecto
        h, w = rgb_frame.shape[:2]
        aspect_ratio = w / h

        # Redimensionar manteniendo relación de aspecto
        if panel_width / panel_height > aspect_ratio:
            # Panel más ancho que el frame
            new_height = panel_height
            new_width = int(panel_height * aspect_ratio)
        else:
            # Panel más alto que el frame
            new_width = panel_width
            new_height = int(panel_width / aspect_ratio)

        # Asegurar que las dimensiones no sean 0
        new_width = max(new_width, 1)
        new_height = max(new_height, 1)

        # Redimensionar
        resized_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Convertir a formato PIL y luego a PhotoImage
        pil_image = Image.fromarray(resized_frame)
        tk_image = ImageTk.PhotoImage(image=pil_image)

        # Mostrar en el panel
        panel.configure(image=tk_image)
        panel.image = tk_image  # Mantener referencia

        # Forzar actualización y limpieza de memoria
        self.root.update_idletasks()

    def toggle_play(self):
        """Inicia o pausa la reproducción del video"""
        if self.cap is None:
            self.status_var.set("Error: No hay video cargado")
            return

        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_btn.configure(text="Pausar")
            self.status_var.set("Reproduciendo video...")
            self.show_frame()
        else:
            self.play_btn.configure(text="Reproducir")
            self.status_var.set("Video pausado")

    def reset_video(self):
        """Reinicia el video desde el principio"""
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_number = 0
        self.video_progress["value"] = 0
        self.status_var.set("Video reiniciado")

        # Mostrar primer frame
        self.show_frame()

    def apply_parameters(self):
        """Aplica los parámetros actuales al frame actual"""
        if self.current_frame is not None:
            self.processed_frame = self.detect_lanes(self.current_frame)
            self.display_frame(self.processed_frame, self.result_panel)
            self.status_var.set("Parámetros aplicados")


# Iniciar aplicación si se ejecuta como script principal
if __name__ == "__main__":
    root = tk.Tk()
    app = LaneDetectionApp(root)
    root.mainloop()