import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from tensorflow.keras.datasets import mnist
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ========== КОНФИГУРАЦИЯ ПУТЕЙ ==========
PATH_TO_MLP = "MLP_optimized.keras"
PATH_TO_CNN = "CNN.keras"

# ========== ГЛОБАЛЬНЫЕ СОСТОЯНИЯ ==========
neural_network = None
current_model = "MLP"
prediction_engine = None

if not os.path.exists(PATH_TO_MLP):
    messagebox.showerror("Файл отсутствует", f"Не найден файл модели: {PATH_TO_MLP}")
    exit()

try:
    neural_network = load_model(PATH_TO_MLP)
except Exception as load_error:
    messagebox.showerror("Ошибка загрузки", f"Сбой при загрузке модели:\n{load_error}")
    exit()


# ========== ФУНКЦИИ АНАЛИЗА ИЗОБРАЖЕНИЙ ==========
def analyze_with_mlp(drawn_image):
    """
    Обрабатывает рисунок с использованием многослойного перцептрона.
    """
    processed_img = drawn_image.resize((28, 28), Image.LANCZOS).convert('L')
    pixel_data = np.array(processed_img).astype('float32') / 255.0
    pixel_data = pixel_data.reshape(1, 28, 28)
    
    model_output = neural_network.predict(pixel_data, verbose=0)[0]
    predicted_number = int(np.argmax(model_output))
    certainty_level = float(model_output[predicted_number])
    
    return predicted_number, certainty_level

def analyze_with_cnn(drawn_image):
    """
    Обрабатывает рисунок с использованием сверточной нейронной сети.
    """
    processed_img = drawn_image.resize((28, 28), Image.LANCZOS).convert('L')
    pixel_data = np.array(processed_img).astype('float32') / 255.0
    pixel_data = pixel_data.reshape(1, 28, 28, 1)
    
    model_output = neural_network.predict(pixel_data, verbose=0)[0]
    predicted_number = int(np.argmax(model_output))
    certainty_level = float(model_output[predicted_number])
    
    return predicted_number, certainty_level

# ========== УПРАВЛЕНИЕ МОДЕЛЯМИ ==========
def activate_mlp_model():
    """
    Активирует и загружает модель MLP для использования.
    """
    global neural_network, current_model, prediction_engine
    
    try:
        neural_network = load_model(PATH_TO_MLP)
        current_model = "MLP"
        prediction_engine = analyze_with_mlp
        ui_status_display.config(text="Активна: Многослойный перцептрон (MLP)", foreground="#2E8B57")
        messagebox.showinfo("Модель изменена", "Переключено на архитектуру MLP")
    except Exception as load_error:
        messagebox.showerror("Сбой загрузки", f"Не удалось загрузить MLP:\n{load_error}")

def activate_cnn_model():
    """
    Активирует и загружает модель CNN для использования.
    """
    global neural_network, current_model, prediction_engine
    
    if not os.path.exists(PATH_TO_CNN):
        messagebox.showerror("Файл отсутствует", f"Файл модели CNN не обнаружен:\n{PATH_TO_CNN}")
        return
    
    try:
        neural_network = load_model(PATH_TO_CNN)
        current_model = "CNN"
        prediction_engine = analyze_with_cnn
        ui_status_display.config(text="Активна: Сверточная сеть (CNN)", foreground="#1E90FF")
        messagebox.showinfo("Модель изменена", "Переключено на архитектуру CNN")
    except Exception as load_error:
        messagebox.showerror("Сбой загрузки", f"Не удалось загрузить CNN:\n{load_error}")

# Установка движка по умолчанию
prediction_engine = analyze_with_mlp

# ========== ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ ==========
class DigitRecognizerApp:
    def __init__(self, master_window):
        self.master = master_window
        self.master.title("Нейросетевой классификатор цифр")
        self.master.geometry("1024x600")
        self.master.configure(bg='#F0F0F0')
        
        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        """Настраивает стили для элементов интерфейса."""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Primary.TButton', 
                       font=('Segoe UI', 10, 'bold'),
                       padding=8,
                       background='#4A6FA5',
                       foreground='white')
        
        style.configure('Secondary.TButton',
                       font=('Segoe UI', 9),
                       padding=6,
                       background='#6C757D',
                       foreground='white')

    def create_widgets(self):
        """Создает и размещает все элементы интерфейса."""
        # Заголовок
        title_frame = tk.Frame(self.master, bg='#2C3E50', height=60)
        title_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(title_frame, 
                text="🧠 Визуальный классификатор рукописных цифр", 
                font=('Segoe UI', 16, 'bold'),
                bg='#2C3E50',
                fg='white').pack(pady=15)
        
        # Панель управления моделями
        control_panel = tk.Frame(self.master, bg='#F0F0F0')
        control_panel.pack(pady=15)
        
        ttk.Button(control_panel, 
                  text="🔄 Активировать MLP", 
                  command=activate_mlp_model,
                  style='Primary.TButton',
                  width=20).pack(side='left', padx=8)
        
        ttk.Button(control_panel, 
                  text="🔄 Активировать CNN", 
                  command=activate_cnn_model,
                  style='Primary.TButton',
                  width=20).pack(side='left', padx=8)
        
        global ui_status_display
        ui_status_display = tk.Label(control_panel, 
                                    text="Активна: Многослойный перцептрон (MLP)", 
                                    font=('Segoe UI', 10, 'bold'),
                                    bg='#F0F0F0',
                                    fg='#2E8B57')
        ui_status_display.pack(side='left', padx=30)
        
        # Область рисования
        canvas_frame = tk.Frame(self.master, bg='#F0F0F0')
        canvas_frame.pack(pady=15)
        
        tk.Label(canvas_frame, 
                text="Область для ввода цифры (рисуйте ниже)", 
                font=('Segoe UI', 10),
                bg='#F0F0F0').pack()
        
        self.drawing_surface = Image.new("L", (280, 280), 0)
        self.drawing_tool = ImageDraw.Draw(self.drawing_surface)
        self.previous_x = self.previous_y = None
        
        self.canvas_area = tk.Canvas(self.master, 
                                     width=300, 
                                     height=300, 
                                     bg='white',
                                     highlightthickness=2,
                                     highlightbackground='#4A6FA5',
                                     relief='solid')
        self.canvas_area.pack(pady=10)
        self.canvas_area.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas_area.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Панель инструментов
        tool_panel = tk.Frame(self.master, bg='#F0F0F0')
        tool_panel.pack(pady=15)
        
        button_configs = [
            ("🗑 Очистить поле", self.clear_canvas, '#DC3545'),
            ("🔍 Проанализировать", self.perform_analysis, '#28A745'),
            ("💾 Сохранить образец", self.store_image, '#17A2B8'),
            ("📂 Загрузить образец", self.retrieve_image, '#6C757D')
        ]
        
        for text, command, color in button_configs:
            btn = tk.Button(tool_panel,
                           text=text,
                           command=command,
                           font=('Segoe UI', 9),
                           bg=color,
                           fg='white',
                           padx=15,
                           pady=6,
                           relief='flat',
                           cursor='hand2')
            btn.pack(side='left', padx=6)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.adjust_color(b.cget('bg'), 20)))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.configure(bg=c))
        
        # Область результатов
        self.result_display = tk.Label(self.master,
                                      text="Нарисуйте цифру и нажмите 'Проанализировать'",
                                      font=('Segoe UI', 12),
                                      bg='#F8F9FA',
                                      fg='#495057',
                                      padx=20,
                                      pady=15,
                                      relief='solid',
                                      borderwidth=1)
        self.result_display.pack(pady=20, ipadx=50)
        
        # Информационная панель
        info_frame = tk.Frame(self.master, bg='#E9ECEF')
        info_frame.pack(fill='x', pady=10)
        
        info_text = "• MLP: быстрее, требует меньше ресурсов | • CNN: точнее, лучше для сложных образцов"
        tk.Label(info_frame,
                text=info_text,
                font=('Segoe UI', 8),
                bg='#E9ECEF',
                fg='#6C757D',
                wraplength=800).pack(pady=5)

    def adjust_color(self, color, delta):
        """Осветляет цвет для эффекта наведения."""
        if color.startswith('#'):
            rgb = [int(color[i:i+2], 16) for i in (1, 3, 5)]
            rgb = [min(255, c + delta) for c in rgb]
            return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
        return color

    def draw_on_canvas(self, event):
        """Обрабатывает процесс рисования на холсте."""
        x_coord, y_coord = event.x, event.y
        brush_size = 12
        
        if self.previous_x and self.previous_y:
            self.canvas_area.create_line(self.previous_x, self.previous_y,
                                        x_coord, y_coord,
                                        width=brush_size*2,
                                        fill="black",
                                        capstyle=tk.ROUND,
                                        smooth=True)
            self.drawing_tool.line([self.previous_x, self.previous_y,
                                   x_coord, y_coord],
                                   fill=255,
                                   width=brush_size*2)
        else:
            self.canvas_area.create_oval(x_coord-brush_size, y_coord-brush_size,
                                        x_coord+brush_size, y_coord+brush_size,
                                        fill="black",
                                        outline="")
            self.drawing_tool.ellipse([x_coord-brush_size, y_coord-brush_size,
                                      x_coord+brush_size, y_coord+brush_size],
                                      fill=255)
        
        self.previous_x, self.previous_y = x_coord, y_coord

    def stop_drawing(self, _=None):
        """Сбрасывает координаты при завершении рисования."""
        self.previous_x = self.previous_y = None

    def clear_canvas(self):
        """Полностью очищает область рисования."""
        self.canvas_area.delete("all")
        self.drawing_surface = Image.new("L", (280, 280), 0)
        self.drawing_tool = ImageDraw.Draw(self.drawing_surface)
        self.result_display.config(text="Холст очищен. Нарисуйте новую цифру.")

    def perform_analysis(self):
        """Выполняет анализ нарисованного изображения с помощью активной модели."""
        if prediction_engine is None:
            messagebox.showwarning("Модель не активна", "Сначала активируйте модель нейронной сети.")
            return
        
        try:
            identified_digit, confidence_value = prediction_engine(self.drawing_surface)
            confidence_percent = confidence_value * 100
            self.result_display.config(
                text=f"🔍 Результат анализа: цифра '{identified_digit}' | "
                     f"Уверенность модели: {confidence_percent:05.2f}%",
                fg='#155724',
                bg='#D4EDDA'
            )
        except Exception as e:
            messagebox.showerror("Ошибка анализа", f"Не удалось обработать изображение:\n{e}")

    def store_image(self):
        """Сохраняет текущий рисунок в файл."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("Изображения PNG", "*.png"),
                ("Изображения JPEG", "*.jpg *.jpeg"),
                ("Все файлы", "*.*")
            ],
            title="Сохранить рисунок как"
        )
        
        if file_path:
            try:
                # Сохраняем в обучающем формате 28x28
                self.drawing_surface.resize((28, 28), Image.LANCZOS).save(file_path)
                filename = os.path.basename(file_path)
                self.result_display.config(text=f"✓ Рисунок сохранён: {filename}")
            except Exception as save_error:
                messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл:\n{save_error}")

    def retrieve_image(self):
        """Загружает изображение из файла для анализа."""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Графические файлы", "*.png *.jpg *.jpeg *.bmp"),
                ("Все файлы", "*.*")
            ],
            title="Выберите изображение для загрузки"
        )
        
        if file_path:
            try:
                loaded_image = Image.open(file_path).convert('L')
                self.drawing_surface = loaded_image.resize((280, 280), Image.LANCZOS)
                self.drawing_tool = ImageDraw.Draw(self.drawing_surface)
                self.display_current_canvas()
                self.result_display.config(text=f"Изображение загружено: {os.path.basename(file_path)}")
            except Exception as load_error:
                messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить изображение:\n{load_error}")

    def display_current_canvas(self):
        """Отображает текущее изображение на холсте."""
        self.canvas_area.delete("all")
        canvas_preview = ImageTk.PhotoImage(self.drawing_surface)
        self.canvas_area.create_image(0, 0, anchor=tk.NW, image=canvas_preview)
        self.canvas_area.image = canvas_preview

# ========== ТОЧКА ВХОДА ПРИЛОЖЕНИЯ ==========
if __name__ == "__main__":
    main_window = tk.Tk()
    app_instance = DigitRecognizerApp(main_window)
    main_window.mainloop()