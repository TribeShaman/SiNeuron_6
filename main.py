import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from adaline import Adaline
import data_generator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

training_data = []
iterations = 1000
learning_threshold = 0.01
learning_rate = 0.01
input_size = 3

perceptron = Perceptron(learning_rate, input_size)
adaline = Adaline(learning_rate,input_size)
def load_file():
    file_path = filedialog.askopenfilename()
    training_data.clear()
    if file_path:
        messagebox.showinfo("Plik wczytany", f"Wczytano plik: {file_path}")
        with open(file_path) as file:
            for line in file:
                # Wczytywanie danych i dzielenie ich na listę wartości oddzielonych przecinkami
                data = line.strip().split(',')
                # Konwersja danych na liczby zmiennoprzecinkowe
                data = [float(x) for x in data]
                # Dodanie wczytanych danych do listy danych treningowych
                training_data.append(data)

def plot_graph():
    for widget in plot_frame.winfo_children():
        widget.destroy()
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Wykres 1: Historia błędów treningowych
    error_history = perceptron.get_error_history()
    ax1.plot(error_history)
    ax1.set_title('Perceptron Training Error History')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Misclassifications')
    ax1.set_xlim(-30, iterations + 30)
    ax1.set_ylim(-0.1, 0.4)
    ax1.grid(True)

    # Wykres 2: Klasyfikacja z perceptronem
    class_0_x = []
    class_0_y = []
    class_1_x = []
    class_1_y = []

    for inputs in training_data:
        label = int(inputs[-1])
        if label == 0:
            class_0_x.append(inputs[0])
            class_0_y.append(inputs[1])
        else:
            class_1_x.append(inputs[0])
            class_1_y.append(inputs[1])

    ax2.scatter(class_0_x, class_0_y, color='blue', label='Class 0')
    ax2.scatter(class_1_x, class_1_y, color='red', label='Class 1')

    x_values = np.linspace(0, 30, 100)
    y_values = (-perceptron.weights[1] * x_values - perceptron.weights[0]) / perceptron.weights[2]
    ax2.plot(x_values, y_values, color='green', label='Decision Boundary')

    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Classification with Perceptron')
    ax2.grid(True)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(0, 25)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_graph_adaline():
    for widget in plot_frame.winfo_children():
        widget.destroy()
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Wykres 1: Historia błędów treningowych
    error_history = adaline.get_error_history()
    ax1.plot(error_history)
    ax1.set_title('Adaline Training Error History')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Misclassifications')
    ax1.set_xlim(-30, iterations + 30)
    ax1.set_ylim(-0.1, 0.4)
    ax1.grid(True)

    # Wykres 2: Klasyfikacja z perceptronem
    class_0_x = []
    class_0_y = []
    class_1_x = []
    class_1_y = []

    for inputs in training_data:
        label = int(inputs[-1])
        if label == 0:
            class_0_x.append(inputs[0])
            class_0_y.append(inputs[1])
        else:
            class_1_x.append(inputs[0])
            class_1_y.append(inputs[1])

    ax2.scatter(class_0_x, class_0_y, color='blue', label='Class 0')
    ax2.scatter(class_1_x, class_1_y, color='red', label='Class 1')

    x_values = np.linspace(0, 30, 100)
    y_values = (-adaline.weights[1] * x_values - adaline.weights[0]) / perceptron.weights[2]
    ax2.plot(x_values, y_values, color='green', label='Decision Boundary')

    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Classification with Perceptron')
    ax2.grid(True)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(0, 25)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def learn():
    perceptron.weights = np.random.rand(input_size)
    perceptron.errors = []
    for _ in range(iterations):
        perceptron.train(training_data)
        if perceptron.errors[-1] <= learning_threshold:
            break
def learn_adaline():
    adaline.weights = np.random.rand(input_size)
    adaline.errors = []
    for _ in range(iterations):
        adaline.train(training_data)
        if adaline.errors[-1] <= learning_threshold:
            break
def learnOne():
    perceptron.train(training_data)
def learnOne_adaline():
    adaline.train(training_data)
def open_custom_point_window():
    custom_window = tk.Toplevel(window)
    custom_window.title("Custom Point Selector")
    custom_window.geometry("300x400")

    points = training_data.copy()  # Kopiowanie istniejących danych treningowych

    coord_label = tk.Label(custom_window, text="X: 0, Y: 0")
    coord_label.pack(pady=5)

    def on_click(event):
        canvas_height = 250
        x, y = event.x, event.y
        x_scaled = x / 10
        y_scaled = (canvas_height - y) / 10
        label = label_var.get()
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='black')
        points.append((x_scaled, y_scaled, label))
        coord_label.config(text=f"X: {x_scaled:.2f}, Y: {y_scaled:.2f}")

    def save_points():
        training_data.clear()
        for point in points:
            x, y, label = point
            training_data.append(point)
        messagebox.showinfo("Punkty wczytane", "Wczytano punkty")
        custom_window.destroy()

    canvas = tk.Canvas(custom_window, bg="white", width=250, height=250)
    canvas.pack(pady=10)
    canvas.bind("<Button-1>", on_click)

    # Wyświetlanie istniejących punktów
    for point in points:
        x, y, label = point
        canvas_x = x * 10
        canvas_y = 250 - (y * 10)
        canvas.create_oval(canvas_x - 2, canvas_y - 2, canvas_x + 2, canvas_y + 2, fill='black')

    label_var = tk.IntVar()
    label_frame = tk.Frame(custom_window)
    label_frame.pack(pady=5)
    tk.Radiobutton(label_frame, text="Label 0", variable=label_var, value=0).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(label_frame, text="Label 1", variable=label_var, value=1).pack(side=tk.LEFT, padx=10)
    label_var.set(0)

    save_button = tk.Button(custom_window, text="Save Points", command=save_points)
    save_button.pack(pady=10)

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets" / "frame0"

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()

window.geometry("1200x950")
window.configure(bg = "#F6F9FF")


canvas = Canvas(
    window,
    bg = "#F6F9FF",
    height = 800,
    width = 1200,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    1200.0,
    129.0,
    fill="#B4C5E4",
    outline="")

canvas.create_text(
    208.0,
    0.0,
    anchor="nw",
    text="Uczenie neuronu",
    fill="#F6F9FF",
    font=("Noto Sans", 96 * -1,  "bold")
)

canvas.create_text(
    800.0,
    244.0,
    anchor="nw",
    text="Adaline",
    fill="#000000",
    font=("Noto Sans", 48 * -1,  "bold")
)

canvas.create_text(
    225.0,
    244.0,
    anchor="nw",
    text="Percepton",
    fill="#000000",
    font=("Noto Sans", 48 * -1,  "bold")
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=data_generator.generate_data,
    relief="flat"
)
button_1.place(
    x=208.0,
    y=140.0,
    width=372.0,
    height=81.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("Zapisz stan"),
    relief="flat"
)
button_2.place(
    x=414.0,
    y=860.0,
    width=372.0,
    height=81.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=load_file,
    relief="flat"
)
button_3.place(
    x=596.0,
    y=140.0,
    width=184.0,
    height=81.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=open_custom_point_window,
    relief="flat"
)
button_4.place(
    x=795.0,
    y=140.0,
    width=184.0,
    height=81.0
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=learn,
    relief="flat"
)
button_5.place(
    x=24.0,
    y=244.0,
    width=184.0,
    height=81.0
)

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=learnOne_adaline,
    relief="flat"
)
button_6.place(
    x=994.0,
    y=339.0,
    width=184.0,
    height=81.0
)

button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=learnOne,
    relief="flat"
)
button_7.place(
    x=24.0,
    y=339.0,
    width=184.0,
    height=81.0
)

button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_8 = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    command=learn_adaline,
    relief="flat"
)
button_8.place(
    x=994.0,
    y=239.0,
    width=184.0,
    height=81.0
)

button_image_9 = PhotoImage(
    file=relative_to_assets("button_9.png"))
button_9 = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    command=plot_graph,
    relief="flat"
)
button_9.place(
    x=230.0,
    y=339.0,
    width=184.0,
    height=81.0
)

button_image_10 = PhotoImage(
    file=relative_to_assets("button_10.png"))
button_10 = Button(
    image=button_image_10,
    borderwidth=0,
    highlightthickness=0,
    command=plot_graph_adaline,
    relief="flat"
)
button_10.place(
    x=795.0,
    y=339.0,
    width=184.0,
    height=81.0
)

plot_frame = tk.Frame(window)
plot_frame.place(
    x=50.0,
    y=460.0,
    width=1100.0,
    height=400.0
)

window.resizable(False, False)
window.mainloop()
