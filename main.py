import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
import data_generator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

training_data = []
iterations = 1000
learning_threshold = 0.01
learning_rate = 0.01
input_size = 3

neuron = Perceptron(learning_rate,input_size)

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
    error_history = neuron.get_error_history()
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
    y_values = (-neuron.weights[1] * x_values - neuron.weights[0]) / neuron.weights[2]
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
    neuron.weights = np.random.rand(input_size)
    neuron.errors = []
    for _ in range(iterations):
        neuron.train(training_data)
        if neuron.errors[-1] <= learning_threshold:
            break

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets" / "frame0"


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1200x900")
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
    600.0,
    55.0,
    anchor="center",
    text="Perceptron",
    fill="#F6F9FF",
    font=("Arial", 96, "bold")
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
    x=114.0,
    y=174.0,
    width=372.0,
    height=81.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=load_file,
    relief="flat"
)
button_2.place(
    x=714.0,
    y=174.0,
    width=372.0,
    height=81.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=learn,
    relief="flat"
)
button_3.place(
    x=114.0,
    y=327.0,
    width=372.0,
    height=81.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=plot_graph,
    relief="flat"
)
button_4.place(
    x=714.0,
    y=327.0,
    width=372.0,
    height=81.0
)

plot_frame = tk.Frame(window)
plot_frame.place(
    x=50.0,
    y=450.0,
    width=1100.0,
    height=400.0
)
window.resizable(False, False)
window.mainloop()
