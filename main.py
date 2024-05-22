import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
import data_generator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

    # Wykres 2: Klasyczacja z perceptronem
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

    # Rysowanie punktów testowych na wykresie
    ax2.scatter(class_0_x, class_0_y, color='blue', label='Class 0')
    ax2.scatter(class_1_x, class_1_y, color='red', label='Class 1')

    # Rysowanie prostej separującej klasy na podstawie danych treningowych
    x_values = np.linspace(0, 30, 100)
    y_values = (-neuron.weights[1] * x_values - neuron.weights[0]) / neuron.weights[2]
    ax2.plot(x_values, y_values, color='green', label='Decision Boundary')

    # Dodanie legendy i wyświetlenie wykresu
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Classification with Perceptron')
    ax2.grid(True)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(0, 25)

    # Utwórz obiekt FigureCanvasTkAgg
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

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    window.geometry(f"{width}x{height}+{x}+{y}")

window = tk.Tk()

window_width = 1200
window_height = 800
center_window(window, window_width, window_height)
window.title("Perceptron v0.1")

button = tk.Button(window, text="Wygeneruj dane", command=data_generator.generate_data)
button.pack(pady=10)

load_button = tk.Button(window, text="Wczytaj plik", command=load_file)
load_button.pack(pady=10)

plot_button = tk.Button(window, text="Naucz perceptron", command=learn)
plot_button.pack(pady=10)

plot_button = tk.Button(window, text="Pokaż wykresy", command=plot_graph)
plot_button.pack(pady=10)

# Utwórz ramkę dla wykresów
plot_frame = tk.Frame(window)
plot_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)



window.mainloop()
