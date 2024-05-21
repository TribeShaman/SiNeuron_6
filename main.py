import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron

# Tworzenie instancji perceptronu
neuron = Perceptron(3)

# Wczytywanie danych treningowych z pliku
training_data = []
with open('Dane/in.tab') as file:
    for line in file:
        # Wczytywanie danych i dzielenie ich na listę wartości oddzielonych przecinkami
        data = line.strip().split(',')
        # Konwersja danych na liczby zmiennoprzecinkowe
        data = [float(x) for x in data]
        # Dodanie wczytanych danych do listy danych treningowych
        training_data.append(data)

# Wczytywanie danych testowych z pliku
test_data = []
with open('Dane/cal.tab') as file:
    for line in file:
        # Wczytywanie danych i dzielenie ich na listę wartości oddzielonych przecinkami
        data = line.strip().split(',')
        # Konwersja danych na liczby zmiennoprzecinkowe
        data = [float(x) for x in data]
        # Dodanie wczytanych danych do listy danych testowych
        test_data.append(data)

for _ in range(100):
    neuron.train(training_data)
    if neuron.errors[-1] == 0:
        break



error_history = neuron.get_error_history()
plt.plot(error_history)
plt.title('Perceptron Training Error History')
plt.xlabel('Iteration')
plt.ylabel('Number of Misclassifications')
plt.grid(True)
plt.show()

# Podział danych testowych na dwie klasy na podstawie predykcji perceptronu
class_0_x = []
class_0_y = []
class_1_x = []
class_1_y = []

for inputs in test_data:
    label = int(inputs[-1])
    if label == 0:
        class_0_x.append(inputs[0])
        class_0_y.append(inputs[1])
    else:
        class_1_x.append(inputs[0])
        class_1_y.append(inputs[1])

# Rysowanie punktów testowych na wykresie
plt.scatter(class_0_x, class_0_y, color='blue', label='Class 0')
plt.scatter(class_1_x, class_1_y, color='red', label='Class 1')

neuron.train(test_data)

# Rysowanie prostej separującej klasy na podstawie danych treningowych
x_values = np.linspace(0, 30, 100)
y_values = (-neuron.weights[1] * x_values - neuron.weights[0]) / neuron.weights[2]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

# Dodanie legendy i wyświetlenie wykresu
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Classification with Perceptron')
plt.grid(True)
plt.xlim(0,25)
plt.ylim(0,25)
plt.show()
