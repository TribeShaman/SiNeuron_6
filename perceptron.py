import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, iterations=100):
        self.weights = np.random.rand(input_size)  # Inicjalizacja losowych wag (wliczając bias)
        self.learning_rate = learning_rate
        self.iterations = iterations

    def predict(self, inputs):
        summation = np.dot(inputs[:-1], self.weights[1:]) + self.weights[0]  # Sumowanie ważonych wejść (bez label)
        if summation > 0:
            activation = 1  # Funkcja aktywacji (np. skokowa)
        else:
            activation = 0
        return activation

    def train(self, training_inputs):
        for inputs in training_inputs:
            label = inputs[-1]  # Pobranie etykiety z danych wejściowych
            prediction = self.predict(inputs)
            self.weights[1:] += self.learning_rate * (label - prediction) * np.array(inputs[:-1], dtype=float)  # Korekcja wag (bez biasu i label)
            self.weights[0] += self.learning_rate * (label - prediction)  # Korekcja biasu
            a = round(-self.weights[1] / self.weights[2],2)  # Współczynnik kierunkowy dla x
            b = round(-self.weights[0] / self.weights[2],2)  # Przesunięcie w osi y

            print("y = ", a, "x + ", b)
