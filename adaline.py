import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, input_size=3):
        self.weights = np.random.rand(input_size)  # Inicjalizacja losowych wag (wliczając bias)
        self.learning_rate = learning_rate

        self.errors = []

    def net_input(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

    def predict(self, inputs):
        return self.net_input(inputs)

    def train(self, training_inputs):
            total_error = 0
            for inputs in training_inputs:
                label = inputs[-1]  # Pobranie etykiety z danych wejściowych
                inputs_without_label = inputs[:-1]  # Dane wejściowe bez etykiety
                output = self.predict(inputs_without_label)
                error = label - output
                self.weights[1:] += self.learning_rate * error * np.array(inputs_without_label, dtype=float)
                self.weights[0] += self.learning_rate * error
                total_error += error**2
            self.errors.append(total_error / len(training_inputs))

    def get_error_history(self):
        return self.errors
