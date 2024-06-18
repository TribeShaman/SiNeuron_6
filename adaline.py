import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, input_size=3):
        self.weights = np.random.rand(input_size)  # Inicjalizacja losowych wag
        self.learning_rate = learning_rate
        self.errors = []
        self.min_vals = None
        self.max_vals = None

    def fit(self, X, y):
        activation_function_output = self.activation_function(self.net_input(X))
        errors = y - activation_function_output
        self.weights[1:] += self.learning_rate * X.T.dot(errors)
        self.weights[0] += self.learning_rate * errors.sum()
        total_error = np.mean(errors ** 2)
        self.errors.append(total_error)

    def net_input(self, X):
        weighted_sum = np.dot(X, self.weights[1:]) + self.weights[0]
        return weighted_sum

    def activation_function(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def train(self, training_inputs):
        training_data = self.normalize(training_inputs)
        X = training_data[:, :-1]
        y = training_data[:, -1]
        self.fit(X, y)

    def normalize(self, training_inputs):
        inputs = np.array(training_inputs)
        if self.min_vals is None or self.max_vals is None:
            self.min_vals = np.min(inputs[:, :-1], axis=0)
            self.max_vals = np.max(inputs[:, :-1], axis=0)
        inputs[:, :-1] = (inputs[:, :-1] - self.min_vals) / (self.max_vals - self.min_vals)
        return inputs
    def get_error_history(self):
        return self.errors