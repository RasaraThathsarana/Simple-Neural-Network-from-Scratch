import numpy as np
import pandas as pd
import csv


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def relu(logits):
    return np.maximum(0, logits)


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


class Layer:
    def __init__(self, activation_func):
        self.data = np.array([], dtype=np.float32)
        self.activation_func = activation_func
        self.output_data = np.array([], dtype=np.float32)

    def feed_data(self, X):
        self.data = np.array(X, dtype=np.float32)

    def run(self, input_data):
        self.feed_data(input_data)
        if self.activation_func == "relu":
            self.output_data = relu(self.data)
        elif self.activation_func == "softmax":
            self.output_data = softmax(self.data)
        return self.output_data


class NN:
    def __init__(self):
        b = pd.read_csv("Task_1/b/b-100-40-4.csv", header=None).drop(columns=0)
        layer1_b = np.array(b[:1].dropna(axis=1)).reshape(1, -1)
        layer2_b = np.array(b[1:2].dropna(axis=1)).reshape(1, -1)
        layer3_b = np.array(b[2:].dropna(axis=1)).reshape(1, -1)

        w = pd.read_csv("Task_1/b/w-100-40-4.csv", header=None).drop(columns=0)
        layer1_w = np.array(w[:14].dropna(axis=1))
        layer2_w = np.array(w[14:114].dropna(axis=1))
        layer3_w = np.array(w[114:].dropna(axis=1))

        self.weights_layers = [layer1_w, layer2_w, layer3_w]
        self.bias_layers = [layer1_b, layer2_b, layer3_b]
        self.layers = []
        self.outputs = []
        self.pre_layer_size = 0

        self.gra_b = []
        self.gra_w = []

    def add_layer(self, nodes, activation_func, num_features=None):
        if num_features is None:
            num_features = self.pre_layer_size
        layer = Layer(activation_func)
        self.layers.append(layer)
        # weights = np.random.randn(num_features, nodes) * np.sqrt(2.0 / num_features)
        # self.weights_layers.append(weights)
        # bias = np.zeros(nodes)
        # self.bias_layers.append(bias)
        self.pre_layer_size = nodes

    def forward(self, input_data):
        output = input_data
        self.outputs = [output]
        for i in range(len(self.layers)):
            Y = output @ self.weights_layers[i] + self.bias_layers[i]
            if self.layers[i].activation_func == "relu":
                output = relu(Y)
            elif self.layers[i].activation_func == "softmax":
                output = softmax(Y)
            self.outputs.append(output)
        return output

    def backward(self, prediction, actual, learning_rate):
        def relu_der(x):
            return np.where(x > 0, 1, 0)

        loss = cross_entropy_loss(actual, prediction)

        der_err = prediction - actual  # Starting point for backpropagation

        # Iterate over layers in reverse order
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # Handle ReLU and softmax activation derivatives
            if layer.activation_func == "relu":
                activation_der = relu_der(self.outputs[i + 1])  # For ReLU
                der_err = der_err * activation_der
            elif layer.activation_func == "softmax":
                der_err = der_err  # Derivative for softmax already calculated

            # Gradient of weights and biases
            der_w = (
                self.outputs[i].T @ der_err / actual.shape[0]
            )  # Gradient w.r.t. weights
            der_b = (
                np.sum(der_err, axis=0, keepdims=True) / actual.shape[0]
            )  # Gradient w.r.t. bias
            self.append_gradient_b(der_b)
            self.append_gradient_w(der_w)

            der_err = der_err @ self.weights_layers[i].T
            # Update weights and biases using gradients
            self.weights_layers[i] -= learning_rate * der_w
            self.bias_layers[i] -= learning_rate * der_b

    def train(self, X, Y, learning_rate, epochs=1):
        for epoch in range(epochs):
            prediction = self.forward(X)
            loss = cross_entropy_loss(Y, prediction)
            self.backward(prediction=prediction, actual=Y, learning_rate=learning_rate)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        return self.get_gradient()

    def predict(self, X):
        return self.forward(X)

    def append_gradient_b(self, der_bias):
        self.gra_b.append(der_bias)

    def append_gradient_w(self, der_weight):
        self.gra_w.append(der_weight)

    def get_gradient(self):
        return self.gra_b, self.gra_w


x_train = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]], dtype=np.float32)
y_train = np.array([[0, 0, 0, 1]], dtype=np.float32)

nn = NN()
nn.add_layer(40, "relu", num_features=14)  # First hidden layer (40 nodes)
nn.add_layer(100, "relu")  # Second hidden layer (100 nodes)
nn.add_layer(4, "softmax")  # Output layer (4 nodes)

bias, weight = nn.train(x_train, y_train, 0.1, 1)

with open("db.csv", "w", newline="") as file:
    writer = csv.writer(file)
    for arr in reversed(bias):
        writer.writerow(arr.flatten())

with open("dw.csv", "w", newline="") as file:
    writer = csv.writer(file)
    for arr in weight[2]:
        writer.writerow(arr.flatten())
    for arr in weight[1]:
        writer.writerow(arr.flatten())
    for arr in weight[0]:
        writer.writerow(arr.flatten())
