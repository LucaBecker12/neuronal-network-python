import numpy as np
import csv
from scipy.special import expit
import matplotlib.pyplot as plt


class neuralNetwork():
    def __init__(self, in_nodes, h_nodes, out_nodes, learning_rate=0.5):
        self.in_nodes = in_nodes
        self.h_nodes = h_nodes
        self.out_nodes = out_nodes

        self.learning_rate = learning_rate

        self.w_ih = np.random.normal(
            0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.in_nodes))
        self.w_ho = np.random.normal(
            0.0, pow(self.out_nodes, -0.5), (self.out_nodes, self.h_nodes))

        self.activation_function = lambda x: expit(x)

    def train(self, data_inputs, data_targets):
        inputs = np.array(data_inputs, ndmin=2).T
        targets = np.array(data_targets, ndmin=2).T

        hidden_inputs = np.dot(self.w_ih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.w_ho, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.w_ho.T, output_errors)

        self.w_ho += self.learning_rate * \
            np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                   np.transpose(hidden_outputs))

        self.w_ih += self.learning_rate * \
            np.dot((hidden_errors * hidden_outputs *
                    (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, data):
        inputs = np.array(data, ndmin=2).T

        hidden_inputs = np.dot(self.w_ih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.w_ho, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def main():
    in_nodes = 784
    h_nodes = 100
    out_nodes = 10

    learning_rate = 0.3

    network = neuralNetwork(in_nodes, h_nodes, out_nodes, learning_rate)

    for i in range(10):
        train(out_nodes, network)

    with open('test_set.csv') as test_file:
        reader = csv.reader(test_file)

        scorecard = []
        for i, data in enumerate(list(reader)):
            inputs = (np.asfarray(data[1:]) / 255 * 0.99) + 0.01
            answer = int(data[0])

            output = network.query(inputs)

            label = np.argmax(output)

            if label == answer:
                scorecard.append(1)
            else:
                scorecard.append(0)

        arr = np.asarray(scorecard)
        performance = sum(arr) / arr.size

        print(performance)


def train(out_nodes, network):
    reader = csv.reader(open("train_set.csv"))

    for data in list(reader):
        inputs = (np.asfarray(data[1:]) / 255 * 0.99) + 0.01

        targets = np.zeros(out_nodes) + 0.01
        targets[int(data[0])] = 0.99

        network.train(inputs, targets)


if __name__ == "__main__":
    main()
