from os import error
import random as rndm
import json
import math
import sys
import ai
import numpy as np
import tensorflow
import matplotlib.pyplot as plt

random = rndm.Random()
random.seed()

class NeuralNetwork:
    def __init__(self):

        # initializing member variables
        self.layer_sizes = 0
        self.biases = []
        self.weights = []
        self.z = []
        self.activations = []
        self.gradient = {
            'd_cost_d_bias': [],
            'd_cost_d_activation': [],
            'd_cost_d_weight': []
        }
        self.num_feed_forwards_since_last_epoch = 0
        self.num_feed_forwards_since_last_back_propagate = 0
        self.num_back_propagates_since_last_epoch = 0
        self.num_layers = 0
        self.num_inputs = 0
        self.num_outputs = 0
        self.learning_rate = 0

        return

    def activation_func(self, x):
        # this models a sigmoid function, currently
        try:
            sig = (1.0 / (1.0 + math.exp(-1 * x)))
            return sig
        except OverflowError:
            print("x-val: " + str(x))
            raise OverflowError('overflow')

    def d_activation_func(self, x):
        # this models the derivative of a sigmoid function, currently
        sig = self.activation_func(x)
        return (sig * (1 - sig))

    def cost_func(self, current, intended):
        # cost = (current-intended)^2
        return (math.pow(current - intended, 3))

    def d_cost_func(self, current, intended):
        # models d_cost/d_activation
        # d_cost/d_activation = 2(current-intended)
        return (2 * (current - intended))

    def setup_new(self, lr, args):
        self.learning_rate = lr
        self.layer_sizes = args

        num_neurons_in_prev_layer = 0

        # for each layer
        layer = 0
        for num_neurons_per_layer in args:

            # append an array to each network array for each layer
            self.biases.append([])
            self.weights.append([])
            self.z.append([])
            self.activations.append([])
            self.gradient['d_cost_d_bias'].append([])
            self.gradient['d_cost_d_activation'].append([])
            self.gradient['d_cost_d_weight'].append([])

            # for each neuron in the layer
            for neuron in range(num_neurons_per_layer):

                # creating weight array that contains all weights attached to the current neuron
                weights = []
                d_cost_d_weight = []
                for prev_neuron in range(num_neurons_in_prev_layer):
                    weights.append(random.random())
                    d_cost_d_weight.append(0)

                #  appending weight array for each neuron
                self.weights[layer].append(weights)
                self.gradient['d_cost_d_weight'][layer].append(d_cost_d_weight)

                #  appending random value for each neuron's bias
                self.biases[layer].append(random.random())

                # appending 0 for each neuron's z value
                self.z[layer].append(0)

                #  appending 0 for each neuron's activation
                self.activations[layer].append(0)

                #
                self.gradient['d_cost_d_bias'][layer].append(0)

                #
                self.gradient['d_cost_d_activation'][layer].append(0)

            layer += 1
            num_neurons_in_prev_layer = num_neurons_per_layer

        self.num_layers = layer
        self.num_inputs = len(self.activations[0])
        self.num_outputs = len(self.activations[self.num_layers - 1])

        return

    def setup_existing(self, layer_sizes, vars, biases, weights, z, activations, gradient):
        self.layer_sizes = layer_sizes
        self.biases = biases
        self.weights = weights
        self.z = z
        self.activations = activations
        self.gradient = gradient
        self.num_feed_forwards_since_last_epoch = vars[0]
        self.num_feed_forwards_since_last_back_propagate = vars[1]
        self.num_back_propagates_since_last_epoch = vars[2]
        self.num_layers = vars[3]
        self.num_inputs = vars[4]
        self.num_outputs = vars[5]
        self.learning_rate = vars[6]

        return

    def save_to_file(self, path):
        data = [
            self.layer_sizes,
            [
                self.num_feed_forwards_since_last_epoch,
                self.num_feed_forwards_since_last_back_propagate,
                self.num_back_propagates_since_last_epoch,
                self.num_layers,
                self.num_inputs,
                self.num_outputs,
                self.learning_rate
            ],
            self.biases,
            self.weights,
            self.z,
            self.activations,
            self.gradient
        ]

        try:
            with open(path, 'w') as writer:
                json.dump(data, writer)

            return True
        except error:
            print("save err\n")
            print(error.strerror)
            return False

    def setup_from_file(self, path):
        try:
            with open(path, 'r') as reader:
                data = json.load(reader)

            self.setup_existing(
                layer_sizes = data[0],
                vars = data[1],
                biases = data[2],
                weights = data[3],
                z = data[4],
                activations = data[5],
                gradient = data[6])

            return True
        except:
            print("setup err\n")
            return False

    def activate(self, L, n):
        z = 0
        weights = self.weights[L][n]
        bias = self.biases[L][n]
        activations = self.activations[L - 1]

        i = 0
        for weight in weights:
            z += weight * activations[i]
            i += 1

        z += bias

        self.z[L][n] = z

        activation = self.activation_func(z)
        # print(activation)
        self.activations[L][n] = activation

        return

    def feed_forward(self, inp):
        if type(inp) == np.ndarray:
            inputs = inp.tolist()
        else:
            inputs = inp

        if len(inputs) != self.num_inputs:
            print("Wrong number of inputs.\n")
            return []

        # grabbing the inputs
        self.activations[0] = inputs

        # for each layer in the network
        for L in range(1, self.num_layers):

            # for each neuron in the layer
            for n in range(self.layer_sizes[L]):
                self.activate(L, n)

        # activations from the output layer are the outputs
        outputs = self.activations[self.num_layers - 1]

        self.num_feed_forwards_since_last_epoch += 1
        self.num_feed_forwards_since_last_back_propagate += 1

        return outputs

    def back_propagate(self, corr_vals):
        if type(corr_vals) == np.ndarray:
            correct_vals = corr_vals.tolist()
        else:
            correct_vals = corr_vals

        if len(correct_vals) != self.num_outputs:
            print("Wrong number of outputs.\n")
            return 0

        total_cost = 0
        last_layer_index = self.num_layers - 1
        for n in range(len(correct_vals)):
            # calculating cost as C = summation(from n=0 to j)[ (a_L_n - correct_val) ^ 2 ]
            total_cost += self.cost_func(self.activations[last_layer_index][n], correct_vals[n])
            self.gradient['d_cost_d_activation'][last_layer_index][n] = self.d_cost_func(self.activations[last_layer_index][n], correct_vals[n])

        # for each layer, incrementing backwards, from the last layer to the second layer ([1, L], inclusive)
        for L in range(last_layer_index, 0, -1):

            # for each n-th neuron in the L-th layer
            for n in range(len(self.activations[L])):

                # the derivative of the activation function with respect to z
                d_activation_d_z = self.d_activation_func(self.z[L][n])

                # the derivative of cost with respect to activation
                d_cost_d_activation = self.gradient['d_cost_d_activation'][L][n]
                self.gradient['d_cost_d_activation'][L][n] = 0

                if L != 1:
                    for n_prev in range(len(self.activations[L - 1])):
                        self.gradient['d_cost_d_activation'][L - 1][n_prev] += self.weights[L][n][n_prev] * \
                            d_activation_d_z * d_cost_d_activation

                # the derivative of cost with respect to bias
                self.gradient['d_cost_d_bias'][L][n] += (d_activation_d_z * d_cost_d_activation)
                # print(self.gradient['d_cost_d_bias'][L][n])
                # for each weight connecting to the n-th neuron in the L-th layer
                for w in range(len(self.weights[L][n])):
                    # the derivative of cost with respect to weight
                    self.gradient['d_cost_d_weight'][L][n][w] += (
                        self.activations[L - 1][w] * d_activation_d_z * d_cost_d_activation)

        self.num_feed_forwards_since_last_back_propagate = 0
        self.num_back_propagates_since_last_epoch += 1

        return total_cost

    def apply_gradient(self):
        for L in range(self.num_layers):
            for n in range(len(self.activations[L])):
                self.biases[L][n] -= (self.gradient['d_cost_d_bias'][L][n] / self.num_back_propagates_since_last_epoch) * self.learning_rate
                # print((self.gradient['d_cost_d_bias'][L][n]))
                self.gradient['d_cost_d_bias'][L][n] = 0
                for w in range(len(self.weights[L][n])):
                    self.weights[L][n][w] -= (self.gradient['d_cost_d_weight'][L][n][w] / self.num_back_propagates_since_last_epoch) * self.learning_rate
                    # print((self.gradient['d_cost_d_weight'][L][n][w]))
                    self.gradient['d_cost_d_weight'][L][n][w] = 0

        self.num_feed_forwards_since_last_epoch = 0
        self.num_back_propagates_since_last_epoch = 0

        return

mnist = tensorflow.keras.datasets.mnist

(tr_images, train_labels), (te_images, test_labels) = mnist.load_data()

train_images = tr_images / 255.0

test_images = te_images / 255.0

data_file = "nn_data.txt"

input_file = "neural_network_record.txt"

num_inputs = 784

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

nn = NeuralNetwork()
if sys.argv[1] == "1":
    nn.setup_from_file(data_file)
else:
    nn.setup_new(1, [num_inputs, 128, 10])



if sys.argv[2] == "1":
    for test in range(int(sys.argv[3])):
        result = nn.feed_forward(test_images[test].flatten().tolist())
        c_vals = [0] * 10
        c_vals[test_labels[test]] = 1
        print(c_vals)
        print("correct: " + str(test_labels[test]))
        print("nn: " + str(result))
    
else:
    try:
        for epoch in range(int(sys.argv[3])):
            for i in range(len(train_images)):
                arr = train_images[i].flatten().tolist()
                nn.feed_forward(arr)
                c_vals = [0] * 10
                c_vals[train_labels[i]] = 1
                nn.back_propagate(c_vals)
                nn.apply_gradient()
                # if (i + 1) % 100 == 0:
                #     nn.apply_gradient()
                #     print(str(i))
    except KeyboardInterrupt:
        nn.save_to_file(data_file)
        sys.exit()


    nn.save_to_file(data_file)