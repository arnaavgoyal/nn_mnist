from os import error
import random as rndm
import json
import math
import sys
import ai
import numpy as np
import tensorflow
import matplotlib.pyplot as mpl


class NeuralNetwork:
    def create(self, lr: float, layers: np.ndarray, *, using_data=False, data: dict = None):

        net_shape = (0, 0)
        net_weight_shape = (0, 0)

        if using_data and (data != None):
            bi = data['b']
            we = data['w']
            self.learning_rate = data['lr']
            self.layers = data['l']
        else:
            self.learning_rate = lr
            self.layers = layers

        self.num_layers = len(self.layers)
        self.max_num = self.layers.max()
        self.num_inputs = self.layers[0]
        self.num_outputs = self.layers[-1]

        net_shape = (self.num_layers, self.max_num)
        net_weight_shape = (self.num_layers, self.max_num, self.max_num)

        if not using_data or (data is None):
            bi = np.zeros(net_shape)
            we = np.zeros(net_weight_shape)

        self.biases = bi
        self.weights = we

        self.num_feed_forwards_since_last_epoch = 0
        self.num_back_propagates_since_last_epoch = 0
        self.num_feed_forwards_since_last_back_propagate = 0

        self.activations = np.empty(net_shape)
        self.z = np.empty(net_shape)
        self.gradient = {
            'd_cost_d_activation': np.zeros(net_shape),
            'd_cost_d_bias': np.zeros(net_shape),
            'd_cost_d_weight': np.zeros(net_weight_shape)
        }

        random = rndm.Random()
        random.seed()

        prev_layer = self.layers[0]
        
        for L in range(1, self.num_layers):
            for n in range(self.layers[L]):
                self.biases[L, n] = random.random()
                for w in range(prev_layer):
                    self.weights[L, n, w] = random.random()
            prev_layer = self.layers[L]

        del random

        return

    def activation_func(self, x: np.ndarray) -> np.ndarray:
        # this models a sigmoid function, currently
        return ((1.0 / (1.0 + np.exp(-1 * x))))

    def d_activation_func(self, x: np.ndarray) -> np.ndarray:
        # this models the derivative of a sigmoid function, currently
        sig = self.activation_func(x)
        return (sig * (1 - sig))

    def cost_func(self, current: np.ndarray, intended: np.ndarray) -> np.ndarray:
        # cost = (current-intended)^2
        i_l = len(intended)
        intended_fit = np.zeros(self.max_num)
        intended_fit[:i_l] = intended
        return (np.power(current - intended_fit, 3))

    def d_cost_func(self, current: np.ndarray, intended: np.ndarray) -> np.ndarray:
        # models d_cost/d_activation
        # d_cost/d_activation = 2(current-intended)
        i_l = len(intended)
        intended_fit = np.zeros(self.max_num)
        intended_fit[:i_l] = intended
        return (2 * (current - intended_fit))

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        if len(inputs) != self.num_inputs:
            raise Exception('Wrong number of inputs')

        # grabbing the inputs
        self.activations[0] = inputs

        for L in range(1, self.num_layers):
            self.z[L] = np.add(np.dot(self.weights[L], self.activations[L - 1]), self.biases[L])
            # np.add(np.einsum('ij,j->i', self.weights[L], self.activations[L - 1]), self.biases[L])
            # works as well

            self.activations[L] = self.activation_func(self.z[L])

        # activations from the output layer are the outputs
        outputs = self.activations[-1]

        self.num_feed_forwards_since_last_epoch += 1
        self.num_feed_forwards_since_last_back_propagate += 1

        return outputs

    def back_propagate(self, correct_vals: np.ndarray):
        if len(correct_vals) != self.num_outputs:
            raise Exception('Wrong number of expected outputs')

        # calculating cost as C = summation(from n=0 to j)[ (a_L_n - correct_val) ^ 2 ]
        total_cost = np.sum(self.cost_func(self.activations[-1], correct_vals))
        self.gradient['d_cost_d_activation'][-1] = self.d_cost_func(self.activations[-1], correct_vals)

        # for each layer, incrementing backwards, from the last layer to the second layer ([1, L], inclusive)
        for L in range(-1, -1 * self.num_layers, -1):

            # the derivative of the activation function with respect to z
            d_activation_d_z = self.d_activation_func(self.z[L])

            # the derivative of cost with respect to activation
            d_cost_d_activation = self.gradient['d_cost_d_activation'][L]
            self.gradient['d_cost_d_activation'][L] = np.zeros(self.max_num)

            # the d_cost_d_activation of the previous layer
            if L != 1:
                self.gradient['d_cost_d_activation'][L - 1] = np.swapaxes(self.weights[L], 0, 1) @ (d_activation_d_z * d_cost_d_activation)

            # the derivative of cost with respect to bias
            self.gradient['d_cost_d_bias'][L] += (d_activation_d_z * d_cost_d_activation)

            # for each weight connecting to the n-th neuron in the L-th layer
            # the derivative of cost with respect to weight
            self.gradient['d_cost_d_weight'][L] += np.einsum('i,j->ij', (d_activation_d_z * d_cost_d_activation), self.activations[L - 1])

        self.num_feed_forwards_since_last_back_propagate = 0
        self.num_back_propagates_since_last_epoch += 1

        return total_cost

    def apply_gradient(self):

        self.biases[1:] -= self.gradient['d_cost_d_bias'][1:] * (self.learning_rate / self.num_back_propagates_since_last_epoch)

        self.gradient['d_cost_d_bias'][1:].fill(0)
        self.weights[1:] -= self.gradient['d_cost_d_weight'][1:] * (self.learning_rate / self.num_back_propagates_since_last_epoch)

        self.gradient['d_cost_d_weight'][1:].fill(0)

        self.num_feed_forwards_since_last_epoch = 0
        self.num_back_propagates_since_last_epoch = 0

        return

    def save_to(self, path: str):
        try:
            np.savez(path,
                b  = self.biases,
                w  = self.weights,
                lr = self.learning_rate,
                l  = self.layers)
        except Exception:
            print('save err')
        finally:
            return

    def load_from(self, path: str):
        
        data_arrays = np.load(path)
        self.create(0, np.ndarray([0]), using_data=True, data=data_arrays)
        data_arrays.close()
        



mnist = tensorflow.keras.datasets.mnist

(tr_images, train_labels), (te_images, test_labels) = mnist.load_data()

train_images = tr_images / 255.0

test_images = te_images / 255.0

data_file = "nn_data.npz"

input_file = "neural_network_record.txt"

num_inputs = 784

# fig = mpl.figure()
# fig, ax_list = mpl.subplots()
# mpl.imshow(train_images[0])
# mpl.colorbar()
# mpl.grid(False)
# mpl.show()

nn = NeuralNetwork()
if sys.argv[1] == "1":
    nn.load_from(data_file)
else:
    nn.create(1, np.array([num_inputs, 128, 10]))


if sys.argv[2] == "1":
    for test in range(int(sys.argv[3])):
        result = nn.feed_forward(test_images[test].flatten())
        c_vals = [0] * 10
        c_vals[test_labels[test]] = 1
        print(c_vals)
        print("correct: " + str(test_labels[test]))
        print("nn: " + str(result))
    
else:
    try:
        for epoch in range(int(sys.argv[3])):
            for i in range(len(train_images)):
                arr = train_images[i].flatten()
                nn.feed_forward(arr)
                c_vals = [0] * 10
                c_vals[train_labels[i]] = 1
                nn.back_propagate(c_vals)
                nn.apply_gradient()
                # if (i + 1) % 100 == 0:
                #     nn.apply_gradient()
                #     print(str(i))
    except KeyboardInterrupt:
        nn.save_to(data_file)
        sys.exit()


    nn.save_to(data_file)