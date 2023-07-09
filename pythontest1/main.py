# This is a sample Python script.
import copy

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import openpyxl
from numpy import double
import torch

# 男0女1

# Our activation function: f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


# y_true and y_pred are numpy arrays of the same length.
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()  # mean()求均值


# 神经元
'''class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # Weight inputs, add bias, then use the activation function
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


weights = np.array([0, 1])  # w1=0,w2=1
bias = 4  # b=4
n = Neuron(weights, bias)

x = np.array([2, 3])  # x1 = 2, x2 = 3
print(n.feedforward(x))  # 0.9990889488055994s
'''


# 神经网络
class OurNeuralNetwork:
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000  # number of times to loop through the entire dataset
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # --- Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)  # 得到预估的o1 np.apply_along_axis()沿轴调用函数
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))
        print(self.w1)
        print(self.w2)
        print(self.w3)
        print(self.w4)
        print(self.w5)
        print(self.w6)
        print(self.b1)
        print(self.b2)
        print(self.b3)


def kg_to_p(weight):
    return weight * 2.2 - 135


def cm_to_inch(height):
    return height * 0.39 - 66


# Man 0  Female 1
def sex_invert(sex):
    return sex ^ 1


'''network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))  # 0.7216325609518421

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred))  # 0.5


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

'''
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define dataset
    csv = pd.read_excel("/home/lzt/桌面/Data.xlsx", header=None, dtype=int, skiprows=1)
    initial_data = np.array(csv)
    data = copy.deepcopy(initial_data)
    data[:, 0] = np.apply_along_axis(kg_to_p, 0, initial_data[:, 2])
    data[:, 1] = np.apply_along_axis(cm_to_inch, 0, initial_data[:, 1])
    data[:, 2] = np.apply_along_axis(sex_invert, 0, initial_data[:, 0])
    # print(f"{data}")
    # print(kg_to_p(58.5))

    all_y_trues = copy.deepcopy(data[:, 2])
    '''data = np.array([
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ])
    all_y_trues = np.array([
        1,
        0,
        0,
        1,
    ])'''

    # Train our neural network!
    network = OurNeuralNetwork()
    # network.train(data, all_y_trues)
    network.w1 = 1.0712953533268055
    network.w2 = 1.1435723968284752
    network.w3 = -1.4216670725389606
    network.w4 = 0.5014047324702274
    network.w5 = -2.1556450786754042
    network.w6 = -2.1556450786754042
    network.b1 = -10.733199658761652
    network.b2 = -15.017140556133981
    network.b3 = -0.2111903409175054

    # Make some predictions
    emily = np.array([-7, -3])  # 128 pounds, 63 inches
    luziteng = np.array([-7, 5.5])
    dongchengri = np.array([-12, 5.5])
    shenhaoran = np.array([5, 5.5])

    print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
    print("luziteng: %.3f" % network.feedforward(luziteng))
    print("dongchengri: %.3f" % network.feedforward(dongchengri))
    print("shenhaoran: %.3f" % network.feedforward(shenhaoran))

    # print(type(data))
    # print(f"{data}")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
