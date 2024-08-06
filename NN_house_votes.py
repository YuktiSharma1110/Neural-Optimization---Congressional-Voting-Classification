import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class NN:
    def __init__(self, layer_sizes, initialization='uniform',lam=0.1):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.initialization = initialization
        self.lam = lam
        self.weights = []
        self.biases = []
        self.init_weights_and_biases()

    def custom_accuracy_score(y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions

    def custom_precision_score(self,y_true, y_pred):
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        precision = 0

        for label in unique_classes:
            tp = np.sum((y_pred == label) & (y_true == label))
            fp = np.sum((y_pred == label) & (y_true != label))
            if (tp + fp) > 0:
                label_precision = tp / (tp + fp)
            else:
                label_precision = 0
            precision += label_precision

        precision /= len(unique_classes)
        return precision

    def custom_recall_score(self,y_true, y_pred):
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        recall = 0

        for label in unique_classes:
            tp = np.sum((y_pred == label) & (y_true == label))
            fn = np.sum((y_pred != label) & (y_true == label))
            if (tp + fn) > 0:
                label_recall = tp / (tp + fn)
            else:
                label_recall = 0
            recall += label_recall

        # Averaging over all classes
        recall /= len(unique_classes)
        return recall

    def custom_f1_score(self,y_true, y_pred):
        precision = self.custom_precision_score(y_true, y_pred)
        recall = self.custom_recall_score(y_true, y_pred)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return f1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def init_weights_and_biases(self):
        for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if self.initialization == 'uniform':
                self.weights.append(np.random.uniform(-1, 1, (y, x)))
            elif self.initialization == 'gaussian':
                self.weights.append(np.random.randn(y, x))
            self.biases.append(np.random.randn(y, 1))

    def compute_loss(self, y_true, y_pred,lam):
        """Compute the categorical cross-entropy loss with L2 regularization."""
        m = y_true.shape[0]
        loss = -np.sum(np.log(y_pred[np.arange(m), y_true])) / m
        reg_term = 0
        for weights in self.weights:
            reg_term += np.sum(weights ** 2)
        loss += lam * reg_term / (2 * m)

        return loss

    def forward(self, X):
        activations = [X]
        input = X

        for i in range(len(self.weights) - 1):
            input = self.sigmoid(np.dot(input, self.weights[i].T))
            activations.append(input)

        output = self.softmax(np.dot(input, self.weights[-1].T))
        activations.append(output)
        return output, activations

    def backprop(self, X, y_true, activations, learning_rate):
        m = y_true.shape[0]
        output = activations[-1]
        y_true_onehot = np.zeros_like(output)
        y_true_onehot[np.arange(m), y_true] = 1

        d_output = -(y_true_onehot - output)
        d_weights = []
        d_biases = []

        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                d_input = d_output
            else:
                d_input = np.dot(d_hidden, self.weights[i+1]) * self.sigmoid_derivative(activations[i+1])

            d_weights.append(np.dot(d_input.T, activations[i]) / m)

            #d_biases.append(np.sum(d_input, axis=0, keepdims=True) / m)

            d_hidden = d_input

        d_weights = d_weights[::-1]
        d_biases = d_biases[::-1]

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * (d_weights[i] + 0.01 * self.weights[i])
            #self.biases[i] -= learning_rate * np.squeeze(d_biases[i], axis=1)

    def train(self, X, y, X_test, y_test,layer_sizes, epochs=10, learning_rate=0.01, lam=0.01):
        losses = []
        train_accs = []
        test_accs = []

        train_f1Scores = []
        test_f1Scores = []

        mean_train_f1Scores = []  # List to hold mean F1 scores for training data for each epoch
        mean_test_f1Scores = []  # List to hold mean F1 scores for testing data for each epoch

        sum_of_accuracies = 0
        sum_of_f1Scores = 0

        losses = []
        loss_tests=[]

        for epoch in range(epochs):
            y_pred, activations = self.forward(X)
            loss = self.compute_loss(y, y_pred, lam)
            losses.append(loss)
            self.backprop(X, y, activations, learning_rate)

            y_pred_class = np.argmax(y_pred, axis=1)
            train_acc = np.mean(y_pred_class == y)
            train_accs.append(train_acc)

            train_f1 = self.custom_f1_score(y, y_pred_class)
            train_f1Scores.append(train_f1)

            y_pred_test, _ = self.forward(X_test)

            loss_test = self.compute_loss(y_test, y_pred_test,lam)
            loss_tests.append(loss_test)

            y_pred_class_test = np.argmax(y_pred_test, axis=1)

            test_acc = np.mean(y_pred_class_test  == y_test)
            test_accs.append(test_acc)
            sum_of_accuracies+=test_acc

            test_f1 = self.custom_f1_score(y_test, y_pred_class_test )
            test_f1Scores.append(test_f1)
            sum_of_f1Scores+=test_f1

            # Calculate and append the mean F1 scores for current epoch
            mean_train_f1 = np.mean(train_f1Scores)
            mean_test_f1 = np.mean(test_f1Scores)
            mean_train_f1Scores.append(mean_train_f1)
            mean_test_f1Scores.append(mean_test_f1)


        epoch_accuracy = sum_of_accuracies/epochs
        epoch_f1_score = sum_of_f1Scores/epochs
        #print("layers size ", layer_sizes, "accuracy ", epoch_accuracy)

        return epoch_accuracy, epoch_f1_score, loss_tests[-1]

