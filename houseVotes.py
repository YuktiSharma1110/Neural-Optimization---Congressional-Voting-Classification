import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from NN_house_votes import NN
from sklearn.preprocessing import StandardScaler



def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def custom_stratified_k_fold(X, y, k):
    # Convert to numpy arrays if not already, to facilitate easy indexing
    X = np.asarray(X)
    y = np.asarray(y)

    # Determine unique classes and their distribution
    classes, y_indices = np.unique(y, return_inverse=True)
    class_counts = np.bincount(y_indices)

    number_of_samples = len(y)
    number_of_classes = len(classes)

    # Initialize the folds by lists of indices
    folds = [[] for _ in range(k)]

    # For each class, distribute the samples across folds
    for cls_index, cls in enumerate(classes):
        # Find the indices of all samples belonging to the current class
        indices = np.where(y == cls)[0]
        np.random.shuffle(indices)  # Shuffle to ensure random distribution

        # Evenly distribute indices of the current class across folds
        n_samples_for_cls = len(indices)
        n_samples_per_fold = np.full(k, n_samples_for_cls // k, dtype=int)

        # Handle remainder of the distribution by adding one more sample to some folds
        remainder = n_samples_for_cls % k
        n_samples_per_fold[:remainder] += 1

        # Distribute the samples across folds
        current_idx = 0
        for fold_index in range(k):
            start, stop = current_idx, current_idx + n_samples_per_fold[fold_index]
            folds[fold_index].extend(indices[start:stop])
            current_idx = stop

    # Shuffle indices within each fold to ensure random ordering
    for fold in folds:
        np.random.shuffle(fold)

    # Display the class distribution in each fold

    for i, fold in enumerate(folds):
        print(f"Fold {i + 1}:")
        # Initialize a dictionary to count the occurrences of each class in the fold
        class_distribution = {cls: 0 for cls in classes}
        for index in fold:
            class_distribution[y[index]] += 1
        # Print the distribution
        for cls, count in class_distribution.items():
            print(f"  Class {cls}: {count} instances")

    return folds


def upload_dataset():
    df = pd.read_csv('hw4_house_votes_84.csv')

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # print(y)
    # print(X)

    folds = custom_stratified_k_fold(X, y, 10)

    for fold_idx in range(len(folds)):
        test_index = folds[fold_idx]
        train_index = np.hstack([folds[i] for i in range(len(folds)) if i != fold_idx])

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = upload_dataset()
    input_size = X_train.shape[1]
    scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    output_size = 2

    #[input_size,2,6,output_size],[input_size,20,30,30,output_size],[input_size,20,10,10,output_size]
    layers = [[input_size,200,100,250,300,64,output_size],[input_size,8,50,250,300,64,8,output_size],[input_size,64,128,256,64,64,output_size],[input_size,200,100,250,output_size],[input_size,300,250,200,output_size],[input_size,300,250,output_size]]
    lr = [1e-3,1e-3,0.9,0.2,0.1,0.25,0,5,0.7,0.8]
    lamVal = [1e-3,1e-4,0.5,0.25,0.1,0.65,0.9]

    max_acc = 0
    max_layer_size = []
    max_lr = 0
    max_lam_val = 0

    # Loop over configurations
    for layer_sizes in layers:
        print("***********************************************************************************************************************")
        for learning_rate in lr:
            for lam in lamVal:
                nn = NN(layer_sizes, initialization='uniform', lam=lam)
                print(f"Training NN with layers {layer_sizes}, lr {learning_rate}, lambda {lam}")
                layer_acc,layer_F1, loss_val = nn.train(X_train, y_train, X_test, y_test,layer_sizes, epochs=200, learning_rate=learning_rate)
                predictions, _ = nn.forward(X_test)

                print("Layer Accuracy: ", layer_acc, "Layer F1 ",layer_F1, "Layer Size : ", layer_sizes, "Learning rate ", learning_rate, " Lam: ",
                      lam)

                if(layer_acc > max_acc):
                    max_acc = layer_acc
                    max_layer_size = layer_sizes
                    max_lr = learning_rate
                    max_lam_val = lam

                predicted_classes = np.argmax(predictions, axis=1)
                #print(f"Predicted Classes: {predicted_classes}")
                #print(f"actual Classes: {y_test}")


    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    print("Max Accuracy: ", max_acc , " Layer Size : ", max_layer_size , "Learning rate ",max_lr, " Lam: ", max_lam_val)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


    test_losses = []
    training_examples = [5, 10, 20,  30, 40, 50, 60, 70, 80,100,135,140,150]
    for example in training_examples:
        nn =  NN(max_layer_size, initialization='uniform', lam=max_lam_val)

        test_acc ,test_f1, test_loss = nn.train(X_train[:example], y_train[:example], X_test, y_test,layer_sizes, epochs=100, learning_rate=learning_rate)
        print("train eg ", example, "test_acc", test_acc, "test f1 ",test_f1,"test loss", test_loss)
        test_losses.append(test_loss)

    plt.figure(figsize=(8, 6))
    plt.plot(training_examples, test_losses, marker='o')
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs. Number of Training Examples")
    plt.grid(True)

    # Save the plot as a file
    plt.savefig("House_Votes_training_vs_loss.png")


