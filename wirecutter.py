import numpy as np # import NumPy library
from wirediagrams import wire_diagram_dataset # import our 'wire_diagram_dataset' generated from wirediagrams.py

# SOFTMAX REGRESSION MODEL FOR MULTI-CLASS CLASSIFICATION

def softmax_func(m): # function to apply the softmax function to input matrix 'm'
    exponentials = np.exp(m - np.max(m, axis=1, keepdims=True)) # calculate the exponentials
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True) # calculate probabilities by using the softmax function to normalize the calculated exponentials

    return probabilities # return the calculated probability matrix 

def initialize_parameters(input_size, output_size): # function to parameterize our softmax regression model with weights and bias
    w = np.zeros((input_size, output_size))
    b = np.zeros((1, output_size))
    return w, b

def concatenate_nonlinear_features(M): # function that adds squared features to the model to introduce non-linearity
    # computes the square of each of the features in input matrix 'M' and returns concatenated array of M and M_sqrd together
    M_sqrd = M*M
    return np.concatenate((M, M_sqrd), axis=1)

# function to compute the cross-entropy loss and its gradient during forward/backward propagation. 
# takes in the weight, bias, input data matrix 'X', the vector containing the diagram's labels 'Y', and the lambda regularization parameter for controlling the regularization strength
def propagate(w, b, X, Y, lambda_regularization):
    m = X.shape[0] # retrieve the number of training wire_diagrams in the set
    A = softmax_func(np.dot(X, w) + b) # apply our softmax function to the linear transformation

    cross_entropy_loss = -1/m * np.sum(Y * np.log(A + 1e-15)) # calculate the cross-entropy loss value
    l2_regularization = (lambda_regularization / (2 * m)) * np.sum(w**2) # compute our L2 regularization term by multiplying lambda by the squared weight of every feature

    cross_entropy_loss += l2_regularization # add our cross_entropy_loss with our l2_regularization value (to prevent overfitting) to compute our final 'cross_entropy_loss'

    dw = (1/m) * np.dot(X.T, (A - Y)) + (lambda_regularization / m) * w # calculate the gradient loss with respect to the weight
    db = 1/m * np.sum(A - Y) # calculate the gradient loss with respect to the bias

    return dw, db, cross_entropy_loss, A #returns the gradient loss with respect to the weight, bias, and logistic regression (including L2 regularization), and the softmax activation value A

def calculate_prediction_accuracy(A, Y):
    predictions = np.argmax(A, axis=1)
    true_labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == true_labels)
    return accuracy

def encode_categorical_labels(labels): # function to convert the categorical labels (1-4 , representing each color Red, Blue, Yellow and Green) to a binary matrix

    unique_labels = np.unique(labels) # use np.unique to retrieve the unique labels out of the labels in the input matrix/array
    encoded_labels = np.zeros((len(labels), len(unique_labels))) # initialize a matrix 'encoded_labels' to all 0s, it will store the binary version of the corresponding labels

    # use a for loop to populate the 'encoded_labels' matrix, assigning 1 to the corresponding position(s) in the current row 'i' of the 'encoded_labels' matrix where the label occurs.
    for i, label in enumerate(labels):
        encoded_labels[i, np.where(unique_labels == label)] = 1

    return encoded_labels

def train(X_train, Y_train, learning_rate, num_iterations, display_step, lambda_regularization):
    
    # retrieve both input/output sizes
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1]

    # call 'init_params' to initialize the weights and bias, taking in the calculated number of features
    w, b = initialize_parameters(input_size, output_size)

    # initialize empty lists to store and keep track of cross-entropy loss and accuracy throughout the training period
    cross_entropy_loss_overtime = []
    accuracy_overtime = []

    for i in range(num_iterations):
        # call 'propagate' to retrieve the calculated cross-entropy loss and the gradient of the loss
        dw, db, cross_entropy_loss, A = propagate(w, b, X_train, Y_train, lambda_regularization)

        # update weights and bias w/ gradient descent
        w -= learning_rate * dw
        b -= learning_rate * db

        if i % display_step == 0: # display the current accuracy and cross entropy loss after every 100 passes.

            accuracy = calculate_prediction_accuracy(A, Y_train)

            print(f"Pass #{i}: Cross-Entropy loss: {cross_entropy_loss}, Accuracy: {accuracy}%")

            # append both values to our two lists to keep track of "learning" progress
            cross_entropy_loss_overtime.append(cross_entropy_loss)
            accuracy_overtime.append(accuracy)

    return w, b, cross_entropy_loss_overtime, accuracy_overtime

def test(X_test, Y_test, w, b): # function to test the softmax regression model using the 1000 wire diagrams dataset that was split from the training dataset

    # calculate the softmax activation value, the cross_entropy_loss, and the accuracy
    A = softmax_func(np.dot(X_test, w) + b)
    cross_entropy_loss = -1 / X_test.shape[0] * np.sum(Y_test * np.log(A + 1e-15))
    accuracy = calculate_prediction_accuracy(A, Y_test)

    # display softmax regression model testing results
    print(f"\nTesting wire diagram dataset... Cross-entropy loss: {cross_entropy_loss}, Accuracy: {accuracy}\n")

# filter our dataset to only retrieve wire diagrams with an associated label of 'Dangerous' because our softmax regression model only takes in Dangerous wire diagrams as input
dangerous_diagrams = [(diagram, label, cut_color) for diagram, label, cut_color in wire_diagram_dataset if label == 'Dangerous']

features = np.array([np.ravel(diagram) for diagram, _, _ in dangerous_diagrams])  # iterates over each wire_diagram in 'wire_diagram_dataset', flattens the wire diagrams and store them in 'features' 
cut_colors = np.array([cut_color for _, _, cut_color in dangerous_diagrams]) # stores the color to cut associated with each 'Dangerous' wire diagram, calculated from wirediagrams.py

encoded_cut_colors = encode_categorical_labels(cut_colors) #store the resulting labels retrieved from calling 'encode_categorical_labels'

# call the 'concatenate_nonlinear_features' to add the squared non-linear features to the original features
nonlinear_features = concatenate_nonlinear_features(features)

# shuffle the dataset once more to prevent pattern recognition from the model
shuffled_index_order = np.random.permutation(len(nonlinear_features))
shuffled_features = nonlinear_features[shuffled_index_order]
shuffled_encoded_cut_colors = encoded_cut_colors[shuffled_index_order]

# training parameters
learning_rate = 0.01 # set the learning_rate to 0.01, which means at each iteration of training, the model parameters (weights and biases) will be updated by a fraction of 0.01 times the gradient of the loss function with respect to those parameters
passes = 2000 # total number of passes the model will take 
display_checkpoint = 100 # every 100 passes, display the log loss/accuracy
lambda_regularization = 0.1  # Adjust the regularization strength as needed

# train the model and test its performance for different dataset sizes of 500, 1000, 2500, and 5000 (all unique wire diagrams, non-repeating)
dataset_subset_sizes = [500, 1000, 2500, 5000] 
for subset_size in dataset_subset_sizes:
    # shuffle the training datasets to acquire new and unique training wire diagrams
    X_train_subset, Y_train_subset = shuffled_features[:subset_size], shuffled_encoded_cut_colors[:subset_size]
    w, b, cross_entropy_loss_overtime, accuracy_overtime = train(X_train_subset, Y_train_subset, learning_rate, passes, display_checkpoint, lambda_regularization)

    # print the avg cross-entropy loss and accuracy for each of the different training subsets (500, 1k, 2.5k, and 5k)
    average_loss = np.mean(cross_entropy_loss_overtime)
    average_accuracy = np.mean(accuracy_overtime)
    print(f"\nAvg cross-entropy loss after training on {subset_size} wire diagrams: {average_loss}")
    print(f"Avg Accuracy after training on {subset_size} wire diagrams: {average_accuracy}%")

# use the remainining 1000 wire diagrams for testing the softmax regression model
X_test, Y_test = shuffled_features[-1000:], shuffled_encoded_cut_colors[-1000:]

# test the model
test(X_test, Y_test, w, b)