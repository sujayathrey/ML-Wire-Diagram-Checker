import numpy as np #import NumPy library
from wirediagrams import wire_diagram_dataset # import our wire_diagram_dataset created in wirediagrams.py

# LOGISTIC REGRESSION MODEL FOR BINARY CLASSIFICATION

def sigmoid_func(m): # function to apply the sigmoid function 1/1-e^-z to input matrix 'm'
    return 1 / (1 + np.exp(-m))

def init_params(num_features): # function to parameterize our logistic regression model with weights and bias, taking in the 'num_features' (number of pixels in "flattened" wiring diagram = 400)
    # initialize weight vector with 0's, and bias to 0
    w = np.zeros((num_features, 1))
    b = 0
    return w, b

def concatenate_nonlinear_features(M): # function that adds squared features to the model to introduce non-linearity
    M_sqrd = M*M # computes the square of each of the features in input matrix 'M'
    return np.concatenate((M, M_sqrd), axis=1) # return the NumPy array as a result of concatenating array M and M_sqrd together 

# function to compute the log loss or "cost" and the gradient of the loss during forward/backward propagation. 
# It takes in the weight, bias, input data matrix 'X', the vector containing the diagram's labels 'Y', and the lambda regularization parameter for controlling the regularization strength
def propagate(w, b, X, Y, lambda_regularization): 
    
    m = X.shape[0] # calculate the number of training examples by using X.shape which returns a tuple representing the dimensions of the input array
    A = sigmoid_func(np.dot(X, w) + b) # compute sigmoid activation value using the 'sigmoid_func' function
    log_loss = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) # compute the logistic regression loss value

    l2_regularization = (lambda_regularization / (2 * m)) * np.sum(w**2) # compute our L2 regularization term by multiplying lambda by the squared weight of every feature

    log_loss += l2_regularization # add our regularization term to our current loss to prevent overfitting

    dw = 1/m * np.dot(X.T, (A - Y)) + (lambda_regularization / m) * w # calculate the gradient loss with respect to the weight
    db = 1/m * np.sum(A - Y) # calculate the gradient loss with respect to the bias

    return dw, db, log_loss, A #returns the gradient loss with respect to the weight, bias, and logistic regression (including L2 regularization), the activation values 'A' for the input data

def calculate_prediction_accuracy(A, Y): # function to calculate the accuracy of the predictions, taking in 'A' --> the predicted scores from 0-1 representing Safe -> Dangerous and Y, which are there actual Safe/Dangerous labels
    binary_prediction_vals = (A > 0.5).astype(int) # converts the prediction proabilities to binary predictions with 0.5 as the classification threshold, meaning anything > 0.5 is = 1 and equal to 'Dangerous' and vice versa for 'Safe'

    accuracy = np.mean(binary_prediction_vals == Y) # compares the binary predictions with their true labels 'Y' and uses np.mean to calculate the average of the resulting boolean array, where True = 1/Dangerous and False = 0/Safe
    return accuracy

def train(X_train, Y_train, learning_rate, passes, display_checkpoint, lambda_regularization): # function to train our logistic regression model
    # retrieve the number of total features in training data
    num_features = X_train.shape[1]

    # call 'init_params' to initialize the weights and bias, taking in the calculated number of features
    w, b = init_params(num_features)

    # initialize empty lists to store and keep track of the log loss and accuracy throughout the training period
    log_loss_overtime = []
    accuracy_overtime = []

    # loop through the dataset for 2000 passes
    for i in range(passes):
        # call 'propagate' to retrieve the calculated log loss and the gradient of the loss
        dw, db, log_loss, A = propagate(w, b, X_train, Y_train, lambda_regularization)

        # update both weights and bias using gradient descent
        w -= learning_rate * dw
        b -= learning_rate * db

        # checks if we have passsed over 100 iterations or of the dataset
        if i % display_checkpoint == 0:
            # Calculate accuracy on the training set
            accuracy = calculate_prediction_accuracy(A, Y_train)

            # print the corresponding log loss and accuracy at that number of passes during training
            print(f"Pass #{i}: Log loss: {log_loss}, Accuracy: {accuracy}")
            
            # append these log loss and accuracy values to our list to store and keep track of the improvements or "learning process"
            log_loss_overtime.append(log_loss)
            accuracy_overtime.append(accuracy)

    # returned the weights, bias, and loss/accuracy improvements after training
    return w, b, log_loss_overtime, accuracy_overtime

def test(X_test, Y_test, w, b): # function to test our logistic regression model using the remaining split 1000 wire diagrams

    m = X_test.shape[0]  # retrieve the number of testing wire diagrams
    A = sigmoid_func(np.dot(X_test, w) + b) # calculate the predicted proabilities using the trained model

    # calc log loss and accuracy on test set
    log_loss = -1/m * np.sum(Y_test * np.log(A) + (1 - Y_test) * np.log(1 - A))
    accuracy = calculate_prediction_accuracy(A, Y_test) 

    # display testing results
    print(f"\nTesting wire diagram dataset... Log loss: {log_loss}, Accuracy: {accuracy}\n")

features = np.array([np.ravel(diagram) for diagram, _, _ in wire_diagram_dataset]) # iterates over each wire_diagram in 'wire_diagram_dataset', flattens the wire diagrams and store them in 'features' 
labels = np.array([1 if label == 'Dangerous' else 0 for _, label, _ in wire_diagram_dataset]) # creates a binary list of 0's and 1's to represent the labels if they are 'Safe' or 'Dangerous'

# call the 'concatenate_nonlinear_features' to add the squared non-linear features to the original features
nonlinear_features = concatenate_nonlinear_features(features)

# randomly shuffle the training dataset
# this will ensure that the model sees a diverse set of examples during each training iteration and helps prevent the model from learning any patterns that may arise
shuffled_index_order = np.random.permutation(len(nonlinear_features))
shuffled_features = nonlinear_features[shuffled_index_order]
shuffled_labels = labels[shuffled_index_order]

# training parameters
learning_rate = 0.01 # set the learning_rate to 0.01, which means at each iteration of training, the model parameters (weights and biases) will be updated by a fraction of 0.01 times the gradient of the loss function with respect to those parameters
passes = 2000 # total number of passes the model will take 
display_checkpoint = 100 # every 100 passes, display the log loss/accuracy
lambda_regularization = 0.1  # Adjust the regularization strength as needed

# train the model and test its performance for different dataset sizes of 500, 1000, 2500, and 5000 (all unique wire diagrams, non-repeating)
dataset_subset_sizes = [500, 1000, 2500, 5000] 
for subset_size in dataset_subset_sizes:
    # shuffle the training datasets to acquire new and unique training wire diagrams
    X_train_subset, Y_train_subset = shuffled_features[:subset_size], shuffled_labels[:subset_size]
    w, b, log_loss_overtime, accuracy_overtime = train(X_train_subset, Y_train_subset, learning_rate, passes, display_checkpoint, lambda_regularization)

    # print the average loss and average accuracy for each of the different training subsets (500, 1k, 2.5k, and 5k)
    average_loss = np.mean(log_loss_overtime)
    average_accuracy = np.mean(accuracy_overtime)
    print(f"\nAvg log loss after training on {subset_size} wire diagrams: {average_loss}")
    print(f"Avg accuracy after training on {subset_size} wire diagrams: {average_accuracy}%")

# use the remainining 1000 wire diagrams for our testing dataset
X_test, Y_test = shuffled_features[-1000:], shuffled_labels[-1000:]

# test the model on the remaining 1000 wire diagrams
test(X_test, Y_test, w, b)