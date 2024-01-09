# ML-Wire-Diagram-Checker

For the first task, I built a logistic regression model that would take in a wiring diagram as its input and give as output whether the wiring diagram is ‘Safe’ or ‘Dangerous’. This model is built in wiresafetychecker.py
Given that the wiring diagrams are 20x20 pixel images, I represented this by having each wire diagram be a 20x20 two-dimensional array where each element position would correspond to a pixel. To represent the colors Red, Blue, Yellow, and Green - I used the numbers 1, 2, 3, and 4 respectively.
The logistic regression model is best suited for this task for many reasons...
- Given that the output is either the wiring diagram being deemed ‘Safe’ or ‘Dangerous’, we have
two possible classifications, or outputs. This makes the problem a binary classification problem, where the output would be a set of real numbers between 0 and 1... Y = [0, 1] with 0.5 as the classification threshold. This means that any output value > 0.5 would be deemed as y = 1, or ‘Dangerous’, and any output value < 0.5 would be deemed as y = 0, or ‘Safe’.
- We can apply the sigmoid function in logistic regression to minimize the output range to [0, 1] which is perfect for binary classification, as mentioned before. This will additionally be beneficial since the resulting probability output corresponds to higher probabilities > 0.5 equal ‘Dangerous’ and lower probabilities < 0.5 equal ‘Safe’.
Defining the input space:
- The input space is defined by essentially “flattening” the 20x20 two-dimensional arrays of wiring
diagrams into a one-dimensional array of 400 elements (this flattening process is done with every generated wire diagram). I used NumPy’s ravel() method to transform the wire diagram array into a contiguous flattened array.
- In addition to the wire diagrams themselves, the logistic regression model also takes in the associated classification label of each wired diagram, which is calculated in wirediagrams.py. This is needed for the training of the model, since the accuracy is calculated by comparing the predicted to the true label (either ‘Safe’ or ‘Dangerous’)
Defining the output space:
- The output space, as mentioned before when dealing with binary classification, is a binary value
of 0 or 1. If the model outputs a 0, that means its calculated probability was < 0.5, and it predicts that the wired diagram is ‘Safe’. Likewise, if the model outputs a 1, then its calculated probability was > 0.5, meaning it predicts the wired diagram as ‘Dangerous’
Model space, parameters, design choices, measuring loss, preventing overfitting, training algorithm:
- The model I built for this task uses the logistic regression method for binary classification.
- How the model works:
- The weight vector is initialized as a column vector of zeros and the bias is also initialized to 0.
    
 - During the forward propagation, the sigmoid function activation value is calculated by applying the sigmoid function to the dot product of the input features (flattened array) and its weight vector plus the bias.
- The log loss is calculated and the L2 regularization is also calculated by applying the lambda regularization parameter by the squared weight of every feature. The L2 regularization value is then added to the log loss to prevent overfitting.
- The weights and bias are then updated to reflect the gradient descent, where the gradient loss (dw and db) were calculated during backpropagation). (The weights and bias are constantly being adjusted to minimize log loss)
- Lastly, the probability obtained by the sigmoid function is converted to binary predictions of 0 or 1 by the classification threshold of 0.5.
- * My model is trained on 9,000 unique wire diagrams in total, and tested on 1000 unique wire diagrams, meaning that I generated a total of 10,000 different wire diagrams in wirediagrams.py *
Model Performance:
Logistic Regression Model Performance on Different Training Sets
As can be observed, the average log loss after the logistic regression model is trained on 500 wire diagrams, to 1000 → 5000 wire diagrams reduces, and the accuracy percentage (the number of times the predicted label of ‘Safe’ or ‘Dangerous’ matches the true label) increases. This demonstrates how the model “learns” through training.
Second Task
For the second task, I built a softmax regression model that would take in a ‘Dangerous’ wiring diagram as its input and give as output which of the four colors (Red, Blue, Yellow, Green) should be cut in order to defuse the self-destruct and save the ship. This model is built in wirecutter.py.
Defining the input space:
- Similar to the first task (logistic regression model), the input space is also defined by essentially
“flattening” the 20x20 2-D arrays of wiring diagrams into a one-dimensional array of 400 elements using np.ravel(). However, there are a couple of differences...
- Firstly, prior to the flattening, the generated wire diagrams from wirediagrams.py have to be “filtered” such that only the wire diagrams with an associated label of ‘Dangerous’ can be considered. This ensures that no faulty inputs are processed by the model
- Just like how the first logistic regression model took in the wire diagram and its ‘Safe’/ ‘Dangerous’ label in order for prediction/accuracy comparison, this softmax regression
  500
1000
2500
5000
Avg log loss
0.457
0.446
0.432
0.418
Avg accuracy
80%
82%
85%
88%
  
model must take in the correct ‘cut_color’ which represents the right answer pertaining to
which color wire must be cut in the ‘Dangerous’ wire diagram. Defining the output space:
- Unlike the logistic regression model that dealt with binary classification, this model deals with multi-class classification, with there being 4 different possible output types, either Red, Blue, Yellow, or Green (each color here is a ‘class’)
Model space, parameters, design choices, measuring loss, preventing overfitting, training algorithm:
- The model I built for this task uses the softmax regression method for multi-class classification.
- How the model works:
- The weight and bias matrices are initialized with zeros with the dimensions determined by the input size of flattened ‘Dangerous’ wire diagrams as well as the number of output classes for each color
- During forward propagation, the softmax function is then applied to the linear transformation of the input features, the weight and bias matrices. The probabilities for each respective class (color) is calculated.
- To measure the dissimilarity between the predicted vs the actual probability distribution, we calculate the cross-entropy loss, which the L2 regularization is then also calculated using the same process (applying the lambda regularization parameter) and then added to the cross-entropy loss to prevent overfitting.
- It prevents overfitting by discouraging the model from assigning too much importance to any input feature, which essentially punishes large weights.
- During back propagation, the gradient loss with respect to both the weights and bias is calculated. This is done by calculating the derivative of the cross-entropy loss function with respect to the weights and bias (the parameters of the model)
- Update the weights and bias using gradient descent, which in hand adjusts the weights and bias to minimize the cross-entropy loss and improve classification accuracy.
- Lastly, the model, using the softmax function, will yield a prob. distribution across each of the 4 color classes, with the highest prob. in the distribution being selected, and whichever color is associated with that highest prob. becomes the model’s output
Model Performance:
Softmax Regression Model Performance on Different Training Sets
Similarly to the logistic regression model, the softmax regression model also demonstrates learning as the cross-entropy loss decreases a total of 0.308 as the model is trained on 500 wire diagrams, to 1000 → 5000 wire diagrams reduces. In addition, the accuracy percentage (the number of times the highest probability corresponding to the correct color to “cut” off and defuse the self-destruct) increased by 27% in total.
