import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        # else
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        while True:
            num_misclassifications = 0
            for x, y in dataset.iterate_once(batch_size):
                y_pred = self.get_prediction(x)
                if y_pred != nn.as_scalar(y):
                    num_misclassifications += 1
                    self.w.update(x, nn.as_scalar(y))

            if num_misclassifications == 0:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.input_features = 1
        self.output_features = 1
        self.num_hidden_layers = 2
        self.hidden_layer_sizes = [256, 128]
        self.batch_size = 10
        self.learning_rate = 0.01
        self.max_loss = 0.005
        self.weights, self.bias = self.initialize_nn()

    def initialize_nn(self):
        weights = []
        bias = []
        weight_dims = [self.input_features] + self.hidden_layer_sizes + [self.output_features]

        for hidden_layer in range(self.num_hidden_layers + 1):
            w = nn.Parameter(weight_dims[hidden_layer], weight_dims[hidden_layer + 1])
            b = nn.Parameter(1, weight_dims[hidden_layer + 1])
            weights.append(w)
            bias.append(b)

        return weights, bias

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        z = x
        for hidden_layer in range(self.num_hidden_layers):
            xw = nn.Linear(z, self.weights[hidden_layer])
            xw = nn.AddBias(xw, self.bias[hidden_layer])
            z = nn.ReLU(xw)

        # Calculate the output without ReLU
        xw = nn.Linear(z, self.weights[-1])
        z = nn.AddBias(xw, self.bias[-1])
        return z

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        while loss >= self.max_loss:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # Calculate the gradients
                gradients = nn.gradients(loss, self.weights + self.bias)

                for i in range(self.num_hidden_layers + 1):
                    self.weights[i].update(gradients[i], -self.learning_rate)
                    self.bias[i].update(gradients[i + self.num_hidden_layers + 1], -self.learning_rate)

            loss = nn.as_scalar(loss)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.input_features = 784
        self.output_features = 10
        self.num_hidden_layers = 4
        self.hidden_layer_sizes = [256, 128, 64, 32]
        self.batch_size = 50
        self.learning_rate = 0.1
        self.min_validation_accuracy = 0.98
        self.weights, self.bias = self.initialize_nn()

    def initialize_nn(self):
        weights = []
        bias = []
        weight_dims = [self.input_features] + self.hidden_layer_sizes + [self.output_features]

        for hidden_layer in range(self.num_hidden_layers + 1):
            w = nn.Parameter(weight_dims[hidden_layer], weight_dims[hidden_layer + 1])
            b = nn.Parameter(1, weight_dims[hidden_layer + 1])
            weights.append(w)
            bias.append(b)

        return weights, bias

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z = x
        for hidden_layer in range(self.num_hidden_layers):
            xw = nn.Linear(z, self.weights[hidden_layer])
            xw = nn.AddBias(xw, self.bias[hidden_layer])
            z = nn.ReLU(xw)

        # Calculate the output without ReLU
        xw = nn.Linear(z, self.weights[-1])
        z = nn.AddBias(xw, self.bias[-1])
        return z

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        validation_accuracy = 0
        while validation_accuracy <= self.min_validation_accuracy:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # Calculate the gradients
                gradients = nn.gradients(loss, self.weights + self.bias)

                for i in range(self.num_hidden_layers + 1):
                    self.weights[i].update(gradients[i], -self.learning_rate)
                    self.bias[i].update(gradients[i + self.num_hidden_layers + 1], -self.learning_rate)

            validation_accuracy = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 100
        self.w_input = nn.Parameter(self.num_chars, self.hidden_size)
        self.w_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.w_output = nn.Parameter(self.hidden_size, 5)

        self.learning_rate = 0.1
        self.min_validation_accuracy = 0.89
        self.batch_size = 50

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.ReLU(nn.Linear(xs[0], self.w_input))
        for i in range(1, len(xs)):
            h = nn.ReLU(nn.Add(nn.Linear(xs[i], self.w_input), nn.Linear(h, self.w_hidden)))

        h = nn.Linear(h, self.w_output)

        return h

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(xs)
        return nn.SoftmaxLoss(predicted_y, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        validation_accuracy = 0
        while validation_accuracy <= self.min_validation_accuracy:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # Calculate the gradients
                gradients = nn.gradients(loss, [self.w_input, self.w_hidden, self.w_output])

                self.w_input.update(gradients[0], -self.learning_rate)
                self.w_hidden.update(gradients[1], -self.learning_rate)
                self.w_output.update(gradients[2], -self.learning_rate)

            validation_accuracy = dataset.get_validation_accuracy()
