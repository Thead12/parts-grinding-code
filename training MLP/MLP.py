import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def linear(x):
    return x

def linear_derivative(x):
    return 1

class MultiLayerPerceptron():
    def __init__(self, num_layers=3, layer_sizes=[1, 10, 1],
                output_activation_function=linear, output_activation_derivative=linear_derivative,
                hidden_activation_function=relu, hidden_activation_derivative=relu_derivative):
        
        if num_layers != len(layer_sizes):
            raise Exception("Number of layers and number of layer sizes do not match")
        
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes

        self.output_activation_function = output_activation_function
        self.output_activation_derivative = output_activation_derivative
        self.hidden_activation_function = hidden_activation_function
        self.hidden_activation_derivative = hidden_activation_derivative

        self.losses = []          

        self.weights = []
        self.bias = []  
        self.momentums = []

        # initialise hidden layers
        for i in range(1, num_layers-1, 1):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i-1])) # He initialisation
            self.bias.append(np.zeros(layer_sizes[i]))
            self.momentums.append(np.zeros([layer_sizes[i-1], layer_sizes[i]]))
        
        # initialise final weights between hidden and output layers
        self.weights.append(np.random.randn(layer_sizes[-2], layer_sizes[-1]) * np.sqrt(1 / layer_sizes[-2])) # Xavier Initialization
        self.bias.append(np.zeros(layer_sizes[-1]))
        self.momentums.append(np.zeros([layer_sizes[-2], layer_sizes[-1]]))


    def loss(self, y):
        return np.mean(np.square(y - self.prediction))

    def forward(self, x):
        inputs = [x] # container to store pre activations for backprop calculation
        # input layer
        input = np.dot(x, self.weights[0]) + self.bias[0]
        inputs.append(input)    # store pre activation
        output = self.hidden_activation_function(input)
        # hidden layers
        for i in range(1, self.num_layers - 2):
            input = np.dot(output, self.weights[i]) + self.bias[i]
            inputs.append(input)    # store pre activation
            output = self.hidden_activation_function(input)
        # output layers
        output_layer_input = np.dot(output, self.weights[-1]) + self.bias[-1]
        inputs.append(output_layer_input)        
        self.prediction = self.output_activation_function(output_layer_input)

        self.pre_activations = inputs
        return self.prediction

    def backprop(self, x, y, learning_rate=0.001, momentum_rate=0.9):


        # output layer
        #delta = (dL/dy)*g'(z) = (y_hat - y)g'(z)
        delta = 2*(self.prediction - y)*self.output_activation_derivative(self.pre_activations[-1])
        #print(f"Output layer delta shape", delta.shape)

        # update weights and biases for output layer
        # W = W - learning_rate*a(h-1).T*delta
        activation = self.hidden_activation_function(self.pre_activations[-2])

        delta_w = learning_rate * np.dot(activation.T, delta)
        
        self.weights[-1] -= (delta_w + momentum_rate*self.momentums[-1])

        self.momentums[-1] = delta_w

        # b  = b-learning_rate*delta
        self.bias[-1] -= learning_rate * np.sum(delta)    

        # hidden layers
        for i in range(-2, -self.num_layers, -1):
            delta = (np.dot(delta, self.weights[i+1].T))*self.hidden_activation_derivative(self.pre_activations[i])
            
            #print(f"Layer {self.num_layers + i + 1}: weights shape", self.weights[i].shape)
            #print(f"Layer {self.num_layers + i + 1}: delta shape", delta.shape)

            #print("i:", i)
            #print("i-1:", i-1)
            #for y in range(len(self.pre_activations)):
                #print(f"Activations layer: {y}", self.pre_activations[y].shape)

            activation = self.hidden_activation_function(self.pre_activations[i-1])

            delta_w = learning_rate * np.dot(activation.T, delta)
            #print(f"Layer {self.num_layers + i + 1}: activatioins shape", activation.shape)

            self.weights[i] -= (delta_w + momentum_rate * self.momentums[i])

            self.momentums[i] = delta_w

            self.bias[i] -= learning_rate * np.sum(delta)

        # input layer
        delta = (np.dot(delta, self.weights[0].T))*self.hidden_activation_derivative(self.pre_activations[1])
        delta_w = learning_rate * np.dot(x.T, delta)
        self.weights[0] -= (delta_w + momentum_rate * self.momentums[0])
        self.momentums[0] = delta_w
        self.bias[0] -= learning_rate * np.sum(delta)

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(x)
            self.backprop(x, y, learning_rate)
            loss_value = self.loss(y)
            self.losses.append(loss_value)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.5f}")

    def predict(self, x):
        return self.forward(x)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show(block=False)

    def print_architecture(self):
        print("Number of weight layers: ", len(self.weights))
        print("Number of bias layers: ", len(self.bias))

        print("Dimensions of weights")
        for weight in self.weights:
            print(weight.shape)

        print("Dimensions of biases")
        for layer in self.bias:
            print(layer.shape)


