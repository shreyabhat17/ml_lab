import numpy as np

# Input data
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)

# Corresponding output data
y = np.array(([92], [86], [89]), dtype=float)

# Normalize input data
x = x / np.amax(x, axis=0)
y = y / 100

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def derivatives_sigmoid(x):
    return x * (1 - x)

# Hyperparameters
epoch = 5000
lr = 0.1

# Neural network architecture
inputlayer_neuron = 2
hiddenlayer_neuron = 3
output_neurons = 1

# Weight initialization
wh = np.random.uniform(size=(inputlayer_neuron, hiddenlayer_neuron))
bh = np.random.uniform(size=(1, hiddenlayer_neuron))
wout = np.random.uniform(size=(hiddenlayer_neuron, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training loop
for i in range(epoch):
    # Forward pass
    hinp1 = np.dot(x, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    # Calculate error
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad

    # Backpropagation
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad

    # Update weights
    wout += hlayer_act.T.dot(d_output) * lr
    wh += x.T.dot(d_hiddenlayer) * lr

# Print results
print("Input:\n" + str(x))
print("Actual Output:\n" + str(y))
print("Predicted Output:\n", output)
