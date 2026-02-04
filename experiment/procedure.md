### Procedure

The objective of this experiment is to study and compare the effect of different activation functions and optimization algorithms on the training dynamics of a neural network. A simple Multilayer Perceptron (MLP) is trained on the Fashion-MNIST dataset, and the impact of Sigmoid, Tanh, and ReLU activation functions in combination with SGD and Adam optimizers is analysed. The experiment focuses on understanding convergence speed, gradient flow behaviour, training stability, and classification performance using loss and accuracy curves.

1.  **Import Libraries**
    *   Import PyTorch for tensor computation and neural network implementation, torchvision for dataset loading and data transformations, `torch.optim` for optimization algorithms, NumPy for numerical operations, and Matplotlib with Seaborn for visualization of training dynamics.

2.  **Dataset Loading and Description**
    *   Load the Fashion-MNIST dataset using `torchvision.datasets.FashionMNIST`.
    *   The dataset consists of 60,000 training images and 10,000 test images, each of size 28 × 28 pixels in grayscale, belonging to 10 classes.
    *   Split the training data into 90% training and 10% validation subsets for performance monitoring.

3.  **Data Preprocessing**
    *   Convert input images into tensors using `ToTensor()`.
    *   Normalize pixel values using `Normalize((0.5,), (0.5,))` to scale inputs to the range $[-1, 1]$.
    *   Load the data using `DataLoader` with a batch size of 128, enabling shuffling for training data and disabling shuffling for validation and test data.

4.  **Model Architecture Definition**
    *   Define a Multilayer Perceptron (MLP) with an input layer of 784 neurons, two hidden layers of 256 and 128 neurons, and an output layer of 10 neurons.
    *   Apply different activation functions (Sigmoid, Tanh and ReLU) in the hidden layers to study their impact on learning behaviour and gradient flow.

5.  **Optimizer Selection**
    *   Train the MLP using two optimization algorithms: Stochastic Gradient Descent (SGD) and Adaptive Moment Estimation (Adam).
    *   Evaluate all combinations of activation functions and optimizers under identical training conditions.

6.  **Model Training**
    *   Train the network for a fixed number of epochs (short training runs).
    *   During each epoch, perform forward propagation, compute loss using cross-entropy loss, and update model parameters using backpropagation.
    *   Record training loss, validation accuracy, learning rate, and epoch training time at each epoch.

7.  **Model Evaluation**
    *   Evaluate the model on the validation dataset after each epoch.
    *   Track the best validation accuracy and the corresponding epoch for each experimental configuration.

8.  **Training Dynamics Analysis**
    *   Plot training loss versus epochs for individual activation–optimizer combinations to analyse learning behaviour.
    *   Plot validation accuracy versus epochs to study convergence speed and generalization performance.
    *   Analyse gradient flow by computing per-layer gradient statistics and visualizing gradient magnitudes over epochs.
    *   Identify vanishing gradients and dead neurons. Vanishing gradients occur when gradients become very small during backpropagation, hindering effective learning and slowing or preventing convergence.

9.  **Comparative Study**
    *   Compare the effect of Sigmoid, Tanh, and ReLU activation functions on convergence speed and gradient propagation.
    *   Compare SGD and Adam optimizers across different learning rates in terms of training stability and final validation accuracy.
    *   Summarize observations related to vanishing gradients, learning efficiency, and overall model performance.
