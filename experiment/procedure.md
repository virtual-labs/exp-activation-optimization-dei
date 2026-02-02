The objective of this experiment is to study and compare the effect of different activation functions and optimization algorithms on the training dynamics of a neural network. A simple Multilayer Perceptron (MLP) is trained on the Fashion-MNIST dataset, and the impact of Sigmoid, Tanh, and ReLU activation functions in combination with SGD and Adam optimizers is analysed. The experiment focuses on understanding convergence speed, gradient flow behaviour, training stability, and classification performance using loss and accuracy curves.

1. **Import Libraries:**  
   o Import PyTorch for tensor computation and neural network implementation, torchvision for dataset loading and data transformations, torch.optim for optimization algorithms, NumPy for numerical operations, and Matplotlib for visualization of training dynamics.

2. **Dataset Loading and Description:**  
   o Load the Fashion-MNIST dataset using torchvision.datasets.FashionMNIST.  
   o The dataset contains 60,000 training images and 10,000 test images, each of size 28 × 28 pixels in grayscale, divided into 10 distinct classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

3. **Data Preprocessing:**  
   o Convert input images into tensors using ToTensor().  
   o Normalize pixel values using Normalize((0.5,), (0.5,)) to scale inputs to the range [-1, 1].  
   o Load the data using DataLoader with a batch size of 64, enabling shuffling for training data and disabling shuffling for test data.

4. **Model Architecture Definition:**  
   o Define a simple Multilayer Perceptron (MLP) with an input layer of 784 neurons (flattened 28 × 28 images), two hidden layers with 256 and 128 neurons respectively, and an output layer with 10 neurons representing class logits.  
   o Apply different activation functions (Sigmoid, Tanh and ReLU) in the hidden layers to study their impact on learning behaviour.

5. **Optimizer Selection:**  
   o Train the MLP using two optimization algorithms: Stochastic Gradient Descent (SGD) and Adaptive Moment Estimation (Adam).  
   o Evaluate all combinations of activation functions and optimizers under identical training conditions.

6. **Model Training:**  
   o Train the network for a fixed number of epochs (short training runs).  
   o During each epoch, perform forward propagation, compute loss using cross-entropy loss, and update model parameters using backpropagation.  
   o Record training loss and training accuracy at each epoch.

7. **Model Evaluation:**  
   o Evaluate the trained model on the test dataset.  
   o Compute overall test accuracy and class-wise accuracy for all ten classes.  
   o Generate and visualize the confusion matrix for detailed error analysis.

8. **Training Dynamics Analysis:**  
   o Plot training loss and test loss versus epochs for individual activation–optimizer combinations to analyse learning behaviour.  
   o Plot training accuracy and test accuracy curves to study convergence speed and generalization performance.  
   o Observe differences in convergence rate, stability, and oscillations across different activation functions and optimizers.  
   o Compute and visualize gradient norms for each layer to analyse gradient flow and detect vanishing gradients. Vanishing gradients occur when gradients become very small during backpropagation, hindering effective learning and slowing or preventing convergence.

9. **Comparative Study:**  
   o Compare the effect of Sigmoid, Tanh, and ReLU activation functions on convergence speed and gradient propagation.  
   o Compare SGD and Adam optimizers in terms of training stability, convergence rate, and final classification accuracy.  
   o Summarize observations related to vanishing gradients, learning efficiency, and overall model performance.