**Experiment-3** 

**Activation Functions & Optimization**

**1\. Aim**

To study and compare different activation functions and optimizers and analyse their effect on training dynamics.

**3\. Theory**

Neural network learning is governed by the interaction between activation functions and optimization algorithms. Activation functions introduce non-linearity into the network, enabling it to learn complex input–output mappings, while optimization algorithms determine how model parameters are updated to minimize the loss function. The choice of activation function and optimizer significantly affects gradient propagation, convergence speed, training stability, and overall model performance. This experiment studies these effects using a Multilayer Perceptron trained on the Fashion-MNIST dataset.

**2.1   Activation Functions**

Activation functions are a fundamental component of artificial neural networks, as they introduce non-linearity into the model and enable learning of complex input–output relationships. In a Multilayer Perceptron (MLP), the activation function determines the output of a neuron based on the weighted sum of its inputs and bias. Without activation functions, even deep neural networks would behave like linear models and fail to capture complex patterns.

The choice of activation function significantly influences training dynamics, including gradient propagation, convergence speed, and stability during optimization. In this experiment, commonly used activation functions: Sigmoid, Hyperbolic Tangent (Tanh), and Rectified Linear Unit (ReLU) are employed in a Multilayer Perceptron (MLP) to analyse their effect on learning bahaviour

centred when trained on the Fashion-MNIST dataset.

Mathematical Formulation

Let the net input to a neuron be:

z=i=1nwixi+b

where wi represents weights, xi represents input features, and b is the bias term. The output of the neuron is obtained by applying an activation function f(z).

1. **Sigmoid Activation Function: \-** 

The sigmoid activation function is defined as:

σ(z)=11+e-z

The output of the sigmoid function lies in the range (0, 1). It was commonly used in early neural networks due to its smooth and differentiable nature. However, for large positive or negative input values, the function saturates, resulting in very small gradients and slow convergence during training. The Figure 1 given below shows the behaviour of the Sigmoid activation function.

![][image1]

*Figure 1: Sigmoid Activation Function*

2. **Hyperbolic Tangent (Tanh) Activation Function: \-**

 The hyperbolic tangent activation function is given by:

tanh⁡(z)=ez-e-zez+e-z

The output range of tanh is \-1,1, making it zero-centred. Compared to sigmoid, tanh provides better gradient flow near zero and generally results in faster convergence. However, it still suffers from gradient saturation for large input values. The Figure 2 given below illustrates the behaviour of the Tanh activation function.

![][image2]

*Figure 2: Tanh Activation Function*

3. **Rectified Linear Unit (ReLU): \-**

The Rectified Linear Unit activation function is defined as:

ReLU(z)=max⁡(0,z)

It can also be expressed in piecewise form as:

ReLU(z)={0, if z\<0 z, if z≥0 

ReLU outputs zero for negative input values and a linear output for positive values. This behaviour significantly reduces the vanishing gradient problem and improves training efficiency. Due to its simplicity and effectiveness, ReLU is widely used in modern deep learning models. The Figure 3 given below represents the ReLU activation function curve.

![][image3]

*Figure 3: ReLU Activation Function*

**Merits of Activation Functions**

* **Introduce non-linearity:**  
  Activation functions enable neural networks to learn complex and non-linear relationships from data.

* **Smooth and differentiable (Sigmoid, Tanh):**  
  These properties support gradient-based optimization and stable training.

* **Zero-centred output (Tanh):**  
  Helps improve gradient flow and speeds up convergence compared to sigmoid.

* **Efficient training (ReLU):**  
  Reduces vanishing gradient issues and accelerates training in deep networks.

* **Sparsity in activations (ReLU):**  
  Leads to efficient computation by activating only a subset of neurons.

**Demerits of Activation Functions**

* **Vanishing gradient problem (Sigmoid, Tanh):**  
  Gradients become very small for extreme input values, slowing down learning.

* **Non-zero-centred output (Sigmoid):**  
  Causes slower convergence due to zigzagging updates during training.

* **Dying ReLU problem:**  
  Some neurons permanently output zero, stopping learning.

* **Non-differentiable at zero (ReLU):**  
  The function is not differentiable at zero, though this is handled in practice.

* **Saturation in Sigmoid and Tanh:**  
  Output saturates for large inputs, causing slow convergence.


**2.2   Optimization Algorithms**

Optimization algorithms are used to minimize the loss function of a neural network by iteratively adjusting its parameters (weights and biases). During training, the optimizer determines the direction and magnitude of parameter updates based on the gradients of the loss function with respect to the model parameters. An efficient optimization algorithm ensures faster convergence, numerical stability, and improved generalization performance**.**

In this experiment, Stochastic Gradient Descent (SGD) and Adaptive Moment Estimation (Adam) optimizers are studied and compared while training a Multilayer Perceptron (MLP) on the Fashion-MNIST dataset.

1.  **Stochastic Gradient Descent (SGD): \-** 

Stochastic Gradient Descent is a fundamental optimization algorithm based on the theory of stochastic approximation, introduced by Herbert Robbins and Sutton Monro (1951). Unlike batch gradient descent, SGD updates model parameters using a single training example or a small mini-batch, which reduces computational cost.

The parameter update rule is:

t+1=t-L(t)

where η is the learning rate.

SGD is simple and memory-efficient, and the randomness in updates can help escape shallow local minima. However, it is sensitive to the learning rate and may converge slowly or exhibit oscillations during training.

2. **Adaptive Moment Estimation (Adam): \-**

Adaptive Moment Estimation (Adam) was proposed by Diederik P. Kingma and Jimmy Lei Ba (2015). It is an advanced optimization algorithm that combines the benefits of momentum-based methods and adaptive learning rate techniques. Adam adapts the learning rate for each parameter individually by maintaining exponentially decaying averages of past gradients and squared gradients.

The update rule for Adam is:

t+1=t-mtvt+

**Where,**

t is the model parameter at iteration t

t+1 is the updated parameter

η is the learning rate

mt is the bias-corrected first moment estimate (mean of gradients)

vt is the bias-corrected second moment estimate (mean of squared gradients)

 is a small constant added for numerical stability

Adam provides faster convergence, stable training, and performs well with noisy or sparse gradients. Due to these advantages, it is widely used in deep learning applications.

**Merits of Optimization Algorithms**

* **Efficient parameter updates:**  
  Optimization algorithms iteratively adjust model parameters to minimize loss, enabling effective training of neural networks.

* **Stochastic Gradient Descent (SGD) simplicity:**  
  SGD is straightforward, memory-efficient, and easy to implement, making it suitable for a wide range of problems.

* **Escape local minima (SGD):**  
  The stochastic nature helps avoid shallow local minima by introducing noise in updates.

* **Adaptive learning rates (Adam):**  
  Adam adjusts the learning rate for each parameter individually, leading to faster and more stable convergence.

* **Robustness to noisy and sparse gradients (Adam):**  
  Adam performs well in scenarios with noisy gradients or sparse data, common in deep learning.


**Demerits of Optimization Algorithms**

* **Sensitivity to learning rate (SGD):**  
  Choosing an inappropriate learning rate can cause slow convergence or training instability.

* **Slow convergence (SGD):**  
  SGD may require many iterations to reach the optimum, especially in complex or noisy datasets.

* **No adaptive learning rate (SGD):**  
  SGD uses a fixed learning rate which can limit performance if not well-tuned.

* **Increased computational cost (Adam):**  
  Adam requires more computation per iteration due to maintaining moment estimates.

* **Risk of overfitting (Adam):**  
  Adam’s aggressive optimization can sometimes lead to overfitting on training data.


**3\. Pre-Test (MCQs)**

1. What is the primary role of activation functions in neural networks?  
   a) To linearly transform input data  
   (Incorrect because linear transformation alone cannot model complex relationships.)  
   b) To introduce non-linearity into the model  
   (Correct because activation functions enable learning complex patterns.)  
   c) To reduce the dataset size  
   (Incorrect because activation functions do not affect dataset size.)  
   d) To initialize weights  
   (Incorrect because weight initialization is a separate step.)  
   **Answer:** b

2. Which activation function outputs values between 0 and 1?  
   a) ReLU  
   (Incorrect because ReLU outputs zero or positive values with no upper bound.)  
   b) Tanh  
   (Incorrect because Tanh outputs values between \-1 and 1.)  
   c) Sigmoid  
   (Correct because Sigmoid outputs values in the range (0, 1).)  
   d) Linear  
   (Incorrect because linear activation outputs any real number.)  
   **Answer:** c

3. Which optimizer adjusts the learning rate individually for each parameter?  
   a) SGD  
   (Incorrect because SGD uses a fixed learning rate for all parameters.)  
   b) Adam  
   (Correct because Adam adapts learning rates based on past gradients per parameter.)  
   c) Batch Gradient Descent  
   (Incorrect because it uses a fixed learning rate.)  
   d) RMSProp  
   (Incorrect because although RMSProp adapts learning rates, Adam also incorporates momentum.)  
   **Answer:** b

4. What problem is commonly associated with Sigmoid and Tanh activation functions?  
   a) Dying neurons  
   (Incorrect because dying neurons are mostly associated with ReLU.)  
   b) Vanishing gradients  
   (Correct because sigmoid and tanh saturate for large inputs causing gradients to vanish.)  
   c) Overfitting  
   (Incorrect because overfitting relates to model complexity, not activation function.)  
   d) Data normalization  
   (Incorrect because normalization is a preprocessing step.)  
   **Answer:** b

5. What does SGD stand for?  
   a) Stochastic Gradient Descent  
   (Correct because SGD updates weights using small batches or single samples, making training faster and more flexible.)  
   b) Standard Gradient Descent  
   (Incorrect; usually called Batch Gradient Descent.)  
   c) Simple Gradient Descent  
   (Incorrect; not a standard term.)  
   d) Steady Gradient Descent  
   (Incorrect; not a recognized term.)  
   **Answer:** a

6. Which activation function outputs zero for negative inputs?  
   a) Sigmoid  
   (Incorrect because Sigmoid outputs values between 0 and 1.)  
   b) Tanh  
   (Incorrect because Tanh outputs negative values for negative inputs.)  
   c) ReLU  
   (Correct because ReLU outputs zero for negative inputs and positive linear values otherwise.)  
   d) Linear  
   (Incorrect because Linear outputs the input as is.)  
   Answer: c

7. What is the learning rate in optimization algorithms?  
   a) Speed of updating parameters  
   (Correct because learning rate controls step size in parameter updates.)  
   b) Number of layers in the network  
   (Incorrect as this is architectural.)  
   c) Size of input data  
   (Incorrect as data size doesn’t control learning rate.)  
   d) Number of neurons  
   (Incorrect, network size is unrelated.)  
   **Answer:** a

8. Which activation function is zero-centred?  
   a) Sigmoid  
   (Incorrect because sigmoid outputs in (0,1).)  
   b) Tanh  
   (Correct because tanh outputs in (-1,1), centred at zero.)  
   c) ReLU  
   (Incorrect since output is always ≥ 0.)  
   d) Linear  
   (Incorrect as linear just outputs the input.)  
   **Answer:** b

9.  Which optimizer uses momentum and adaptive learning rates?  
   a) SGD  
   (Incorrect because SGD does not use adaptive learning rates or momentum by default.)  
   b) Adam  
   (Correct because Adam combines momentum and adaptive learning rates.)  
   c) Batch Gradient Descent  
   (Incorrect, similar to SGD.)  
   d) RMSProp  
   (Incorrect because although RMSProp uses adaptive learning rates, Adam also uses momentum.)  
   **Answer:** b

10. Which activation function helps reduce the vanishing gradient problem?  
    a) Sigmoid  
    (Incorrect because sigmoid saturates causing vanishing gradients.)  
    b) Tanh  
    (Incorrect for similar reasons.)  
    c) ReLU  
    (Correct because ReLU outputs zero or positive linear values, mitigating vanishing gradients.)  
    d) Linear  
    (Incorrect because linear activation does not add non-linearity.)  
    **Answer:** c

    

**4\. Procedure**

The objective of this experiment is to study and compare the effect of different activation functions and optimization algorithms on the training dynamics of a neural network. A simple Multilayer Perceptron (MLP) is trained on the Fashion-MNIST dataset, and the impact of Sigmoid, Tanh, and ReLU activation functions in combination with SGD and Adam optimizers is analysed. The experiment focuses on understanding convergence speed, gradient flow behaviour, training stability, and classification performance using loss and accuracy curves.

1. **Import Libraries:**  
   o Import PyTorch for tensor computation and neural network implementation, torchvision for dataset loading and data transformations, torch.optim for optimization algorithms, NumPy for numerical operations, and Matplotlib for visualization of training dynamics.

2. **Dataset Loading and Description:**  
   o Load the Fashion-MNIST dataset using torchvision.datasets.FashionMNIST.  
   o The dataset contains 60,000 training images and 10,000 test images, each of size 28 × 28 pixels in grayscale, divided into 10 distinct classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

3. **Data Preprocessing:**  
   o Convert input images into tensors using ToTensor().  
   o Normalize pixel values using Normalize((0.5,), (0.5,)) to scale inputs to the range \[-1, 1\].  
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

   

**5\. Post-Test (MCQs)**

1. Which activation function is zero-centred with output range (-1, 1)?  
   a) Sigmoid  
   (Incorrect because Sigmoid outputs (0,1).)  
   b) Tanh  
   (Correct because Tanh outputs values from \-1 to 1 and is zero-centred.)  
   c) ReLU  
   (Incorrect because ReLU outputs zero or positive.)  
   d) Linear  
   (Incorrect because linear has no bounds.)  
   Answer: b

2. What is a potential drawback of using ReLU activation?  
   a) It saturates for large positive inputs  
   (Incorrect because ReLU is linear for positive inputs.)  
   b) It causes vanishing gradients for all inputs  
   (Incorrect because vanishing gradients mostly occur in Sigmoid/Tanh.)  
   c) Dying ReLU problem where neurons stop learning  
   (Correct because neurons output zero for all inputs and stop updating.)  
   d) It outputs values between 0 and 1 only  
   (Incorrect because ReLU outputs zero or any positive value.)  
   **Answer:** c

3. What is the update rule of parameters in Stochastic Gradient Descent?  
   a) θ\_(t+1) \= θ\_t \+ η ∇θ L(θ\_t)  
   (Incorrect because update subtracts gradient to minimize loss.)  
   b) θ(t+1) \= θ\_t \- η ∇θ L(θ\_t)  
   (Correct because parameters are updated opposite to the gradient, which points toward the steepest increase in loss. Subtracting it reduces the loss. The learning rate  controls the update size.)  
   c) θ(t+1) \= η ∇θ L(θ\_t)  
   (Incorrect because parameters are not replaced by gradient.)  
   d) θ(t+1) \= θ\_t / η  
   (Incorrect because parameters are not divided by learning rate.)  
   **Answer:** b

4. What does the term "moment" refer to in Adam optimizer?  
   a) The number of epochs  
   (Incorrect because moments are related to gradients, not epochs.)  
   b) Running averages of past gradients and squared gradients  
   (Correct because Adam uses the first moment, which is the average of past gradients to capture update direction, and the second moment, which is the average of squared gradients to adjust step size based on gradient variability, enabling adaptive and efficient optimization.)  
   c) The time taken per update  
   (Incorrect as moment is not related to time.)  
   d) The size of the training batch  
   (Incorrect as batch size is independent.)  
   **Answer:** b

5. What is a drawback of the "dying ReLU" problem?  
   a) Neurons output zero and stop learning  
   (Correct because neurons that output zero for all inputs receive no gradient updates, causing them to stop learning and reducing the model’s capacity.)  
   b) Neurons output large positive values  
   (Incorrect because dying ReLU neurons output zero, not positive values.)  
   c) Training is faster  
   (Incorrect because dying ReLU slows or halts learning, negatively affecting training.)  
   d) Model becomes more complex  
   (Incorrect because dying ReLU reduces effective model capacity by disabling neurons.)  
   **Answer:** a

6. Which optimizer combines momentum and adaptive learning rates?  
   a) SGD  
   (Incorrect because SGD does not adapt learning rates.)  
   b) Adam  
   (Correct because Adam uses both momentum and adaptive rates.)  
   c) RMSProp  
   (Incorrect because it does not use momentum in the same way.)  
   d) AdaGrad  
   (Incorrect as it only adapts learning rate without momentum.)  
   **Answer:** b

7. Which activation function is non-differentiable at zero?  
   a) Sigmoid  
   (Incorrect because sigmoid is differentiable everywhere.)  
   b) ReLU  
   (Correct because ReLU has a sharp corner at zero.)  
   c) Tanh  
   (Incorrect as tanh is differentiable everywhere.)  
   d) Softmax  
   (Incorrect as softmax is differentiable.)  
   **Answer:** b

8. Why can Sigmoid activation slow down training?  
   a) Because it causes neurons to die  
   (Incorrect because dying neurons are mostly related to ReLU.)  
   b) Because it is not differentiable  
   (Incorrect because Sigmoid is differentiable.)  
   c) Because it saturates and causes vanishing gradients  
   (Correct because large inputs push gradients close to zero.)  
   d) Because it outputs values only in negative range  
   (Incorrect because Sigmoid outputs between 0 and 1.)  
   **Answer:** c

9. How does the Adam optimizer differ from standard SGD?  
   a) Adam uses fixed learning rate for all parameters  
   (Incorrect because Adam adapts learning rates individually.)  
   b) Adam maintains moving averages of past gradients and squared gradients  
   (Correct because this enables adaptive moment estimation.)  
   c) Adam ignores gradient information during updates  
   (Incorrect because Adam heavily relies on gradients.)  
   d) Adam always requires manual tuning of momentum separately  
   (Incorrect because momentum is part of Adam’s built-in mechanism.)  
   **Answer:** b

10. What is the main advantage of using adaptive optimizers like Adam?  
    a) They adjust learning rates individually for each parameter  
    (Correct because this allows faster and more stable convergence.)  
    b) They require no hyperparameter tuning at all  
    (Incorrect because learning rate and other params still need tuning.)  
    c) They eliminate the need for backpropagation  
    (Incorrect because backpropagation is essential for gradient calculation.)  
    d) They always prevent overfitting  
    (Incorrect because overfitting depends on model complexity and data.)

    **Answer:** a

    

**6\. References**

1. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA, USA: MIT Press, 2016\.

2. M. A. Nielsen, Neural Networks and Deep Learning. Determination Press, 2015\.

3. D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” *International Conference on Learning Representations (ICLR)*, 2015\.

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAVYAAAD+CAYAAABhly20AAA24ElEQVR4Xu2dCXwURf72/6uu53rger3qerArHhweiLqiwHoroCKigooHCqjgiQp44IGCgogsogIKgiSEm3DJGQkQblguBWISAiSc4b5CSH7vPBV66Kk5MpXM9KQzz/fz+cGkqmd6urr729U1XVX/J4QQQiLK/+kJhBBCygbFSgghEYZiJYSQCEOxEkJIhKFYCSEkwlCsxEtRUZFfhMvBgwdl//79enLEWLt2raxevVpPNmb58uWyfft2PdmY0pSRCQUFBbJ48WI9mbgEipUomjZtKrfffrtP1KlTR18sKOvWrZP09HQ9OSAjR46U7OxsPdkHiHrUqFHev998801p27atbYnwSEpKkr1793r/fvTRR+XXX3+1LWHO//73P28Z3XnnnTJnzhx9kVJhF+nu3bvlrrvusuUSN0GxEi+ofd1zzz3emuFvv/2mxNG+fXt1olvLPPPMM9KsWTN55JFHvDKAWCdMmKDSIa8//vjD+7l21qxZI7Vq1ZLGjRv71PZ27Nghd999t7Ro0UKysrKkV69ectpppyl5obb63//+V77++mvZs2ePNGnSxPu+6dOny1dffSVHjhxRFwd8/65du8q+fftUDfrkk0+Wm266SX2nzZs3y/vvvy/z58+X/Px86dKli9x7773SsmVL9f3BAw88IKNHj1afhe8YiJkzZ/rVWO3S/+ijj2Tq1KnqOyUmJsp7772ntg2vreUTEhKkfv368vDDD8vPP/8sc+fOlUsuuURtb//+/dX3f+2119SyKI/nnntOfcYPP/ygPgPbgPX89NNPcv/998sbb7zhXT+JPRQr8QFigvwATlxIBiKDfABO6ksvvVQGDx6sTmaIALKDmCClX375Rdq1aye1a9dWYtGxRFOjRg35888/VRqkis8ZMWKE9OzZU5YuXSoLFy6Uyy+/XAkKn48aa5s2bdT6GzZsqN6H10888YSMGTNGDhw4oAQ1dOhQueOOO5SYcTt9/vnny5AhQ2T27NmqFmzVWCHCm2++WeXhc+vWras+76STTlKvkY7t27lzp/3rKyBWfFfUvMePH68+t169et58SBnixPZfe+21SvQDBgyQKlWqKLmDG264QUnRWs+2bdvk6aefVtubkZGhLmS4qKFczznnHLX9qH3feOON0r17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>

[image3]: <data:image/png;base64,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>