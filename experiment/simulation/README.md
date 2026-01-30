# Activation Functions & Optimization Simulation

## Overview
This simulation is based on **Experiment 3 – Activation Functions & Optimization** from the theory folder. It provides an interactive visualization of how different activation functions (ReLU, Sigmoid, Tanh) and optimizers (SGD, Adam) perform on the Fashion-MNIST dataset.

## Structure

### Files Created:
1. **index.html** - Main HTML structure with 6 interactive steps
2. **main.css** - Styling copied from reference (Virtual Labs theme)
3. **script.js** - Interactive JavaScript with neural network visualization

### Experiment Steps:

#### Step 1: Import Libraries
- Shows the necessary PyTorch and data processing imports
- Demonstrates setting up the device (CPU/GPU)

#### Step 2: Load Dataset
- Loads Fashion-MNIST dataset (60,000 training + 10,000 test samples)
- Shows data transformation and normalization
- Displays the 10 clothing categories

#### Step 3: Model Definition
- Defines a Simple MLP architecture: 784 → 256 → 128 → 10
- **Interactive Element 1**: Dropdown to select activation function (ReLU/Sigmoid/Tanh)
- Architecture visualization updates based on selection

#### Step 4: Training Loop
- Shows the complete training function code
- Demonstrates how to set up optimizer and loss function

#### Step 5: Run Experiments
- **Interactive Element 2**: Multiple parameter controls:
  - Activation Function selector
  - Optimizer selector (SGD/Adam)
  - Epochs (5 or 10)
  - Learning Rate (0.001 or 0.01)
- Real-time neural network visualization with:
  - Animated data flow particles
  - Node activation visualization
  - Weight connection display
- Live training metrics (Epoch, Loss, Accuracy)
- Uses actual training results from the theory notebook

#### Step 6: Visualization
- Displays training loss and accuracy charts
- Interactive line graphs showing performance over epochs

## Key Features

### Interactive Elements:
1. **Activation Function Selector** - Changes the activation function used in the model
2. **Training Parameter Controls** - Allows experimentation with different hyperparameters

### Training Results:
The simulation uses real training results from the theory notebook for different configurations:
- ReLU + SGD: 73.61% accuracy (5 epochs)
- ReLU + Adam: 89.49% accuracy (5 epochs)
- Sigmoid + SGD: 39.33% accuracy (5 epochs)
- Sigmoid + Adam: 89.02% accuracy (5 epochs)
- Tanh + SGD: 74.52% accuracy (5 epochs)
- Tanh + Adam: 88.74% accuracy (5 epochs)

### Visual Features:
- Animated neural network with pulsing nodes
- Data flow particles showing information propagation
- Color-coded activation states
- Smooth transitions and animations
- Responsive design matching Virtual Labs theme

## How to Use

1. Open `index.html` in a web browser
2. Click "Run" on each step to progress through the experiment
3. In Step 3, select different activation functions to see architecture changes
4. In Step 5, configure training parameters and click "Run Training"
5. Watch the neural network animate during training
6. View final results and charts in Step 6

## Technical Details

- **Architecture**: 3-layer MLP (784 input → 256 → 128 → 10 output)
- **Dataset**: Fashion-MNIST (28×28 grayscale images)
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Optimizers**: SGD, Adam
- **Loss Function**: Cross-Entropy Loss
- **Visualization**: HTML5 Canvas with real-time animation

## Learning Objectives

1. Understand different activation functions and their impact on training
2. Compare SGD vs Adam optimizer performance
3. Visualize how neural networks process information
4. Observe the effect of hyperparameters on model performance
5. Interpret training metrics (loss and accuracy curves)

## Notes

- All training results are pre-computed from the actual experiment
- The simulation provides instant feedback without requiring actual model training
- The neural network visualization is simplified for educational purposes
- The UI follows the Virtual Labs design system for consistency
