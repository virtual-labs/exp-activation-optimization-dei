// ===================================
// GLOBAL STATE MANAGEMENT
// ===================================
const appState = {
    completedSteps: [],
    currentStep: 1,
    maxUnlockedStep: 1
};

// ===================================
// TRAINING RESULTS FROM THEORY
// ===================================
const trainingResults = {
    // ReLU + SGD
    'relu-sgd-5-0.001': {
        loss: [2.1610, 1.6609, 1.1640, 0.9216, 0.8014],
        accuracy: [38.62, 61.47, 68.22, 72.11, 73.61]
    },
    'relu-sgd-10-0.001': {
        loss: [2.1610, 1.6609, 1.1640, 0.9216, 0.8014, 0.7245, 0.6723, 0.6312, 0.5987, 0.5721],
        accuracy: [38.62, 61.47, 68.22, 72.11, 73.61, 75.23, 76.45, 77.32, 78.01, 78.56]
    },
    // ReLU + Adam
    'relu-adam-5-0.001': {
        loss: [0.4945, 0.3669, 0.3271, 0.3041, 0.2838],
        accuracy: [82.00, 86.47, 87.94, 88.60, 89.49]
    },
    'relu-adam-10-0.001': {
        loss: [0.4945, 0.3669, 0.3271, 0.3041, 0.2838, 0.2678, 0.2541, 0.2423, 0.2319, 0.2227],
        accuracy: [82.00, 86.47, 87.94, 88.60, 89.49, 90.12, 90.67, 91.15, 91.58, 91.95]
    },
    // Sigmoid + SGD
    'sigmoid-sgd-5-0.001': {
        loss: [2.3063, 2.2939, 2.2875, 2.2806, 2.2728],
        accuracy: [11.94, 21.95, 31.33, 36.35, 39.33]
    },
    'sigmoid-sgd-10-0.001': {
        loss: [2.3063, 2.2939, 2.2875, 2.2806, 2.2728, 2.2645, 2.2558, 2.2467, 2.2372, 2.2273],
        accuracy: [11.94, 21.95, 31.33, 36.35, 39.33, 41.78, 43.89, 45.73, 47.34, 48.75]
    },
    // Sigmoid + Adam
    'sigmoid-adam-5-0.001': {
        loss: [0.6339, 0.3940, 0.3497, 0.3201, 0.2993],
        accuracy: [78.55, 85.81, 87.24, 88.41, 89.02]
    },
    'sigmoid-adam-10-0.001': {
        loss: [0.6339, 0.3940, 0.3497, 0.3201, 0.2993, 0.2832, 0.2697, 0.2582, 0.2482, 0.2395],
        accuracy: [78.55, 85.81, 87.24, 88.41, 89.02, 89.56, 90.03, 90.45, 90.82, 91.15]
    },
    // Tanh + SGD
    'tanh-sgd-5-0.001': {
        loss: [1.8644, 1.3072, 1.0410, 0.8947, 0.8047],
        accuracy: [50.02, 67.32, 71.50, 73.33, 74.52]
    },
    'tanh-sgd-10-0.001': {
        loss: [1.8644, 1.3072, 1.0410, 0.8947, 0.8047, 0.7389, 0.6876, 0.6458, 0.6108, 0.5809],
        accuracy: [50.02, 67.32, 71.50, 73.33, 74.52, 75.56, 76.48, 77.29, 78.01, 78.66]
    },
    // Tanh + Adam
    'tanh-adam-5-0.001': {
        loss: [0.4941, 0.3737, 0.3429, 0.3206, 0.3052],
        accuracy: [82.12, 86.28, 87.39, 88.21, 88.74]
    },
    'tanh-adam-10-0.001': {
        loss: [0.4941, 0.3737, 0.3429, 0.3206, 0.3052, 0.2926, 0.2820, 0.2729, 0.2650, 0.2580],
        accuracy: [82.12, 86.28, 87.39, 88.21, 88.74, 89.21, 89.62, 89.99, 90.32, 90.62]
    },
    // Additional LR variations
    'relu-sgd-5-0.01': {
        loss: [1.8234, 1.2456, 0.9823, 0.8234, 0.7123],
        accuracy: [45.23, 68.34, 73.45, 76.23, 78.12]
    },
    'relu-adam-5-0.01': {
        loss: [0.3234, 0.2156, 0.1823, 0.1634, 0.1489],
        accuracy: [88.23, 91.34, 92.45, 93.12, 93.67]
    }
};

// ===================================
// NEURAL NETWORK VISUALIZATION CLASS
// ===================================
class FashionMNISTNN {
    constructor(activation = 'relu') {
        this.layers = [784, 256, 128, 10];
        this.activation = activation;
        this.weights = [];
        this.biases = [];
        this.activations = [];
        this.initializeWeights();
    }

    initializeWeights() {
        for (let i = 0; i < this.layers.length - 1; i++) {
            const inputSize = this.layers[i];
            const outputSize = this.layers[i + 1];
            const scale = Math.sqrt(2.0 / inputSize);

            const w = [];
            for (let j = 0; j < outputSize; j++) {
                const row = [];
                for (let k = 0; k < inputSize; k++) {
                    row.push((Math.random() - 0.5) * 2 * scale);
                }
                w.push(row);
            }
            this.weights.push(w);

            const b = new Array(outputSize).fill(0);
            this.biases.push(b);
        }
    }

    activationFunction(x) {
        switch (this.activation) {
            case 'relu':
                return Math.max(0, x);
            case 'sigmoid':
                return 1 / (1 + Math.exp(-x));
            case 'tanh':
                return Math.tanh(x);
            default:
                return Math.max(0, x);
        }
    }

    forward(input) {
        this.activations = [input];

        for (let i = 0; i < this.weights.length; i++) {
            const z = [];
            for (let j = 0; j < this.weights[i].length; j++) {
                let sum = this.biases[i][j];
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    sum += this.weights[i][j][k] * this.activations[i][k];
                }
                z.push(sum);
            }

            let a = (i < this.weights.length - 1)
                ? z.map(val => this.activationFunction(val))
                : this.softmax(z);
            this.activations.push(a);
        }

        return this.activations[this.activations.length - 1];
    }

    softmax(arr) {
        const maxVal = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - maxVal));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sumExps);
    }
}

// ===================================
// GLOBAL VARIABLES
// ===================================
let network = new FashionMNISTNN('relu');
let currentEpoch = 0;
let isTraining = false;
let trainingInterval = null;
let flowParticles = [];

// ===================================
// STEP MANAGEMENT FUNCTIONS
// ===================================
function unlockNextStep() {
    appState.maxUnlockedStep = Math.min(appState.maxUnlockedStep + 1, 6);
    updateStepButtons();
}

function setStepRunning(stepNum) {
    if (appState.completedSteps.includes(stepNum)) {
        return;
    }

    const stepBtn = document.querySelector(`.step-item[data-step="${stepNum}"]`);
    if (stepBtn) {
        stepBtn.classList.remove('idle', 'completed');
        stepBtn.classList.add('running');
    }
}

function completeStep(stepNum) {
    const stepBtn = document.querySelector(`.step-item[data-step="${stepNum}"]`);
    if (stepBtn) {
        stepBtn.classList.remove('idle', 'running');
        stepBtn.classList.add('completed');
    }

    const stepContent = document.getElementById(`step-${stepNum}`);
    if (stepContent) {
        stepContent.classList.add('completed');
    }

    if (!appState.completedSteps.includes(stepNum)) {
        appState.completedSteps.push(stepNum);
        unlockNextStep();
    }
}

function updateStepButtons() {
    const stepButtons = document.querySelectorAll('.step-item');
    stepButtons.forEach(btn => {
        const stepNum = parseInt(btn.dataset.step);

        if (appState.completedSteps.includes(stepNum)) {
            btn.style.backgroundColor = '#d1e7dd';
            btn.style.borderColor = '#badbcc';
            btn.style.color = '#0f5132';

            if (!btn.classList.contains('completed')) {
                btn.classList.add('completed', 'unlocked');
            }
            return;
        }

        btn.style.backgroundColor = '';
        btn.style.borderColor = '';
        btn.style.color = '';

        if (!btn.classList.contains('running') && !btn.classList.contains('completed')) {
            btn.classList.remove('active', 'unlocked', 'idle');

            if (stepNum === appState.currentStep) {
                btn.classList.add('active', 'unlocked', 'idle');
            } else if (stepNum <= appState.maxUnlockedStep) {
                btn.classList.add('unlocked', 'idle');
            } else {
                btn.classList.add('idle');
            }
        }
    });
}

// ===================================
// STEP NAVIGATION
// ===================================
const stepButtons = document.querySelectorAll('.step-item');
const stepContents = document.querySelectorAll('.step-content');

stepButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const stepNum = parseInt(btn.dataset.step);

        if (!btn.classList.contains('unlocked')) {
            alert('Please complete the current step first!');
            return;
        }

        const isCompleted = appState.completedSteps.includes(stepNum);

        appState.currentStep = stepNum;

        stepContents.forEach(c => c.classList.remove('active'));
        document.getElementById(`step-${stepNum}`).classList.add('active');

        if (!isCompleted) {
            updateStepButtons();
        }

        if (stepNum === 3) {
            updateArchitecture();
        } else if (stepNum === 5) {
            initializeCanvas();
        } else if (stepNum === 6) {
            displayResults();
        }
    });
});

// ===================================
// UPDATE CODE DISPLAY
// ===================================
function updateCodeDisplay() {
    const activation = document.getElementById('activation').value;
    const optimizer = document.getElementById('optimizer').value;
    const epochs = document.getElementById('epochs').value;
    const learningRate = document.getElementById('learningRate').value;

    document.getElementById('config-activation').textContent = activation;
    document.getElementById('config-optimizer').textContent = optimizer;
    document.getElementById('config-epochs').textContent = epochs;
    document.getElementById('config-lr').textContent = learningRate;
}

// ===================================
// UPDATE ARCHITECTURE DISPLAY
// ===================================
function updateArchitecture() {
    const activation = document.getElementById('activation')?.value || 'relu';
    const activationDisplay = activation.charAt(0).toUpperCase() + activation.slice(1);

    const archElements = document.querySelectorAll('#arch-activation, #arch-activation2');
    archElements.forEach(el => {
        if (el) el.textContent = activationDisplay;
    });
}

// ===================================
// PARAMETER SELECTS UPDATES
// ===================================
const paramSelects = document.querySelectorAll('.param-select');
paramSelects.forEach(select => {
    select.addEventListener('change', () => {
        updateCodeDisplay();
        if (select.id === 'activation') {
            updateArchitecture();
            network = new FashionMNISTNN(select.value);
        }
    });
});

// ===================================
// CANVAS DRAWING
// ===================================
const canvas = document.getElementById('networkCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;

function initializeCanvas() {
    if (canvas && ctx) {
        resizeCanvas();
    }
}

function resizeCanvas() {
    if (!canvas) return;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    drawNetwork();
}

if (canvas) {
    window.addEventListener('resize', resizeCanvas);
}

function drawNetwork() {
    if (!ctx || !canvas) return;

    if (canvas.offsetWidth === 0 || canvas.offsetHeight === 0) {
        return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layerSpacing = canvas.width / (network.layers.length + 1);
    const nodePositions = [];

    for (let l = 0; l < network.layers.length; l++) {
        const x = layerSpacing * (l + 1);
        const maxNodes = Math.min(network.layers[l], 8);
        const spacing = canvas.height / (maxNodes + 1);
        const positions = [];

        for (let n = 0; n < maxNodes; n++) {
            positions.push({ x, y: spacing * (n + 1) });
        }
        nodePositions.push(positions);
    }

    // Draw connections
    for (let l = 0; l < nodePositions.length - 1; l++) {
        for (let j = 0; j < nodePositions[l + 1].length; j++) {
            for (let k = 0; k < nodePositions[l].length; k++) {
                const from = nodePositions[l][k];
                const to = nodePositions[l + 1][j];

                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = isTraining ? '#f39c12' : '#27ae60';
                ctx.lineWidth = 1;
                ctx.globalAlpha = 0.3;
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
        }
    }

    // Draw nodes
    for (let l = 0; l < nodePositions.length; l++) {
        for (let n = 0; n < nodePositions[l].length; n++) {
            const pos = nodePositions[l][n];
            const activation = network.activations[l]?.[n] || 0;
            const intensity = Math.min(Math.abs(activation), 1);

            let radius = 15;
            if (isTraining && activation > 0.1) {
                const pulse = Math.sin(Date.now() / 150 + n * 0.5) * 2 + 15;
                radius = pulse;
            }

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = activation > 0
                ? `rgba(46, 204, 113, ${0.3 + intensity * 0.5})`
                : `rgba(231, 76, 60, ${0.3 + intensity * 0.5})`;
            ctx.fill();
            ctx.strokeStyle = '#2c3e50';
            ctx.lineWidth = 2;
            ctx.stroke();

            if (n === 0) {
                ctx.fillStyle = '#555';
                ctx.font = 'bold 11px Arial';
                const labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
                ctx.fillText(labels[l], pos.x, 20);
            }
        }
    }

    if (isTraining || flowParticles.length > 0) {
        drawDataFlow(nodePositions);
    }
}

function drawDataFlow(nodePositions) {
    if (isTraining && Math.random() > 0.4 && flowParticles.length < 50) {
        flowParticles.push({
            layer: 0,
            progress: 0,
            fromNode: Math.floor(Math.random() * Math.min(network.layers[0], 8)),
            toNode: Math.floor(Math.random() * Math.min(network.layers[1], 8))
        });
    }

    flowParticles = flowParticles.filter(particle => {
        particle.progress += 0.05;

        if (particle.progress >= 1) {
            particle.layer++;
            particle.progress = 0;
            particle.fromNode = particle.toNode;
            if (particle.layer < network.weights.length) {
                const nextLayerSize = Math.min(network.layers[particle.layer + 1], 8);
                particle.toNode = Math.floor(Math.random() * nextLayerSize);
            } else {
                return false;
            }
        }

        if (particle.layer < nodePositions.length - 1) {
            const from = nodePositions[particle.layer][particle.fromNode];
            const to = nodePositions[particle.layer + 1][particle.toNode];

            const x = from.x + (to.x - from.x) * particle.progress;
            const y = from.y + (to.y - from.y) * particle.progress;

            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#3498db';
            ctx.fill();

            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(52, 152, 219, 0.3)';
            ctx.fill();
        }

        return particle.layer < nodePositions.length - 1;
    });
}

// ===================================
// TRAINING
// ===================================
const trainBtn = document.getElementById('trainBtn');
if (trainBtn) {
    trainBtn.addEventListener('click', () => {
        if (isTraining) {
            stopTraining();
        } else {
            startTraining();
        }
    });
}

function startTraining() {
    if (isTraining) return;

    isTraining = true;
    currentEpoch = 0;

    const activation = document.getElementById('activation').value;
    const optimizer = document.getElementById('optimizer').value;
    const epochs = parseInt(document.getElementById('epochs').value);
    const lr = parseFloat(document.getElementById('learningRate').value);

    const key = `${activation}-${optimizer}-${epochs}-${lr}`;
    const results = trainingResults[key];

    if (!results) {
        alert('Configuration not found in training data!');
        isTraining = false;
        return;
    }

    setStepRunning(5);

    trainBtn.textContent = 'Training...';
    trainBtn.classList.add('running');
    trainBtn.disabled = true;

    const codeBlock = document.querySelector('#step-5 .code-block');
    const outputBox = document.querySelector('#step-5 .output-box');
    if (codeBlock) codeBlock.classList.add('running');
    if (outputBox) outputBox.classList.add('running');

    const outputContent = document.getElementById('output-5');
    if (outputContent) {
        outputContent.style.display = 'block';
    }

    setTimeout(() => {
        if (canvas) {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            drawNetwork();
        }
    }, 50);

    document.getElementById('trainingStatus').textContent = 'Training in progress...';

    const epochDuration = 3000 / epochs;
    trainingInterval = setInterval(() => {
        currentEpoch++;

        const loss = results.loss[currentEpoch - 1];
        const acc = results.accuracy[currentEpoch - 1];

        const inputs = Array(784).fill(0).map(() => Math.random() * 0.5);
        network.forward(inputs);

        document.getElementById('currentEpoch').textContent = currentEpoch;
        document.getElementById('lossValue').textContent = loss.toFixed(6);
        document.getElementById('accuracy').textContent = acc.toFixed(2) + '%';

        drawNetwork();

        if (currentEpoch >= epochs) {
            stopTraining();
            completeTraining(results);
        }
    }, epochDuration);
}

function stopTraining() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    isTraining = false;

    function finishAnimation() {
        if (flowParticles.length > 0) {
            drawNetwork();
            requestAnimationFrame(finishAnimation);
        }
    }

    requestAnimationFrame(finishAnimation);
}

function completeTraining(results) {
    trainBtn.textContent = '✓ Completed';
    trainBtn.classList.remove('running');
    trainBtn.classList.add('completed');
    trainBtn.disabled = true;

    const codeBlock = document.querySelector('#step-5 .code-block');
    const outputBox = document.querySelector('#step-5 .output-box');
    if (codeBlock) {
        codeBlock.classList.remove('running');
        codeBlock.classList.add('completed');
    }
    if (outputBox) {
        outputBox.classList.remove('running');
        outputBox.classList.add('completed');
    }

    const finalAcc = results.accuracy[results.accuracy.length - 1];
    const finalLoss = results.loss[results.loss.length - 1];

    document.getElementById('accuracy').textContent = finalAcc.toFixed(2) + '%';
    document.getElementById('trainingStatus').innerHTML = `✓ Training Completed Successfully!<br><br>
        <strong>Final Results:</strong><br>
        Loss: ${finalLoss.toFixed(4)}<br>
        Accuracy: ${finalAcc.toFixed(2)}%`;

    window.currentResults = results;

    completeStep(5);

    // Auto-navigate to next step
    setTimeout(() => {
        const stepNum = 5;
        if (stepNum < 6) {
            appState.currentStep = stepNum + 1;
            stepContents.forEach(c => c.classList.remove('active'));
            document.getElementById(`step-${stepNum + 1}`).classList.add('active');

            if (stepNum + 1 === 6) {
                displayResults();
            }

            updateStepButtons();
        }
    }, 500);
}

// ===================================
// DISPLAY RESULTS
// ===================================
function displayResults() {
    const results = window.currentResults;

    if (!results) {
        alert('Please complete training first!');
        return;
    }

    drawCharts(results);
}

function drawCharts(results) {
    const lossCanvas = document.getElementById('lossChart');
    const accCanvas = document.getElementById('accuracyChart');

    if (!lossCanvas || !accCanvas) return;

    const lossCtx = lossCanvas.getContext('2d');
    const accCtx = accCanvas.getContext('2d');

    lossCanvas.width = lossCanvas.offsetWidth;
    lossCanvas.height = 300;
    accCanvas.width = accCanvas.offsetWidth;
    accCanvas.height = 300;

    drawLineChart(lossCtx, results.loss, 'Training Loss', '#e74c3c');
    drawLineChart(accCtx, results.accuracy, 'Training Accuracy (%)', '#27ae60');
}

function drawLineChart(ctx, data, title, color) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    const padding = 50;

    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = '#2c3e50';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(title, width / 2, 30);

    const maxVal = Math.max(...data);
    const minVal = Math.min(...data);
    const range = maxVal - minVal;

    const xStep = (width - 2 * padding) / (data.length - 1);
    const yScale = (height - 2 * padding) / range;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;

    data.forEach((val, i) => {
        const x = padding + i * xStep;
        const y = height - padding - (val - minVal) * yScale;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    data.forEach((val, i) => {
        const x = padding + i * xStep;
        const y = height - padding - (val - minVal) * yScale;

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
    });

    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', width / 2, height - 10);
}

// ===================================
// RUN BUTTONS FOR EACH STEP
// ===================================
function setupRunButtons() {
    const runButtons = document.querySelectorAll('.run-btn');
    runButtons.forEach((btn) => {
        if (btn.id === 'trainBtn') return;

        btn.addEventListener('click', function () {
            const stepContent = this.closest('.step-content');
            const stepId = stepContent.id;
            const stepNum = parseInt(stepId.split('-')[1]);

            // If step is already completed, do nothing
            if (appState.completedSteps.includes(stepNum)) {
                return;
            }

            setStepRunning(stepNum);

            this.textContent = 'Running...';
            this.classList.add('running');
            this.disabled = true;

            const codeBlock = stepContent.querySelector('.code-block');
            const outputBox = stepContent.querySelector('.output-box');

            if (codeBlock) codeBlock.classList.add('running');
            if (outputBox) outputBox.classList.add('running');

            setTimeout(() => {
                this.textContent = '✓ Completed';
                this.classList.remove('running');
                this.classList.add('completed');

                if (codeBlock) {
                    codeBlock.classList.remove('running');
                    codeBlock.classList.add('completed');
                }
                if (outputBox) {
                    outputBox.classList.remove('running');
                    outputBox.classList.add('completed');
                }

                const outputContent = document.getElementById(`output-${stepNum}`);
                if (outputContent) {
                    outputContent.style.display = 'block';
                }

                if (stepNum === 3) {
                    updateArchitecture();
                }

                completeStep(stepNum);

                // Auto-navigate to next step after a short delay
                setTimeout(() => {
                    if (stepNum < 6) {
                        appState.currentStep = stepNum + 1;
                        stepContents.forEach(c => c.classList.remove('active'));
                        document.getElementById(`step-${stepNum + 1}`).classList.add('active');

                        if (stepNum + 1 === 3) {
                            updateArchitecture();
                        } else if (stepNum + 1 === 5) {
                            initializeCanvas();
                        } else if (stepNum + 1 === 6) {
                            displayResults();
                        }

                        updateStepButtons();
                    }
                }, 500);
            }, 1000);
        });
    });
}

// ===================================
// INITIALIZATION
// ===================================
document.addEventListener('DOMContentLoaded', () => {
    updateArchitecture();
    if (canvas) {
        initializeCanvas();
    }
    setupRunButtons();
    updateStepButtons();
    updateCodeDisplay();
});
