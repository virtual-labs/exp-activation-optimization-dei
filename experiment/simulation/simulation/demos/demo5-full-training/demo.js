// Demo 5: Full Training Overlay with TensorFlow.js

// State
let state = {
    activation: 'relu',
    optimizer: 'adam',
    learningRate: 0.001,
    batchSize: 128,
    epochs: 10,
    trainingRuns: [],
    isTraining: false,
    currentModel: null,
    stopRequested: false
};

// Canvas contexts
let lossCtx, accuracyCtx;
let lossCanvas, accuracyCanvas;

// Dataset
let trainData = null;
let testData = null;

// Colors for different runs
const runColors = [
    '#2563eb', '#10b981', '#f59e0b', '#ef4444',
    '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'
];

// Initialize
async function init() {
    lossCanvas = document.getElementById('loss-canvas');
    accuracyCanvas = document.getElementById('accuracy-canvas');
    lossCtx = lossCanvas.getContext('2d');
    accuracyCtx = accuracyCanvas.getContext('2d');

    resizeCanvases();

    // Event listeners
    document.getElementById('activation-select').addEventListener('change', (e) => {
        state.activation = e.target.value;
    });

    document.getElementById('optimizer-select').addEventListener('change', (e) => {
        state.optimizer = e.target.value;
    });

    document.getElementById('learning-rate').addEventListener('input', (e) => {
        state.learningRate = parseFloat(e.target.value);
        document.getElementById('lr-value').textContent = state.learningRate.toFixed(4);
    });

    document.getElementById('batch-size').addEventListener('input', (e) => {
        state.batchSize = parseInt(e.target.value);
        document.getElementById('batch-value').textContent = state.batchSize;
    });

    document.getElementById('epochs').addEventListener('input', (e) => {
        state.epochs = parseInt(e.target.value);
        document.getElementById('epochs-value').textContent = state.epochs;
    });

    document.getElementById('train-btn').addEventListener('click', startTraining);
    document.getElementById('stop-btn').addEventListener('click', stopTraining);
    document.getElementById('clear-btn').addEventListener('click', clearRuns);

    // Initial chart draw
    drawCharts();

    // Load dataset
    updateStatus('Loading dataset...');
    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = true;
    trainBtn.textContent = 'Loading...';

    try {
        // Wait for tf to be ready
        await tf.ready();
        await loadData();

        updateStatus('Ready');
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train Model';
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error loading data: ' + error.message);
    }
}

// Resize canvases
function resizeCanvases() {
    const width = lossCanvas.clientWidth;
    lossCanvas.width = width;
    accuracyCanvas.width = width;
}

// Load data (Synthetic for demo stability)
async function loadData() {
    // Generate synthetic data for robust demo experience without external dependencies
    // In a production environment, you would use tf.data.csv/web

    tf.tidy(() => {
        // 1000 training examples, 784 features (28x28)
        const trainXs = tf.randomNormal([1000, 784]);
        const trainYs = tf.oneHot(tf.randomUniform([1000], 0, 10, 'int32'), 10);

        // 200 test examples
        const testXs = tf.randomNormal([200, 784]);
        const testYs = tf.oneHot(tf.randomUniform([200], 0, 10, 'int32'), 10);

        // Store globally, keeping them in memory (not disposed by tidy)
        trainData = {
            xs: tf.keep(trainXs),
            ys: tf.keep(trainYs)
        };

        testData = {
            xs: tf.keep(testXs),
            ys: tf.keep(testYs)
        };
    });
}

// Create model
function createModel() {
    const model = tf.sequential();

    // Input layer
    model.add(tf.layers.dense({
        inputShape: [784],
        units: 128,
        activation: state.activation
    }));

    // Hidden layer
    model.add(tf.layers.dense({
        units: 64,
        activation: state.activation
    }));

    // Output layer
    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax'
    }));

    // Compile model
    let optimizer;
    switch (state.optimizer) {
        case 'sgd':
            optimizer = tf.train.sgd(state.learningRate);
            break;
        case 'adam':
            optimizer = tf.train.adam(state.learningRate);
            break;
        case 'rmsprop':
            optimizer = tf.train.rmsprop(state.learningRate);
            break;
        default:
            optimizer = tf.train.adam(state.learningRate);
    }

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// Start training
async function startTraining() {
    if (state.isTraining) return;

    if (!trainData || !trainData.xs) {
        updateStatus('Error: Data not loaded yet');
        return;
    }

    state.isTraining = true;
    state.stopRequested = false;

    document.getElementById('train-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;

    // Create new training run
    const run = {
        activation: state.activation,
        optimizer: state.optimizer,
        learningRate: state.learningRate,
        batchSize: state.batchSize,
        epochs: state.epochs,
        losses: [],
        accuracies: [],
        color: runColors[state.trainingRuns.length % runColors.length],
        label: `${state.activation} + ${state.optimizer}`
    };

    state.trainingRuns.push(run);

    // Create and train model
    state.currentModel = createModel();

    updateStatus('Training...');
    document.getElementById('progress-container').style.display = 'block';
    document.getElementById('total-epochs').textContent = state.epochs;

    try {
        await state.currentModel.fit(trainData.xs, trainData.ys, {
            batchSize: state.batchSize,
            epochs: state.epochs,
            validationData: [testData.xs, testData.ys],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (state.stopRequested) {
                        state.currentModel.stopTraining = true;
                        return;
                    }

                    run.losses.push(logs.loss);
                    run.accuracies.push(logs.val_acc || logs.acc);

                    document.getElementById('current-epoch').textContent = epoch + 1;
                    document.getElementById('current-loss').textContent = logs.loss.toFixed(4);
                    document.getElementById('current-accuracy').textContent =
                        ((logs.val_acc || logs.acc) * 100).toFixed(2) + '%';

                    drawCharts();

                    await tf.nextFrame();
                }
            }
        });

        if (!state.stopRequested) {
            updateStatus('Training complete!');
        } else {
            updateStatus('Training stopped');
        }
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Training failed: ' + error.message);
    }

    state.isTraining = false;
    document.getElementById('train-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;

    // Cleanup
    if (state.currentModel) {
        state.currentModel.dispose();
        state.currentModel = null;
    }
}

// Stop training
function stopTraining() {
    state.stopRequested = true;
    updateStatus('Stopping training...');
}

// Clear all runs
function clearRuns() {
    state.trainingRuns = [];
    drawCharts();
}

// Update status
function updateStatus(message) {
    document.getElementById('status-text').textContent = message;
}

// Draw charts
function drawCharts() {
    drawLossChart();
    drawAccuracyChart();
}

// Draw loss chart
function drawLossChart() {
    const width = lossCanvas.width;
    const height = lossCanvas.height;

    lossCtx.clearRect(0, 0, width, height);

    if (state.trainingRuns.length === 0 || state.trainingRuns.every(r => r.losses.length === 0)) {
        drawEmptyMessage(lossCtx, width, height, 'Training loss curves will appear here');
        return;
    }

    const maxEpochs = Math.max(...state.trainingRuns.map(r => r.losses.length));
    const allLosses = state.trainingRuns.flatMap(r => r.losses);
    const maxLoss = Math.max(...allLosses);
    const minLoss = Math.min(...allLosses);

    CanvasUtils.drawAxes(lossCtx, width, height, [0, maxEpochs], [minLoss * 0.9, maxLoss * 1.1], {
        xLabel: 'Epoch',
        yLabel: 'Loss',
        gridLines: true
    });

    // Plot each run
    for (const run of state.trainingRuns) {
        if (run.losses.length === 0) continue;

        const epochs = Array.from({ length: run.losses.length }, (_, i) => i + 1);

        CanvasUtils.plotFunction(lossCtx, width, height, epochs, run.losses,
            [0, maxEpochs], [minLoss * 0.9, maxLoss * 1.1], {
            color: run.color,
            lineWidth: 2
        });
    }

    // Draw legend
    drawLegend(lossCtx, width, 10);
}

// Draw accuracy chart
function drawAccuracyChart() {
    const width = accuracyCanvas.width;
    const height = accuracyCanvas.height;

    accuracyCtx.clearRect(0, 0, width, height);

    if (state.trainingRuns.length === 0 || state.trainingRuns.every(r => r.accuracies.length === 0)) {
        drawEmptyMessage(accuracyCtx, width, height, 'Validation accuracy curves will appear here');
        return;
    }

    const maxEpochs = Math.max(...state.trainingRuns.map(r => r.accuracies.length));

    CanvasUtils.drawAxes(accuracyCtx, width, height, [0, maxEpochs], [0, 1], {
        xLabel: 'Epoch',
        yLabel: 'Accuracy',
        gridLines: true
    });

    // Plot each run
    for (const run of state.trainingRuns) {
        if (run.accuracies.length === 0) continue;

        const epochs = Array.from({ length: run.accuracies.length }, (_, i) => i + 1);

        CanvasUtils.plotFunction(accuracyCtx, width, height, epochs, run.accuracies,
            [0, maxEpochs], [0, 1], {
            color: run.color,
            lineWidth: 2
        });
    }

    // Draw legend
    drawLegend(accuracyCtx, width, 10);
}

// Draw legend
function drawLegend(ctx, width, y) {
    if (state.trainingRuns.length === 0) return;

    ctx.save();

    const startX = width - 250;
    let currentY = y + 20;

    for (const run of state.trainingRuns) {
        // Draw line
        ctx.strokeStyle = run.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(startX, currentY);
        ctx.lineTo(startX + 30, currentY);
        ctx.stroke();

        // Draw label
        ctx.fillStyle = '#1a1a1a';
        ctx.font = '11px Inter';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(run.label, startX + 40, currentY);

        currentY += 18;
    }

    ctx.restore();
}

// Draw empty message
function drawEmptyMessage(ctx, width, height, message) {
    ctx.fillStyle = '#666';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, width / 2, height / 2);
}

// Initialize on load
window.addEventListener('load', init);
window.addEventListener('resize', () => {
    resizeCanvases();
    drawCharts();
});
