// Demo 4: Learning Rate Scheduler

// State
let state = {
    scheduler: 'step',
    initialLR: 0.1,
    epochs: 100,
    simulations: []
};

// Canvas contexts
let lrCtx, lossCtx, accuracyCtx;
let lrCanvas, lossCanvas, accuracyCanvas;

// Scheduler colors
const schedulerColors = {
    none: '#666666',
    step: '#2563eb',
    cosine: '#10b981',
    exponential: '#f59e0b'
};

// Initialize
function init() {
    lrCanvas = document.getElementById('lr-canvas');
    lossCanvas = document.getElementById('loss-canvas');
    accuracyCanvas = document.getElementById('accuracy-canvas');

    lrCtx = lrCanvas.getContext('2d');
    lossCtx = lossCanvas.getContext('2d');
    accuracyCtx = accuracyCanvas.getContext('2d');

    resizeCanvases();

    // Event listeners
    document.getElementById('scheduler-select').addEventListener('change', (e) => {
        state.scheduler = e.target.value;
    });

    document.getElementById('initial-lr').addEventListener('input', (e) => {
        state.initialLR = parseFloat(e.target.value);
        document.getElementById('initial-lr-value').textContent = state.initialLR.toFixed(2);
    });

    document.getElementById('epochs').addEventListener('input', (e) => {
        state.epochs = parseInt(e.target.value);
        document.getElementById('epochs-value').textContent = state.epochs;
    });

    document.getElementById('simulate-btn').addEventListener('click', simulateTraining);
    document.getElementById('compare-btn').addEventListener('click', compareSchedulers);
    document.getElementById('reset-btn').addEventListener('click', reset);

    // Initial render
    reset();

    // ResizeObserver: redraw when container changes size
    if (window.ResizeObserver) {
        const ro = new ResizeObserver(() => {
            resizeCanvases();
            drawAllCharts();
        });
        const vizPanel = document.querySelector('.visualization-panel');
        if (vizPanel) ro.observe(vizPanel);
    }
}

// Resize canvases
// Resize canvases
function resizeCanvases() {
    const scrollContainer = document.getElementById('scroll-container');
    const width = scrollContainer ? (scrollContainer.clientWidth - 40) : lrCanvas.clientWidth;

    lrCanvas.width = width;
    lossCanvas.width = width;
    accuracyCanvas.width = width;
}

// Reset
function reset() {
    state.simulations = [];
    drawAllCharts();
}

// Simulate training with current settings
function simulateTraining() {
    const simulation = {
        scheduler: state.scheduler,
        initialLR: state.initialLR,
        epochs: state.epochs,
        learningRates: [],
        losses: [],
        accuracies: []
    };

    // Generate learning rate schedule
    for (let epoch = 0; epoch < state.epochs; epoch++) {
        const lr = getLearningRate(state.scheduler, state.initialLR, epoch, state.epochs);
        simulation.learningRates.push(lr);
    }

    // Simulate training curves
    simulateTrainingCurves(simulation);

    // Replace or add simulation
    state.simulations = [simulation];

    drawAllCharts();
}

// Compare all schedulers
function compareSchedulers() {
    state.simulations = [];

    const schedulers = ['none', 'step', 'cosine', 'exponential'];

    for (const scheduler of schedulers) {
        const simulation = {
            scheduler: scheduler,
            initialLR: state.initialLR,
            epochs: state.epochs,
            learningRates: [],
            losses: [],
            accuracies: []
        };

        // Generate learning rate schedule
        for (let epoch = 0; epoch < state.epochs; epoch++) {
            const lr = getLearningRate(scheduler, state.initialLR, epoch, state.epochs);
            simulation.learningRates.push(lr);
        }

        // Simulate training curves
        simulateTrainingCurves(simulation);

        state.simulations.push(simulation);
    }

    drawAllCharts();
}

// Get learning rate for given scheduler
function getLearningRate(scheduler, initialLR, epoch, totalEpochs) {
    switch (scheduler) {
        case 'none':
            return initialLR;

        case 'step':
            // Reduce by 0.1 at 50% and 75% of training
            if (epoch >= totalEpochs * 0.75) {
                return initialLR * 0.01;
            } else if (epoch >= totalEpochs * 0.5) {
                return initialLR * 0.1;
            }
            return initialLR;

        case 'cosine':
            // Cosine annealing
            const minLR = initialLR * 0.01;
            return minLR + (initialLR - minLR) * 0.5 * (1 + Math.cos(Math.PI * epoch / totalEpochs));

        case 'exponential':
            // Exponential decay
            const decayRate = Math.pow(0.01, 1 / totalEpochs);
            return initialLR * Math.pow(decayRate, epoch);

        default:
            return initialLR;
    }
}

// Simulate training curves based on learning rate schedule
function simulateTrainingCurves(simulation) {
    const { learningRates, epochs } = simulation;

    let loss = 2.3; // Initial loss (similar to random initialization)
    let accuracy = 0.1; // Initial accuracy (10% for 10-class problem)

    for (let epoch = 0; epoch < epochs; epoch++) {
        const lr = learningRates[epoch];

        // Simulate loss decrease
        // Higher LR = faster initial decrease but potential instability
        // Lower LR = slower but more stable
        const lossDecrease = lr * 0.5 * (1 - accuracy) * (1 + 0.1 * (Math.random() - 0.5));
        loss = Math.max(0.1, loss - lossDecrease);

        // Add noise based on learning rate (higher LR = more noise)
        const noise = lr * 0.2 * (Math.random() - 0.5);
        loss += noise;

        // Simulate accuracy increase
        const accuracyIncrease = lr * 0.15 * (1 - accuracy) * (1 + 0.1 * (Math.random() - 0.5));
        accuracy = Math.min(0.95, accuracy + accuracyIncrease);

        simulation.losses.push(loss);
        simulation.accuracies.push(accuracy);
    }

    // Apply smoothing to make curves more realistic
    simulation.losses = smoothCurve(simulation.losses, 3);
    simulation.accuracies = smoothCurve(simulation.accuracies, 3);
}

// Smooth curve using moving average
function smoothCurve(data, windowSize) {
    const smoothed = [];
    for (let i = 0; i < data.length; i++) {
        let sum = 0;
        let count = 0;
        for (let j = Math.max(0, i - windowSize); j <= Math.min(data.length - 1, i + windowSize); j++) {
            sum += data[j];
            count++;
        }
        smoothed.push(sum / count);
    }
    return smoothed;
}

// Draw all charts
function drawAllCharts() {
    drawLRChart();
    drawLossChart();
    drawAccuracyChart();
}

// Draw learning rate chart
function drawLRChart() {
    const width = lrCanvas.width;
    const height = lrCanvas.height;

    lrCtx.clearRect(0, 0, width, height);

    if (state.simulations.length === 0) {
        drawEmptyMessage(lrCtx, width, height, 'Click "Simulate Training" to see learning rate schedule');
        return;
    }

    const maxEpochs = Math.max(...state.simulations.map(s => s.epochs));
    const maxLR = Math.max(...state.simulations.flatMap(s => s.learningRates));

    CanvasUtils.drawAxes(lrCtx, width, height, [0, maxEpochs], [0, maxLR * 1.1], {
        xLabel: 'Epoch',
        yLabel: 'Learning Rate',
        gridLines: true
    });

    // Plot each simulation
    for (const sim of state.simulations) {
        const epochs = Array.from({ length: sim.epochs }, (_, i) => i);
        const color = schedulerColors[sim.scheduler] || '#666';

        CanvasUtils.plotFunction(lrCtx, width, height, epochs, sim.learningRates,
            [0, maxEpochs], [0, maxLR * 1.1], {
            color: color,
            lineWidth: 2
        });
    }

    // Draw legend
    drawLegend(lrCtx, width, 10);
}

// Draw loss chart
function drawLossChart() {
    const width = lossCanvas.width;
    const height = lossCanvas.height;

    lossCtx.clearRect(0, 0, width, height);

    if (state.simulations.length === 0) {
        drawEmptyMessage(lossCtx, width, height, 'Training loss will appear here');
        return;
    }

    const maxEpochs = Math.max(...state.simulations.map(s => s.epochs));
    const maxLoss = Math.max(...state.simulations.flatMap(s => s.losses));
    const minLoss = Math.min(...state.simulations.flatMap(s => s.losses));

    CanvasUtils.drawAxes(lossCtx, width, height, [0, maxEpochs], [minLoss * 0.9, maxLoss * 1.1], {
        xLabel: 'Epoch',
        yLabel: 'Loss',
        gridLines: true
    });

    // Plot each simulation
    for (const sim of state.simulations) {
        const epochs = Array.from({ length: sim.epochs }, (_, i) => i);
        const color = schedulerColors[sim.scheduler] || '#666';

        CanvasUtils.plotFunction(lossCtx, width, height, epochs, sim.losses,
            [0, maxEpochs], [minLoss * 0.9, maxLoss * 1.1], {
            color: color,
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

    if (state.simulations.length === 0) {
        drawEmptyMessage(accuracyCtx, width, height, 'Validation accuracy will appear here');
        return;
    }

    const maxEpochs = Math.max(...state.simulations.map(s => s.epochs));

    CanvasUtils.drawAxes(accuracyCtx, width, height, [0, maxEpochs], [0, 1], {
        xLabel: 'Epoch',
        yLabel: 'Accuracy',
        gridLines: true
    });

    // Plot each simulation
    for (const sim of state.simulations) {
        const epochs = Array.from({ length: sim.epochs }, (_, i) => i);
        const color = schedulerColors[sim.scheduler] || '#666';

        CanvasUtils.plotFunction(accuracyCtx, width, height, epochs, sim.accuracies,
            [0, maxEpochs], [0, 1], {
            color: color,
            lineWidth: 2
        });
    }

    // Draw legend
    drawLegend(accuracyCtx, width, 10);
}

// Draw legend
function drawLegend(ctx, width, y) {
    if (state.simulations.length === 0) return;

    ctx.save();

    const startX = width - 200;
    let currentY = y + 20;

    for (const sim of state.simulations) {
        const color = schedulerColors[sim.scheduler] || '#666';
        const label = sim.scheduler.charAt(0).toUpperCase() + sim.scheduler.slice(1);

        // Draw line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(startX, currentY);
        ctx.lineTo(startX + 30, currentY);
        ctx.stroke();

        // Draw label
        ctx.fillStyle = '#1a1a1a';
        ctx.font = '12px Inter';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, startX + 40, currentY);

        currentY += 20;
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
    drawAllCharts();
});
