// Demo 3: Optimizer Step Visualizer

// State
let state = {
    optimizer: 'adam',
    learningRate: 0.1,
    momentumBeta: 0.9,
    position: { x: -1.5, y: 1.5 }, // Starting position
    velocity: { x: 0, y: 0 },
    m: { x: 0, y: 0 }, // First moment (Adam)
    v: { x: 0, y: 0 }, // Second moment (Adam)
    iteration: 0,
    path: [],
    isRunning: false,
    animationId: null
};

// Default parameters
const defaults = {
    sgd:      { lr: 0.002, beta: 0.0 },
    momentum: { lr: 0.001, beta: 0.9 },
    adam:     { lr: 0.1,   beta: 0.9 }
};

// Per-optimizer animation config
const optimizerConfig = {
    sgd:      { maxIter: 5000, stepsPerFrame: 20, delay: 30 },
    momentum: { maxIter: 1500, stepsPerFrame:  3, delay: 30 },
    adam:     { maxIter:  500, stepsPerFrame:  1, delay: 50 }
};

// Canvas
let canvas, ctx;

// Loss function (Rosenbrock function - classic optimization test)
function rosenbrock(x, y) {
    const a = 1;
    const b = 100;
    return Math.pow(a - x, 2) + b * Math.pow(y - x * x, 2);
}

// Gradient of Rosenbrock function
function rosenbrockGradient(x, y) {
    const a = 1;
    const b = 100;
    const dx = -2 * (a - x) - 4 * b * x * (y - x * x);
    const dy = 2 * b * (y - x * x);
    return { x: dx, y: dy };
}

// Initialize
function init() {
    canvas = document.getElementById('surface-canvas');
    ctx = canvas.getContext('2d');

    resizeCanvas();

    // Event listeners
    document.getElementById('optimizer-select').addEventListener('change', (e) => {
        state.optimizer = e.target.value;
        updateControls();
        resetDefaults();
        reset();
    });

    document.getElementById('learning-rate').addEventListener('input', (e) => {
        state.learningRate = parseFloat(e.target.value);
        document.getElementById('lr-value').textContent = state.learningRate.toFixed(4);
    });

    document.getElementById('momentum-beta').addEventListener('input', (e) => {
        state.momentumBeta = parseFloat(e.target.value);
        document.getElementById('momentum-value').textContent = state.momentumBeta.toFixed(2);
    });

    document.getElementById('start-btn').addEventListener('click', toggleOptimization);
    document.getElementById('step-btn').addEventListener('click', singleStep);
    document.getElementById('reset-btn').addEventListener('click', reset);
    document.getElementById('reset-params-btn').addEventListener('click', resetDefaults);

    // Initial setup
    updateControls();
    resetDefaults();
    reset();
}

// Resize canvas
function resizeCanvas() {
    const size = Math.min(canvas.clientWidth, 800);
    canvas.width = size;
    canvas.height = size;
}

// Update controls based on optimizer selection
function updateControls() {
    const momentumControl = document.getElementById('momentum-control-group');
    if (state.optimizer === 'momentum') {
        momentumControl.style.display = 'flex';
    } else {
        momentumControl.style.display = 'none';
    }
}

// Reset defaults
function resetDefaults() {
    const def = defaults[state.optimizer];
    if (def) {
        state.learningRate = def.lr;
        document.getElementById('learning-rate').value = def.lr;
        document.getElementById('lr-value').textContent = def.lr.toFixed(4);

        if (def.beta !== undefined) {
            state.momentumBeta = def.beta;
            document.getElementById('momentum-beta').value = def.beta;
            document.getElementById('momentum-value').textContent = def.beta.toFixed(2);
        }
    }
}

// Status update helper
function updateStatus(message, color = '#2563eb') {
    const el = document.getElementById('optimizer-status');
    if (el) {
        el.textContent = message;
        el.style.color = color;
    }
}

// Reset
function reset() {
    state.position = { x: -1.5, y: 1.5 };
    state.velocity = { x: 0, y: 0 };
    state.m = { x: 0, y: 0 };
    state.v = { x: 0, y: 0 };
    state.iteration = 0;
    state.path = [{ ...state.position }];
    state.isRunning = false;

    if (state.animationId) {
        cancelAnimationFrame(state.animationId);
        state.animationId = null;
    }

    document.getElementById('start-btn').textContent = 'Start Optimization';
    updateStatus('Ready');
    updateDisplay();
    drawVisualization();
}

// Toggle optimization
function toggleOptimization() {
    state.isRunning = !state.isRunning;

    if (state.isRunning) {
        document.getElementById('start-btn').textContent = 'Pause';
        updateStatus('Running...', '#10b981');
        runOptimization();
    } else {
        document.getElementById('start-btn').textContent = 'Resume';
        updateStatus('Paused', '#f59e0b');
        if (state.animationId) {
            cancelAnimationFrame(state.animationId);
            state.animationId = null;
        }
    }
}

// Single step
function singleStep() {
    if (state.isRunning) return;
    optimizationStep();
    drawVisualization();
    updateStatus('Stepped', '#666');
}

// Run optimization loop
function runOptimization() {
    if (!state.isRunning) return;

    const cfg = optimizerConfig[state.optimizer] || { maxIter: 500, stepsPerFrame: 1, delay: 50 };
    let stoppedEarly = false;

    for (let k = 0; k < cfg.stepsPerFrame && state.iteration < cfg.maxIter; k++) {
        optimizationStep();

        const loss = rosenbrock(state.position.x, state.position.y);

        if (loss < 1e-4) {
            state.isRunning = false;
            document.getElementById('start-btn').textContent = 'Converged ✓';
            updateStatus(`Converged at step ${state.iteration} (Loss < 1e-4)`, '#10b981');
            stoppedEarly = true;
            break;
        }

        if (!isFinite(loss) || Math.abs(state.position.x) > 6 || Math.abs(state.position.y) > 6) {
            state.isRunning = false;
            document.getElementById('start-btn').textContent = 'Start Optimization';
            updateStatus('Diverged — try a smaller learning rate', '#ef4444');
            stoppedEarly = true;
            break;
        }
    }

    drawVisualization();

    if (!stoppedEarly) {
        if (state.iteration >= cfg.maxIter) {
            state.isRunning = false;
            document.getElementById('start-btn').textContent = 'Start Optimization';
            updateStatus('Max iterations reached', '#666');
        } else if (state.isRunning) {
            state.animationId = setTimeout(() => runOptimization(), cfg.delay);
        }
    }
}

// Perform one optimization step
function optimizationStep() {
    const grad = rosenbrockGradient(state.position.x, state.position.y);

    switch (state.optimizer) {
        case 'sgd':
            sgdStep(grad);
            break;
        case 'momentum':
            momentumStep(grad);
            break;
        case 'adam':
            adamStep(grad);
            break;
    }

    state.iteration++;
    state.path.push({ ...state.position });
    updateDisplay();
}

// Clip gradient to prevent explosion
function clipGradient(grad, maxNorm = 5.0) {
    const norm = Math.sqrt(grad.x * grad.x + grad.y * grad.y);
    if (norm > maxNorm) {
        const scale = maxNorm / norm;
        return { x: grad.x * scale, y: grad.y * scale };
    }
    return grad;
}

// SGD step
function sgdStep(grad) {
    const clipped = clipGradient(grad);
    state.position.x -= state.learningRate * clipped.x;
    state.position.y -= state.learningRate * clipped.y;
}

// Momentum step
function momentumStep(grad) {
    const clipped = clipGradient(grad);
    state.velocity.x = state.momentumBeta * state.velocity.x - state.learningRate * clipped.x;
    state.velocity.y = state.momentumBeta * state.velocity.y - state.learningRate * clipped.y;

    state.position.x += state.velocity.x;
    state.position.y += state.velocity.y;
}

// Adam step
function adamStep(grad) {
    const clipped = clipGradient(grad); // prevent exploding gradients on steep surfaces
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

    // Update biased first moment estimate
    state.m.x = beta1 * state.m.x + (1 - beta1) * clipped.x;
    state.m.y = beta1 * state.m.y + (1 - beta1) * clipped.y;

    // Update biased second moment estimate
    state.v.x = beta2 * state.v.x + (1 - beta2) * clipped.x * clipped.x;
    state.v.y = beta2 * state.v.y + (1 - beta2) * clipped.y * clipped.y;

    // Bias correction
    const mHatX = state.m.x / (1 - Math.pow(beta1, state.iteration + 1));
    const mHatY = state.m.y / (1 - Math.pow(beta1, state.iteration + 1));
    const vHatX = state.v.x / (1 - Math.pow(beta2, state.iteration + 1));
    const vHatY = state.v.y / (1 - Math.pow(beta2, state.iteration + 1));

    // Update parameters
    state.position.x -= state.learningRate * mHatX / (Math.sqrt(vHatX) + epsilon);
    state.position.y -= state.learningRate * mHatY / (Math.sqrt(vHatY) + epsilon);
}

// Update display
function updateDisplay() {
    document.getElementById('iteration-display').textContent = state.iteration;
    if (Math.abs(state.position.x) > 100 || Math.abs(state.position.y) > 100) {
        document.getElementById('loss-display').textContent = 'Exploded';
    } else {
        document.getElementById('loss-display').textContent =
            rosenbrock(state.position.x, state.position.y).toFixed(4);
    }
    document.getElementById('pos-x-display').textContent = state.position.x.toFixed(3);
    document.getElementById('pos-y-display').textContent = state.position.y.toFixed(3);
}

// Draw visualization
function drawVisualization() {
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw contour plot
    drawContours();

    // Draw path
    drawPath();

    // Draw current position
    drawCurrentPosition();

    // Draw gradient arrow
    drawGradientArrow();
}

// Draw contour plot
function drawContours() {
    const width = canvas.width;
    const height = canvas.height;
    const xRange = [-2, 2];
    const yRange = [-1, 3];

    const resolution = 100;
    const imageData = ctx.createImageData(width, height);

    // Find min/max loss for color scaling
    let minLoss = Infinity;
    let maxLoss = -Infinity;

    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            const x = xRange[0] + (xRange[1] - xRange[0]) * i / resolution;
            const y = yRange[0] + (yRange[1] - yRange[0]) * j / resolution;
            const loss = rosenbrock(x, y);
            minLoss = Math.min(minLoss, loss);
            maxLoss = Math.max(maxLoss, loss);
        }
    }

    // Use log scale for better visualization
    const logMin = Math.log(minLoss + 1);
    const logMax = Math.log(Math.min(maxLoss, 1000) + 1); // Cap max for better colors

    // Draw pixels
    for (let px = 0; px < width; px++) {
        for (let py = 0; py < height; py++) {
            const x = xRange[0] + (xRange[1] - xRange[0]) * px / width;
            const y = yRange[0] + (yRange[1] - yRange[0]) * (height - py) / height;
            const loss = rosenbrock(x, y);
            const logLoss = Math.log(loss + 1);

            // Normalize to 0-1
            const normalized = (logLoss - logMin) / (logMax - logMin);
            const clamped = Math.max(0, Math.min(1, normalized));

            // Color: blue (low) to red (high)
            const idx = (py * width + px) * 4;
            const r = Math.floor(clamped * 255);
            const g = Math.floor((1 - clamped) * 100);
            const b = Math.floor((1 - clamped) * 255);

            imageData.data[idx] = r;
            imageData.data[idx + 1] = g;
            imageData.data[idx + 2] = b;
            imageData.data[idx + 3] = 255;
        }
    }

    ctx.putImageData(imageData, 0, 0);

    // Draw contour lines
    drawContourLines(xRange, yRange);
}

// Draw contour lines
function drawContourLines(xRange, yRange) {
    const width = canvas.width;
    const height = canvas.height;
    const levels = [1, 5, 10, 50, 100, 200, 500];

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;

    for (const level of levels) {
        ctx.beginPath();
        let started = false;

        for (let i = 0; i <= 100; i++) {
            for (let j = 0; j <= 100; j++) {
                const x = xRange[0] + (xRange[1] - xRange[0]) * i / 100;
                const y = yRange[0] + (yRange[1] - yRange[0]) * j / 100;
                const loss = rosenbrock(x, y);

                if (Math.abs(loss - level) < level * 0.1) {
                    const px = (x - xRange[0]) / (xRange[1] - xRange[0]) * width;
                    const py = height - (y - yRange[0]) / (yRange[1] - yRange[0]) * height;

                    if (!started) {
                        ctx.moveTo(px, py);
                        started = true;
                    } else {
                        ctx.lineTo(px, py);
                    }
                }
            }
        }

        ctx.stroke();
    }
}

// Draw path
function drawPath() {
    if (state.path.length < 2) return;

    const width = canvas.width;
    const height = canvas.height;
    const xRange = [-2, 2];
    const yRange = [-1, 3];

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();

    for (let i = 0; i < state.path.length; i++) {
        const pos = state.path[i];
        const px = (pos.x - xRange[0]) / (xRange[1] - xRange[0]) * width;
        const py = height - (pos.y - yRange[0]) / (yRange[1] - yRange[0]) * height;

        // Skip if off-screen (prevent weird lines)
        if (px < -100 || px > width + 100 || py < -100 || py > height + 100) continue;

        if (i === 0) {
            ctx.moveTo(px, py);
        } else {
            ctx.lineTo(px, py);
        }
    }

    ctx.stroke();

    // Draw dots at each step
    ctx.fillStyle = '#10b981';
    let dotCount = 0;
    // Only draw last 50 dots to prevent clutter
    const startIndex = Math.max(0, state.path.length - 50);

    for (let i = startIndex; i < state.path.length; i++) {
        const pos = state.path[i];
        const px = (pos.x - xRange[0]) / (xRange[1] - xRange[0]) * width;
        const py = height - (pos.y - yRange[0]) / (yRange[1] - yRange[0]) * height;

        if (px < -10 || px > width + 10 || py < -10 || py > height + 10) continue;

        ctx.beginPath();
        ctx.arc(px, py, 3, 0, 2 * Math.PI);
        ctx.fill();
        dotCount++;
    }
}

// Draw current position
function drawCurrentPosition() {
    const width = canvas.width;
    const height = canvas.height;
    const xRange = [-2, 2];
    const yRange = [-1, 3];

    const px = (state.position.x - xRange[0]) / (xRange[1] - xRange[0]) * width;
    const py = height - (state.position.y - yRange[0]) / (yRange[1] - yRange[0]) * height;

    if (px < -20 || px > width + 20 || py < -20 || py > height + 20) return;

    // Outer circle
    ctx.fillStyle = '#2563eb';
    ctx.beginPath();
    ctx.arc(px, py, 10, 0, 2 * Math.PI);
    ctx.fill();

    // Inner circle
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, 2 * Math.PI);
    ctx.fill();
}

// Draw gradient arrow
function drawGradientArrow() {
    const width = canvas.width;
    const height = canvas.height;
    const xRange = [-2, 2];
    const yRange = [-1, 3];

    const grad = rosenbrockGradient(state.position.x, state.position.y);
    const scale = 0.02; // Scale for visualization

    // Clip gradient for visualization so arrow doesn't get huge
    const clippedGrad = clipGradient(grad, 10.0);

    const px = (state.position.x - xRange[0]) / (xRange[1] - xRange[0]) * width;
    const py = height - (state.position.y - yRange[0]) / (yRange[1] - yRange[0]) * height;

    if (px < -20 || px > width + 20 || py < -20 || py > height + 20) return;

    const dx = -clippedGrad.x * scale * width / (xRange[1] - xRange[0]);
    const dy = clippedGrad.y * scale * height / (yRange[1] - yRange[0]);

    // Draw arrow
    ctx.strokeStyle = '#f59e0b';
    ctx.fillStyle = '#f59e0b';
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + dx, py + dy);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(dy, dx);
    const headLength = 10;

    ctx.beginPath();
    ctx.moveTo(px + dx, py + dy);
    ctx.lineTo(
        px + dx - headLength * Math.cos(angle - Math.PI / 6),
        py + dy - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
        px + dx - headLength * Math.cos(angle + Math.PI / 6),
        py + dy - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
}

// Initialize on load
window.addEventListener('load', init);
window.addEventListener('resize', () => {
    resizeCanvas();
    drawVisualization();
});
