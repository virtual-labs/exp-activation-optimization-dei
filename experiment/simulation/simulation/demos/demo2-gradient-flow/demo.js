// Demo 2: Gradient Flow Visualizer

// State
let state = {
    activation: 'relu',
    layerCount: 5,
    inputValue: 2.0,
    layerValues: [],
    layerGradients: [],
    isAnimating: false
};

// Canvas contexts
let networkCtx, gradientChartCtx;
let networkCanvas, gradientChartCanvas;

// Network parameters
const NEURON_RADIUS = 20;
const LAYER_SPACING = 80;

// Initialize
function init() {
    networkCanvas = document.getElementById('network-canvas');
    gradientChartCanvas = document.getElementById('gradient-chart');
    networkCtx = networkCanvas.getContext('2d');
    gradientChartCtx = gradientChartCanvas.getContext('2d');

    resizeCanvases();

    // Event listeners
    document.getElementById('activation-select').addEventListener('change', (e) => {
        state.activation = e.target.value;
        reset();
    });

    document.getElementById('layer-count').addEventListener('input', (e) => {
        state.layerCount = parseInt(e.target.value);
        document.getElementById('layer-count-value').textContent = state.layerCount;
        reset();
    });

    document.getElementById('input-value').addEventListener('input', (e) => {
        state.inputValue = parseFloat(e.target.value);
        document.getElementById('input-value-display').textContent = state.inputValue.toFixed(1);
        reset();
    });

    document.getElementById('forward-btn').addEventListener('click', runForwardPass);
    document.getElementById('backward-btn').addEventListener('click', runBackwardPass);
    document.getElementById('reset-btn').addEventListener('click', reset);

    // Initial render
    reset();
}

// Resize canvases
function resizeCanvases() {
    const scrollContainer = document.getElementById('scroll-container');
    const width = scrollContainer ? (scrollContainer.clientWidth - 40) : networkCanvas.clientWidth;

    // Calculate required height based on layer count
    // Base height 600 or enough to fit all layers with spacing
    // Also include padding
    const minHeight = 600;
    const requiredHeight = (state.layerCount * LAYER_SPACING) + 200; // 200 for padding/margins
    const height = Math.max(minHeight, requiredHeight);

    networkCanvas.width = width;
    networkCanvas.height = height; // Set explicit height attribute
    networkCanvas.style.height = `${height}px`; // Update CSS height

    gradientChartCanvas.width = width;
}

// Reset visualization
function reset() {
    state.layerValues = [];
    state.layerGradients = [];
    resizeCanvases(); // Update canvas size based on new layer count
    drawNetwork();
    drawGradientChart();

    // Trigger zoom update in parent window if available
    window.dispatchEvent(new CustomEvent('resize-content'));
}

// Run forward pass
function runForwardPass() {
    if (state.isAnimating) return;
    state.isAnimating = true;

    state.layerValues = [state.inputValue];
    const activationFn = getActivation(state.activation);

    // Compute forward pass values
    for (let i = 1; i <= state.layerCount; i++) {
        // Simple linear transformation: multiply by 0.8 (simulating weights)
        const linearOutput = state.layerValues[i - 1] * 0.8;
        const activatedOutput = activationFn(linearOutput);
        state.layerValues.push(activatedOutput);
    }

    // Animate forward pass
    animateForwardPass(() => {
        state.isAnimating = false;
    });
}

// Run backward pass
function runBackwardPass() {
    if (state.isAnimating) return;
    if (state.layerValues.length === 0) {
        alert('Please run forward pass first');
        return;
    }

    state.isAnimating = true;

    // Compute gradients
    state.layerGradients = new Array(state.layerCount + 1).fill(0);
    state.layerGradients[state.layerCount] = 1.0; // Start with gradient of 1 at output

    const derivativeFn = getActivationDerivative(state.activation);

    for (let i = state.layerCount - 1; i >= 0; i--) {
        // Gradient flows backward: multiply by derivative and weight
        const linearInput = state.layerValues[i] * 0.8;
        const activationGrad = derivativeFn(linearInput);
        state.layerGradients[i] = state.layerGradients[i + 1] * activationGrad * 0.8;
    }

    // Animate backward pass
    animateBackwardPass(() => {
        state.isAnimating = false;
        drawGradientChart();
    });
}

// Animate forward pass
function animateForwardPass(callback) {
    const duration = 0.3;
    let currentLayer = 0;

    function animateLayer() {
        if (currentLayer >= state.layerCount) {
            callback();
            return;
        }

        drawNetwork(currentLayer);

        // Animate data packet
        const startY = getLayerY(currentLayer);
        const endY = getLayerY(currentLayer + 1);
        const centerX = networkCanvas.width / 2;

        const packet = { y: startY };
        gsap.to(packet, {
            y: endY,
            duration: duration,
            ease: 'power2.inOut',
            onUpdate: () => {
                drawNetwork(currentLayer);
                drawDataPacket(centerX, packet.y, '#2563eb');
            },
            onComplete: () => {
                currentLayer++;
                animateLayer();
            }
        });
    }

    animateLayer();
}

// Animate backward pass
function animateBackwardPass(callback) {
    const duration = 0.3;
    let currentLayer = state.layerCount;

    function animateLayer() {
        if (currentLayer <= 0) {
            callback();
            return;
        }

        drawNetwork(null, currentLayer);

        // Animate gradient packet
        const startY = getLayerY(currentLayer);
        const endY = getLayerY(currentLayer - 1);
        const centerX = networkCanvas.width / 2;

        const gradMagnitude = Math.abs(state.layerGradients[currentLayer]);
        const color = getGradientColor(gradMagnitude);

        const packet = { y: startY };
        gsap.to(packet, {
            y: endY,
            duration: duration,
            ease: 'power2.inOut',
            onUpdate: () => {
                drawNetwork(null, currentLayer);
                drawDataPacket(centerX, packet.y, color);
            },
            onComplete: () => {
                currentLayer--;
                animateLayer();
            }
        });
    }

    animateLayer();
}

// Draw network
function drawNetwork(highlightForward = null, highlightBackward = null) {
    const width = networkCanvas.width;
    const height = networkCanvas.height;

    networkCtx.clearRect(0, 0, width, height);

    const centerX = width / 2;

    // Draw connections
    for (let i = 0; i < state.layerCount; i++) {
        const y1 = getLayerY(i);
        const y2 = getLayerY(i + 1);

        networkCtx.strokeStyle = '#e0e0e0';
        networkCtx.lineWidth = 2;
        networkCtx.beginPath();
        networkCtx.moveTo(centerX, y1);
        networkCtx.lineTo(centerX, y2);
        networkCtx.stroke();
    }

    // Draw neurons
    for (let i = 0; i <= state.layerCount; i++) {
        const y = getLayerY(i);

        // Determine color based on gradient if backward pass has run
        let color = '#f8f9fa';
        let strokeColor = '#2563eb';

        if (state.layerGradients.length > 0 && i < state.layerGradients.length) {
            const gradMagnitude = Math.abs(state.layerGradients[i]);
            color = getGradientColor(gradMagnitude);
            strokeColor = color;
        }

        // Highlight if animating
        if (highlightForward === i || highlightBackward === i) {
            strokeColor = '#2563eb';
            networkCtx.lineWidth = 4;
        } else {
            networkCtx.lineWidth = 2;
        }

        // Draw neuron
        networkCtx.fillStyle = color;
        networkCtx.strokeStyle = strokeColor;
        networkCtx.beginPath();
        networkCtx.arc(centerX, y, NEURON_RADIUS, 0, 2 * Math.PI);
        networkCtx.fill();
        networkCtx.stroke();

        // Draw value if computed
        if (i < state.layerValues.length) {
            networkCtx.fillStyle = '#1a1a1a';
            networkCtx.font = '12px Inter';
            networkCtx.textAlign = 'center';
            networkCtx.textBaseline = 'middle';
            networkCtx.fillText(state.layerValues[i].toFixed(2), centerX, y);
        }

        // Draw layer label
        networkCtx.fillStyle = '#666';
        networkCtx.font = '14px Inter';
        networkCtx.textAlign = 'left';
        const label = i === 0 ? 'Input' : i === state.layerCount ? 'Output' : `Layer ${i}`;
        networkCtx.fillText(label, centerX + NEURON_RADIUS + 10, y);

        // Draw gradient magnitude if available
        if (state.layerGradients.length > 0 && i < state.layerGradients.length) {
            networkCtx.fillStyle = '#666';
            networkCtx.font = '11px Inter';
            networkCtx.textAlign = 'right';
            networkCtx.fillText(`∇: ${state.layerGradients[i].toExponential(2)}`,
                centerX - NEURON_RADIUS - 10, y);
        }
    }
}

// Draw data packet
function drawDataPacket(x, y, color) {
    networkCtx.fillStyle = color;
    networkCtx.beginPath();
    networkCtx.arc(x, y, 8, 0, 2 * Math.PI);
    networkCtx.fill();

    networkCtx.strokeStyle = '#fff';
    networkCtx.lineWidth = 2;
    networkCtx.stroke();
}

// Get Y position for layer
function getLayerY(layerIndex) {
    const height = networkCanvas.height;
    const totalHeight = state.layerCount * LAYER_SPACING;
    const startY = (height - totalHeight) / 2;
    return startY + layerIndex * LAYER_SPACING;
}

// Get gradient color based on magnitude
function getGradientColor(magnitude) {
    if (magnitude >= 0.1) {
        return '#10b981'; // Green - healthy
    } else if (magnitude >= 0.01) {
        return '#f59e0b'; // Yellow - weakening
    } else {
        return '#ef4444'; // Red - vanishing
    }
}

// Draw gradient chart
function drawGradientChart() {
    const width = gradientChartCanvas.width;
    const height = gradientChartCanvas.height;

    gradientChartCtx.clearRect(0, 0, width, height);

    if (state.layerGradients.length === 0) {
        gradientChartCtx.fillStyle = '#666';
        gradientChartCtx.font = '14px Inter';
        gradientChartCtx.textAlign = 'center';
        gradientChartCtx.fillText('Run backward pass to see gradient magnitudes', width / 2, height / 2);
        return;
    }

    const padding = 50;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Find max gradient for scaling
    const maxGrad = Math.max(...state.layerGradients.map(Math.abs));

    // Draw axes
    gradientChartCtx.strokeStyle = '#666';
    gradientChartCtx.lineWidth = 2;
    gradientChartCtx.beginPath();
    gradientChartCtx.moveTo(padding, padding);
    gradientChartCtx.lineTo(padding, height - padding);
    gradientChartCtx.lineTo(width - padding, height - padding);
    gradientChartCtx.stroke();

    // Draw bars
    const barWidth = plotWidth / (state.layerGradients.length + 1);

    for (let i = 0; i < state.layerGradients.length; i++) {
        const magnitude = Math.abs(state.layerGradients[i]);
        const barHeight = (magnitude / maxGrad) * plotHeight;
        const x = padding + (i + 0.5) * barWidth;
        const y = height - padding - barHeight;

        const color = getGradientColor(magnitude);

        gradientChartCtx.fillStyle = color;
        gradientChartCtx.fillRect(x - barWidth * 0.4, y, barWidth * 0.8, barHeight);

        // Draw layer label
        gradientChartCtx.fillStyle = '#666';
        gradientChartCtx.font = '11px Inter';
        gradientChartCtx.textAlign = 'center';
        const label = i === 0 ? 'In' : i === state.layerGradients.length - 1 ? 'Out' : `L${i}`;
        gradientChartCtx.fillText(label, x, height - padding + 15);
    }

    // Y-axis label
    gradientChartCtx.fillStyle = '#666';
    gradientChartCtx.font = '12px Inter';
    gradientChartCtx.textAlign = 'right';
    gradientChartCtx.fillText('Gradient Magnitude', padding - 5, padding - 10);
}

// Initialize on load
window.addEventListener('load', init);
window.addEventListener('resize', () => {
    resizeCanvases();
    drawNetwork();
    drawGradientChart();
});
