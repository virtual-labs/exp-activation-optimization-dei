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

    reset();
}

// Resize canvases — canvas column is fixed at 480px
function resizeCanvases() {
    const width = 480;
    const minHeight = 600;
    const requiredHeight = (state.layerCount * LAYER_SPACING) + 200;
    const height = Math.max(minHeight, requiredHeight);

    networkCanvas.width = width;
    networkCanvas.height = height;
    networkCanvas.style.height = `${height}px`;

    gradientChartCanvas.width = width;
}

// Reset visualization
function reset() {
    state.layerValues = [];
    state.layerGradients = [];
    resizeCanvases();
    drawNetwork();
    drawGradientChart();

    const ids = ['math-forward-steps', 'math-backward-steps'];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.style.display = 'none'; el.innerHTML = ''; }
    });
    const lossEl = document.getElementById('math-loss-box');
    const mathPanel = document.getElementById('math-panel');
    const fwdSum = document.getElementById('fwd-summary-line');
    const bwdSum = document.getElementById('bwd-summary-line');
    if (lossEl)    lossEl.style.display = 'none';
    if (mathPanel) mathPanel.style.display = 'none';
    if (fwdSum)  { fwdSum.style.display = 'none'; fwdSum.innerHTML = ''; }
    if (bwdSum)  { bwdSum.style.display = 'none'; bwdSum.innerHTML = ''; }

    window.dispatchEvent(new CustomEvent('resize-content'));
}

// ── Forward pass ──────────────────────────────────────────────────────────────

function runForwardPass() {
    if (state.isAnimating) return;
    state.isAnimating = true;

    // Pre-compute all values
    state.layerValues = [state.inputValue];
    const activationFn = getActivation(state.activation);
    for (let i = 1; i <= state.layerCount; i++) {
        state.layerValues.push(activationFn(state.layerValues[i - 1] * 0.8));
    }

    // Initialise the math panel (shows before animation)
    initForwardMathPanel();

    animateForwardPass(() => {
        state.isAnimating = false;
        finalizeForwardMath();
    });
}

function initForwardMathPanel() {
    const mathPanel = document.getElementById('math-panel');
    const el = document.getElementById('math-forward-steps');
    if (!el) return;
    if (mathPanel) mathPanel.style.display = 'block';
    el.innerHTML =
        `<div class="math-steps-block" id="fwd-steps-inner">
            <div class="math-steps-header fwd">Forward Pass — Step by Step</div>
            <div class="math-step fwd">
                <div class="math-step-label">Input (Layer 0)</div>
                a[0] = <b>${fmt(state.inputValue)}</b>
            </div>
         </div>`;
    el.style.display = 'block';
    window.dispatchEvent(new CustomEvent('resize-content'));
}

// Called once per layer as its animation completes
function appendForwardStep(i) {
    const block = document.getElementById('fwd-steps-inner');
    if (!block || i >= state.layerValues.length) return;

    const aIn  = state.layerValues[i - 1];
    const z    = aIn * 0.8;
    const aOut = state.layerValues[i];
    const layerLabel = i === state.layerCount ? 'Output' : `Layer ${i}`;

    const step = document.createElement('div');
    step.className = 'math-step fwd';
    step.style.cssText = 'opacity:0; transform:translateY(-4px);';
    step.innerHTML =
        `<div class="math-step-label">${layerLabel} (i = ${i})</div>` +
        `z[${i}] = a[${i-1}] &times; w = ${fmt(aIn)} &times; 0.8 = ${fmt(z)}<br>` +
        `a[${i}] = f(${fmt(z)}) = <b>${fmt(aOut)}</b>`;
    block.appendChild(step);

    requestAnimationFrame(() => {
        step.style.transition = 'opacity 0.25s, transform 0.25s';
        step.style.opacity = '1';
        step.style.transform = 'translateY(0)';
    });

    step.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    window.dispatchEvent(new CustomEvent('resize-content'));
}

function finalizeForwardMath() {
    const n = state.layerCount;
    const aOut = state.layerValues[n];
    const el = document.getElementById('fwd-summary-line');
    if (el) {
        el.innerHTML =
            `Input <b>${fmt(state.layerValues[0])}</b> &rarr; ${n} layers &rarr; ` +
            `Output <b>${fmt(aOut)}</b><br>Each layer: z&nbsp;=&nbsp;a&times;0.8,&nbsp;a&nbsp;=&nbsp;f(z)`;
        el.style.display = 'block';
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    window.dispatchEvent(new CustomEvent('resize-content'));
}

// ── Backward pass ─────────────────────────────────────────────────────────────

function runBackwardPass() {
    if (state.isAnimating) return;
    if (state.layerValues.length === 0) {
        alert('Please run forward pass first');
        return;
    }
    state.isAnimating = true;

    // Pre-compute all gradients
    state.layerGradients = new Array(state.layerCount + 1).fill(0);
    const aOut = state.layerValues[state.layerCount];
    state.layerGradients[state.layerCount] = aOut - 1.0; // dL/da_out from L = ½(a-1)²

    const derivFn = getActivationDerivative(state.activation);
    for (let i = state.layerCount - 1; i >= 0; i--) {
        const z = state.layerValues[i] * 0.8;
        state.layerGradients[i] = state.layerGradients[i + 1] * derivFn(z) * 0.8;
    }

    // Initialise the backward math panel (shows loss before animation)
    initBackwardMathPanel();

    animateBackwardPass(() => {
        state.isAnimating = false;
        drawGradientChart();
        finalizeBackwardMath();
    });
}

function initBackwardMathPanel() {
    const mathPanel = document.getElementById('math-panel');
    const lossEl    = document.getElementById('math-loss-box');
    const el        = document.getElementById('math-backward-steps');
    if (!el) return;
    if (mathPanel) mathPanel.style.display = 'block';
    if (lossEl)    lossEl.style.display = 'block';

    const n     = state.layerCount;
    const aOut  = state.layerValues[n];
    const loss  = 0.5 * Math.pow(aOut - 1, 2);
    const dLoss = aOut - 1;

    el.innerHTML =
        `<div class="math-steps-block" id="bwd-steps-inner">
            <div class="math-steps-header bwd">Backward Pass — Chain Rule</div>
            <div class="math-note" style="margin-bottom:0.4rem;">
                <b>∇ (left of each neuron)</b> = &part;L/&part;a[i] &mdash; how much total loss
                changes per unit change in that neuron's activation.
            </div>
            <div class="math-step bwd">
                <div class="math-step-label">Loss at output &nbsp;(target&nbsp;=&nbsp;1.0)</div>
                L = &frac12;(${fmt(aOut)} &minus; 1)&sup2; = <b>${fmt(loss)}</b><br>
                &part;L/&part;a[${n}] = a[${n}] &minus; 1 = ${fmt(aOut)} &minus; 1 = <b>${fmt(dLoss)}</b>
            </div>
         </div>`;
    el.style.display = 'block';

    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    window.dispatchEvent(new CustomEvent('resize-content'));
}

// Called once per layer as its backward animation completes
// fromLayer: gradient packet just moved from fromLayer → fromLayer-1
function appendBackwardStep(fromLayer) {
    const block = document.getElementById('bwd-steps-inner');
    if (!block) return;

    const i      = fromLayer;
    const aIn    = state.layerValues[i - 1];
    const z      = aIn * 0.8;
    const gOut   = state.layerGradients[i];
    const gIn    = state.layerGradients[i - 1];
    const toLabel = (i - 1) === 0 ? 'Input' : `Layer ${i-1}`;

    const step = document.createElement('div');
    step.className = 'math-step bwd';
    step.style.cssText = 'opacity:0; transform:translateY(-4px);';
    step.innerHTML =
        `<div class="math-step-label">Layer ${i} &rarr; ${toLabel}</div>` +
        `${derivHtml(state.activation, z)}<br>` +
        `&part;L/&part;a[${i-1}] = &part;L/&part;a[${i}] &times; f&prime; &times; w<br>` +
        `&nbsp;&nbsp;= ${fmt(gOut)} &times; f&prime;(${fmt(z)}) &times; 0.8 = <b>${fmt(gIn)}</b>`;
    block.appendChild(step);

    requestAnimationFrame(() => {
        step.style.transition = 'opacity 0.25s, transform 0.25s';
        step.style.opacity = '1';
        step.style.transform = 'translateY(0)';
    });

    step.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    window.dispatchEvent(new CustomEvent('resize-content'));
}

function finalizeBackwardMath() {
    const block = document.getElementById('bwd-steps-inner');
    if (block) {
        const note = document.createElement('div');
        note.className = 'math-note';
        note.textContent = 'w = 0.8 (fixed weight in this illustration). ∇ on each neuron = ∂L/∂a[i].';
        block.appendChild(note);
    }

    const n      = state.layerCount;
    const aOut   = state.layerValues[n];
    const dLoss  = aOut - 1;
    const gInput = state.layerGradients[0];
    const mag    = Math.abs(gInput);
    const status = mag >= 0.1 ? 'healthy' : mag >= 0.01 ? 'weakening' : 'vanishing';

    const el = document.getElementById('bwd-summary-line');
    if (el) {
        el.innerHTML =
            `Seed &part;L/&part;a[out] = <b>${fmt(dLoss)}</b><br>` +
            `Input gradient = <b>${fmt(gInput)}</b> &mdash; ${status}`;
        el.style.display = 'block';
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    window.dispatchEvent(new CustomEvent('resize-content'));
}

// ── Animations ────────────────────────────────────────────────────────────────

function animateForwardPass(callback) {
    const duration = 0.35;
    let currentLayer = 0;

    function animateLayer() {
        if (currentLayer >= state.layerCount) { callback(); return; }

        drawNetwork(currentLayer);

        const startY  = getLayerY(currentLayer);
        const endY    = getLayerY(currentLayer + 1);
        const centerX = networkCanvas.width / 2;
        const packet  = { y: startY };

        gsap.to(packet, {
            y: endY,
            duration,
            ease: 'power2.inOut',
            onUpdate: () => {
                drawNetwork(currentLayer);
                drawDataPacket(centerX, packet.y, '#2563eb');
            },
            onComplete: () => {
                appendForwardStep(currentLayer + 1); // show calc for destination layer
                currentLayer++;
                animateLayer();
            }
        });
    }

    animateLayer();
}

function animateBackwardPass(callback) {
    const duration = 0.35;
    let currentLayer = state.layerCount;

    function animateLayer() {
        if (currentLayer <= 0) { callback(); return; }

        drawNetwork(null, currentLayer);

        const startY  = getLayerY(currentLayer);
        const endY    = getLayerY(currentLayer - 1);
        const centerX = networkCanvas.width / 2;
        const color   = getGradientColor(Math.abs(state.layerGradients[currentLayer]));
        const packet  = { y: startY };

        gsap.to(packet, {
            y: endY,
            duration,
            ease: 'power2.inOut',
            onUpdate: () => {
                drawNetwork(null, currentLayer);
                drawDataPacket(centerX, packet.y, color);
            },
            onComplete: () => {
                appendBackwardStep(currentLayer); // show chain-rule from currentLayer → currentLayer-1
                currentLayer--;
                animateLayer();
            }
        });
    }

    animateLayer();
}

// ── Canvas drawing ────────────────────────────────────────────────────────────

function drawNetwork(highlightForward = null, highlightBackward = null) {
    const width   = networkCanvas.width;
    const height  = networkCanvas.height;
    const centerX = width / 2;

    networkCtx.clearRect(0, 0, width, height);

    // Connections
    for (let i = 0; i < state.layerCount; i++) {
        networkCtx.strokeStyle = '#e0e0e0';
        networkCtx.lineWidth = 2;
        networkCtx.beginPath();
        networkCtx.moveTo(centerX, getLayerY(i));
        networkCtx.lineTo(centerX, getLayerY(i + 1));
        networkCtx.stroke();
    }

    // Neurons
    for (let i = 0; i <= state.layerCount; i++) {
        const y = getLayerY(i);

        let color = '#f8f9fa';
        let strokeColor = '#2563eb';

        if (state.layerGradients.length > 0 && i < state.layerGradients.length) {
            color = getGradientColor(Math.abs(state.layerGradients[i]));
            strokeColor = color;
        }

        if (highlightForward === i || highlightBackward === i) {
            strokeColor = '#2563eb';
            networkCtx.lineWidth = 4;
        } else {
            networkCtx.lineWidth = 2;
        }

        networkCtx.fillStyle = color;
        networkCtx.strokeStyle = strokeColor;
        networkCtx.beginPath();
        networkCtx.arc(centerX, y, NEURON_RADIUS, 0, 2 * Math.PI);
        networkCtx.fill();
        networkCtx.stroke();

        // Activation value inside neuron
        if (i < state.layerValues.length) {
            networkCtx.fillStyle = '#1a1a1a';
            networkCtx.font = '12px Inter';
            networkCtx.textAlign = 'center';
            networkCtx.textBaseline = 'middle';
            networkCtx.fillText(state.layerValues[i].toFixed(2), centerX, y);
        }

        // Layer label (right side)
        networkCtx.fillStyle = '#666';
        networkCtx.font = '14px Inter';
        networkCtx.textAlign = 'left';
        networkCtx.textBaseline = 'middle';
        const label = i === 0 ? 'Input' : i === state.layerCount ? 'Output' : `Layer ${i}`;
        networkCtx.fillText(label, centerX + NEURON_RADIUS + 10, y);

        // Gradient ∂L/∂a[i] (left side) — appears after backward pass
        if (state.layerGradients.length > 0 && i < state.layerGradients.length) {
            networkCtx.fillStyle = '#444';
            networkCtx.font = '11px Inter';
            networkCtx.textAlign = 'right';
            networkCtx.fillText(
                `\u2207: ${state.layerGradients[i].toExponential(2)}`,
                centerX - NEURON_RADIUS - 10, y
            );
        }
    }
}

function drawDataPacket(x, y, color) {
    networkCtx.fillStyle = color;
    networkCtx.beginPath();
    networkCtx.arc(x, y, 8, 0, 2 * Math.PI);
    networkCtx.fill();
    networkCtx.strokeStyle = '#fff';
    networkCtx.lineWidth = 2;
    networkCtx.stroke();
}

function getLayerY(layerIndex) {
    const height = networkCanvas.height;
    const totalHeight = state.layerCount * LAYER_SPACING;
    const startY = (height - totalHeight) / 2;
    return startY + layerIndex * LAYER_SPACING;
}

function getGradientColor(magnitude) {
    if (magnitude >= 0.1)  return '#10b981'; // green  — healthy
    if (magnitude >= 0.01) return '#f59e0b'; // yellow — weakening
    return '#ef4444';                         // red    — vanishing
}

function drawGradientChart() {
    const width  = gradientChartCanvas.width;
    const height = gradientChartCanvas.height;
    gradientChartCtx.clearRect(0, 0, width, height);

    if (state.layerGradients.length === 0) {
        gradientChartCtx.fillStyle = '#666';
        gradientChartCtx.font = '14px Inter';
        gradientChartCtx.textAlign = 'center';
        gradientChartCtx.fillText('Run backward pass to see gradient magnitudes', width / 2, height / 2);
        return;
    }

    const padding   = 50;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;
    const maxGrad   = Math.max(...state.layerGradients.map(Math.abs));

    // Axes
    gradientChartCtx.strokeStyle = '#666';
    gradientChartCtx.lineWidth = 2;
    gradientChartCtx.beginPath();
    gradientChartCtx.moveTo(padding, padding);
    gradientChartCtx.lineTo(padding, height - padding);
    gradientChartCtx.lineTo(width - padding, height - padding);
    gradientChartCtx.stroke();

    const barWidth = plotWidth / (state.layerGradients.length + 1);
    for (let i = 0; i < state.layerGradients.length; i++) {
        const magnitude = Math.abs(state.layerGradients[i]);
        const barHeight = (magnitude / maxGrad) * plotHeight;
        const x = padding + (i + 0.5) * barWidth;
        const y = height - padding - barHeight;

        gradientChartCtx.fillStyle = getGradientColor(magnitude);
        gradientChartCtx.fillRect(x - barWidth * 0.4, y, barWidth * 0.8, barHeight);

        gradientChartCtx.fillStyle = '#666';
        gradientChartCtx.font = '11px Inter';
        gradientChartCtx.textAlign = 'center';
        const lbl = i === 0 ? 'In' : i === state.layerGradients.length - 1 ? 'Out' : `L${i}`;
        gradientChartCtx.fillText(lbl, x, height - padding + 15);
    }

    gradientChartCtx.fillStyle = '#666';
    gradientChartCtx.font = '12px Inter';
    gradientChartCtx.textAlign = 'right';
    gradientChartCtx.fillText('Gradient Magnitude', padding - 5, padding - 10);
}

// ── Math helpers ──────────────────────────────────────────────────────────────

function fmt(v) {
    const abs = Math.abs(v);
    if (abs === 0) return '0.0000';
    if (abs < 0.0001 || abs >= 10000) return v.toExponential(4);
    return v.toFixed(4);
}

function derivHtml(name, z) {
    const v = getActivationDerivative(name)(z);
    switch (name) {
        case 'sigmoid': {
            const s = 1 / (1 + Math.exp(-z));
            return `f&prime;(z) = &sigma;(${fmt(z)})&middot;(1&minus;&sigma;(${fmt(z)})) = ${fmt(s)}&middot;${fmt(1-s)} = <b>${fmt(v)}</b>`;
        }
        case 'tanh': {
            const t = Math.tanh(z);
            return `f&prime;(z) = 1&minus;tanh&sup2;(${fmt(z)}) = 1&minus;${fmt(t*t)} = <b>${fmt(v)}</b>`;
        }
        case 'relu':
            return v === 1
                ? `f&prime;(z) = <b>1</b> &nbsp;(z=${fmt(z)} &gt; 0)`
                : `f&prime;(z) = <b>0</b> &nbsp;(z=${fmt(z)} &le; 0)`;
        case 'leakyrelu':
            return v === 1
                ? `f&prime;(z) = <b>1</b> &nbsp;(z=${fmt(z)} &gt; 0)`
                : `f&prime;(z) = <b>0.01</b> (z=${fmt(z)} &le; 0)`;
        default:
            return `f&prime;(z) = <b>${fmt(v)}</b>`;
    }
}

// ── Init ──────────────────────────────────────────────────────────────────────

window.addEventListener('load', init);
window.addEventListener('resize', () => {
    resizeCanvases();
    drawNetwork();
    drawGradientChart();
});
