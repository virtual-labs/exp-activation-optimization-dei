// Demo 1: Activation Function Shape Explorer

// State
let state = {
  activation: 'relu',
  xRange: 6,
  alpha: 0.01
};

// Canvas contexts
let functionCtx, derivativeCtx;
let functionCanvas, derivativeCanvas;

// Intuition content for each activation
const intuitionContent = {
  linear: {
    title: 'Linear Activation',
    description: `
      <h3>Try This</h3>
      <p>Notice that f'(x) is always 1 (horizontal green line). No matter what the input is, the gradient is constant. This means no non-linearity is learned.</p>
      
      <p style="margin-top: 1.5rem; border-top: 1px solid #e0e0e0; padding-top: 1rem;">The linear activation is simply <strong>f(x) = x</strong>. It passes the input unchanged.</p>
      
      <h3>Key Properties</h3>
      <ul>
        <li>Output range: (-∞, +∞)</li>
        <li>Derivative is constant: f'(x) = 1</li>
        <li>No non-linearity introduced</li>
      </ul>
      
      <h3>Why This Matters</h3>
      <p>A neural network with only linear activations is equivalent to a single linear transformation, no matter how many layers you stack. <strong>Non-linearity is essential for learning complex patterns.</strong></p>
    `
  },

  sigmoid: {
    title: 'Sigmoid Activation',
    description: `
      <h3>Try This</h3>
      <p>Increase the <strong>X-axis Range</strong> to 10. Observe how the derivative (green line) becomes almost zero for inputs > 4 or < -4. This is saturation.</p>
      
      <p style="margin-top: 1.5rem; border-top: 1px solid #e0e0e0; padding-top: 1rem;">The sigmoid function is <strong>f(x) = 1 / (1 + e<sup>-x</sup>)</strong>. It squashes inputs to the range (0, 1).</p>
      
      <h3>Key Properties</h3>
      <ul>
        <li>Output range: (0, 1)</li>
        <li>S-shaped curve</li>
        <li>Derivative peaks at x=0 with f'(0) ≈ 0.25</li>
        <li>Saturates for |x| > 3</li>
      </ul>
      
      <h3>The Vanishing Gradient Problem</h3>
      <p>Notice how the derivative (f'(x)) approaches zero for large positive or negative inputs. This is called <strong>saturation</strong>.</p>
      
      <div class="intuition-highlight">
        <strong>Critical Issue:</strong> When gradients are near zero, learning stops. In deep networks, these small gradients multiply together during backpropagation, causing gradients to vanish exponentially.
      </div>
    `
  },

  tanh: {
    title: 'Tanh Activation',
    description: `
      <h3>Try This</h3>
      <p>Compare the peak derivative value with Sigmoid. Tanh peaks at 1.0 (vs 0.25), which allows for stronger gradients during initial training.</p>
      
      <p style="margin-top: 1.5rem; border-top: 1px solid #e0e0e0; padding-top: 1rem;">The hyperbolic tangent is <strong>f(x) = tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)</strong>. It squashes inputs to (-1, 1).</p>
      
      <h3>Key Properties</h3>
      <ul>
        <li>Output range: (-1, 1)</li>
        <li>Zero-centered (unlike sigmoid)</li>
        <li>Derivative peaks at x=0 with f'(0) = 1</li>
        <li>Still saturates for |x| > 2</li>
      </ul>
      
      <h3>Improvement Over Sigmoid</h3>
      <p>Tanh is zero-centered, which helps with gradient flow. The derivative at zero is 1 (vs 0.25 for sigmoid), allowing stronger gradients initially.</p>
    `
  },

  relu: {
    title: 'ReLU (Rectified Linear Unit)',
    description: `
      <h3>Try This</h3>
      <p>Look at the negative side (x < 0). The derivative (green line) is exactly zero. This creates "dead" neurons that stop learning entirely if they get stuck here.</p>
      
      <p style="margin-top: 1.5rem; border-top: 1px solid #e0e0e0; padding-top: 1rem;">ReLU is defined as <strong>f(x) = max(0, x)</strong>. It's the most popular activation in modern deep learning.</p>
      
      <h3>Key Properties</h3>
      <ul>
        <li>Output range: [0, +∞)</li>
        <li>Derivative is 1 for x > 0, and 0 for x ≤ 0</li>
        <li>No saturation for positive values</li>
        <li>Computationally efficient</li>
      </ul>
      
      <h3>Why ReLU Works So Well</h3>
      <p>The constant derivative of 1 for positive inputs means <strong>no vanishing gradient problem</strong> during backpropagation. Gradients flow unchanged through active neurons.</p>
      
      <div class="intuition-highlight">
        <strong>The "Dying ReLU" Problem:</strong> If a neuron's output is always negative, its gradient is always 0, and it never updates. This neuron is "dead". Proper initialization and learning rates help prevent this.
      </div>
    `
  },

  leakyrelu: {
    title: 'Leaky ReLU',
    description: `
      <h3>Try This</h3>
      <p>Adjust the <strong>Alpha</strong> slider. Notice how the derivative on the negative side (green line) becomes non-zero. This small slope keeps the neuron "alive".</p>
      
      <p style="margin-top: 1.5rem; border-top: 1px solid #e0e0e0; padding-top: 1rem;">Leaky ReLU is <strong>f(x) = max(αx, x)</strong> where α is a small constant (typically 0.01).</p>
      
      <h3>Key Properties</h3>
      <ul>
        <li>Output range: (-∞, +∞)</li>
        <li>Derivative is 1 for x > 0, and α for x ≤ 0</li>
        <li>Small negative slope prevents dying neurons</li>
      </ul>
      
      <h3>Solving the Dying ReLU Problem</h3>
      <p>By allowing a small gradient (α) for negative inputs, Leaky ReLU ensures that neurons can recover even if they output negative values.</p>
    `
  },

  gelu: {
    title: 'GELU (Gaussian Error Linear Unit)',
    description: `
      <h3>Try This</h3>
      <p>Zoom in near x=0. Notice the smooth curve compared to ReLU's sharp corner. This smoothness helps optimization in very deep networks like Transformers.</p>
      
      <p style="margin-top: 1.5rem; border-top: 1px solid #e0e0e0; padding-top: 1rem;">GELU is a smooth approximation: <strong>f(x) ≈ x · Φ(x)</strong> where Φ is the cumulative distribution function of the standard normal distribution.</p>
      
      <h3>Key Properties</h3>
      <ul>
        <li>Output range: (-∞, +∞)</li>
        <li>Smooth, differentiable everywhere</li>
        <li>Non-monotonic (slight dip for negative x)</li>
        <li>Used in transformers (BERT, GPT)</li>
      </ul>
      
      <h3>Why GELU for Transformers?</h3>
      <p>GELU provides a smooth, probabilistic gating mechanism. Unlike ReLU's hard cutoff, GELU gradually weights inputs based on their magnitude.</p>
    `
  },

  swish: {
    title: 'Swish (SiLU)',
    description: `
      <h3>Try This</h3>
      <p>Observe the "dip" in the negative region. It allows a small amount of negative information to flow through, unlike ReLU which cuts it off completely.</p>
      
      <p style="margin-top: 1.5rem; border-top: 1px solid #e0e0e0; padding-top: 1rem;">Swish is defined as <strong>f(x) = x · sigmoid(x) = x / (1 + e<sup>-x</sup>)</strong>.</p>
      
      <h3>Key Properties</h3>
      <ul>
        <li>Output range: (-∞, +∞)</li>
        <li>Smooth and non-monotonic</li>
        <li>Self-gated (input modulates itself)</li>
        <li>Discovered through neural architecture search</li>
      </ul>
      
      <h3>Self-Gating Mechanism</h3>
      <p>Swish multiplies the input by its sigmoid, creating a smooth gating effect. For large positive x, it behaves like linear; for large negative x, it approaches zero.</p>
    `
  }
};

// Initialize
function init() {
  functionCanvas = document.getElementById('function-canvas');
  derivativeCanvas = document.getElementById('derivative-canvas');
  functionCtx = functionCanvas.getContext('2d');
  derivativeCtx = derivativeCanvas.getContext('2d');

  // Set canvas size
  resizeCanvases();

  // Check initial state for slider
  toggleAlphaSlider();

  // Event listeners
  document.getElementById('activation-select').addEventListener('change', (e) => {
    state.activation = e.target.value;
    toggleAlphaSlider();
    updateVisualization();
    updateIntuition();
  });

  document.getElementById('x-range').addEventListener('input', (e) => {
    state.xRange = parseFloat(e.target.value);
    document.getElementById('x-range-value').textContent = `-${state.xRange} to ${state.xRange}`;
    updateVisualization();
  });

  document.getElementById('alpha-slider').addEventListener('input', (e) => {
    state.alpha = parseFloat(e.target.value);
    document.getElementById('alpha-value').textContent = state.alpha.toFixed(2);
    if (state.activation === 'leakyrelu') {
      updateVisualization();
    }
  });

  document.getElementById('reset-btn').addEventListener('click', () => {
    state.xRange = 6;
    state.alpha = 0.01;
    document.getElementById('x-range').value = 6;
    document.getElementById('alpha-slider').value = 0.01;
    document.getElementById('x-range-value').textContent = '-6 to 6';
    document.getElementById('alpha-value').textContent = '0.01';
    updateVisualization();
  });

  // Initial render
  updateVisualization();
  updateIntuition();

  // ResizeObserver: redraw when container changes size (orientation change, panel resize)
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => {
      resizeCanvases();
      updateVisualization();
    });
    const vizPanel = document.querySelector('.visualization-panel');
    if (vizPanel) ro.observe(vizPanel);
  }
}

// Toggle alpha slider
function toggleAlphaSlider() {
  const alphaControl = document.getElementById('alpha-control-group');
  if (state.activation === 'leakyrelu') {
    alphaControl.style.display = 'flex';
  } else {
    alphaControl.style.display = 'none';
  }
}

// Resize canvases
function resizeCanvases() {
  const width = functionCanvas.clientWidth;
  functionCanvas.width = width;
  derivativeCanvas.width = width;
}

// Update visualization
function updateVisualization() {
  const xValues = linspace(-state.xRange, state.xRange, 500);

  // Get activation functions
  const activationFn = getActivation(state.activation, state.alpha);
  const derivativeFn = getActivationDerivative(state.activation, state.alpha);

  // Calculate y values
  const yValues = applyFunction(activationFn, xValues);
  const dyValues = applyFunction(derivativeFn, xValues);

  // Determine y ranges
  const yRange = getYRange(yValues);
  const dyRange = getYRange(dyValues);

  // Draw function
  drawFunction(functionCtx, functionCanvas, xValues, yValues, yRange);

  // Draw derivative
  drawDerivative(derivativeCtx, derivativeCanvas, xValues, dyValues, dyRange);
}

// Get appropriate y range
function getYRange(yValues) {
  const min = Math.min(...yValues);
  const max = Math.max(...yValues);
  const range = max - min;

  if (range === 0) return [min - 0.5, min + 0.5];

  const padding = range * 0.1;
  return [min - padding, max + padding];
}

// Draw function plot
function drawFunction(ctx, canvas, xValues, yValues, yRange) {
  const width = canvas.width;
  const height = canvas.height;

  CanvasUtils.clear(ctx, width, height);
  CanvasUtils.drawAxes(ctx, width, height, [-state.xRange, state.xRange], yRange, {
    xLabel: 'x',
    yLabel: 'f(x)'
  });

  CanvasUtils.plotFunction(ctx, width, height, xValues, yValues,
    [-state.xRange, state.xRange], yRange, {
    color: '#2563eb',
    lineWidth: 3
  });
}

// Draw derivative plot
function drawDerivative(ctx, canvas, xValues, dyValues, dyRange) {
  const width = canvas.width;
  const height = canvas.height;

  CanvasUtils.clear(ctx, width, height);
  CanvasUtils.drawAxes(ctx, width, height, [-state.xRange, state.xRange], dyRange, {
    xLabel: 'x',
    yLabel: "f'(x)"
  });

  // Highlight saturation regions (where derivative is near zero)
  highlightSaturation(ctx, width, height, xValues, dyValues, dyRange);

  CanvasUtils.plotFunction(ctx, width, height, xValues, dyValues,
    [-state.xRange, state.xRange], dyRange, {
    color: '#10b981',
    lineWidth: 3
  });
}

// Highlight saturation regions
function highlightSaturation(ctx, width, height, xValues, dyValues, dyRange) {
  const padding = 50;
  const threshold = 0.1;

  ctx.save();
  ctx.fillStyle = 'rgba(239, 68, 68, 0.1)';

  // Clip to content area just like plotFunction
  ctx.beginPath();
  ctx.rect(padding, padding, width - 2 * padding, height - 2 * padding);
  ctx.clip();

  for (let i = 0; i < xValues.length - 1; i++) {
    if (Math.abs(dyValues[i]) < threshold) {
      const x1 = CanvasUtils.mapValue(xValues[i], -state.xRange, state.xRange, padding, width - padding);
      const x2 = CanvasUtils.mapValue(xValues[i + 1], -state.xRange, state.xRange, padding, width - padding);

      // Draw full height of plot area
      ctx.fillRect(x1, padding, (x2 - x1) + 1, height - 2 * padding);
    }
  }

  ctx.restore();
}

// Update intuition panel
function updateIntuition() {
  const content = intuitionContent[state.activation];
  const html = `
    <h3>${content.title}</h3>
    ${content.description}
  `;

  document.getElementById('intuition-content').innerHTML = html;
}

// Initialize on load
window.addEventListener('load', init);
window.addEventListener('resize', () => {
  resizeCanvases();
  updateVisualization();
});
