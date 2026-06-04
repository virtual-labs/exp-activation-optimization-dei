// Mathematical functions for activation functions and their derivatives

const ActivationFunctions = {
  // Linear
  linear: (x) => x,
  linearDerivative: (x) => 1,
  
  // Sigmoid
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  sigmoidDerivative: (x) => {
    const s = 1 / (1 + Math.exp(-x));
    return s * (1 - s);
  },
  
  // Tanh
  tanh: (x) => Math.tanh(x),
  tanhDerivative: (x) => {
    const t = Math.tanh(x);
    return 1 - t * t;
  },
  
  // ReLU
  relu: (x) => Math.max(0, x),
  reluDerivative: (x) => x > 0 ? 1 : 0,
  
  // Leaky ReLU
  leakyRelu: (x, alpha = 0.01) => x > 0 ? x : alpha * x,
  leakyReluDerivative: (x, alpha = 0.01) => x > 0 ? 1 : alpha,
  
  // GELU (Gaussian Error Linear Unit)
  gelu: (x) => {
    const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    return x * cdf;
  },
  geluDerivative: (x) => {
    const sqrt2OverPi = Math.sqrt(2 / Math.PI);
    const coeff = 0.044715;
    const inner = sqrt2OverPi * (x + coeff * Math.pow(x, 3));
    const tanh_inner = Math.tanh(inner);
    const sech2 = 1 - tanh_inner * tanh_inner;
    const cdf = 0.5 * (1 + tanh_inner);
    const pdf_term = 0.5 * x * sech2 * sqrt2OverPi * (1 + 3 * coeff * x * x);
    return cdf + pdf_term;
  },
  
  // Swish (SiLU)
  swish: (x) => x / (1 + Math.exp(-x)),
  swishDerivative: (x) => {
    const sig = 1 / (1 + Math.exp(-x));
    return sig + x * sig * (1 - sig);
  }
};

// Get activation function by name
function getActivation(name, alpha = 0.01) {
  const lowerName = name.toLowerCase().replace(/\s+/g, '');
  
  switch(lowerName) {
    case 'linear': return (x) => ActivationFunctions.linear(x);
    case 'sigmoid': return (x) => ActivationFunctions.sigmoid(x);
    case 'tanh': return (x) => ActivationFunctions.tanh(x);
    case 'relu': return (x) => ActivationFunctions.relu(x);
    case 'leakyrelu': return (x) => ActivationFunctions.leakyRelu(x, alpha);
    case 'gelu': return (x) => ActivationFunctions.gelu(x);
    case 'swish': return (x) => ActivationFunctions.swish(x);
    default: return (x) => x;
  }
}

// Get derivative function by name
function getActivationDerivative(name, alpha = 0.01) {
  const lowerName = name.toLowerCase().replace(/\s+/g, '');
  
  switch(lowerName) {
    case 'linear': return (x) => ActivationFunctions.linearDerivative(x);
    case 'sigmoid': return (x) => ActivationFunctions.sigmoidDerivative(x);
    case 'tanh': return (x) => ActivationFunctions.tanhDerivative(x);
    case 'relu': return (x) => ActivationFunctions.reluDerivative(x);
    case 'leakyrelu': return (x) => ActivationFunctions.leakyReluDerivative(x, alpha);
    case 'gelu': return (x) => ActivationFunctions.geluDerivative(x);
    case 'swish': return (x) => ActivationFunctions.swishDerivative(x);
    default: return (x) => 1;
  }
}

// Generate array of x values
function linspace(start, end, num) {
  const step = (end - start) / (num - 1);
  return Array.from({ length: num }, (_, i) => start + step * i);
}

// Apply function to array
function applyFunction(fn, xValues) {
  return xValues.map(x => fn(x));
}
