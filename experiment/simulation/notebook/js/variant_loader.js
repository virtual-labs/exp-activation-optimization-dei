// Dynamic Variant Loader for Activation Functions Experiment
// Adapts the ViT experiment logic to the new data structure

let currentVariant = null;
let completedSteps = new Set();
// Global debug flag
const DEBUG = true;

function log(msg, ...args) {
    if (DEBUG) console.log(`[VariantLoader] ${msg}`, ...args);
}

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    log("DOM Loaded");

    // Check if data is loaded
    if (typeof EXPERIMENTS_DATA === 'undefined') {
        document.getElementById('cellsContainer').innerHTML =
            `<div class="loading-message error" style="color:red;">
                Error: EXPERIMENTS_DATA is not defined.<br>
                Check if experiments_data.js is loaded correctly.
             </div>`;
        return;
    }

    log("Data loaded, count:", EXPERIMENTS_DATA.length);

    initializeSelectors();
    setupEventListeners();

    // Initial Load - force load the first one if defaults fail
    if (!loadVariantBySelectors()) {
        log("Default selection failed, loading first experiment");
        const firstExp = EXPERIMENTS_DATA[0];
        if (firstExp) {
            setSelectValues(firstExp.config);
            loadVariant(firstExp);
        }
    }
});

function initializeSelectors() {
    // Extract unique values
    const activations = new Set();
    const optimizers = new Set();
    const lrs = new Set();

    EXPERIMENTS_DATA.forEach(exp => {
        if (exp.config.ACTIVATION) activations.add(exp.config.ACTIVATION);
        if (exp.config.OPTIMIZER) optimizers.add(exp.config.OPTIMIZER);
        // Handle LR as string for consistent Set behavior, convert back for sorting
        if (exp.config.LEARNING_RATE !== undefined) lrs.add(String(exp.config.LEARNING_RATE));
    });

    log("Unique values:", {
        act: activations.size,
        opt: optimizers.size,
        lr: lrs.size
    });

    // Sort values
    const sortedActivations = Array.from(activations).sort();
    const sortedOptimizers = Array.from(optimizers).sort();
    // Sort LRs numerically
    const sortedLrs = Array.from(lrs).sort((a, b) => parseFloat(a) - parseFloat(b));

    // Populate Dropdowns
    populateSelect('activationSelect', sortedActivations);
    populateSelect('optimizerSelect', sortedOptimizers);
    populateSelect('lrSelect', sortedLrs);

    // Set defaults (try to find 'relu', 'adam', '0.01' if possible)
    setSelectDefault('activationSelect', 'relu');
    setSelectDefault('optimizerSelect', 'adam');

    // Find 0.01 in LRs (handle string/number differences)
    const defaultLr = sortedLrs.find(lr => Math.abs(parseFloat(lr) - 0.01) < 0.0001) || sortedLrs[0];
    setSelectDefault('lrSelect', defaultLr);
}

function populateSelect(id, values) {
    const select = document.getElementById(id);
    if (!select) return;
    select.innerHTML = '';
    values.forEach(val => {
        const option = document.createElement('option');
        option.value = val;
        option.textContent = val;
        select.appendChild(option);
    });
}

function setSelectDefault(id, preferredVal) {
    const select = document.getElementById(id);
    if (!select) return;
    // Try exact match
    for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].value === String(preferredVal)) {
            select.selectedIndex = i;
            return;
        }
    }
    // If not found, select first
    if (select.options.length > 0) select.selectedIndex = 0;
}

function setSelectValues(config) {
    setSelectDefault('activationSelect', config.ACTIVATION);
    setSelectDefault('optimizerSelect', config.OPTIMIZER);
    setSelectDefault('lrSelect', config.LEARNING_RATE);
}

function findExperiment(activation, optimizer, lr) {
    return EXPERIMENTS_DATA.find(exp =>
        String(exp.config.ACTIVATION) === String(activation) &&
        String(exp.config.OPTIMIZER) === String(optimizer) &&
        // Robust number comparison
        Math.abs(parseFloat(exp.config.LEARNING_RATE) - parseFloat(lr)) < 0.000001
    );
}

function setupEventListeners() {
    document.getElementById('activationSelect').addEventListener('change', handleVariantChange);
    document.getElementById('optimizerSelect').addEventListener('change', handleVariantChange);
    document.getElementById('lrSelect').addEventListener('change', handleVariantChange);

    document.getElementById('resetBtn').addEventListener('click', resetExperiment);
}

function handleVariantChange() {
    resetExperiment();
    loadVariantBySelectors();
}

function loadVariantBySelectors() {
    const activation = document.getElementById('activationSelect').value;
    const optimizer = document.getElementById('optimizerSelect').value;
    const lr = document.getElementById('lrSelect').value;

    log("Looking for:", activation, optimizer, lr);

    const experiment = findExperiment(activation, optimizer, lr);

    if (!experiment) {
        console.warn("Experiment combination not found");
        return false;
    }

    loadVariant(experiment);
    return true;
}

function loadVariant(experiment) {
    currentVariant = experiment;
    log("Loading experiment:", experiment.config.EXPERIMENT_ID);

    // Render Sidebar
    renderSidebar(experiment);

    // Render Cells
    renderCells(experiment);
}

function inferCellTitle(cell, index) {
    if (cell.type === 'markdown') {
        const match = cell.source.match(/^#+ (.+)$/m);
        if (match) {
            return match[1].replace(/\*\*/g, '').replace(/\*/g, '');
        }
        return "Note";
    }

    const code = cell.source;

    // 1. Explicit Comment Titles (e.g. "# Plot 2: Learning Rate Schedule")
    const plotMatch = code.match(/# Plot \d+:\s*(.+)/);
    if (plotMatch) return plotMatch[1].trim();

    const stepMatch = code.match(/# Step:\s*(.+)/);
    if (stepMatch) return stepMatch[1].trim();

    // 2. High-level Logic Checking
    if (code.includes('ACTIVATION =') && code.includes('OPTIMIZER =')) return "Configuration";
    if (code.includes('import torch') && code.includes('import matplotlib')) return "Import Libraries";
    if (code.includes('transforms.Compose') || (code.includes('FashionMNIST') && !code.includes('def train'))) return "Load Dataset";
    if (code.includes('class FashionMLP') || (code.includes('nn.Linear') && code.includes('forward'))) return "Define Model";
    if (code.includes('get_optimizer') || code.includes('get_scheduler')) return "Optimizer Setup";
    if (code.includes('def train_epoch') || (code.includes('def validate') && code.includes('correct'))) return "Training Logic";
    if ((code.includes('train_loader') && code.includes('epochs')) || code.includes('for epoch in range')) return "Main Training Loop";

    // 3. Visualization Specifics
    if (code.includes('plt.plot') || code.includes('plt.figure')) {
        if (code.includes('Training Dynamics') || code.includes('Loss')) return "Plot: Training Dynamics";
        if (code.includes('Learning Rate Schedule')) return "Plot: Learning Rate Schedule";
        if (code.includes('Dead Neuron') || code.includes('pct_zero')) return "Plot: Dead Neuron Analysis";
        if (code.includes('Gradient Norm') || code.includes('grad_norm')) return "Plot: Gradient Analysis";
        return "Visualization";
    }

    // 4. Results & Logs
    if (code.includes('Compute summary statistics') || (code.includes('final_loss') && code.includes('best_val_acc'))) return "Calculate Statistics";
    if (code.includes('Save complete training log') || code.includes('json.dump(training_log')) return "Save Experiment Logs";

    // 5. Fallbacks based on content
    if (code.includes('print(') && code.includes('Saved')) return "Save Results";

    // Fallback for unnamed steps
    return `Step ${index}`;
}

function parseMarkdown(text) {
    if (!text) return '';
    let html = text
        // Encode html to prevent XSS (basic)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")

        // Headers
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')

        // Horizontal Rule
        .replace(/^\-\-\-$/gim, '<hr>')

        // Blockquotes
        .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>')

        // Bold
        .replace(/\*\*(.*?)\*\*/gim, '<b>$1</b>')

        // Italic
        .replace(/\*(.*?)\*/gim, '<i>$1</i>')

        // Code Blocks
        .replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>')

        // Inline Code
        .replace(/`([^`]+)`/gim, '<code>$1</code>')

        // Lists (Unordered)
        .replace(/^\- (.*$)/gim, '<ul><li>$1</li></ul>')
        .replace(/^\* (.*$)/gim, '<ul><li>$1</li></ul>')

        // Lists (Ordered)
        .replace(/^\d+\. (.*$)/gim, '<ol><li>$1</li></ol>')

        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank">$1</a>')

        // Newlines to breaks
        .replace(/\n$/gim, '')
        .replace(/\n/gim, '<br>');

    // Fix adjacent lists
    html = html.replace(/<\/ul><br><ul>/gim, '')
        .replace(/<\/ol><br><ol>/gim, '')
        .replace(/<\/ul><ul>/gim, '')
        .replace(/<\/ol><ol>/gim, '');

    return html;
}

function renderSidebar(experiment) {
    const stepList = document.getElementById('stepList');
    stepList.innerHTML = '';

    let allCells = [];
    let stepCount = 0;

    experiment.cells.forEach((cell, index) => {
        // Skip hidden/system cells
        if (cell.source.includes('PAPERMILL PARAMETERS')) return;
        if (cell.source === undefined) return;

        allCells.push(cell);

        if (cell.type === 'code' || !cell.type) {
            stepCount++;
            const currentStep = stepCount;
            const title = inferCellTitle(cell, currentStep);

            const stepItem = document.createElement('div');
            stepItem.className = 'step-item';
            stepItem.dataset.step = currentStep;

            stepItem.innerHTML = `
                <div class="step-indicator">
                    <span class="step-number">${currentStep}</span>
                </div>
                <div class="step-content">
                    <span class="step-title">${title}</span>
                </div>
            `;

            stepItem.addEventListener('click', function () {
                scrollToCell(currentStep);
            });

            stepList.appendChild(stepItem);
        }
    });

    // Store ALL cells for the renderer, but we will index only code cells for execution
    currentVariant.allCells = allCells;
    currentVariant.totalSteps = stepCount;

    if (stepCount === 0) {
        stepList.innerHTML = '<div style="padding:10px;">No executable steps found.</div>';
    } else {
        const firstStep = stepList.querySelector('.step-item[data-step="1"]');
        if (firstStep) firstStep.classList.add('active');
    }
}

function renderCells(experiment) {
    const container = document.getElementById('cellsContainer');
    container.innerHTML = '';

    if (!currentVariant.allCells || currentVariant.allCells.length === 0) {
        container.innerHTML = '<div class="loading-message">No cells to display.</div>';
        return;
    }

    let codeStepCounter = 0;

    currentVariant.allCells.forEach((cell) => {
        if (cell.type === 'code' || !cell.type) {
            codeStepCounter++;
            const cellDiv = createCellElement(cell, codeStepCounter);
            container.appendChild(cellDiv);
        } else if (cell.type === 'markdown') {
            const cellDiv = createCellElement(cell, null);
            container.appendChild(cellDiv);
        }
    });
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
}

function createCellElement(cell, stepNum) {
    const cellDiv = document.createElement('div');
    // Common base class
    cellDiv.className = 'notebook-cell';

    // Handle Markdown
    if (cell.type === 'markdown') {
        // No step number for markdown
        cellDiv.className += ' markdown-cell';
        cellDiv.innerHTML = `<div class="markdown-content">${parseMarkdown(cell.source)}</div>`;
        return cellDiv;
    }

    // Handle Code
    cellDiv.dataset.step = stepNum;

    // Title
    const title = inferCellTitle(cell, stepNum);

    // Header
    const header = document.createElement('div');
    header.className = 'cell-header';
    header.innerHTML = `
        <span class="cell-label">${title}</span>
        <button class="run-btn" onclick="runCell(${stepNum})">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
            </svg>
            Run
        </button>
    `;
    cellDiv.appendChild(header);

    // Code
    const codeDiv = document.createElement('div');
    codeDiv.className = 'cell-code';
    const pre = document.createElement('pre');
    pre.textContent = cell.source || "";
    codeDiv.appendChild(pre);
    cellDiv.appendChild(codeDiv);

    // Output
    const outputDiv = document.createElement('div');
    outputDiv.className = 'cell-output hidden';

    const outputLabel = document.createElement('div');
    outputLabel.className = 'output-label';
    outputLabel.textContent = 'Output:';
    outputDiv.appendChild(outputLabel);

    const outputContent = document.createElement('div');
    outputContent.className = 'output-content';

    let hasOutput = false;

    if (cell.outputs && cell.outputs.length > 0) {
        cell.outputs.forEach(output => {
            // Text output
            if (output.type === 'text' || output.name === 'stdout') {
                const text = output.content || output.text; // Support both formats
                if (text && (Array.isArray(text) ? text.length > 0 : text.trim().length > 0)) {
                    const textDiv = document.createElement('div');
                    textDiv.className = 'text-output';
                    const tPre = document.createElement('pre');
                    tPre.textContent = Array.isArray(text) ? text.join('') : text;
                    textDiv.appendChild(tPre);
                    outputContent.appendChild(textDiv);
                    hasOutput = true;
                }
            }
            // Image output
            else if (output.type === 'image' || (output.data && output.data['image/png'])) {
                let imgData = output.content;
                if (!imgData && output.data) imgData = output.data['image/png'];

                if (imgData) {
                    const imgDiv = document.createElement('div');
                    imgDiv.className = 'output-image';
                    const img = document.createElement('img');
                    // Standardize base64
                    img.src = imgData.startsWith('data:') ? imgData : `data:image/png;base64,${imgData}`;
                    imgDiv.appendChild(img);
                    outputContent.appendChild(imgDiv);
                    hasOutput = true;
                }
            }
        });
    }

    if (hasOutput) {
        outputDiv.appendChild(outputContent);
    } else {
        const noOutput = document.createElement('div');
        noOutput.className = 'output-content';
        noOutput.innerHTML = '<em style="color: #888;">No output</em>';
        outputDiv.appendChild(noOutput);
    }

    cellDiv.appendChild(outputDiv);
    return cellDiv;
}

// Execution Logic
function runCell(step) {
    return new Promise(resolve => {
        const cell = document.querySelector(`.notebook-cell[data-step="${step}"]`);
        if (!cell) { resolve(false); return; }

        if (step > 1 && !completedSteps.has(step - 1)) {
            // Sequential enforcement
            resolve(false);
            return;
        }

        if (completedSteps.has(step)) {
            resolve(true);
            return;
        }

        // Scroll Code into view
        // Changed to center to ensure better visibility
        cell.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Sync Sidebar
        const stepItem = document.querySelector(`.step-item[data-step="${step}"]`);
        if (stepItem) {
            stepItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
            stepItem.classList.add('running');
        }

        cell.classList.add('running');

        // Wait 1 second at code
        setTimeout(() => {
            cell.classList.remove('running');
            cell.classList.add('completed');
            if (stepItem) {
                stepItem.classList.remove('running');
                stepItem.classList.add('completed');
            }
            completedSteps.add(step);

            // Show Output Immediately
            const output = cell.querySelector('.cell-output');
            if (output) {
                output.classList.remove('hidden');

                // Scroll to output immediately
                // Using center block for output visibility
                output.scrollIntoView({ behavior: 'smooth', block: 'start' });

                // Wait 3 seconds at output before resolving
                setTimeout(() => {
                    if (currentVariant && completedSteps.size === currentVariant.totalSteps) {
                        showCompletionMessage();
                    }
                    resolve(true);
                }, 3000);
            } else {
                resolve(true);
            }

        }, 1000);
    });
}

function scrollToCell(step) {
    const cell = document.querySelector(`.notebook-cell[data-step="${step}"]`);
    // Added scroll-margin-top in CSS, so block: 'start' works better now
    if (cell) cell.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetExperiment() {
    completedSteps.clear();
    document.querySelectorAll('.notebook-cell').forEach(c => {
        c.classList.remove('running', 'completed');
        const out = c.querySelector('.cell-output');
        if (out) out.classList.add('hidden');
    });
    document.querySelectorAll('.step-item').forEach(s => {
        s.classList.remove('running', 'completed', 'active');
    });

    document.getElementById('completionMessage').classList.add('hidden');

    const first = document.querySelector('.step-item[data-step="1"]');
    if (first) {
        first.classList.add('active');
        scrollToCell(1);
    }
}

function showCompletionMessage() {
    const msg = document.getElementById('completionMessage');
    msg.classList.remove('hidden');
    msg.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

let ticking = false;
window.addEventListener('scroll', function () {
    if (!ticking) {
        window.requestAnimationFrame(() => {
            const cells = document.querySelectorAll('.notebook-cell');
            const scrollPos = window.scrollY + 100;

            cells.forEach(cell => {
                const top = cell.offsetTop;
                const bottom = top + cell.offsetHeight;
                if (scrollPos >= top && scrollPos < bottom) {
                    const step = cell.dataset.step;
                    document.querySelectorAll('.step-item').forEach(s => s.classList.remove('active'));
                    const activeStep = document.querySelector(`.step-item[data-step="${step}"]`);
                    if (activeStep) activeStep.classList.add('active');
                }
            });
            ticking = false;
        });
        ticking = true;
    }
});
