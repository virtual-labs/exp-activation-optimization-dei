// Shared utility functions for all demos

// Canvas utilities
const CanvasUtils = {
    // Clear canvas
    clear(ctx, width, height) {
        ctx.clearRect(0, 0, width, height);
    },

    // Draw axes with numbers
    drawAxes(ctx, width, height, xRange, yRange, options = {}) {
        const {
            xLabel = 'x',
            yLabel = 'y',
            gridLines = true,
            axisColor = '#666',
            gridColor = '#e0e0e0',
            fontSize = 12
        } = options;

        const padding = 50;
        const plotWidth = width - 2 * padding;
        const plotHeight = height - 2 * padding;

        ctx.save();

        // Draw grid lines
        if (gridLines) {
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;

            // Vertical grid lines
            for (let i = 0; i <= 10; i++) {
                const x = padding + (plotWidth * i) / 10;
                ctx.beginPath();
                ctx.moveTo(x, padding);
                ctx.lineTo(x, height - padding);
                ctx.stroke();
            }

            // Horizontal grid lines
            for (let i = 0; i <= 10; i++) {
                const y = padding + (plotHeight * i) / 10;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
                ctx.stroke();
            }
        }

        // Draw axes lines
        ctx.strokeStyle = axisColor;
        ctx.lineWidth = 2;

        // X-axis (find y=0 position)
        let yZero = this.mapValue(0, yRange[0], yRange[1], height - padding, padding);
        // Clamp yZero to plot area
        yZero = Math.max(padding, Math.min(height - padding, yZero));

        ctx.beginPath();
        ctx.moveTo(padding, yZero);
        ctx.lineTo(width - padding, yZero);
        ctx.stroke();

        // Y-axis (find x=0 position)
        let xZero = this.mapValue(0, xRange[0], xRange[1], padding, width - padding);
        // Clamp xZero to plot area
        xZero = Math.max(padding, Math.min(width - padding, xZero));

        ctx.beginPath();
        ctx.moveTo(xZero, padding);
        ctx.lineTo(xZero, height - padding);
        ctx.stroke();

        // Draw numbers
        ctx.fillStyle = axisColor;
        ctx.font = `${fontSize}px Inter`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // X-axis numbers
        const xStep = (xRange[1] - xRange[0]) / 5; // 5 steps
        for (let i = 0; i <= 5; i++) {
            const val = xRange[0] + i * xStep;
            const x = padding + (plotWidth * i) / 5;
            // Don't draw if too close to y-axis (clutter) unless it is near 0
            if (Math.abs(x - xZero) > 20 || Math.abs(val) < 0.001) {
                ctx.fillText(Number(val.toFixed(1)).toString(), x, height - padding + 5);
            }
        }

        // Y-axis numbers
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const yStep = (yRange[1] - yRange[0]) / 5;
        for (let i = 0; i <= 5; i++) {
            const val = yRange[0] + i * yStep;
            const y = height - padding - (plotHeight * i) / 5;
            // Don't draw if too close to x-axis
            if (Math.abs(y - yZero) > 15 || Math.abs(val) < 0.001) {
                ctx.fillText(Number(val.toFixed(1)).toString(), padding - 8, y);
            }
        }

        // Labels
        ctx.textAlign = 'center';
        ctx.fillText(xLabel, width - padding + 20, yZero);
        ctx.save();
        ctx.textAlign = 'center';
        ctx.fillText(yLabel, xZero, padding - 20);
        ctx.restore();

        ctx.restore();
    },

    // Plot a function
    plotFunction(ctx, width, height, xValues, yValues, xRange, yRange, options = {}) {
        const {
            color = '#2563eb',
            lineWidth = 2,
            fill = false,
            fillColor = 'rgba(37, 99, 235, 0.1)'
        } = options;

        const padding = 50;

        ctx.save();

        // Clip to plot area
        ctx.beginPath();
        ctx.rect(padding, padding, width - 2 * padding, height - 2 * padding);
        ctx.clip();

        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        ctx.beginPath();

        let started = false;
        for (let i = 0; i < xValues.length; i++) {
            const x = this.mapValue(xValues[i], xRange[0], xRange[1], padding, width - padding);
            const y = this.mapValue(yValues[i], yRange[0], yRange[1], height - padding, padding);

            // Skip if completely wildly out of bounds (optimization)
            if (Math.abs(y) > height * 2) {
                started = false;
                continue;
            }

            if (!started) {
                ctx.moveTo(x, y);
                started = true;
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Fill under curve if requested
        if (fill) {
            const yZero = this.mapValue(0, yRange[0], yRange[1], height - padding, padding);
            // Clamp yZero for fill
            const clampedYZero = Math.max(padding, Math.min(height - padding, yZero));

            ctx.lineTo(this.mapValue(xValues[xValues.length - 1], xRange[0], xRange[1], padding, width - padding), clampedYZero);
            ctx.lineTo(this.mapValue(xValues[0], xRange[0], xRange[1], padding, width - padding), clampedYZero);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
        }

        ctx.restore();
    },

    // Map value from one range to another
    mapValue(value, inMin, inMax, outMin, outMax) {
        return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
    },

    // Draw a point
    drawPoint(ctx, x, y, options = {}) {
        const { radius = 4, color = '#2563eb', stroke = true } = options;

        ctx.save();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();

        if (stroke) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        ctx.restore();
    },

    // Draw text
    drawText(ctx, text, x, y, options = {}) {
        const {
            font = '12px Inter',
            color = '#666',
            align = 'center',
            baseline = 'middle'
        } = options;

        ctx.save();
        ctx.font = font;
        ctx.fillStyle = color;
        ctx.textAlign = align;
        ctx.textBaseline = baseline;
        ctx.fillText(text, x, y);
        ctx.restore();
    }
};

// Color utilities
const ColorUtils = {
    // Interpolate between two colors
    interpolate(color1, color2, factor) {
        const c1 = this.hexToRgb(color1);
        const c2 = this.hexToRgb(color2);

        const r = Math.round(c1.r + factor * (c2.r - c1.r));
        const g = Math.round(c1.g + factor * (c2.g - c1.g));
        const b = Math.round(c1.b + factor * (c2.b - c1.b));

        return `rgb(${r}, ${g}, ${b})`;
    },

    // Convert hex to RGB
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    },

    // Get gradient color based on value (0-1)
    getGradientColor(value, colorStops = null) {
        if (!colorStops) {
            // Default: red -> yellow -> green
            colorStops = [
                { stop: 0, color: '#ef4444' },    // red
                { stop: 0.5, color: '#f59e0b' },  // yellow
                { stop: 1, color: '#10b981' }     // green
            ];
        }

        // Clamp value
        value = Math.max(0, Math.min(1, value));

        // Find the two color stops to interpolate between
        for (let i = 0; i < colorStops.length - 1; i++) {
            if (value >= colorStops[i].stop && value <= colorStops[i + 1].stop) {
                const localValue = (value - colorStops[i].stop) / (colorStops[i + 1].stop - colorStops[i].stop);
                return this.interpolate(colorStops[i].color, colorStops[i + 1].color, localValue);
            }
        }

        return colorStops[colorStops.length - 1].color;
    }
};

// Animation utilities
const AnimationUtils = {
    // Request animation frame with fallback
    requestFrame(callback) {
        return window.requestAnimationFrame(callback);
    },

    // Cancel animation frame
    cancelFrame(id) {
        window.cancelAnimationFrame(id);
    },

    // Easing functions
    easing: {
        linear: t => t,
        easeInQuad: t => t * t,
        easeOutQuad: t => t * (2 - t),
        easeInOutQuad: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
        easeInCubic: t => t * t * t,
        easeOutCubic: t => (--t) * t * t + 1,
        easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
    }
};

// Data normalization utilities
const DataUtils = {
    // Normalize array to [0, 1]
    normalize(arr) {
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        const range = max - min;

        if (range === 0) return arr.map(() => 0.5);
        return arr.map(val => (val - min) / range);
    },

    // Standardize array (mean=0, std=1)
    standardize(arr) {
        const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length;
        const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
        const std = Math.sqrt(variance);

        if (std === 0) return arr.map(() => 0);
        return arr.map(val => (val - mean) / std);
    },

    // Get min and max
    getRange(arr) {
        return {
            min: Math.min(...arr),
            max: Math.max(...arr)
        };
    }
};

// Format utilities
const FormatUtils = {
    // Format number to fixed decimal places
    toFixed(num, decimals = 2) {
        return Number(num.toFixed(decimals));
    },

    // Format as percentage
    toPercent(num, decimals = 1) {
        return `${(num * 100).toFixed(decimals)}%`;
    },

    // Format scientific notation
    toScientific(num, decimals = 2) {
        return num.toExponential(decimals);
    }
};
