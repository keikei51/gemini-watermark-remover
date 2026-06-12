import { createAlphaGradientMask } from './alphaGradientMask.js';

const ALLENK_FDNCNN_MODEL = Object.freeze({
    name: 'FDnCNN Color FP16',
    upstream: 'allenk/GeminiWatermarkTool',
    license: 'MIT',
    runtime: 'NCNN',
    inputBlob: 0,
    outputBlob: 20,
    inputLayout: '[R, G, B, sigma] CHW float32',
    outputLayout: '[R, G, B] CHW float32',
    defaultSigma: 25,
    defaultStrength: 0.85,
    defaultPadding: 16,
    maxSigma: 150,
    maxStrength: 3
});

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function createGaussianKernel(sigma, radius = Math.ceil(sigma * 3)) {
    const safeSigma = Math.max(0.01, sigma);
    const safeRadius = Math.max(1, Math.round(radius));
    const kernel = new Float32Array(safeRadius * 2 + 1);
    let sum = 0;

    for (let i = -safeRadius; i <= safeRadius; i++) {
        const value = Math.exp(-(i * i) / (2 * safeSigma * safeSigma));
        kernel[i + safeRadius] = value;
        sum += value;
    }

    for (let i = 0; i < kernel.length; i++) {
        kernel[i] /= sum;
    }

    return { kernel, radius: safeRadius };
}

function gaussianBlurFloatMap(source, width, height, sigma, radius = Math.ceil(sigma * 3)) {
    if (!source || width <= 0 || height <= 0 || !Number.isFinite(sigma) || sigma <= 0) {
        return new Float32Array(source || 0);
    }

    const { kernel, radius: r } = createGaussianKernel(sigma, radius);
    const temp = new Float32Array(source.length);
    const output = new Float32Array(source.length);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sum = 0;
            for (let dx = -r; dx <= r; dx++) {
                const xx = clamp(x + dx, 0, width - 1);
                sum += source[y * width + xx] * kernel[dx + r];
            }
            temp[y * width + x] = sum;
        }
    }

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sum = 0;
            for (let dy = -r; dy <= r; dy++) {
                const yy = clamp(y + dy, 0, height - 1);
                sum += temp[yy * width + x] * kernel[dy + r];
            }
            output[y * width + x] = sum;
        }
    }

    return output;
}

function inferSquareSize(alphaMap) {
    const size = Math.round(Math.sqrt(alphaMap?.length || 0));
    return size > 0 && size * size === alphaMap.length ? size : 0;
}

function resizeSquareAlphaMapArea(sourceAlpha, sourceSize, targetWidth, targetHeight = targetWidth) {
    if (!sourceAlpha || sourceSize <= 0 || targetWidth <= 0 || targetHeight <= 0) {
        return new Float32Array(0);
    }
    if (sourceSize === targetWidth && sourceSize === targetHeight) {
        return new Float32Array(sourceAlpha);
    }

    const output = new Float32Array(targetWidth * targetHeight);
    const scaleX = sourceSize / targetWidth;
    const scaleY = sourceSize / targetHeight;

    for (let y = 0; y < targetHeight; y++) {
        const yStart = y * scaleY;
        const yEnd = (y + 1) * scaleY;
        const y0 = Math.floor(yStart);
        const y1 = Math.ceil(yEnd);

        for (let x = 0; x < targetWidth; x++) {
            const xStart = x * scaleX;
            const xEnd = (x + 1) * scaleX;
            const x0 = Math.floor(xStart);
            const x1 = Math.ceil(xEnd);

            let sum = 0;
            let areaSum = 0;
            for (let sy = y0; sy < y1; sy++) {
                if (sy < 0 || sy >= sourceSize) continue;
                const wy = Math.max(0, Math.min(yEnd, sy + 1) - Math.max(yStart, sy));
                for (let sx = x0; sx < x1; sx++) {
                    if (sx < 0 || sx >= sourceSize) continue;
                    const wx = Math.max(0, Math.min(xEnd, sx + 1) - Math.max(xStart, sx));
                    const area = wx * wy;
                    sum += sourceAlpha[sy * sourceSize + sx] * area;
                    areaSum += area;
                }
            }

            output[y * targetWidth + x] = areaSum > 0 ? sum / areaSum : 0;
        }
    }

    return output;
}

function normalizeAllenkFdncnnOptions(options = {}) {
    const sigma = Number.isFinite(options.sigma)
        ? clamp(options.sigma, 0, ALLENK_FDNCNN_MODEL.maxSigma)
        : ALLENK_FDNCNN_MODEL.defaultSigma;
    const strength = Number.isFinite(options.strength)
        ? clamp(options.strength, 0, ALLENK_FDNCNN_MODEL.maxStrength)
        : ALLENK_FDNCNN_MODEL.defaultStrength;
    const padding = Number.isFinite(options.padding)
        ? Math.max(0, Math.round(options.padding))
        : ALLENK_FDNCNN_MODEL.defaultPadding;

    return { sigma, strength, padding };
}

function createAllenkGradientMask({
    alphaMap,
    width,
    height = width,
    strength = ALLENK_FDNCNN_MODEL.defaultStrength
} = {}) {
    const sourceSize = inferSquareSize(alphaMap);
    const resizedAlphaMap = sourceSize > 0
        ? resizeSquareAlphaMapArea(alphaMap, sourceSize, width, height)
        : alphaMap;

    return createAlphaGradientMask({
        alphaMap: resizedAlphaMap,
        width,
        height,
        strength,
        gamma: 0.5,
        dilateRadius: 2,
        blurSigma: 2
    });
}

function calculateAllenkPaddedRoi({ imageWidth, imageHeight, region, padding = ALLENK_FDNCNN_MODEL.defaultPadding } = {}) {
    if (!region || imageWidth <= 0 || imageHeight <= 0 || region.width <= 0 || region.height <= 0) {
        return null;
    }

    const safePadding = Math.max(0, Math.round(padding));
    const x = clamp(Math.round(region.x - safePadding), 0, imageWidth);
    const y = clamp(Math.round(region.y - safePadding), 0, imageHeight);
    const right = clamp(Math.round(region.x + region.width + safePadding), 0, imageWidth);
    const bottom = clamp(Math.round(region.y + region.height + safePadding), 0, imageHeight);
    const width = right - x;
    const height = bottom - y;

    if (width < 4 || height < 4) return null;

    return {
        x,
        y,
        width,
        height,
        inner: {
            x: clamp(Math.round(region.x - x), 0, width),
            y: clamp(Math.round(region.y - y), 0, height),
            width: clamp(Math.round(region.width), 0, width),
            height: clamp(Math.round(region.height), 0, height)
        }
    };
}

function embedAllenkRoiWeights({ roiWeights, roiWidth, roiHeight, paddedRoi, blurSigma = 1 } = {}) {
    if (!roiWeights || !paddedRoi?.inner || roiWidth <= 0 || roiHeight <= 0) {
        return new Float32Array(0);
    }

    const weights = new Float32Array(paddedRoi.width * paddedRoi.height);
    const inner = paddedRoi.inner;

    for (let y = 0; y < roiHeight; y++) {
        const py = inner.y + y;
        if (py < 0 || py >= paddedRoi.height) continue;

        for (let x = 0; x < roiWidth; x++) {
            const px = inner.x + x;
            if (px < 0 || px >= paddedRoi.width) continue;
            weights[py * paddedRoi.width + px] = clamp(roiWeights[y * roiWidth + x] || 0, 0, 1);
        }
    }

    return gaussianBlurFloatMap(weights, paddedRoi.width, paddedRoi.height, blurSigma);
}

function buildAllenkFdncnnInput({ imageData, sigma = ALLENK_FDNCNN_MODEL.defaultSigma } = {}) {
    if (!imageData?.data || imageData.width <= 0 || imageData.height <= 0) {
        return new Float32Array(0);
    }

    const { width, height, data } = imageData;
    const pixelCount = width * height;
    const input = new Float32Array(pixelCount * 4);
    const sigmaNorm = clamp(sigma, 0, ALLENK_FDNCNN_MODEL.maxSigma) / 255;
    const stride = data.length >= pixelCount * 4 ? 4 : 3;

    for (let i = 0; i < pixelCount; i++) {
        const src = i * stride;
        input[i] = (data[src] || 0) / 255;
        input[pixelCount + i] = (data[src + 1] || 0) / 255;
        input[pixelCount * 2 + i] = (data[src + 2] || 0) / 255;
        input[pixelCount * 3 + i] = sigmaNorm;
    }

    return input;
}

function convertAllenkFdncnnOutputToRgba({ output, width, height, alpha = 255 } = {}) {
    if (!output || width <= 0 || height <= 0) {
        return new Uint8ClampedArray(0);
    }

    const pixelCount = width * height;
    const rgba = new Uint8ClampedArray(pixelCount * 4);

    for (let i = 0; i < pixelCount; i++) {
        rgba[i * 4] = Math.round(clamp(output[i] || 0, 0, 1) * 255);
        rgba[i * 4 + 1] = Math.round(clamp(output[pixelCount + i] || 0, 0, 1) * 255);
        rgba[i * 4 + 2] = Math.round(clamp(output[pixelCount * 2 + i] || 0, 0, 1) * 255);
        rgba[i * 4 + 3] = alpha;
    }

    return rgba;
}

function getLocalMeanRgb(data, width, height, x, y, channel) {
    let sum = 0;
    let count = 0;
    for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            sum += data[(yy * width + xx) * 4 + channel];
            count++;
        }
    }
    return count > 0 ? sum / count : data[(y * width + x) * 4 + channel];
}

function blendAllenkDenoisedRoi({
    originalData,
    denoisedData,
    weights,
    width = 0,
    height = 0,
    preserveHighpassStrength = 0
} = {}) {
    if (!originalData || !denoisedData || !weights || originalData.length !== denoisedData.length) {
        return new Uint8ClampedArray(originalData || 0);
    }

    const output = new Uint8ClampedArray(originalData);
    const pixelCount = Math.min(weights.length, Math.floor(originalData.length / 4));
    const canPreserveHighpass = (
        Number.isFinite(preserveHighpassStrength) &&
        preserveHighpassStrength > 0 &&
        Number.isFinite(width) &&
        Number.isFinite(height) &&
        width > 0 &&
        height > 0 &&
        width * height <= pixelCount
    );

    for (let pixel = 0; pixel < pixelCount; pixel++) {
        const weight = clamp(weights[pixel] || 0, 0, 1);
        if (weight <= 0) continue;

        const idx = pixel * 4;
        const x = canPreserveHighpass ? pixel % width : 0;
        const y = canPreserveHighpass ? Math.floor(pixel / width) : 0;
        const highpassGain = canPreserveHighpass
            ? Math.min(0.28, weight * preserveHighpassStrength)
            : 0;
        for (let c = 0; c < 3; c++) {
            const blended = (
                originalData[idx + c] * (1 - weight) + denoisedData[idx + c] * weight
            );
            const highpass = highpassGain > 0
                ? clamp(originalData[idx + c] - getLocalMeanRgb(originalData, width, height, x, y, c), -14, 14)
                : 0;
            output[idx + c] = Math.round(clamp(blended + highpass * highpassGain, 0, 255));
        }
    }

    return output;
}

export {
    ALLENK_FDNCNN_MODEL,
    blendAllenkDenoisedRoi,
    buildAllenkFdncnnInput,
    calculateAllenkPaddedRoi,
    convertAllenkFdncnnOutputToRgba,
    createAllenkGradientMask,
    embedAllenkRoiWeights,
    normalizeAllenkFdncnnOptions
};
