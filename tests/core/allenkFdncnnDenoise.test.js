import test from 'node:test';
import assert from 'node:assert/strict';

import {
    ALLENK_FDNCNN_MODEL,
    blendAllenkDenoisedRoi,
    buildAllenkFdncnnInput,
    calculateAllenkPaddedRoi,
    convertAllenkFdncnnOutputToRgba,
    createAllenkGradientMask,
    embedAllenkRoiWeights,
    normalizeAllenkFdncnnOptions
} from '../../src/core/allenkFdncnnDenoise.js';

test('ALLENK_FDNCNN_MODEL should document the imported allenk model contract', () => {
    assert.equal(ALLENK_FDNCNN_MODEL.upstream, 'allenk/GeminiWatermarkTool');
    assert.equal(ALLENK_FDNCNN_MODEL.license, 'MIT');
    assert.equal(ALLENK_FDNCNN_MODEL.inputBlob, 0);
    assert.equal(ALLENK_FDNCNN_MODEL.outputBlob, 20);
});

test('normalizeAllenkFdncnnOptions should keep allenk defaults and clamp unsafe values', () => {
    assert.deepEqual(normalizeAllenkFdncnnOptions(), {
        sigma: 25,
        strength: 0.85,
        padding: 16
    });

    assert.deepEqual(normalizeAllenkFdncnnOptions({ sigma: 500, strength: 9, padding: -4 }), {
        sigma: 150,
        strength: 3,
        padding: 0
    });
});

test('buildAllenkFdncnnInput should produce 4-channel CHW RGB plus sigma map', () => {
    const input = buildAllenkFdncnnInput({
        imageData: {
            width: 2,
            height: 1,
            data: new Uint8ClampedArray([
                255, 128, 0, 255,
                10, 20, 30, 255
            ])
        },
        sigma: 51
    });

    assert.equal(input.length, 8);
    assert.deepEqual([...input].map((value) => Number(value.toFixed(6))), [
        1, 0.039216,
        0.501961, 0.078431,
        0, 0.117647,
        0.2, 0.2
    ]);
});

test('convertAllenkFdncnnOutputToRgba should restore RGB planes into RGBA pixels', () => {
    const rgba = convertAllenkFdncnnOutputToRgba({
        width: 2,
        height: 1,
        output: new Float32Array([
            1, 0.1,
            0.5, 0.2,
            0, 0.3
        ]),
        alpha: 240
    });

    assert.deepEqual([...rgba], [
        255, 128, 0, 240,
        26, 51, 77, 240
    ]);
});

test('createAllenkGradientMask should emphasize alpha edges and honor strength', () => {
    const size = 25;
    const alphaMap = new Float32Array(size * size);

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dx = x - 12;
            const dy = y - 12;
            alphaMap[y * size + x] = Math.sqrt(dx * dx + dy * dy) <= 5 ? 0.65 : 0;
        }
    }

    const strong = createAllenkGradientMask({ alphaMap, width: size, height: size, strength: 1 });
    const weak = createAllenkGradientMask({ alphaMap, width: size, height: size, strength: 0.5 });

    const center = 12 * size + 12;
    const edge = 12 * size + 17;

    assert.ok(strong[edge] > strong[center], `edge=${strong[edge]}, center=${strong[center]}`);
    assert.ok(strong[center] < strong[edge] * 0.75, `center=${strong[center]}, edge=${strong[edge]}`);
    assert.ok(weak[edge] < strong[edge], `weak=${weak[edge]}, strong=${strong[edge]}`);
});

test('createAllenkGradientMask should resize square alpha maps before computing gradients', () => {
    const sourceSize = 16;
    const targetSize = 8;
    const alphaMap = new Float32Array(sourceSize * sourceSize);
    for (let y = 0; y < sourceSize; y++) {
        for (let x = 0; x < sourceSize; x++) {
            alphaMap[y * sourceSize + x] = x >= 8 ? 0.7 : 0;
        }
    }

    const mask = createAllenkGradientMask({
        alphaMap,
        width: targetSize,
        height: targetSize,
        strength: 1
    });

    assert.equal(mask.length, targetSize * targetSize);
    assert.ok(mask[3 * targetSize + 3] > mask[3 * targetSize + 0]);
});

test('calculateAllenkPaddedRoi should clamp padding and report inner region', () => {
    const padded = calculateAllenkPaddedRoi({
        imageWidth: 100,
        imageHeight: 80,
        region: { x: 88, y: 68, width: 10, height: 10 },
        padding: 16
    });

    assert.deepEqual(padded, {
        x: 72,
        y: 52,
        width: 28,
        height: 28,
        inner: {
            x: 16,
            y: 16,
            width: 10,
            height: 10
        }
    });
});

test('embedAllenkRoiWeights should place ROI weights into padded coordinates and blur boundary', () => {
    const padded = {
        x: 0,
        y: 0,
        width: 6,
        height: 6,
        inner: { x: 2, y: 2, width: 2, height: 2 }
    };
    const weights = embedAllenkRoiWeights({
        roiWeights: new Float32Array([1, 1, 1, 1]),
        roiWidth: 2,
        roiHeight: 2,
        paddedRoi: padded,
        blurSigma: 0.5
    });

    assert.equal(weights.length, 36);
    assert.ok(weights[2 * 6 + 2] > weights[0], `inside=${weights[14]}, outside=${weights[0]}`);
    assert.ok(weights[1 * 6 + 2] > 0, `blurred neighbor=${weights[8]}`);
});

test('blendAllenkDenoisedRoi should blend only weighted RGB pixels', () => {
    const original = new Uint8ClampedArray([
        10, 20, 30, 255,
        100, 120, 140, 255
    ]);
    const denoised = new Uint8ClampedArray([
        110, 120, 130, 255,
        200, 220, 240, 255
    ]);

    const output = blendAllenkDenoisedRoi({
        originalData: original,
        denoisedData: denoised,
        weights: new Float32Array([0.5, 0])
    });

    assert.deepEqual([...output], [
        60, 70, 80, 255,
        100, 120, 140, 255
    ]);
});

test('blendAllenkDenoisedRoi should optionally preserve small local highpass texture', () => {
    const original = new Uint8ClampedArray([
        40, 40, 40, 255,
        50, 50, 50, 255,
        60, 60, 60, 255,
        50, 50, 50, 255,
        80, 80, 80, 255,
        50, 50, 50, 255,
        40, 40, 40, 255,
        50, 50, 50, 255,
        60, 60, 60, 255
    ]);
    const denoised = new Uint8ClampedArray(original.length);
    for (let i = 0; i < denoised.length; i += 4) {
        denoised[i] = 50;
        denoised[i + 1] = 50;
        denoised[i + 2] = 50;
        denoised[i + 3] = 255;
    }

    const flat = blendAllenkDenoisedRoi({
        originalData: original,
        denoisedData: denoised,
        weights: new Float32Array(9).fill(1)
    });
    const preserved = blendAllenkDenoisedRoi({
        originalData: original,
        denoisedData: denoised,
        weights: new Float32Array(9).fill(1),
        width: 3,
        height: 3,
        preserveHighpassStrength: 0.32
    });

    assert.equal(flat[4 * 4], 50);
    assert.ok(preserved[4 * 4] > flat[4 * 4], `center=${preserved[4 * 4]}, flat=${flat[4 * 4]}`);
});
