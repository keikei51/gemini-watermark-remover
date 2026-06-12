import test from 'node:test';
import assert from 'node:assert/strict';

import {
    DEFAULT_DENOISE_BACKEND,
    DEFAULT_EDGE_DENOISE_STRENGTH,
    DEFAULT_HIGH_QUALITY_CLEANUP,
    DEFAULT_RESIDUAL_CLEANUP_STRENGTH,
    DEFAULT_TEXTURE_REPAIR,
    DEFAULT_TEXTURE_REPAIR_STRENGTH,
    VIDEO_CLEANUP_BACKENDS,
    VIDEO_DENOISE_BACKENDS,
    applyVideoResidualCleanup,
    applyVideoResidualCleanupAsync,
    borrowCleanHighpassTexture,
    buildEdgeBandDenoiseWeightMap,
    buildFootprintPolishWeightMap,
    buildGradientWeightMap,
    buildLumaStructureGuard,
    buildTextureRepairWeightMap,
    normalizeVideoCleanupOptions
} from '../../src/video/videoCleanupBackends.js';
import { getVideoAlphaMap } from '../../src/video/videoWatermarkDetector.js';

function createDiamondAlphaMap(width, height) {
    const alphaMap = new Float32Array(width * height);
    const cx = (width - 1) / 2;
    const cy = (height - 1) / 2;
    const radius = Math.min(width, height) / 2;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const distance = (Math.abs(x - cx) + Math.abs(y - cy)) / radius;
            alphaMap[y * width + x] = Math.max(0, 0.28 * (1 - distance));
        }
    }

    return alphaMap;
}

test('normalizeVideoCleanupOptions should keep conservative defaults', () => {
    const normalized = normalizeVideoCleanupOptions();

    assert.equal(normalized.residualCleanupStrength, DEFAULT_RESIDUAL_CLEANUP_STRENGTH);
    assert.equal(normalized.cleanupBackend, VIDEO_CLEANUP_BACKENDS.CANVAS_SOFT);
    assert.equal(normalized.highQualityCleanup, DEFAULT_HIGH_QUALITY_CLEANUP);
    assert.equal(normalized.denoiseBackend, DEFAULT_DENOISE_BACKEND);
    assert.equal(normalized.edgeDenoiseStrength, DEFAULT_EDGE_DENOISE_STRENGTH);
    assert.equal(normalized.textureRepair, DEFAULT_TEXTURE_REPAIR);
    assert.equal(normalized.textureRepairStrength, DEFAULT_TEXTURE_REPAIR_STRENGTH);
    assert.equal(VIDEO_CLEANUP_BACKENDS.CANVAS_SOFT, 'canvas-soft');
    assert.equal(VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE, 'allenk-fdncnn-browser-spike');
    assert.equal(VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_BAND_DENOISE, 'canvas-edge-band-denoise');
    assert.equal(VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_CORE_DENOISE, 'canvas-edge-core-denoise');
    assert.equal(VIDEO_DENOISE_BACKENDS.CANVAS_FOOTPRINT_POLISH, 'canvas-footprint-polish');
    assert.equal(VIDEO_DENOISE_BACKENDS.CANVAS_TEMPORAL_DELTA_STABILIZE, 'canvas-temporal-delta-stabilize');
    assert.equal(VIDEO_DENOISE_BACKENDS.CANVAS_TEMPORAL_MATCH_DELTA_STABILIZE, 'canvas-temporal-match-delta-stabilize');
    assert.equal(VIDEO_DENOISE_BACKENDS.CANVAS_TEMPORAL_STABILIZE, 'canvas-temporal-stabilize');
    assert.equal(VIDEO_DENOISE_BACKENDS.CANVAS_TEXTURE_REPAIR, 'canvas-texture-repair');
});

test('normalizeVideoCleanupOptions should clamp numeric cleanup controls', () => {
    assert.deepEqual(
        normalizeVideoCleanupOptions({
            residualCleanupStrength: 9,
            highQualityCleanup: true,
            textureRepair: true,
            textureRepairStrength: 2
        }),
        {
            residualCleanupStrength: 1.8,
            cleanupBackend: 'canvas-bilateral',
            highQualityCleanup: true,
            denoiseBackend: 'canvas-texture-repair',
            edgeDenoiseStrength: DEFAULT_EDGE_DENOISE_STRENGTH,
            textureRepair: true,
            textureRepairStrength: 1
        }
    );
});

test('normalizeVideoCleanupOptions should prefer explicit denoise backend over legacy texture flag', () => {
    assert.equal(
        normalizeVideoCleanupOptions({
            denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_TEXTURE_REPAIR,
            textureRepair: false
        }).textureRepair,
        true
    );
    assert.equal(
        normalizeVideoCleanupOptions({
            denoiseBackend: VIDEO_DENOISE_BACKENDS.NONE,
            textureRepair: true
        }).textureRepair,
        false
    );
    const edgeDenoise = normalizeVideoCleanupOptions({
        denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_DENOISE,
        textureRepair: true,
        edgeDenoiseStrength: 2
    });

    assert.equal(edgeDenoise.denoiseBackend, VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_DENOISE);
    assert.equal(edgeDenoise.edgeDenoiseStrength, 1);
    assert.equal(edgeDenoise.textureRepair, false);
    assert.equal(
        normalizeVideoCleanupOptions({
            denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_BAND_DENOISE
        }).denoiseBackend,
        VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_BAND_DENOISE
    );
});

test('allenk FDnCNN browser spike backend should be accepted but fail closed without runtime', () => {
    const normalized = normalizeVideoCleanupOptions({
        denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE,
        edgeDenoiseStrength: 1.8
    });

    assert.equal(normalized.denoiseBackend, VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE);
    assert.equal(normalized.edgeDenoiseStrength, 1.8);
    assert.equal(normalized.denoiseRuntimeStatus, 'unavailable');
    assert.match(normalized.denoiseRuntimeReason, /browser GPU inference runtime/);
});

test('buildLumaStructureGuard should protect strong image edges', () => {
    const width = 9;
    const height = 5;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const value = x < 4 ? 32 : 220;
            data[idx] = value;
            data[idx + 1] = value;
            data[idx + 2] = value;
            data[idx + 3] = 255;
        }
    }

    const guard = buildLumaStructureGuard({ width, height, data });
    const edge = guard[2 * width + 4];
    const flat = guard[2 * width + 0];

    assert.ok(edge > 0.6, `edge=${edge}`);
    assert.ok(flat < 0.12, `flat=${flat}`);
});

test('borrowCleanHighpassTexture should borrow texture only from clean low-weight neighbors', () => {
    const width = 9;
    const height = 5;
    const source = new Uint8ClampedArray(width * height * 4);
    const target = new Uint8ClampedArray(width * height * 4);
    const weights = new Float32Array(width * height);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const cleanTexture = x < 3 ? (x + y) % 2 === 0 ? 116 : 84 : 100;
            source[idx] = cleanTexture;
            source[idx + 1] = cleanTexture;
            source[idx + 2] = cleanTexture;
            source[idx + 3] = 255;
            target[idx] = 100;
            target[idx + 1] = 100;
            target[idx + 2] = 100;
            target[idx + 3] = 255;
            weights[y * width + x] = x >= 5 ? 0.5 : 0;
        }
    }

    const output = borrowCleanHighpassTexture({
        targetData: target,
        sourceData: source,
        weights,
        width,
        height,
        strength: 1
    });

    const texturedPixel = (2 * width + 6) * 4;
    const cleanPixel = (2 * width + 1) * 4;

    assert.notEqual(output[texturedPixel], target[texturedPixel]);
    assert.equal(output[cleanPixel], target[cleanPixel]);
});

test('allenk FDnCNN browser spike backend should accept an injected runtime', () => {
    const runtime = { denoiseImageData() {} };
    const normalized = normalizeVideoCleanupOptions({
        denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE,
        allenkFdncnnRuntime: runtime,
        allenkFdncnnSigma: 75,
        allenkFdncnnPadding: 0
    });

    assert.equal(normalized.denoiseBackend, VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE);
    assert.equal(normalized.allenkFdncnnRuntime, runtime);
    assert.equal(normalized.allenkFdncnnSigma, 75);
    assert.equal(normalized.allenkFdncnnPadding, 0);
    assert.equal(normalized.denoiseRuntimeStatus, 'available');
});

test('buildGradientWeightMap should emphasize alpha edges', () => {
    const width = 7;
    const height = 7;
    const alphaMap = createDiamondAlphaMap(width, height);
    const weights = buildGradientWeightMap(alphaMap, width, height, 1.5);
    const center = weights[3 * width + 3];
    const edge = weights[2 * width + 3];

    assert.equal(weights.length, width * height);
    assert.ok(edge > 0);
    assert.ok(Math.max(...weights) <= 1);
    assert.ok(edge >= center);
});

test('buildGradientWeightMap should include crop-boundary alpha edges', () => {
    const width = 3;
    const height = 3;
    const alphaMap = new Float32Array([
        0, 1, 0,
        0, 0, 0,
        0, 0, 0
    ]);
    const weights = buildGradientWeightMap(alphaMap, width, height, 1);

    assert.ok(weights[1] > 0);
    assert.ok(Math.max(...weights) <= 1);
});

test('buildEdgeBandDenoiseWeightMap should include crop-boundary alpha edges', () => {
    const width = 3;
    const height = 3;
    const alphaMap = new Float32Array([
        0, 1, 0,
        0, 0, 0,
        0, 0, 0
    ]);
    const weights = buildEdgeBandDenoiseWeightMap(alphaMap, width, height, 1);

    assert.ok(weights[1] > 0);
    assert.ok(Math.max(...weights) <= 1);
});

test('buildTextureRepairWeightMap should remain disabled at zero strength', () => {
    const width = 7;
    const height = 7;
    const alphaMap = createDiamondAlphaMap(width, height);
    const weights = buildTextureRepairWeightMap(alphaMap, width, height, 0);

    assert.equal(weights.length, width * height);
    assert.equal(Math.max(...weights), 0);
});

test('buildEdgeBandDenoiseWeightMap should guard high-alpha body pixels', () => {
    const width = 72;
    const height = 72;
    const alphaMap = getVideoAlphaMap(width);
    const gradientWeights = buildGradientWeightMap(alphaMap, width, height, 1);
    const bandWeights = buildEdgeBandDenoiseWeightMap(alphaMap, width, height, 1);
    const center = 36 * width + 36;
    const edge = 0 * width + 36;

    assert.equal(bandWeights.length, width * height);
    assert.ok(bandWeights[edge] > bandWeights[center]);
    assert.ok(bandWeights[center] < gradientWeights[center]);
    assert.ok(Math.max(...bandWeights) <= 1);
});

test('buildEdgeBandDenoiseWeightMap should remain disabled at zero strength', () => {
    const width = 7;
    const height = 7;
    const alphaMap = createDiamondAlphaMap(width, height);
    const weights = buildEdgeBandDenoiseWeightMap(alphaMap, width, height, 0);

    assert.equal(weights.length, width * height);
    assert.equal(Math.max(...weights), 0);
});

test('buildTextureRepairWeightMap should activate inside alpha footprint', () => {
    const width = 7;
    const height = 7;
    const alphaMap = createDiamondAlphaMap(width, height);
    const weights = buildTextureRepairWeightMap(alphaMap, width, height, 0.85);

    assert.equal(weights.length, width * height);
    assert.ok(weights[3 * width + 3] > 0);
    assert.ok(Math.max(...weights) <= 1);
});

test('buildFootprintPolishWeightMap should cover alpha body and edges conservatively', () => {
    const width = 72;
    const height = 72;
    const alphaMap = getVideoAlphaMap(width);
    const weights = buildFootprintPolishWeightMap(alphaMap, width, height, 0.65);
    const center = 36 * width + 36;
    const edge = 0 * width + 36;

    assert.equal(weights.length, width * height);
    assert.ok(weights[center] > 0);
    assert.ok(weights[edge] > 0);
    assert.ok(Math.max(...weights) <= 1);
});

test('applyVideoResidualCleanup should not touch canvas when cleanup is disabled', () => {
    const ctx = {
        canvas: { width: 16, height: 16 },
        getImageData() {
            throw new Error('getImageData should not be called');
        },
        putImageData() {
            throw new Error('putImageData should not be called');
        }
    };

    const result = applyVideoResidualCleanup(ctx, {
        x: 4,
        y: 4,
        width: 4,
        height: 4
    }, new Float32Array(16), {
        residualCleanupStrength: 0,
        textureRepair: false
    });

    assert.equal(result.residualCleanupStrength, 0);
    assert.equal(result.textureRepair, false);
});

test('applyVideoResidualCleanup should not touch canvas for allenk FDnCNN before runtime is wired', () => {
    const ctx = {
        canvas: { width: 16, height: 16 },
        getImageData() {
            throw new Error('getImageData should not be called');
        },
        putImageData() {
            throw new Error('putImageData should not be called');
        }
    };

    const result = applyVideoResidualCleanup(ctx, {
        x: 4,
        y: 4,
        width: 4,
        height: 4
    }, new Float32Array(16), {
        residualCleanupStrength: 0,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE
    });

    assert.equal(result.denoiseBackend, VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE);
    assert.equal(result.denoiseRuntimeStatus, 'unavailable');
});

test('applyVideoResidualCleanup should run injected allenk FDnCNN runtime on the padded ROI', () => {
    const source = {
        width: 16,
        height: 16,
        data: new Uint8ClampedArray(16 * 16 * 4)
    };
    for (let i = 0; i < source.data.length; i += 4) {
        source.data[i] = 40;
        source.data[i + 1] = 50;
        source.data[i + 2] = 60;
        source.data[i + 3] = 255;
    }

    let putCalls = 0;
    let runtimeCalls = 0;
    let written = null;
    const runtime = {
        id: 'fake-allenk-runtime',
        denoiseImageData({ imageData, sigma }) {
            runtimeCalls++;
            assert.equal(sigma, 75);
            const data = new Uint8ClampedArray(imageData.data);
            for (let i = 0; i < data.length; i += 4) {
                data[i] = 220;
            }
            return {
                runtime: 'fake-allenk-runtime',
                macs: 1234,
                imageData: {
                    width: imageData.width,
                    height: imageData.height,
                    data
                }
            };
        }
    };
    const ctx = {
        canvas: { width: source.width, height: source.height },
        getImageData(x, y, width, height) {
            assert.equal(x, 0);
            assert.equal(y, 0);
            assert.equal(width, 16);
            assert.equal(height, 16);
            return {
                width: source.width,
                height: source.height,
                data: new Uint8ClampedArray(source.data)
            };
        },
        putImageData(imageData, x, y) {
            putCalls++;
            assert.equal(x, 0);
            assert.equal(y, 0);
            written = imageData;
        }
    };

    const result = applyVideoResidualCleanup(ctx, {
        x: 4,
        y: 4,
        width: 8,
        height: 8
    }, createDiamondAlphaMap(8, 8), {
        residualCleanupStrength: 0,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE,
        edgeDenoiseStrength: 1,
        allenkFdncnnRuntime: runtime,
        allenkFdncnnSigma: 75
    });

    assert.equal(runtimeCalls, 1);
    assert.equal(putCalls, 1);
    assert.equal(result.denoiseRuntimeStatus, 'applied');
    assert.equal(result.denoiseRuntime, 'fake-allenk-runtime');
    assert.equal(result.denoiseRuntimeMacs, 1234);
    assert.ok(written.data.some((value, index) => index % 4 === 0 && value > 40));
});

test('applyVideoResidualCleanupAsync should await injected allenk FDnCNN runtime', async () => {
    const source = {
        width: 8,
        height: 8,
        data: new Uint8ClampedArray(8 * 8 * 4)
    };
    for (let i = 0; i < source.data.length; i += 4) {
        source.data[i] = 40;
        source.data[i + 1] = 50;
        source.data[i + 2] = 60;
        source.data[i + 3] = 255;
    }

    let written = null;
    const runtime = {
        id: 'async-allenk-runtime',
        async denoiseImageData({ imageData, sigma }) {
            assert.equal(sigma, 25);
            assert.equal(imageData.width, 4);
            assert.equal(imageData.height, 4);
            const data = new Uint8ClampedArray(imageData.data);
            for (let i = 0; i < data.length; i += 4) {
                data[i] = 240;
            }
            return {
                runtime: 'async-allenk-runtime',
                macs: 4321,
                runMs: 12.5,
                imageData: {
                    width: imageData.width,
                    height: imageData.height,
                    data
                }
            };
        }
    };
    const ctx = {
        canvas: { width: source.width, height: source.height },
        getImageData(x, y, width, height) {
            assert.equal(x, 2);
            assert.equal(y, 2);
            assert.equal(width, 4);
            assert.equal(height, 4);
            const data = new Uint8ClampedArray(width * height * 4);
            for (let i = 0; i < data.length; i += 4) {
                data[i] = 40;
                data[i + 1] = 50;
                data[i + 2] = 60;
                data[i + 3] = 255;
            }
            return { width, height, data };
        },
        putImageData(imageData, x, y) {
            assert.equal(x, 2);
            assert.equal(y, 2);
            written = imageData;
        }
    };

    const result = await applyVideoResidualCleanupAsync(ctx, {
        x: 2,
        y: 2,
        width: 4,
        height: 4
    }, createDiamondAlphaMap(4, 4), {
        residualCleanupStrength: 0,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE,
        edgeDenoiseStrength: 1,
        allenkFdncnnRuntime: runtime,
        allenkFdncnnSigma: 25,
        allenkFdncnnPadding: 0
    });

    assert.equal(result.denoiseRuntimeStatus, 'applied');
    assert.equal(result.denoiseRuntime, 'async-allenk-runtime');
    assert.equal(result.denoiseRuntimeMacs, 4321);
    assert.equal(result.denoiseRuntimeRunMs, 12.5);
    assert.equal(written.width, 4);
    assert.equal(written.height, 4);
});

test('applyVideoResidualCleanupAsync should run allenk FDnCNN before residual cleanup polish', async () => {
    const events = [];
    const runtime = {
        id: 'ordered-allenk-runtime',
        async denoiseImageData({ imageData }) {
            events.push('runtime');
            return {
                runtime: 'ordered-allenk-runtime',
                imageData: {
                    width: imageData.width,
                    height: imageData.height,
                    data: new Uint8ClampedArray(imageData.data)
                }
            };
        }
    };
    const ctx = {
        canvas: { width: 8, height: 8 },
        getImageData(x, y, width, height) {
            events.push(x === 2 && y === 2 && width === 4 && height === 4 ? 'get-ai-roi' : 'get-polish-roi');
            const data = new Uint8ClampedArray(width * height * 4);
            for (let i = 0; i < data.length; i += 4) {
                data[i] = 80;
                data[i + 1] = 90;
                data[i + 2] = 100;
                data[i + 3] = 255;
            }
            return { width, height, data };
        },
        putImageData(_imageData, x, y) {
            events.push(x === 2 && y === 2 ? 'put-ai-roi' : 'put-polish-roi');
        }
    };

    const result = await applyVideoResidualCleanupAsync(ctx, {
        x: 2,
        y: 2,
        width: 4,
        height: 4
    }, createDiamondAlphaMap(4, 4), {
        residualCleanupStrength: 1.5,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.ALLENK_FDNCNN_BROWSER_SPIKE,
        edgeDenoiseStrength: 1,
        allenkFdncnnRuntime: runtime,
        allenkFdncnnPadding: 0
    });

    assert.equal(result.denoiseRuntimeStatus, 'applied');
    assert.deepEqual(events, ['get-ai-roi', 'runtime', 'put-ai-roi', 'get-polish-roi', 'put-polish-roi']);
});

test('applyVideoResidualCleanup should route canvas edge denoise backend', () => {
    const image = {
        width: 20,
        height: 20,
        data: new Uint8ClampedArray(20 * 20 * 4)
    };
    for (let i = 0; i < image.data.length; i += 4) {
        image.data[i] = 80 + ((i / 4) % 20);
        image.data[i + 1] = 90;
        image.data[i + 2] = 100;
        image.data[i + 3] = 255;
    }

    let putCalls = 0;
    const ctx = {
        canvas: { width: 20, height: 20 },
        getImageData() {
            return {
                width: image.width,
                height: image.height,
                data: new Uint8ClampedArray(image.data)
            };
        },
        putImageData() {
            putCalls++;
        }
    };

    const result = applyVideoResidualCleanup(ctx, {
        x: 7,
        y: 7,
        width: 6,
        height: 6
    }, createDiamondAlphaMap(6, 6), {
        residualCleanupStrength: 0,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_DENOISE,
        edgeDenoiseStrength: 0.5
    });

    assert.equal(result.denoiseBackend, VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_DENOISE);
    assert.equal(putCalls, 1);
});

test('applyVideoResidualCleanup should route canvas edge band denoise backend', () => {
    const image = {
        width: 20,
        height: 20,
        data: new Uint8ClampedArray(20 * 20 * 4)
    };
    for (let i = 0; i < image.data.length; i += 4) {
        image.data[i] = 80 + ((i / 4) % 20);
        image.data[i + 1] = 90;
        image.data[i + 2] = 100;
        image.data[i + 3] = 255;
    }

    let putCalls = 0;
    const ctx = {
        canvas: { width: 20, height: 20 },
        getImageData() {
            return {
                width: image.width,
                height: image.height,
                data: new Uint8ClampedArray(image.data)
            };
        },
        putImageData() {
            putCalls++;
        }
    };

    const result = applyVideoResidualCleanup(ctx, {
        x: 7,
        y: 7,
        width: 6,
        height: 6
    }, createDiamondAlphaMap(6, 6), {
        residualCleanupStrength: 0,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_BAND_DENOISE,
        edgeDenoiseStrength: 0.5
    });

    assert.equal(result.denoiseBackend, VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_BAND_DENOISE);
    assert.equal(putCalls, 1);
});

test('applyVideoResidualCleanup should route canvas edge core denoise backend', () => {
    const image = {
        width: 20,
        height: 20,
        data: new Uint8ClampedArray(20 * 20 * 4)
    };
    for (let i = 0; i < image.data.length; i += 4) {
        image.data[i] = 80 + ((i / 4) % 20);
        image.data[i + 1] = 90;
        image.data[i + 2] = 100;
        image.data[i + 3] = 255;
    }

    let putCalls = 0;
    const ctx = {
        canvas: { width: 20, height: 20 },
        getImageData() {
            return {
                width: image.width,
                height: image.height,
                data: new Uint8ClampedArray(image.data)
            };
        },
        putImageData() {
            putCalls++;
        }
    };

    const result = applyVideoResidualCleanup(ctx, {
        x: 7,
        y: 7,
        width: 6,
        height: 6
    }, createDiamondAlphaMap(6, 6), {
        residualCleanupStrength: 0,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_CORE_DENOISE,
        edgeDenoiseStrength: 0.5
    });

    assert.equal(result.denoiseBackend, VIDEO_DENOISE_BACKENDS.CANVAS_EDGE_CORE_DENOISE);
    assert.equal(putCalls, 1);
});

test('applyVideoResidualCleanup should route canvas footprint polish backend', () => {
    const image = {
        width: 20,
        height: 20,
        data: new Uint8ClampedArray(20 * 20 * 4)
    };
    for (let i = 0; i < image.data.length; i += 4) {
        image.data[i] = 80 + ((i / 4) % 20);
        image.data[i + 1] = 90;
        image.data[i + 2] = 100;
        image.data[i + 3] = 255;
    }

    let putCalls = 0;
    const ctx = {
        canvas: { width: 20, height: 20 },
        getImageData() {
            return {
                width: image.width,
                height: image.height,
                data: new Uint8ClampedArray(image.data)
            };
        },
        putImageData() {
            putCalls++;
        }
    };

    const result = applyVideoResidualCleanup(ctx, {
        x: 7,
        y: 7,
        width: 6,
        height: 6
    }, createDiamondAlphaMap(6, 6), {
        residualCleanupStrength: 0,
        denoiseBackend: VIDEO_DENOISE_BACKENDS.CANVAS_FOOTPRINT_POLISH,
        edgeDenoiseStrength: 0.5
    });

    assert.equal(result.denoiseBackend, VIDEO_DENOISE_BACKENDS.CANVAS_FOOTPRINT_POLISH);
    assert.equal(putCalls, 1);
});
