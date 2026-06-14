import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { access } from 'node:fs/promises';

import { calculateAlphaMap } from '../../src/core/alphaMap.js';
import { getEmbeddedAlphaMap } from '../../src/core/embeddedAlphaMaps.js';
import { processWatermarkImageData } from '../../src/core/watermarkProcessor.js';
import { interpolateAlphaMap, warpAlphaMap, computeRegionSpatialCorrelation } from '../../src/core/adaptiveDetector.js';
import { decodeImageDataInNode } from '../../scripts/sample-benchmark.js';
import {
    applySyntheticWatermark,
    createPatternImageData,
    createSyntheticAlphaMap
} from './syntheticWatermarkTestUtils.js';

test('processWatermarkImageData should run in Node without asset imports and record single-pass meta', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const position = { x: 320 - 32 - 48, y: 320 - 32 - 48, width: 48, height: 48 };
    applySyntheticWatermark(imageData, alpha48, position, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        locatedAggressiveRemoval: false
    });

    assert.equal(result.imageData.width, 320);
    assert.ok(result.meta.applied);
    assert.equal(result.meta.passCount, 1, `passCount=${result.meta.passCount}`);
    assert.equal(result.meta.passStopReason, 'residual-low');
    assert.ok(Array.isArray(result.meta.passes));
    assert.ok(result.meta.detection.processedSpatialScore < 0.25, `score=${result.meta.detection.processedSpatialScore}`);
});

test('processWatermarkImageData should not attempt extra passes when the first pass already clears a single watermark layer', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const position = { x: 320 - 32 - 48, y: 320 - 32 - 48, width: 48, height: 48 };
    applySyntheticWatermark(imageData, alpha48, position, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        locatedAggressiveRemoval: false
    });

    assert.equal(result.meta.passCount, 1, `passCount=${result.meta.passCount}`);
    assert.equal(result.meta.attemptedPassCount, 1, `attemptedPassCount=${result.meta.attemptedPassCount}`);
    assert.equal(result.meta.passStopReason, 'residual-low');
    assert.equal(result.meta.passes.length, 1, `passes=${JSON.stringify(result.meta.passes)}`);
});

test('processWatermarkImageData should interpolate adaptive alpha maps when getAlphaMap is omitted', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(1008, 1071);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96
    });

    assert.equal(result.meta.applied, false);
    assert.equal(result.meta.skipReason, 'no-watermark-detected');
});

test('processWatermarkImageData should not use template warp in fixed-core mode', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const position = { x: 320 - 32 - 48, y: 320 - 32 - 48, width: 48, height: 48 };
    const embeddedWarpedAlpha = warpAlphaMap(alpha48, 48, { dx: -1, dy: 2, scale: 0.95 });
    applySyntheticWatermark(imageData, embeddedWarpedAlpha, position, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
    });

    assert.equal(result.meta.templateWarp, null);
    assert.ok(!String(result.meta.source).includes('+warp'), `source=${result.meta.source}`);
});

test('processWatermarkImageData should select the evidence-gated allenk V2 36px profile on official 1K outputs', () => {
    const alpha48 = getEmbeddedAlphaMap(48);
    const alpha96 = getEmbeddedAlphaMap(96);
    const alpha36V2 = getEmbeddedAlphaMap('36-v2');
    const imageData = createPatternImageData(1376, 768);
    const position = {
        x: 1376 - 96 - 36,
        y: 768 - 96 - 36,
        width: 36,
        height: 36
    };
    applySyntheticWatermark(imageData, alpha36V2, position, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        getAlphaMap: (size) => size === '36-v2'
            ? alpha36V2
            : (size === 48 ? alpha48 : (size === 96 ? alpha96 : interpolateAlphaMap(alpha96, 96, size)))
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(result.meta.config, {
        logoSize: 36,
        marginRight: 96,
        marginBottom: 96,
        alphaVariant: 'v2'
    });
    assert.equal(result.meta.detection.processedSpatialScore < 0.12, true);
});

test('processWatermarkImageData should cleanup residual edges on allenk V2 36px catalog anchors', async () => {
    const alpha48 = getEmbeddedAlphaMap(48);
    const alpha96 = getEmbeddedAlphaMap(96);
    const alpha36V2 = getEmbeddedAlphaMap('36-v2');
    const imageData = await decodeImageDataInNode(path.resolve('tests/fixtures/gemini-v2-36-small-watermark.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        getAlphaMap: (size) => size === '36-v2'
            ? alpha36V2
            : (size === 48 ? alpha48 : (size === 96 ? alpha96 : interpolateAlphaMap(alpha96, 96, size)))
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(result.meta.config, {
        logoSize: 36,
        marginRight: 71,
        marginBottom: 71,
        alphaVariant: 'v2'
    });
    assert.ok(
        result.meta.source.includes('v2-small-edge-cleanup'),
        `expected v2 cleanup source, got ${result.meta.source}`
    );
    assert.ok(
        Math.abs(result.meta.detection.processedSpatialScore) <= 0.08,
        `expected v2 cleanup to preserve low spatial residual, got ${result.meta.detection.processedSpatialScore}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore < 0.08,
        `expected v2 cleanup to reduce gradient residual, got ${result.meta.detection.processedGradientScore}`
    );
    assert.equal(result.meta.detection.residualVisibility?.visiblePositiveHalo, true);
    assert.ok(
        result.meta.detection.residualVisibility.positiveHaloLum > 6,
        `expected final meta to report visible v2 positive halo, got ${JSON.stringify(result.meta.detection.residualVisibility)}`
    );
});

test('processWatermarkImageData should allow weak alpha gain to compete as a first-pass candidate', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260607.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(result.meta.applied);
    assert.ok(result.meta.alphaGain < 1, `expected weak alpha gain candidate, got ${result.meta.alphaGain}`);
    assert.ok(
        result.meta.detection.processedSpatialScore < 0.25,
        `expected first-pass gain candidate to suppress residual, got ${result.meta.detection.processedSpatialScore}`
    );
    assert.ok(
        result.meta.passes[0].afterSpatialScore < 0.25,
        `expected first recorded pass to use gain candidate, got ${result.meta.passes[0].afterSpatialScore}`
    );
});

test('processWatermarkImageData should remove strong white watermarks on near-black backgrounds', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const width = 320;
    const height = 320;
    const data = new Uint8ClampedArray(width * height * 4);

    for (let index = 0; index < width * height; index++) {
        const offset = index * 4;
        data[offset] = 0;
        data[offset + 1] = 0;
        data[offset + 2] = 0;
        data[offset + 3] = 255;
    }

    const imageData = { width, height, data };
    const position = { x: 240, y: 240, width: 48, height: 48 };
    for (let row = 0; row < position.height; row++) {
        for (let col = 0; col < position.width; col++) {
            const alpha = alpha48[row * position.width + col];
            if (alpha <= 0.005) continue;

            const offset = ((position.y + row) * imageData.width + position.x + col) * 4;
            for (let channel = 0; channel < 3; channel++) {
                imageData.data[offset + channel] = Math.round(alpha * 255);
            }
        }
    }

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        locatedAggressiveRemoval: false
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.equal(result.meta.source, 'standard');
    assert.ok(
        result.meta.detection.originalSpatialScore >= 0.9,
        `originalSpatial=${result.meta.detection.originalSpatialScore}`
    );
    assert.ok(
        result.meta.detection.originalGradientScore >= 0.7,
        `originalGradient=${result.meta.detection.originalGradientScore}`
    );
    assert.ok(
        Math.abs(result.meta.detection.processedSpatialScore) <= 0.7,
        `processedSpatial=${result.meta.detection.processedSpatialScore}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore <= 0.32,
        `processedGradient=${result.meta.detection.processedGradientScore}`
    );
});

test('processWatermarkImageData should skip off-catalog adaptive-only positions in fixed-core mode', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const truePosition = { x: 320 - 36 - 48, y: 320 - 20 - 48, width: 48, height: 48 };
    applySyntheticWatermark(imageData, alpha48, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        aggressiveLocatedFallback: false
    });

    assert.equal(result.meta.applied, false);
    assert.equal(result.meta.source, 'skipped');
});

test('processWatermarkImageData should process strong located off-catalog watermarks by default', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const truePosition = { x: 320 - 36 - 48, y: 320 - 20 - 48, width: 48, height: 48 };
    applySyntheticWatermark(imageData, alpha48, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.ok(
        String(result.meta.source).includes('aggressive-located'),
        `source=${result.meta.source}`
    );
    assert.ok(
        Math.abs(result.meta.position.x - truePosition.x) <= 2,
        `x=${result.meta.position.x}`
    );
    assert.ok(
        Math.abs(result.meta.position.y - truePosition.y) <= 2,
        `y=${result.meta.position.y}`
    );
    assert.ok(
        Math.abs(result.meta.detection.processedSpatialScore) < 0.12,
        `processedSpatial=${result.meta.detection.processedSpatialScore}`
    );
});

test('processWatermarkImageData should process dark-polarity 96px 192px-margin watermarks', () => {
    const alpha48 = getEmbeddedAlphaMap(48);
    const alpha96 = getEmbeddedAlphaMap(96);
    const alpha96NewMargin = getEmbeddedAlphaMap('96-20260520');
    const width = 2778;
    const height = 1536;
    const data = new Uint8ClampedArray(width * height * 4);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const value = 110 + Math.round(20 * Math.sin(x / 80) + 12 * Math.cos(y / 60));
            data[idx] = value;
            data[idx + 1] = value + 4;
            data[idx + 2] = value + 14;
            data[idx + 3] = 255;
        }
    }

    const imageData = { width, height, data };
    const position = {
        x: width - 192 - 96,
        y: height - 192 - 96,
        width: 96,
        height: 96
    };

    for (let row = 0; row < position.height; row++) {
        for (let col = 0; col < position.width; col++) {
            const alpha = alpha96NewMargin[row * position.width + col];
            if (alpha <= 0.001) continue;

            const idx = ((position.y + row) * imageData.width + position.x + col) * 4;
            for (let channel = 0; channel < 3; channel++) {
                imageData.data[idx + channel] = Math.round((1 - alpha) * imageData.data[idx + channel]);
            }
        }
    }

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        alpha96Variants: {
            '20260520': alpha96NewMargin
        },
        getAlphaMap: (size) => {
            if (size === 48) return alpha48;
            if (size === 96) return alpha96;
            if (size === '96-20260520') return alpha96NewMargin;
            return interpolateAlphaMap(alpha96, 96, size);
        }
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.ok(
        result.meta.source.includes('dark-polarity'),
        `source=${result.meta.source}`
    );
    assert.deepEqual(result.meta.config, {
        logoSize: 96,
        marginRight: 192,
        marginBottom: 192,
        alphaVariant: '20260520'
    });
    assert.ok(
        result.meta.detection.processedSpatialScore < 0.08,
        `processedSpatial=${result.meta.detection.processedSpatialScore}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore < 0.08,
        `processedGradient=${result.meta.detection.processedGradientScore}`
    );
});

test('processWatermarkImageData should select the near-official scaled large-margin anchor for 20260613 sample', async (t) => {
    const samplePath = path.resolve('src/assets/samples/20260613.png');
    try {
        await access(samplePath);
    } catch {
        t.skip('20260613.png is not present in the current fixture set');
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(samplePath);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.equal(
        result.meta.config.logoSize,
        42,
        `expected projected large-margin logo size to avoid bright edge halo, got ${JSON.stringify(result.meta.config)}`
    );
    assert.ok(
        result.meta.config.marginRight >= 78 && result.meta.config.marginRight <= 88,
        `expected scaled large-margin right anchor, got ${JSON.stringify(result.meta.config)}`
    );
    assert.ok(
        result.meta.config.marginBottom >= 78 && result.meta.config.marginBottom <= 88,
        `expected scaled large-margin bottom anchor, got ${JSON.stringify(result.meta.config)}`
    );
    assert.notDeepEqual(
        result.meta.config,
        { logoSize: 41, marginRight: 34, marginBottom: 38 },
        'should not select the lower-right residual star instead of the primary watermark'
    );
    assert.ok(
        result.meta.detection.processedGradientScore < 0.24,
        `processedGradient=${result.meta.detection.processedGradientScore}`
    );
    assert.equal(
        result.meta.detection.residualVisibility?.visiblePositiveHalo,
        false,
        `residualVisibility=${JSON.stringify(result.meta.detection.residualVisibility)}`
    );
});

test('processWatermarkImageData should relocate visible 45px fixed-local residuals to the matched small anchor', async (t) => {
    const cases = [
        {
            samplePath: 'D:/Project/sample-files/gemini-watermark/2026-06-09/2064244525538217984-source.png',
            expectedConfig: { logoSize: 46, marginRight: 32, marginBottom: 42 },
            maxSpatial: 0.08,
            maxGradient: 0.22
        },
        {
            samplePath: 'D:/Project/sample-files/gemini-watermark/样本/Gemini_Generated_Image_21odi621odi621od.png',
            expectedConfigRange: {
                logoSize: [47, 52],
                marginRight: [39, 42],
                marginBottom: [29, 33]
            },
            maxSpatial: 0.1,
            maxGradient: 0.22
        }
    ];

    try {
        for (const item of cases) await access(path.resolve(item.samplePath));
    } catch {
        t.skip('external small fixed-local residual samples are not available');
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));

    for (const item of cases) {
        const imageData = await decodeImageDataInNode(path.resolve(item.samplePath));
        const result = processWatermarkImageData(imageData, {
            alpha48,
            alpha96,
            adaptiveMode: 'never',
            getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
        });

        assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
        if (item.expectedConfig) {
            assert.deepEqual(result.meta.config, item.expectedConfig);
        } else {
            assert.ok(
                result.meta.config.logoSize >= item.expectedConfigRange.logoSize[0] &&
                    result.meta.config.logoSize <= item.expectedConfigRange.logoSize[1] &&
                    result.meta.config.marginRight >= item.expectedConfigRange.marginRight[0] &&
                    result.meta.config.marginRight <= item.expectedConfigRange.marginRight[1] &&
                    result.meta.config.marginBottom >= item.expectedConfigRange.marginBottom[0] &&
                    result.meta.config.marginBottom <= item.expectedConfigRange.marginBottom[1],
                `expected small-anchor relocation cluster, got ${JSON.stringify(result.meta.config)}`
            );
        }
        assert.ok(
            String(result.meta.source).includes('+small-anchor-relocated'),
            `expected small-anchor relocation, source=${result.meta.source}`
        );
        assert.equal(
            result.meta.detection.residualVisibility?.visible,
            false,
            `expected no visible residual, visibility=${JSON.stringify(result.meta.detection.residualVisibility)}`
        );
        assert.ok(
            Math.abs(result.meta.detection.processedSpatialScore) <= item.maxSpatial,
            `expected bounded spatial residual, got ${result.meta.detection.processedSpatialScore}`
        );
        assert.ok(
            result.meta.detection.processedGradientScore <= item.maxGradient,
            `expected bounded gradient residual, got ${result.meta.detection.processedGradientScore}`
        );
    }
});

test('processWatermarkImageData should allow stronger mid-alpha on strong 48px large-margin residuals', async (t) => {
    const samplePath = path.resolve('D:/Project/sample-files/gemini-watermark/样本/Gemini_Generated_Image_n79y30n79y30n79y.png');
    try {
        await access(samplePath);
    } catch {
        t.skip('external strong 48px large-margin residual sample is not available');
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(samplePath);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(result.meta.config, { logoSize: 48, marginRight: 96, marginBottom: 96 });
    assert.equal(result.meta.alphaGain, 0.7);
    assert.ok(
        String(result.meta.source).includes('+fine-alpha'),
        `expected fine-alpha source, got ${result.meta.source}`
    );
    assert.ok(
        Math.abs(result.meta.detection.processedSpatialScore) <= 0.16,
        `expected mid-alpha to clear residual spatial, got ${result.meta.detection.processedSpatialScore}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore <= 0.25,
        `expected bounded gradient residual, got ${result.meta.detection.processedGradientScore}`
    );
    assert.equal(
        result.meta.detection.residualVisibility?.visible,
        false,
        `expected no visible residual, visibility=${JSON.stringify(result.meta.detection.residualVisibility)}`
    );
});

test('processWatermarkImageData should recover near-official scaled anchor without adaptive search', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const alpha125 = interpolateAlphaMap(alpha96, 96, 125);
    const imageData = createPatternImageData(1000, 1792);
    const truePosition = { x: 792, y: 1584, width: 125, height: 125 };
    applySyntheticWatermark(imageData, alpha125, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.ok(
        Math.abs(result.meta.position.x - truePosition.x) <= 8,
        `x=${result.meta.position.x}`
    );
    assert.ok(
        Math.abs(result.meta.position.y - truePosition.y) <= 8,
        `y=${result.meta.position.y}`
    );
    assert.ok(
        Math.abs(result.meta.position.width - truePosition.width) <= 6,
        `width=${result.meta.position.width}`
    );
    assert.ok(
        result.meta.detection.processedSpatialScore < 0.22,
        `expected scaled standard candidate to suppress residual, got ${result.meta.detection.processedSpatialScore}`
    );
});

test('processWatermarkImageData should not recover small size drift outside fixed combinations', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const alpha54 = interpolateAlphaMap(alpha96, 96, 54);
    const imageData = createPatternImageData(320, 320);
    const truePosition = {
        x: 320 - 32 - 54,
        y: 320 - 32 - 54,
        width: 54,
        height: 54
    };
    applySyntheticWatermark(imageData, alpha54, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true);
    assert.deepEqual(result.meta.position, { x: 240, y: 240, width: 48, height: 48 });
    assert.ok(!String(result.meta.source).includes('+size'), `source=${result.meta.source}`);
});

test('processWatermarkImageData should recover preview-sized bottom-right watermark without adaptive search', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const alpha34 = interpolateAlphaMap(alpha96, 96, 34);
    const imageData = createPatternImageData(1024, 559);
    const truePosition = {
        x: 1024 - 24 - 34,
        y: 559 - 24 - 34,
        width: 34,
        height: 34
    };
    applySyntheticWatermark(imageData, alpha34, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    const residual = computeRegionSpatialCorrelation({
        imageData: result.imageData,
        alphaMap: alpha34,
        region: { x: truePosition.x, y: truePosition.y, size: truePosition.width }
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.ok(
        result.meta.source.startsWith('standard'),
        `expected standard catalog recovery, got ${result.meta.source}`
    );
    assert.ok(Math.abs(result.meta.position.width - truePosition.width) <= 1, `width=${result.meta.position.width}`);
    assert.equal(result.meta.position.x, truePosition.x);
    assert.equal(result.meta.position.y, truePosition.y);
    assert.ok(
        residual < 0.22,
        `expected preview-anchor residual < 0.22, got ${residual}, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should skip repeated preview watermark layers in fixed-core mode', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const alpha34 = interpolateAlphaMap(alpha96, 96, 34);
    const imageData = createPatternImageData(1024, 559);
    const truePosition = {
        x: 1024 - 24 - 34,
        y: 559 - 24 - 34,
        width: 34,
        height: 34
    };
    applySyntheticWatermark(imageData, alpha34, truePosition, 2);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        aggressiveLocatedFallback: false,
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, false);
    assert.equal(result.meta.passCount, 0, `passCount=${result.meta.passCount}`);
    assert.equal(result.meta.attemptedPassCount, 0, `attemptedPassCount=${result.meta.attemptedPassCount}`);
    assert.ok(
        !String(result.meta.source).includes('+multipass'),
        `expected preview-anchor removal to skip multipass, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should expose fixed-core candidate selection debug summary in meta', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const truePosition = {
        x: 320 - 32 - 48,
        y: 320 - 32 - 48,
        width: 48,
        height: 48
    };
    applySyntheticWatermark(imageData, alpha48, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.ok(result.meta.selectionDebug, 'expected selectionDebug to be present');
    assert.equal(result.meta.selectionDebug.usedSizeJitter, false);
    assert.equal(result.meta.selectionDebug.usedLocalShift, false);
    assert.equal(result.meta.selectionDebug.usedAdaptive, false);
    assert.equal(typeof result.meta.selectionDebug.texturePenalty, 'number');
    assert.equal(typeof result.meta.selectionDebug.tooDark, 'boolean');
    assert.equal(typeof result.meta.selectionDebug.tooFlat, 'boolean');
    assert.equal(typeof result.meta.selectionDebug.hardReject, 'boolean');
    assert.equal(result.meta.selectionDebug.sourcePriority, 0);
    assert.ok(Array.isArray(result.meta.selectionDebug.rankingKey));
    assert.equal(result.meta.selectionDebug.originalEvidence?.tier, 'strong');
    assert.equal(result.meta.selectionDebug.residual?.cleared, true);
    assert.equal(result.meta.selectionDebug.damage?.safe, true);
    assert.deepEqual(result.meta.selectionDebug.initialConfig, { logoSize: 48, marginRight: 32, marginBottom: 32 });
    assert.deepEqual(result.meta.selectionDebug.initialPosition, { x: 240, y: 240, width: 48, height: 48 });
    assert.deepEqual(result.meta.selectionDebug.finalConfig, result.meta.config);
    assert.deepEqual(result.meta.selectionDebug.finalPosition, result.meta.position);
});

test('processWatermarkImageData should expose local shift provenance for tall portrait anchor recovery', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(768, 1376);
    const truePosition = {
        x: 768 - 59 - 96,
        y: 1376 - 59 - 96,
        width: 96,
        height: 96
    };
    applySyntheticWatermark(imageData, alpha96, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.equal(result.meta.selectionDebug?.usedLocalShift, false);
    assert.equal(result.meta.selectionDebug?.usedCatalogVariant, true);
    assert.deepEqual(result.meta.selectionDebug?.initialConfig, { logoSize: 96, marginRight: 64, marginBottom: 64 });
    assert.deepEqual(result.meta.selectionDebug?.initialPosition, { x: 608, y: 1216, width: 96, height: 96 });
    assert.deepEqual(result.meta.selectionDebug?.finalPosition, result.meta.position);
});

test('processWatermarkImageData should expose normalized decision tier alongside legacy source tags', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const position = { x: 320 - 32 - 48, y: 320 - 32 - 48, width: 48, height: 48 };
    applySyntheticWatermark(imageData, alpha48, position, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
    });

    assert.ok(result.meta.applied);
    assert.equal(typeof result.meta.decisionTier, 'string');
    assert.ok(
        ['direct-match', 'validated-match'].includes(result.meta.decisionTier),
        `decisionTier=${result.meta.decisionTier}, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should avoid expanded alpha-gain search when the first-pass standard candidate is already clean', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const position = { x: 320 - 32 - 48, y: 320 - 32 - 48, width: 48, height: 48 };
    applySyntheticWatermark(imageData, alpha48, position, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
    });

    assert.ok(result.meta.applied);
    assert.equal(result.meta.alphaGain, 1);
    assert.ok(
        !String(result.meta.source).includes('+gain'),
        `expected no expanded gain search, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should avoid conservative fallback on debug1-source download sample', async (t) => {
    const samplePath = path.resolve('src/assets/samples/debug1-source.png');
    try {
        await access(samplePath);
    } catch {
        t.skip('debug1-source.png is not present in the current fixture set');
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(samplePath);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.ok(
        result.meta.detection.processedGradientScore < 0.12,
        `expected debug1-source residual gradient < 0.12, got ${result.meta.detection.processedGradientScore}, source=${result.meta.source}`
    );
    assert.ok(
        result.meta.position.width >= 48,
        `expected debug1-source to avoid undersized fallback candidate, got ${result.meta.position.width}`
    );
});

test('processWatermarkImageData should avoid local-shift drift on debug2-source portrait sample', async () => {
    const samplePath = path.resolve('src/assets/samples/debug2-source.png');
    try {
        await access(samplePath);
    } catch {
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(samplePath);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.ok(
        result.meta.source.startsWith('standard') && !result.meta.source.includes('+local'),
        `expected standard anchor to win, got ${result.meta.source}`
    );
    assert.deepEqual(
        result.meta.position,
        { x: 608, y: 1216, width: 96, height: 96 },
        `unexpected final position=${JSON.stringify(result.meta.position)}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore < 0.1,
        `expected debug2-source residual gradient < 0.1, got ${result.meta.detection.processedGradientScore}`
    );
});

test('processWatermarkImageData should keep 20260520-1.png on the canonical 48px anchor', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260520-1.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(
        result.meta.position,
        { x: 1328, y: 688, width: 48, height: 48 }
    );
    assert.ok(
        !String(result.meta.source).includes('+local'),
        `expected canonical standard anchor, got source=${result.meta.source}`
    );
});

test('processWatermarkImageData should not promote weak 192px-margin local drift on 20260607 keyboard sample', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const alpha96NewMargin = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96_20260520.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260607.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        alpha96Variants: {
            '20260520': alpha96NewMargin
        },
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(
        result.meta.applied === false || result.meta.position.x >= 1240,
        `expected no off-anchor 192px-margin local drift, applied=${result.meta.applied}, position=${JSON.stringify(result.meta.position)}, source=${result.meta.source}`
    );
    assert.ok(
        !String(result.meta.source).includes('192') &&
        !(String(result.meta.source).includes('+catalog') && String(result.meta.source).includes('+local')),
        `expected weak catalog local drift to be rejected, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should remove 20260607 samples at the 48px 96px-margin anchor', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const cases = [
        {
            fileName: '20260607.png',
            position: { x: 1264, y: 624, width: 48, height: 48 },
            maxAlphaGain: 0.7
        },
        {
            fileName: '20260607-2.png',
            position: { x: 576, y: 1313, width: 48, height: 48 },
            maxAlphaGain: 1
        }
    ];

    for (const item of cases) {
        const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples', item.fileName));

        const result = processWatermarkImageData(imageData, {
            alpha48,
            alpha96,
            adaptiveMode: 'never',
            getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
        });

        assert.equal(result.meta.applied, true, `${item.fileName} skipReason=${result.meta.skipReason}`);
        assert.deepEqual(
            result.meta.position,
            item.position,
            `${item.fileName} unexpected position=${JSON.stringify(result.meta.position)}, source=${result.meta.source}`
        );
        assert.ok(
            result.meta.alphaGain <= item.maxAlphaGain,
            `${item.fileName} expected a safe alpha gain <= ${item.maxAlphaGain}, got ${result.meta.alphaGain}`
        );
        assert.ok(
            result.meta.detection.processedGradientScore < result.meta.detection.originalGradientScore,
            `${item.fileName} expected residual gradient to improve, before=${result.meta.detection.originalGradientScore}, after=${result.meta.detection.processedGradientScore}`
        );
    }
});

test('processWatermarkImageData should keep visually balanced weak alpha when 20260608 catalog sample keeps a light residual', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260608-3.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size),
        locatedAggressiveRemoval: false
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.equal(result.meta.passCount, 1, `passCount=${result.meta.passCount}`);
    assert.equal(
        result.meta.alphaGain,
        0.7,
        `expected fine alpha around the visually balanced weak-alpha point, got ${result.meta.alphaGain}`
    );
    assert.deepEqual(
        result.meta.alphaAdjustmentStages,
        [],
        'expected direct selected alpha without post-selection alpha adjustment'
    );
    assert.ok(
        result.meta.detection.processedSpatialScore < 0.1,
        `expected light residual to be reduced, got ${result.meta.detection.processedSpatialScore}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore <= result.meta.detection.originalGradientScore - 0.55,
        `expected gradient signal to remain strongly reduced, before=${result.meta.detection.originalGradientScore}, after=${result.meta.detection.processedGradientScore}`
    );
});

test('processWatermarkImageData should keep strong 20260608 catalog evidence ahead of weak preview anchor', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260608-4.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(
        result.meta.position,
        { x: 576, y: 1313, width: 48, height: 48 },
        `expected 48px large-margin catalog anchor, got ${JSON.stringify(result.meta.position)} source=${result.meta.source}`
    );
    assert.equal(result.meta.alphaGain, 0.95);
    assert.equal(result.meta.alphaAdjustmentStages?.[0]?.stage, 'dark-catalog-fine-alpha');
    assert.equal(result.meta.alphaAdjustmentStages?.[0]?.fromAlphaGain, 1);
    assert.equal(result.meta.alphaAdjustmentStages?.[0]?.toAlphaGain, 0.95);
    assert.ok(
        String(result.meta.source).includes('catalog'),
        `expected catalog source, got ${result.meta.source}`
    );
    assert.ok(
        !String(result.meta.source).includes('preview-anchor'),
        `expected strong catalog evidence to beat preview-anchor, got ${result.meta.source}`
    );
    assert.ok(
        result.meta.detection.processedSpatialScore < 0.05,
        `expected residual spatial to be near zero, got ${result.meta.detection.processedSpatialScore}`
    );
});

test('processWatermarkImageData should best-effort remove confirmed 48px large-margin weak-alpha samples', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const samples = [
        { fileName: '20260608-6.png', expectedYMin: 1312, expectedYMax: 1313 },
        { fileName: '20260608-7.png', expectedYMin: 1312, expectedYMax: 1312 }
    ];

    for (const sample of samples) {
        const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples', sample.fileName));

        const result = processWatermarkImageData(imageData, {
            alpha48,
            alpha96,
            adaptiveMode: 'never',
            getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
        });

        assert.equal(result.meta.applied, true, `${sample.fileName} skipReason=${result.meta.skipReason}`);
        assert.equal(result.meta.size, 48, `${sample.fileName} expected 48px watermark`);
        assert.equal(result.meta.position.x, 576, `${sample.fileName} expected 48px large-margin x anchor`);
        assert.ok(
            result.meta.position.y >= sample.expectedYMin && result.meta.position.y <= sample.expectedYMax,
            `${sample.fileName} expected y near 48px large-margin anchor, got ${result.meta.position.y}`
        );
        assert.equal(result.meta.position.width, 48);
        assert.equal(result.meta.config.marginRight, 96);
        assert.equal(result.meta.config.marginBottom, 96);
        assert.ok(
            String(result.meta.source).includes('catalog'),
            `${sample.fileName} expected catalog source, got ${result.meta.source}`
        );
        assert.ok(
            result.meta.alphaGain < 1,
            `${sample.fileName} expected weak-alpha processing, got alphaGain=${result.meta.alphaGain}`
        );
        assert.ok(
            result.meta.detection.originalSpatialScore > 0.9 &&
                result.meta.detection.originalGradientScore > 0.7,
            `${sample.fileName} expected strong original watermark evidence, detection=${JSON.stringify(result.meta.detection)}`
        );
        assert.ok(
            result.meta.detection.processedGradientScore <= result.meta.detection.originalGradientScore - 0.5,
            `${sample.fileName} expected gradient suppression, detection=${JSON.stringify(result.meta.detection)}`
        );
        assert.ok(
            result.meta.detection.suppressionGain > 0.3,
            `${sample.fileName} expected measurable best-effort suppression, gain=${result.meta.detection.suppressionGain}`
        );
    }
});

test('processWatermarkImageData should conservatively remove low-contrast 48px large-margin watermarks', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const samples = [
        {
            fileName: '2-3.png',
            expectedPosition: { x: 704, y: 1120, width: 48, height: 48 },
            maxSpatialResidual: 0.04,
            maxGradientResidual: 0.12
        },
        {
            fileName: '8-1.png',
            expectedPosition: { x: 2784, y: 208, width: 48, height: 48 },
            maxSpatialResidual: 0.12,
            maxGradientResidual: 0.12
        }
    ];

    for (const sample of samples) {
        const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples', sample.fileName));

        const result = processWatermarkImageData(imageData, {
            alpha48,
            alpha96,
            adaptiveMode: 'never',
            getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
        });

        assert.equal(result.meta.applied, true, `${sample.fileName} skipReason=${result.meta.skipReason}`);
        assert.deepEqual(result.meta.position, sample.expectedPosition);
        assert.deepEqual(result.meta.config, { logoSize: 48, marginRight: 96, marginBottom: 96 });
        assert.equal(result.meta.alphaGain, 0.55, `${sample.fileName} expected conservative alpha gain`);
        assert.ok(String(result.meta.source).includes('catalog'), `${sample.fileName} source=${result.meta.source}`);
        assert.ok(
            Math.abs(result.meta.detection.processedSpatialScore) <= sample.maxSpatialResidual,
            `${sample.fileName} spatial residual=${result.meta.detection.processedSpatialScore}`
        );
        assert.ok(
            result.meta.detection.processedGradientScore <= sample.maxGradientResidual,
            `${sample.fileName} gradient residual=${result.meta.detection.processedGradientScore}`
        );
    }
});

test('processWatermarkImageData should cleanup residual edges on known 48px large-margin anchors', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const samples = [
        '4-3.png',
        '9-16.png'
    ];

    for (const fileName of samples) {
        const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples', fileName));
        const result = processWatermarkImageData(imageData, {
            alpha48,
            alpha96,
            adaptiveMode: 'never',
            getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
        });

        assert.equal(result.meta.applied, true, `${fileName} skipReason=${result.meta.skipReason}`);
        assert.deepEqual(result.meta.config, { logoSize: 48, marginRight: 96, marginBottom: 96 });
        assert.ok(
            String(result.meta.source).includes('+edge-cleanup'),
            `${fileName} expected known 48px residual edge cleanup, source=${result.meta.source}`
        );
        assert.ok(
            Math.abs(result.meta.detection.processedSpatialScore) <= 0.12,
            `${fileName} spatial residual=${result.meta.detection.processedSpatialScore}`
        );
        assert.ok(
            result.meta.detection.processedGradientScore <= 0.2,
            `${fileName} gradient residual=${result.meta.detection.processedGradientScore}`
        );
    }
});

test('processWatermarkImageData should flat-fill residual edges only on smooth known 48px backgrounds', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const width = 720;
    const height = 1456;
    const data = new Uint8ClampedArray(width * height * 4);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const offset = (y * width + x) * 4;
            data[offset] = 76;
            data[offset + 1] = Math.round(142 + y * 0.004);
            data[offset + 2] = 148;
            data[offset + 3] = 255;
        }
    }

    const imageData = { width, height, data };
    const position = { x: width - 96 - 48, y: height - 96 - 48, width: 48, height: 48 };
    applySyntheticWatermark(imageData, alpha48, position, 0.85);

    for (let row = 0; row < position.height; row++) {
        for (let col = 0; col < position.width; col++) {
            const alpha = alpha48[row * position.width + col];
            if (alpha < 0.02 || alpha > 0.55) continue;
            const offset = ((position.y + row) * width + position.x + col) * 4;
            data[offset] = Math.max(0, data[offset] - 34);
            data[offset + 1] = Math.max(0, data[offset + 1] - 34);
            data[offset + 2] = Math.max(0, data[offset + 2] - 34);
        }
    }

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.ok(
        String(result.meta.source).includes('+flat-fill'),
        `expected smooth residual to use flat-fill, source=${result.meta.source}`
    );
    const flatFillStage = result.meta.alphaAdjustmentStages?.find((stage) => stage.stage === 'known-48-flat-background-fill');
    assert.ok(flatFillStage, `expected flat-fill stage, stages=${JSON.stringify(result.meta.alphaAdjustmentStages)}`);
    assert.ok(
        flatFillStage.afterGradientScore <= flatFillStage.beforeGradientScore - 0.045,
        `expected flat-fill to suppress residual gradient, stage=${JSON.stringify(flatFillStage)}`
    );
});

test('processWatermarkImageData should allow strong fixed-core standard evidence on dark textured backgrounds', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/5-4.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(
        result.meta.position,
        { x: 1072, y: 848, width: 48, height: 48 },
        `expected canonical 48px anchor, got ${JSON.stringify(result.meta.position)} source=${result.meta.source}`
    );
    assert.equal(result.meta.alphaGain, 1);
    assert.ok(
        result.meta.detection.originalSpatialScore > 0.9,
        `expected strong original spatial evidence, got ${result.meta.detection.originalSpatialScore}`
    );
    assert.ok(
        result.meta.detection.processedSpatialScore < 0.22,
        `expected residual spatial to be accepted, got ${result.meta.detection.processedSpatialScore}`
    );
});

test('processWatermarkImageData should remove the 2816x1536 issue #68 watermark at the new 192px margin', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const alpha96NewMargin = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96_20260520.png')));
    const imageData = await decodeImageDataInNode(path.resolve('tests/fixtures/issue68-new-position.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        alpha96Variants: {
            '20260520': alpha96NewMargin
        },
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(
        result.meta.position,
        { x: 2528, y: 1248, width: 96, height: 96 }
    );
    assert.ok(
        result.meta.detection.processedGradientScore < 0.05,
        `expected residual gradient < 0.05, got ${result.meta.detection.processedGradientScore}, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should allow strong 2752x1536 new-margin alpha evidence through flat-background hard reject', async (t) => {
    const samplePath = path.resolve('D:/Project/sample-files/gemini-watermark/2026-06-09/2064208514779189248-source.png');
    try {
        await access(samplePath);
    } catch {
        t.skip('external 2752x1536 Gemini sample is not available');
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const alpha96NewMargin = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96_20260520.png')));
    const imageData = await decodeImageDataInNode(samplePath);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        alpha96Variants: {
            '20260520': alpha96NewMargin
        },
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size),
        locatedAggressiveRemoval: false
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(
        result.meta.position,
        { x: 2464, y: 1248, width: 96, height: 96 }
    );
    assert.equal(result.meta.alphaGain, 1);
    assert.ok(
        String(result.meta.source).includes('+flat-fill'),
        `expected smooth new-margin residual to use flat-fill, source=${result.meta.source}`
    );
    assert.ok(
        result.meta.alphaAdjustmentStages?.some((stage) => stage.stage === 'new-margin-96-flat-background-fill'),
        `expected new-margin flat-fill stage, stages=${JSON.stringify(result.meta.alphaAdjustmentStages)}`
    );
    assert.ok(
        Math.abs(result.meta.detection.processedSpatialScore) <= 0.03,
        `expected low residual spatial, got ${result.meta.detection.processedSpatialScore}, source=${result.meta.source}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore <= 0.11,
        `expected low residual gradient, got ${result.meta.detection.processedGradientScore}, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should allow strong 96px fixed-core evidence with slight negative overshoot', async (t) => {
    const samplePath = path.resolve('D:/Project/sample-files/gemini-watermark/2026-06-08/2064116984391405568-source.png');
    try {
        await access(samplePath);
    } catch {
        t.skip('external 1792x2400 Gemini sample is not available');
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(samplePath);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(result.meta.config, { logoSize: 96, marginRight: 64, marginBottom: 64 });
    assert.deepEqual(result.meta.position, { x: 1632, y: 2240, width: 96, height: 96 });
    assert.equal(result.meta.alphaGain, 1);
    assert.ok(
        result.meta.detection.originalSpatialScore >= 0.95 &&
            result.meta.detection.originalGradientScore >= 0.9,
        `expected strong original evidence, detection=${JSON.stringify(result.meta.detection)}`
    );
    assert.ok(
        result.meta.detection.processedSpatialScore < 0 &&
            Math.abs(result.meta.detection.processedSpatialScore) <= 0.52,
        `expected bounded negative overshoot, got ${result.meta.detection.processedSpatialScore}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore <= 0.16,
        `expected low residual gradient, got ${result.meta.detection.processedGradientScore}`
    );
    assert.equal(result.meta.selectionDebug?.hardReject, false);
});

test('processWatermarkImageData should keep 1696x2518 portrait sample on the full 96px anchor', async (t) => {
    const samplePath = path.resolve('D:/Project/sample-files/gemini-watermark/2026-06-09/2064204960823775232-source.png');
    try {
        await access(samplePath);
    } catch {
        t.skip('external 1696x2518 Gemini sample is not available');
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(samplePath);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size),
        locatedAggressiveRemoval: false
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(result.meta.config, { logoSize: 96, marginRight: 64, marginBottom: 64 });
    assert.deepEqual(result.meta.position, { x: 1536, y: 2358, width: 96, height: 96 });
    assert.equal(result.meta.source, 'standard');
    assert.equal(result.meta.alphaGain, 1);
    assert.equal(result.meta.selectionDebug?.usedCatalogVariant, false);
    assert.ok(
        result.meta.detection.originalSpatialScore >= 0.4 &&
            result.meta.detection.originalGradientScore >= 0.2,
        `expected strong original 96px evidence, detection=${JSON.stringify(result.meta.detection)}`
    );
    assert.ok(
        Math.abs(result.meta.detection.processedSpatialScore) <= 0.04 &&
            result.meta.detection.processedGradientScore <= 0.04,
        `expected low residual on full 96px anchor, detection=${JSON.stringify(result.meta.detection)}`
    );
});

test('processWatermarkImageData should prefer strong 96px evidence over a weak 48px large-margin crop', async (t) => {
    const samplePath = path.resolve('D:/Project/sample-files/gemini-watermark/2026-06-09/2064229579895083008-source.png');
    try {
        await access(samplePath);
    } catch {
        t.skip('external 1792x2390 Gemini sample is not available');
        return;
    }

    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(samplePath);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size),
        locatedAggressiveRemoval: false
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(result.meta.config, { logoSize: 96, marginRight: 64, marginBottom: 64 });
    assert.deepEqual(result.meta.position, { x: 1632, y: 2230, width: 96, height: 96 });
    assert.equal(result.meta.source, 'standard+gain');
    assert.equal(result.meta.alphaGain, 0.85);
    assert.equal(result.meta.selectionDebug?.usedCatalogVariant, false);
    assert.ok(
        result.meta.detection.originalSpatialScore >= 0.55 &&
            result.meta.detection.originalGradientScore >= 0.5,
        `expected strong original 96px evidence, detection=${JSON.stringify(result.meta.detection)}`
    );
    assert.ok(
        Math.abs(result.meta.detection.processedSpatialScore) <= 0.08 &&
            result.meta.detection.processedGradientScore <= 0.24,
        `expected bounded residual on full 96px anchor, detection=${JSON.stringify(result.meta.detection)}`
    );
    assert.equal(result.meta.detection.residualVisibility?.visible, false);
});

test('processWatermarkImageData should keep a 1024x1024 dark sample on the 48px large-margin anchor', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('tests/fixtures/gemini-1024-large-margin.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(result.meta.config, { logoSize: 48, marginRight: 96, marginBottom: 96 });
    assert.deepEqual(result.meta.position, { x: 880, y: 880, width: 48, height: 48 });
    assert.ok(
        result.meta.detection.originalSpatialScore >= 0.99 &&
            result.meta.detection.originalGradientScore >= 0.99,
        `expected strong original evidence, detection=${JSON.stringify(result.meta.detection)}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore <= 0.02,
        `expected low residual gradient, got ${result.meta.detection.processedGradientScore}, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should remove a near-official 895x1200 large-margin sample', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('tests/fixtures/gemini-near-official-895x1200-large-margin.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(result.meta.config, { logoSize: 48, marginRight: 96, marginBottom: 96 });
    assert.deepEqual(result.meta.position, { x: 751, y: 1056, width: 48, height: 48 });
    assert.ok(
        result.meta.detection.originalSpatialScore >= 0.99 &&
            result.meta.detection.originalGradientScore >= 0.99,
        `expected strong original evidence, detection=${JSON.stringify(result.meta.detection)}`
    );
    assert.ok(
        result.meta.detection.processedGradientScore <= 0.1,
        `expected low residual gradient, got ${result.meta.detection.processedGradientScore}, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should keep 20260520-5.png on the full 96px anchor', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260520-5.png'));

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(result.meta.applied, `skipReason=${result.meta.skipReason}`);
    assert.deepEqual(
        result.meta.position,
        { x: 2548, y: 1376, width: 96, height: 96 }
    );
    assert.ok(
        result.meta.detection.processedGradientScore < 0.1,
        `expected residual gradient < 0.1, got ${result.meta.detection.processedGradientScore}, source=${result.meta.source}`
    );
});

test('processWatermarkImageData should expose stage timings when debugTimings is enabled', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const alpha34 = interpolateAlphaMap(alpha96, 96, 34);
    const imageData = createPatternImageData(1024, 559);
    const truePosition = {
        x: 1024 - 24 - 34,
        y: 559 - 24 - 34,
        width: 34,
        height: 34
    };
    applySyntheticWatermark(imageData, alpha34, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        debugTimings: true,
        getAlphaMap: (size) => interpolateAlphaMap(alpha96, 96, size)
    });

    assert.ok(result.debugTimings, 'expected debugTimings to be present');
    assert.equal(typeof result.debugTimings.initialSelectionMs, 'number');
    assert.equal(typeof result.debugTimings.firstPassMetricsMs, 'number');
    assert.equal(typeof result.debugTimings.totalMs, 'number');
    assert.ok(result.debugTimings.totalMs >= result.debugTimings.initialSelectionMs);
});
