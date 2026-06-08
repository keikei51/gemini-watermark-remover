import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';

import { calculateAlphaMap } from '../../src/core/alphaMap.js';
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
        alpha96
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
        alpha96
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

test('processWatermarkImageData should skip off-catalog adaptive-only positions in fixed-core mode', () => {
    const alpha96 = createSyntheticAlphaMap(96);
    const alpha48 = interpolateAlphaMap(alpha96, 96, 48);
    const imageData = createPatternImageData(320, 320);
    const truePosition = { x: 320 - 36 - 48, y: 320 - 20 - 48, width: 48, height: 48 };
    applySyntheticWatermark(imageData, alpha48, truePosition, 1);

    const result = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
    });

    assert.equal(result.meta.applied, false);
    assert.equal(result.meta.source, 'skipped');
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

test('processWatermarkImageData should avoid conservative fallback on debug1-source download sample', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/debug1-source.png'));

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
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/debug2-source.png'));

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
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    assert.equal(result.meta.applied, true, `skipReason=${result.meta.skipReason}`);
    assert.equal(result.meta.passCount, 1, `passCount=${result.meta.passCount}`);
    assert.equal(
        result.meta.alphaGain,
        0.7,
        `expected fine alpha around the visually balanced weak-alpha point, got ${result.meta.alphaGain}`
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

test('processWatermarkImageData should allow strong fixed-core standard evidence on dark textured backgrounds', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/5-4.webp'));

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
