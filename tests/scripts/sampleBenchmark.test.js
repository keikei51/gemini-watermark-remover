import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';

import {
    buildCandidateRankingReport,
    buildSelectedCandidateDiagnostic,
    classifyFineAlphaSelectionReason,
    classifyBenchmarkCase,
    decodeImageDataInNode,
    listBenchmarkSampleAssets,
    loadSampleGoldManifest,
    summarizeCandidateRankingReport,
    summarizeBenchmarkResults
} from '../../scripts/sample-benchmark.js';
import { calculateAlphaMap } from '../../src/core/alphaMap.js';
import { interpolateAlphaMap } from '../../src/core/adaptiveDetector.js';
import {
    detectWatermarkConfig,
    resolveInitialStandardConfig
} from '../../src/core/watermarkConfig.js';
import { processWatermarkImageData } from '../../src/core/watermarkProcessor.js';

test('classifyBenchmarkCase should mark skipped expected Gemini sample as missed detection', () => {
    const result = classifyBenchmarkCase({
        expectedGemini: true,
        applied: false,
        skipReason: 'no-watermark-detected',
        fileName: 'expected-gemini.png'
    });

    assert.equal(result.status, 'fail');
    assert.equal(result.bucket, 'missed-detection');
});

test('classifyBenchmarkCase should separate weak suppression from residual edge cases', () => {
    const weakSuppression = classifyBenchmarkCase({
        expectedGemini: true,
        applied: true,
        residualScore: 0.31,
        suppressionGain: 0.18,
        decisionTier: 'validated-match',
        fileName: 'weak.png'
    });
    const residualEdge = classifyBenchmarkCase({
        expectedGemini: true,
        applied: true,
        residualScore: 0.31,
        suppressionGain: 0.36,
        decisionTier: 'validated-match',
        fileName: 'edge.png'
    });

    assert.equal(weakSuppression.bucket, 'weak-suppression');
    assert.equal(residualEdge.bucket, 'residual-edge');
});

test('classifyBenchmarkCase should treat changed non-Gemini region as false positive', () => {
    const result = classifyBenchmarkCase({
        expectedGemini: false,
        applied: true,
        changedRatio: 0.08,
        avgAbsoluteDeltaPerChannel: 3.2,
        fileName: '16-9.jpg'
    });

    assert.equal(result.status, 'fail');
    assert.equal(result.bucket, 'false-positive');
});

test('listBenchmarkSampleAssets should include every primary sample image under the sample directory', async () => {
    const sampleDir = path.resolve('src/assets/samples');
    const items = await listBenchmarkSampleAssets(sampleDir);

    assert.ok(items.length > 0, 'expected benchmark sample enumeration to find sample images');
    assert.ok(
        items.some((item) => item.expectedGemini === true),
        'expected directory-driven samples to include supported Gemini fixtures'
    );
    assert.ok(items.every((item) => !item.fileName.includes('-fix.')), 'expected fix snapshots to be excluded');
    assert.ok(items.every((item) => !item.fileName.includes('-after.')), 'expected derived after snapshots to be excluded');
    assert.equal(items.some((item) => item.fileName === '1-1.png'), true);
    assert.equal(items.some((item) => item.fileName === '9-16.png'), true);
    assert.equal(items.find((item) => item.fileName === '2-3.png')?.expectedGemini, true);
    assert.equal(items.find((item) => item.fileName === '8-1.png')?.expectedGemini, true);
    assert.deepEqual(
        items.find((item) => item.fileName === '16-9.png')?.gold?.expectedAnchor,
        { logoSize: 48, marginRight: 96, marginBottom: 96 }
    );
});

test('loadSampleGoldManifest should read human-maintained sample expectations', async () => {
    const manifest = await loadSampleGoldManifest(path.resolve('src/assets/samples'));

    assert.equal(manifest.version, 1);
    assert.equal(manifest.samples['8-1.png'].shouldProcess, true);
    assert.deepEqual(
        manifest.samples['20260520-3.png'].expectedAnchor,
        { logoSize: 96, marginRight: 192, marginBottom: 192 }
    );
});

test('summarizeBenchmarkResults should aggregate pass fail and bucket counts', () => {
    const summary = summarizeBenchmarkResults([
        {
            fileName: 'alpha-adjusted.png',
            classification: { status: 'pass', bucket: 'pass' },
            candidateRankingSummary: {
                topAcceptedMatchesSelectedAnchor: true,
                topAcceptedMatchesSelectedAlpha: false,
                selectedAnchorRank: 1,
                selectedExactRank: null,
                earlyAcceptRank: 2
            },
            selectedCandidateDiagnostic: {
                matchesExpectedAnchor: true,
                matchesExpectedAlpha: true,
                fineAlphaNeighborhood: [
                    { selected: false, alphaGain: 0.6 },
                    { selected: true, alphaGain: 0.64 }
                ],
                fineAlphaSelectedRank: 2,
                fineAlphaTopAlphaGain: 0.6,
                fineAlphaTopDelta: -0.04,
                fineAlphaTopDeltaBucket: 'micro-lower',
                fineAlphaSelectedAlphaType: 'fine',
                fineAlphaTopAlphaType: 'discrete',
                fineAlphaSelectionReason: 'dark-catalog-fine-alpha',
                alphaGain: 0.64,
                residual: { score: 0.12 },
                damage: { penalty: 0.02 },
                alphaAdjustmentStages: [
                    { stage: 'dark-catalog-fine-alpha' },
                    { stage: 'weak-positive-residual-fine-alpha' }
                ]
            }
        },
        { classification: { status: 'fail', bucket: 'missed-detection' } },
        { classification: { status: 'fail', bucket: 'missed-detection' } },
        { classification: { status: 'fail', bucket: 'false-positive' } }
    ]);

    assert.equal(summary.total, 4);
    assert.equal(summary.passCount, 1);
    assert.equal(summary.failCount, 3);
    assert.equal(summary.buckets['missed-detection'], 2);
    assert.equal(summary.buckets['false-positive'], 1);
    assert.equal(summary.candidateRanking.topAcceptedMatchesSelectedAnchor, 1);
    assert.equal(summary.candidateRanking.topAcceptedMatchesSelectedAlpha, 0);
    assert.equal(summary.candidateRanking.selectedAnchorInTop, 1);
    assert.equal(summary.candidateRanking.selectedExactInTop, 0);
    assert.equal(summary.candidateRanking.earlyAcceptInTop, 1);
    assert.equal(summary.candidateRanking.selectedFinalDiagnosticCount, 1);
    assert.equal(summary.candidateRanking.selectedFinalExpectedAnchorCount, 1);
    assert.equal(summary.candidateRanking.selectedFinalExpectedAlphaCount, 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaNeighborhoodCount, 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaTopCount, 0);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaSelectedRankCounts['2'], 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaSelectionReasons['dark-catalog-fine-alpha'], 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaSelectedAlphaTypes.fine, 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaTopDeltaBuckets['micro-lower'], 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaNonTopReasonCounts['dark-catalog-fine-alpha'], 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaNonTopSelectedAlphaTypes.fine, 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaNonTopDeltaBuckets['micro-lower'], 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaNonTopWithAdjustmentCount, 1);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaNonTopWithoutAdjustmentCount, 0);
    assert.deepEqual(summary.candidateRanking.selectedFinalFineAlphaNonTopSamples, [
        {
            fileName: 'alpha-adjusted.png',
            selectedRank: 2,
            selectedAlphaGain: 0.64,
            topAlphaGain: 0.6,
            alphaDelta: -0.04,
            alphaDeltaBucket: 'micro-lower',
            reason: 'dark-catalog-fine-alpha',
            selectedAlphaType: 'fine',
            topAlphaType: 'discrete',
            selectedResidualScore: 0.12,
            topResidualScore: null,
            residualScoreDelta: null,
            selectedDamagePenalty: 0.02,
            topDamagePenalty: null,
            topDamageSafe: null,
            topAccepted: null,
            significantDeltaConcern: null,
            alphaAdjustmentStages: ['dark-catalog-fine-alpha', 'weak-positive-residual-fine-alpha']
        }
    ]);
    assert.equal(summary.candidateRanking.selectedFinalFineAlphaSignificantDeltaCount, 0);
    assert.deepEqual(summary.candidateRanking.selectedFinalFineAlphaSignificantDeltaConcerns, {});
    assert.deepEqual(summary.candidateRanking.selectedFinalFineAlphaSignificantDeltaSamples, []);
    assert.equal(summary.candidateRanking.selectedFinalAlphaAdjustmentCount, 1);
    assert.equal(summary.candidateRanking.selectedFinalAlphaAdjustmentStages['dark-catalog-fine-alpha'], 1);
    assert.equal(summary.candidateRanking.selectedFinalAlphaAdjustmentStages['weak-positive-residual-fine-alpha'], 1);
    assert.deepEqual(
        summary.candidateRanking.selectedFinalAlphaAdjustmentStageSamples['dark-catalog-fine-alpha'],
        ['alpha-adjusted.png']
    );
    assert.deepEqual(
        summary.candidateRanking.selectedFinalAlphaAdjustmentStageSamples['weak-positive-residual-fine-alpha'],
        ['alpha-adjusted.png']
    );
});

test('classifyFineAlphaSelectionReason should separate production stages from report preference drift', () => {
    assert.equal(
        classifyFineAlphaSelectionReason({
            alphaGain: 0.64,
            fineAlphaSelectedRank: 3,
            alphaAdjustmentStages: [{ stage: 'weak-positive-residual-fine-alpha' }]
        }),
        'weak-positive-residual-fine-alpha'
    );
    assert.equal(
        classifyFineAlphaSelectionReason({
            alphaGain: 0.6,
            fineAlphaSelectedRank: 4,
            alphaAdjustmentStages: []
        }),
        'production-kept-standard-alpha'
    );
    assert.equal(
        classifyFineAlphaSelectionReason({
            alphaGain: 0.64,
            fineAlphaSelectedRank: 4,
            alphaAdjustmentStages: []
        }),
        'report-prefers-micro-alpha'
    );
    assert.equal(
        classifyFineAlphaSelectionReason({
            alphaGain: 1,
            fineAlphaSelectedRank: 1,
            alphaAdjustmentStages: []
        }),
        'direct-discrete-alpha'
    );
});

test('summarizeCandidateRankingReport should expose selected and expected candidate ranks', () => {
    const summary = summarizeCandidateRankingReport([
        {
            accepted: true,
            earlyAccept: false,
            matchesSelectedAnchor: false,
            matchesSelectedAlpha: false,
            matchesExpectedAnchor: true,
            matchesExpectedAlpha: true
        },
        {
            accepted: true,
            earlyAccept: true,
            matchesSelectedAnchor: true,
            matchesSelectedAlpha: false,
            matchesExpectedAnchor: true,
            matchesExpectedAlpha: true
        },
        {
            accepted: false,
            earlyAccept: false,
            matchesSelectedAnchor: true,
            matchesSelectedAlpha: true,
            matchesExpectedAnchor: false,
            matchesExpectedAlpha: false
        }
    ]);

    assert.equal(summary.total, 3);
    assert.equal(summary.acceptedCount, 2);
    assert.equal(summary.earlyAcceptRank, 2);
    assert.equal(summary.selectedAnchorRank, 2);
    assert.equal(summary.selectedExactRank, 3);
    assert.equal(summary.expectedAnchorRank, 1);
    assert.equal(summary.expectedAlphaRank, 1);
    assert.equal(summary.topAcceptedMatchesSelectedAnchor, false);
});

test('classifyBenchmarkCase should fail expected Gemini samples with the wrong anchor or alpha', () => {
    const anchorMismatch = classifyBenchmarkCase({
        expectedGemini: true,
        applied: true,
        decisionTier: 'direct-match',
        residualScore: 0.01,
        actualAnchor: { logoSize: 48, marginRight: 32, marginBottom: 32 },
        expectedAnchor: { logoSize: 48, marginRight: 96, marginBottom: 96 }
    });
    const alphaMismatch = classifyBenchmarkCase({
        expectedGemini: true,
        applied: true,
        decisionTier: 'direct-match',
        residualScore: 0.01,
        actualAnchor: { logoSize: 48, marginRight: 96, marginBottom: 96 },
        expectedAnchor: { logoSize: 48, marginRight: 96, marginBottom: 96 },
        alphaGain: 0.4,
        expectedAlphaGain: { min: 0.5, max: 1 }
    });

    assert.equal(anchorMismatch.bucket, 'anchor-mismatch');
    assert.equal(alphaMismatch.bucket, 'alpha-mismatch');
});

test('decodeImageDataInNode should decode sample assets without launching a browser', async () => {
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/1-1.png'));

    assert.equal(imageData.width, 1024);
    assert.equal(imageData.height, 1024);
    assert.equal(imageData.data.length, 1024 * 1024 * 4);
});

test('buildCandidateRankingReport should expose sorted top candidate diagnostics', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260608-3.png'));
    const initialConfig = resolveInitialStandardConfig({
        imageData,
        defaultConfig: detectWatermarkConfig(imageData.width, imageData.height),
        alpha48,
        alpha96
    });

    const candidates = buildCandidateRankingReport({
        imageData,
        initialConfig,
        alpha48,
        alpha96,
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size),
        limit: 8
    });

    assert.ok(candidates.length > 0, 'expected top candidate diagnostics');
    assert.ok(candidates.length <= 8);
    assert.equal(candidates[0].accepted, true);
    const largeMarginCandidate = candidates.find((candidate) => (
        candidate.family === 'known-current-variant' &&
        candidate.watermarkSize === 48 &&
        candidate.marginRight === 96 &&
        candidate.marginBottom === 96
    ));

    assert.ok(largeMarginCandidate, 'expected top diagnostics to include the 48px large-margin catalog candidate');
    assert.equal(largeMarginCandidate.catalogMetadata.sourcePriority, 1);
    assert.equal(largeMarginCandidate.originalEvidence.tier, 'strong');
    assert.equal(typeof largeMarginCandidate.earlyAccept, 'boolean');
    assert.ok(Array.isArray(candidates[0].rankingKey));
    assert.equal(typeof candidates[0].residual.score, 'number');
    assert.equal(typeof candidates[0].damage.safe, 'boolean');
});

test('buildSelectedCandidateDiagnostic should score the final processed fine-alpha result', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260608-5.png'));
    const initialConfig = resolveInitialStandardConfig({
        imageData,
        defaultConfig: detectWatermarkConfig(imageData.width, imageData.height),
        alpha48,
        alpha96
    });
    const processed = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        adaptiveMode: 'never',
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    const diagnostic = buildSelectedCandidateDiagnostic({
        originalImageData: imageData,
        processedImageData: processed.imageData,
        meta: processed.meta,
        initialConfig,
        alpha48,
        alpha96,
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size),
        expectedAnchor: { logoSize: 48, marginRight: 96, marginBottom: 96 },
        expectedAlphaGain: { min: 0.5, max: 1 }
    });

    assert.ok(diagnostic, 'expected selected final diagnostic');
    assert.equal(diagnostic.family, 'selected-final');
    assert.equal(diagnostic.matchesExpectedAnchor, true);
    assert.equal(diagnostic.matchesExpectedAlpha, true);
    assert.equal(diagnostic.alphaGain, processed.meta.alphaGain);
    assert.equal(typeof diagnostic.fineAlphaSelectedRank, 'number');
    assert.equal(typeof diagnostic.fineAlphaTopAlphaGain, 'number');
    assert.equal(typeof diagnostic.fineAlphaTopDelta, 'number');
    assert.equal(typeof diagnostic.fineAlphaTopDeltaBucket, 'string');
    assert.equal(typeof diagnostic.fineAlphaSelectedAlphaType, 'string');
    assert.equal(typeof diagnostic.fineAlphaTopAlphaType, 'string');
    assert.equal(typeof diagnostic.fineAlphaSelectionReason, 'string');
    assert.ok(Array.isArray(diagnostic.rankingKey));
    assert.deepEqual(diagnostic.alphaAdjustmentStages, processed.meta.alphaAdjustmentStages);
    assert.ok(Array.isArray(diagnostic.fineAlphaNeighborhood));
    assert.ok(
        diagnostic.fineAlphaNeighborhood.some((candidate) => candidate.selected === true),
        'expected fine alpha neighborhood to mark the selected final alpha'
    );
    assert.equal(typeof diagnostic.residual.score, 'number');
});

test('buildSelectedCandidateDiagnostic should preserve legacy alpha map variants from selected meta', async () => {
    const alpha48 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_48.png')));
    const alpha96 = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96.png')));
    const alpha96NewMargin = calculateAlphaMap(await decodeImageDataInNode(path.resolve('src/assets/bg_96_20260520.png')));
    const imageData = await decodeImageDataInNode(path.resolve('src/assets/samples/20260520-3.png'));
    const initialConfig = resolveInitialStandardConfig({
        imageData,
        defaultConfig: detectWatermarkConfig(imageData.width, imageData.height),
        alpha48,
        alpha96
    });
    const processed = processWatermarkImageData(imageData, {
        alpha48,
        alpha96,
        alpha96Variants: {
            '20260520': alpha96NewMargin
        },
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size)
    });

    const diagnostic = buildSelectedCandidateDiagnostic({
        originalImageData: imageData,
        processedImageData: processed.imageData,
        meta: processed.meta,
        initialConfig,
        alpha48,
        alpha96,
        alpha96Variants: {
            '20260520': alpha96NewMargin
        },
        getAlphaMap: (size) => size === 48 ? alpha48 : interpolateAlphaMap(alpha96, 96, size),
        expectedAnchor: { logoSize: 96, marginRight: 192, marginBottom: 192 },
        expectedAlphaGain: { min: 0.5, max: 1 }
    });

    assert.ok(diagnostic, 'expected selected final diagnostic');
    assert.equal(diagnostic.alphaMapProfile, '96-20260520');
    assert.equal(diagnostic.fineAlphaNeighborhood.some((candidate) => candidate.selected === true), true);
});
