import { removeWatermark } from './blendModes.js';
import {
    computeRegionGradientCorrelation,
    computeRegionSpatialCorrelation,
    warpAlphaMap
} from './adaptiveDetector.js';
import {
    createAlphaGradientMask,
    getAlphaGradientWeight
} from './alphaGradientMask.js';
import {
    calculateNearBlackRatio,
    scoreRegion,
    selectInitialCandidate
} from './candidateSelector.js';
import {
    assessAlphaBandHalo,
    assessRemovalDiffArtifacts,
    assessWatermarkResidualVisibility
} from './restorationMetrics.js';
import { createSelectionDebugSummary } from './selectionDebug.js';
import {
    calculateWatermarkPosition,
    detectWatermarkConfig,
    resolveInitialStandardConfig
} from './watermarkConfig.js';

const RESIDUAL_RECALIBRATION_THRESHOLD = 0.5;
const MIN_SUPPRESSION_FOR_SKIP_RECALIBRATION = 0.18;
const MIN_RECALIBRATION_SCORE_DELTA = 0.18;
const MAX_NEAR_BLACK_RATIO_INCREASE = 0.05;
const OUTLINE_REFINEMENT_THRESHOLD = 0.42;
const OUTLINE_REFINEMENT_MIN_GAIN = 1.2;
const SUBPIXEL_REFINE_SHIFTS = [-0.25, 0, 0.25];
const SUBPIXEL_REFINE_SCALES = [0.99, 1, 1.01];
const ALPHA_PARAMETER_GROUPS = Object.freeze([
    { name: 'gemini-weak-alpha-202606', alphaGain: 0.6, standardPriority: true },
    { name: 'gemini-standard-alpha', alphaGain: 1, standardPriority: true },
    { name: 'gemini-strong-alpha-202606', alphaGain: 1.15 },
    { name: 'gemini-strong-alpha-high-202606', alphaGain: 1.3 },
    { name: 'weak-alpha-extra-conservative', alphaGain: 0.45 },
    { name: 'weak-alpha-light', alphaGain: 0.7 },
    { name: 'weak-alpha-mid', alphaGain: 0.85 },
    { name: 'weak-alpha-conservative', alphaGain: 0.55 }
]);
const ALPHA_GAIN_CANDIDATES = ALPHA_PARAMETER_GROUPS.map((group) => group.alphaGain);
const LOCATED_AGGRESSIVE_ALPHA_GAINS = Object.freeze([0.85, 1, 1.15, 1.3, 1.45, 1.7, 2, 2.4]);
const ENABLE_VISUAL_POST_PROCESSING = false;
const CATALOG_DARK_ALPHA_GAIN_CANDIDATES = Object.freeze([0.9, 0.85, 0.8, 0.95, 0.7, 0.6]);
const STANDARD_ALPHA_PRIORITY_GAINS = ALPHA_PARAMETER_GROUPS
    .filter((group) => group.standardPriority === true)
    .map((group) => group.alphaGain);
const PREVIEW_EDGE_CLEANUP_MAX_SIZE = 40;
const KNOWN_48_EDGE_CLEANUP_MIN_SIZE = 40;
const KNOWN_48_EDGE_CLEANUP_MAX_SIZE = 56;
const KNOWN_48_EDGE_CLEANUP_MIN_GRADIENT = 0.22;
const KNOWN_48_EDGE_CLEANUP_MAX_ABS_SPATIAL = 0.55;
const KNOWN_48_EDGE_CLEANUP_MIN_GRADIENT_IMPROVEMENT = 0.015;
const KNOWN_48_EDGE_CLEANUP_MAX_SPATIAL_DRIFT = 0.06;
const V2_SMALL_EDGE_CLEANUP_SIZE = 36;
const V2_SMALL_EDGE_CLEANUP_SIZE_TOLERANCE = 2;
const V2_SMALL_EDGE_CLEANUP_MIN_GRADIENT = 0.22;
const V2_SMALL_EDGE_CLEANUP_MAX_ABS_SPATIAL = 0.08;
const V2_SMALL_EDGE_CLEANUP_MIN_GRADIENT_IMPROVEMENT = 0.025;
const V2_SMALL_EDGE_CLEANUP_MAX_SPATIAL_DRIFT = 0.035;
const KNOWN_48_FLAT_FILL_MIN_GRADIENT = 0.28;
const KNOWN_48_FLAT_FILL_MAX_BACKGROUND_STD = 6;
const KNOWN_48_FLAT_FILL_MIN_GRADIENT_IMPROVEMENT = 0.045;
const KNOWN_48_FLAT_FILL_SECOND_PASS_MIN_GRADIENT_IMPROVEMENT = 0.025;
const KNOWN_48_FLAT_FILL_MAX_APPLIED_PASSES = 2;
const KNOWN_48_FLAT_FILL_MAX_SPATIAL_DRIFT = 0.12;
const KNOWN_48_FLAT_FILL_MAX_ACCEPTED_ABS_SPATIAL = 0.38;
const KNOWN_48_FLAT_FILL_PAD = 10;
const KNOWN_48_FLAT_FILL_OUTSIDE_ALPHA_MAX = 0.012;
const KNOWN_48_FLAT_FILL_PRESETS = Object.freeze([
    { name: 'edge', minAlpha: 0.012, maxAlpha: 0.5, strength: 0.9 },
    { name: 'wide', minAlpha: 0.008, maxAlpha: 0.99, strength: 0.92 },
    { name: 'hard', minAlpha: 0.004, maxAlpha: 0.99, strength: 1 }
]);
const NEW_MARGIN_96_FLAT_FILL_MIN_GRADIENT = 0.12;
const NEW_MARGIN_96_FLAT_FILL_MAX_BACKGROUND_STD = 7;
const NEW_MARGIN_96_FLAT_FILL_MIN_GRADIENT_IMPROVEMENT = 0.025;
const NEW_MARGIN_96_FLAT_FILL_MAX_SPATIAL_DRIFT = 0.08;
const NEW_MARGIN_96_FLAT_FILL_MAX_ACCEPTED_ABS_SPATIAL = 0.22;
const NEW_MARGIN_96_FLAT_FILL_PRESETS = Object.freeze([
    { name: 'edge', minAlpha: 0.006, maxAlpha: 0.45, strength: 0.75 }
]);
const KNOWN_48_LUMA_EDGE_MIN_GRADIENT = 0.2;
const KNOWN_48_LUMA_EDGE_MIN_GRADIENT_IMPROVEMENT = 0.02;
const KNOWN_48_LUMA_EDGE_MAX_SPATIAL_DRIFT = 0.04;
const KNOWN_48_LUMA_EDGE_MAX_ACCEPTED_ABS_SPATIAL = 0.38;
const KNOWN_48_LUMA_EDGE_PRESETS = Object.freeze([
    { name: 'soft', minAlpha: 0.012, maxAlpha: 0.7, referenceAlphaMax: 0.025, radius: 2, strength: 0.28, colorSigma: 34, maxDelta: 22 },
    { name: 'mid', minAlpha: 0.012, maxAlpha: 0.7, referenceAlphaMax: 0.04, radius: 3, strength: 0.42, colorSigma: 34, maxDelta: 32 },
    { name: 'wide', minAlpha: 0.012, maxAlpha: 0.7, referenceAlphaMax: 0.055, radius: 4, strength: 0.48, colorSigma: 34, maxDelta: 40 }
]);
const PREVIEW_EDGE_CLEANUP_SPATIAL_THRESHOLD = 0.08;
const PREVIEW_EDGE_CLEANUP_GRADIENT_THRESHOLD = 0.1;
const PREVIEW_EDGE_CLEANUP_MIN_GRADIENT_IMPROVEMENT = 0.03;
const PREVIEW_EDGE_CLEANUP_MAX_SPATIAL_DRIFT = 0.04;
const PREVIEW_EDGE_CLEANUP_MAX_APPLIED_PASSES = 3;
const PREVIEW_EDGE_CLEANUP_FINE_GRADIENT_THRESHOLD = 0.16;
const PREVIEW_EDGE_CLEANUP_FINE_MIN_GRADIENT_IMPROVEMENT = 0.005;
const PREVIEW_EDGE_CLEANUP_HALO_RELAXED_MIN_GRADIENT_IMPROVEMENT = 0.01;
const PREVIEW_EDGE_CLEANUP_HALO_WEIGHT = 0.02;
const PREVIEW_EDGE_CLEANUP_MIN_HALO_REDUCTION = 1.5;
const PREVIEW_EDGE_CLEANUP_STRONG_HALO_THRESHOLD = 4;
const PREVIEW_EDGE_CLEANUP_HALO_SPATIAL_THRESHOLD = 0.18;
const PREVIEW_EDGE_CLEANUP_PRESETS = Object.freeze([
    { minAlpha: 0.02, maxAlpha: 0.45, radius: 2, strength: 0.7, outsideAlphaMax: 0.05 },
    { minAlpha: 0.05, maxAlpha: 0.55, radius: 3, strength: 0.7, outsideAlphaMax: 0.08 },
    { minAlpha: 0.1, maxAlpha: 0.7, radius: 3, strength: 0.8, outsideAlphaMax: 0.12 },
    { minAlpha: 0.01, maxAlpha: 0.35, radius: 4, strength: 1.4, outsideAlphaMax: 0.05 }
]);
const PREVIEW_EDGE_CLEANUP_STRONG_GRADIENT_THRESHOLD = 0.45;
const PREVIEW_EDGE_CLEANUP_AGGRESSIVE_PRESETS = Object.freeze([
    {
        minAlpha: 0.01,
        maxAlpha: 0.55,
        radius: 2,
        strength: 1.3,
        outsideAlphaMax: 0.05,
        minGradientImprovement: 0.12,
        maxSpatialDrift: 0.18,
        maxAcceptedSpatial: 0.18
    }
]);
const LOCATED_AGGRESSIVE_EDGE_PRESETS = Object.freeze([
    { minAlpha: 0.004, maxAlpha: 0.99, radius: 2, strength: 0.85, outsideAlphaMax: 0.08 },
    { minAlpha: 0.004, maxAlpha: 0.99, radius: 3, strength: 1.15, outsideAlphaMax: 0.12 },
    { minAlpha: 0.004, maxAlpha: 0.99, radius: 5, strength: 1.45, outsideAlphaMax: 0.18 },
    { minAlpha: 0.02, maxAlpha: 0.99, radius: 6, strength: 1.8, outsideAlphaMax: 0.25 }
]);
const PREVIEW_BACKGROUND_CLEANUP_MAX_SIZE = 52;
const PREVIEW_BACKGROUND_CLEANUP_MIN_RESIDUAL = 0.3;
const PREVIEW_BACKGROUND_CLEANUP_MAX_BORDER_STD = 24;
const PREVIEW_BACKGROUND_CLEANUP_PAD = 8;
const PREVIEW_BACKGROUND_CLEANUP_PRIOR_RADIUS = 10;
const OVER_SUBTRACTION_SPATIAL_THRESHOLD = -0.25;
const OVER_SUBTRACTION_GRADIENT_THRESHOLD = 0.35;
const OVER_SUBTRACTION_MIN_ABS_SPATIAL_IMPROVEMENT = 0.08;
const OVER_SUBTRACTION_MIN_GRADIENT_IMPROVEMENT = 0.08;
const OVER_SUBTRACTION_FINE_ALPHA_STEP = 0.02;
const OVER_SUBTRACTION_FINE_ALPHA_WINDOW = 0.04;
const WEAK_ALPHA_FINE_TUNE_MIN_ORIGINAL_SPATIAL = 0.45;
const WEAK_ALPHA_FINE_TUNE_MIN_POSITIVE_RESIDUAL = 0.05;
const WEAK_ALPHA_FINE_TUNE_MIN_ABS_SPATIAL_IMPROVEMENT = 0.04;
const WEAK_ALPHA_FINE_TUNE_MAX_GRADIENT_INCREASE = 0.08;
const STRONG_POSITIVE_FINE_TUNE_MIN_ORIGINAL_SPATIAL = 0.75;
const STRONG_POSITIVE_FINE_TUNE_MIN_ORIGINAL_GRADIENT = 0.65;
const STRONG_POSITIVE_FINE_TUNE_MIN_CURRENT_SPATIAL = 0.3;
const STRONG_POSITIVE_FINE_TUNE_MID_GAINS = Object.freeze([0.7, 0.85]);
const STRONG_POSITIVE_FINE_TUNE_EXTRA_GAINS = Object.freeze([0.95, 1]);
const STRONG_POSITIVE_FINE_TUNE_MAX_ACCEPTED_SPATIAL = 0.12;
const STRONG_POSITIVE_FINE_TUNE_MAX_ACCEPTED_GRADIENT = 0.25;
const STRONG_POSITIVE_FINE_TUNE_MAX_POSITIVE_HALO_LUM = 4;
const CATALOG_ALPHA_DARK_FINE_TUNE_MIN_ORIGINAL_SPATIAL = 0.6;
const CATALOG_ALPHA_DARK_FINE_TUNE_MIN_ORIGINAL_GRADIENT = 0.45;
const CATALOG_ALPHA_DARK_FINE_TUNE_MAX_NEGATIVE_RESIDUAL = -0.12;
const CATALOG_ALPHA_DARK_FINE_TUNE_MAX_GRADIENT_INCREASE = 0.12;
const SMALL_PREVIEW_REFINEMENT_MAX_SIZE = 40;
const SMALL_PREVIEW_REFINEMENT_MAX_REFINED_SIZE = 56;
const SMALL_PREVIEW_REFINEMENT_MIN_ABS_SPATIAL_IMPROVEMENT = 0.03;
const SMALL_PREVIEW_REFINEMENT_MIN_GRADIENT_IMPROVEMENT = 0.03;
const SMALL_PREVIEW_REFINEMENT_MAX_SOURCE_SIZE = 32;
const SMALL_PREVIEW_REFINEMENT_MAX_ORIGINAL_GRADIENT = 0.15;
const SMALL_PREVIEW_REFINEMENT_MIN_CURRENT_SPATIAL = 0.04;
const SMALL_PREVIEW_REFINEMENT_MAX_CURRENT_GRADIENT = 0.08;
const FIRST_PASS_SIGN_FLIP_GRADIENT_THRESHOLD = 0.08;
const FIRST_PASS_SIGN_FLIP_MIN_GRADIENT_DROP = 0.2;
const SMALL_ANCHOR_RELOCATION_MIN_SIZE = 40;
const SMALL_ANCHOR_RELOCATION_MAX_SIZE = 56;
const SMALL_ANCHOR_RELOCATION_SIZE_DELTA = 5;
const SMALL_ANCHOR_RELOCATION_MARGIN_DELTA = 10;
const SMALL_ANCHOR_RELOCATION_MIN_CURRENT_GRADIENT = 0.24;
const SMALL_ANCHOR_RELOCATION_MAX_ACCEPTED_SPATIAL = 0.14;
const SMALL_ANCHOR_RELOCATION_MAX_ACCEPTED_GRADIENT = 0.24;
const SMALL_ANCHOR_RELOCATION_MIN_ORIGINAL_SPATIAL = 0.32;
const SMALL_ANCHOR_RELOCATION_MIN_SUPPRESSION_GAIN = 0.22;

function nowMs() {
    if (typeof globalThis.performance?.now === 'function') {
        return globalThis.performance.now();
    }
    return Date.now();
}

function cloneImageData(imageData) {
    if (typeof ImageData !== 'undefined' && imageData instanceof ImageData) {
        return new ImageData(
            new Uint8ClampedArray(imageData.data),
            imageData.width,
            imageData.height
        );
    }

    return {
        width: imageData.width,
        height: imageData.height,
        data: new Uint8ClampedArray(imageData.data)
    };
}

function normalizeMetaPosition(position) {
    if (!position) return null;

    const { x, y, width, height } = position;
    if (![x, y, width, height].every((value) => Number.isFinite(value))) {
        return null;
    }

    return { x, y, width, height };
}

function normalizeMetaConfig(config) {
    if (!config) return null;

    const { logoSize, marginRight, marginBottom } = config;
    if (![logoSize, marginRight, marginBottom].every((value) => Number.isFinite(value))) {
        return null;
    }

    return {
        logoSize,
        marginRight,
        marginBottom,
        ...(typeof config.alphaVariant === 'string' && config.alphaVariant.length > 0
            ? { alphaVariant: config.alphaVariant }
            : {})
    };
}

function createWatermarkMeta({
    position = null,
    config = null,
    adaptiveConfidence = null,
    originalSpatialScore = null,
    originalGradientScore = null,
    processedSpatialScore = null,
    processedGradientScore = null,
    suppressionGain = null,
    residualVisibility = null,
    templateWarp = null,
    alphaGain = 1,
    passCount = 0,
    attemptedPassCount = 0,
    passStopReason = null,
    passes = null,
    source = 'standard',
    decisionTier = null,
    applied = true,
    skipReason = null,
    subpixelShift = null,
    selectionDebug = null,
    alphaAdjustmentStages = null
} = {}) {
    const normalizedPosition = normalizeMetaPosition(position);

    return {
        applied,
        skipReason: applied ? null : skipReason,
        size: normalizedPosition ? normalizedPosition.width : null,
        position: normalizedPosition,
        config: normalizeMetaConfig(config),
        detection: {
            adaptiveConfidence,
            originalSpatialScore,
            originalGradientScore,
            processedSpatialScore,
            processedGradientScore,
            suppressionGain,
            residualVisibility
        },
        templateWarp: templateWarp ?? null,
        alphaGain,
        passCount,
        attemptedPassCount,
        passStopReason,
        passes: Array.isArray(passes) ? passes : null,
        // decisionTier is the normalized contract used by UI and attribution.
        // source remains as a verbose execution trace for debugging/tests.
        source,
        decisionTier,
        subpixelShift: subpixelShift ?? null,
        selectionDebug,
        alphaAdjustmentStages: Array.isArray(alphaAdjustmentStages) ? alphaAdjustmentStages : null
    };
}

function shouldRecalibrateAlphaStrength({ originalScore, processedScore, suppressionGain }) {
    return originalScore >= 0.6 &&
        processedScore >= RESIDUAL_RECALIBRATION_THRESHOLD &&
        suppressionGain <= MIN_SUPPRESSION_FOR_SKIP_RECALIBRATION;
}

function shouldStopAfterFirstPass({
    originalSpatialScore,
    originalGradientScore,
    firstPassSpatialScore,
    firstPassGradientScore
}) {
    if (Math.abs(firstPassSpatialScore) <= 0.25) {
        return true;
    }

    return originalSpatialScore >= 0 &&
        firstPassSpatialScore < 0 &&
        firstPassGradientScore <= FIRST_PASS_SIGN_FLIP_GRADIENT_THRESHOLD &&
        (originalGradientScore - firstPassGradientScore) >= FIRST_PASS_SIGN_FLIP_MIN_GRADIENT_DROP;
}

function refineSubpixelOutline({
    sourceImageData,
    alphaMap,
    position,
    alphaGain,
    originalNearBlackRatio,
    baselineSpatialScore,
    baselineGradientScore,
    baselineShift,
    minGain = OUTLINE_REFINEMENT_MIN_GAIN,
    shiftCandidates = SUBPIXEL_REFINE_SHIFTS,
    scaleCandidates = SUBPIXEL_REFINE_SCALES,
    minGradientImprovement = 0.04,
    maxSpatialDrift = 0.08
}) {
    const size = position.width;
    if (!size || size <= 8) return null;
    if (alphaGain < minGain) return null;

    const maxAllowedNearBlackRatio = Math.min(1, originalNearBlackRatio + MAX_NEAR_BLACK_RATIO_INCREASE);
    const gainCandidates = [alphaGain];
    const lower = Math.max(1, Number((alphaGain - 0.01).toFixed(2)));
    const upper = Number((alphaGain + 0.01).toFixed(2));
    if (lower !== alphaGain) gainCandidates.push(lower);
    if (upper !== alphaGain) gainCandidates.push(upper);

    const baseDx = baselineShift?.dx ?? 0;
    const baseDy = baselineShift?.dy ?? 0;
    const baseScale = baselineShift?.scale ?? 1;

    let best = null;
    for (const scaleDelta of scaleCandidates) {
        const scale = Number((baseScale * scaleDelta).toFixed(4));
        for (const dyDelta of shiftCandidates) {
            const dy = baseDy + dyDelta;
            for (const dxDelta of shiftCandidates) {
                const dx = baseDx + dxDelta;
                const warped = warpAlphaMap(alphaMap, size, { dx, dy, scale });
                for (const gain of gainCandidates) {
                    const candidate = cloneImageData(sourceImageData);
                    removeWatermark(candidate, warped, position, { alphaGain: gain });
                    const nearBlackRatio = calculateNearBlackRatio(candidate, position);
                    if (nearBlackRatio > maxAllowedNearBlackRatio) continue;

                    const spatialScore = computeRegionSpatialCorrelation({
                        imageData: candidate,
                        alphaMap: warped,
                        region: { x: position.x, y: position.y, size }
                    });
                    const gradientScore = computeRegionGradientCorrelation({
                        imageData: candidate,
                        alphaMap: warped,
                        region: { x: position.x, y: position.y, size }
                    });

                    const cost = Math.abs(spatialScore) * 0.6 + Math.max(0, gradientScore);
                    if (!best || cost < best.cost) {
                        best = {
                            imageData: candidate,
                            alphaMap: warped,
                            alphaGain: gain,
                            shift: { dx, dy, scale },
                            spatialScore,
                            gradientScore,
                            nearBlackRatio,
                            cost
                        };
                    }
                }
            }
        }
    }

    if (!best) return null;

    const improvedGradient = best.gradientScore <= baselineGradientScore - minGradientImprovement;
    const keptSpatial = Math.abs(best.spatialScore) <= Math.abs(baselineSpatialScore) + maxSpatialDrift;
    if (!improvedGradient || !keptSpatial) return null;

    return best;
}

function recalibrateAlphaStrength({
    sourceImageData,
    alphaMap,
    position,
    originalSpatialScore,
    processedSpatialScore,
    originalNearBlackRatio
}) {
    let bestScore = processedSpatialScore;
    let bestGain = 1;
    let bestImageData = null;
    const maxAllowedNearBlackRatio = Math.min(1, originalNearBlackRatio + MAX_NEAR_BLACK_RATIO_INCREASE);

    for (const alphaGain of ALPHA_GAIN_CANDIDATES) {
        const candidate = cloneImageData(sourceImageData);
        removeWatermark(candidate, alphaMap, position, { alphaGain });
        const candidateNearBlackRatio = calculateNearBlackRatio(candidate, position);
        if (candidateNearBlackRatio > maxAllowedNearBlackRatio) {
            continue;
        }

        const score = computeRegionSpatialCorrelation({
            imageData: candidate,
            alphaMap,
            region: {
                x: position.x,
                y: position.y,
                size: position.width
            }
        });

        if (score < bestScore) {
            bestScore = score;
            bestGain = alphaGain;
            bestImageData = candidate;
        }
    }

    const scoreDelta = processedSpatialScore - bestScore;
    if (!bestImageData || scoreDelta < MIN_RECALIBRATION_SCORE_DELTA) {
        return null;
    }

    return {
        imageData: bestImageData,
        alphaGain: bestGain,
        processedSpatialScore: bestScore,
        suppressionGain: originalSpatialScore - bestScore
    };
}

function recalibrateOverSubtractedAlpha({
    originalImageData,
    alphaMap,
    position,
    currentSpatialScore,
    currentGradientScore,
    currentAlphaGain,
    originalSpatialScore,
    originalNearBlackRatio
}) {
    if (
        currentSpatialScore > OVER_SUBTRACTION_SPATIAL_THRESHOLD ||
        currentGradientScore < OVER_SUBTRACTION_GRADIENT_THRESHOLD
    ) {
        return null;
    }

    const maxAllowedNearBlackRatio = Math.min(1, originalNearBlackRatio + MAX_NEAR_BLACK_RATIO_INCREASE);
    let best = null;

    const evaluateAlphaGain = (alphaGain) => {
        const candidate = cloneImageData(originalImageData);
        removeWatermark(candidate, alphaMap, position, { alphaGain });
        const nearBlackRatio = calculateNearBlackRatio(candidate, position);
        if (nearBlackRatio > maxAllowedNearBlackRatio) return null;

        const spatialScore = computeRegionSpatialCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientScore = computeRegionGradientCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const artifacts = assessRemovalDiffArtifacts({
            originalImageData,
            candidateImageData: candidate,
            alphaMap,
            position,
            alphaGain
        });
        const absSpatialImprovement = Math.abs(currentSpatialScore) - Math.abs(spatialScore);
        const gradientImprovement = currentGradientScore - gradientScore;
        if (
            absSpatialImprovement < OVER_SUBTRACTION_MIN_ABS_SPATIAL_IMPROVEMENT ||
            gradientImprovement < OVER_SUBTRACTION_MIN_GRADIENT_IMPROVEMENT
        ) {
            return null;
        }

        const cost = artifacts?.visualArtifactCost ?? (
            Math.abs(spatialScore) + Math.max(0, gradientScore) * 0.8 + nearBlackRatio * 2
        );
        return {
            imageData: candidate,
            alphaGain,
            spatialScore,
            gradientScore,
            nearBlackRatio,
            suppressionGain: originalSpatialScore - spatialScore,
            cost
        };
    };

    for (const alphaGain of ALPHA_GAIN_CANDIDATES) {
        if (alphaGain >= currentAlphaGain) continue;

        const candidate = evaluateAlphaGain(alphaGain);
        if (!candidate) continue;

        if (!best || candidate.cost < best.cost) {
            best = candidate;
        }
    }

    if (!best) return null;

    const fineGains = new Set();
    const fineStepCount = Math.round(OVER_SUBTRACTION_FINE_ALPHA_WINDOW / OVER_SUBTRACTION_FINE_ALPHA_STEP);
    for (let step = -fineStepCount; step <= fineStepCount; step++) {
        const alphaGain = Number((best.alphaGain + step * OVER_SUBTRACTION_FINE_ALPHA_STEP).toFixed(2));
        if (alphaGain <= 0 || alphaGain >= currentAlphaGain) continue;
        fineGains.add(alphaGain);
    }

    for (const alphaGain of fineGains) {
        if (alphaGain === best.alphaGain) continue;

        const candidate = evaluateAlphaGain(alphaGain);
        if (!candidate) continue;

        if (candidate.cost < best.cost) {
            best = candidate;
        }
    }

    return best;
}

function fineTuneWeakPositiveResidualAlpha({
    originalImageData,
    alphaMap,
    position,
    currentSpatialScore,
    currentGradientScore,
    currentAlphaGain,
    originalSpatialScore,
    originalGradientScore,
    originalNearBlackRatio
}) {
    if (
        currentAlphaGain >= 1 ||
        originalSpatialScore < WEAK_ALPHA_FINE_TUNE_MIN_ORIGINAL_SPATIAL ||
        currentSpatialScore < WEAK_ALPHA_FINE_TUNE_MIN_POSITIVE_RESIDUAL
    ) {
        return null;
    }

    const maxAllowedNearBlackRatio = Math.min(1, originalNearBlackRatio + MAX_NEAR_BLACK_RATIO_INCREASE);
    let best = null;
    const fineStepCount = Math.round(OVER_SUBTRACTION_FINE_ALPHA_WINDOW / OVER_SUBTRACTION_FINE_ALPHA_STEP);
    const alphaGainCandidates = new Set();

    for (let step = 1; step <= fineStepCount; step++) {
        const alphaGain = Number((currentAlphaGain + step * OVER_SUBTRACTION_FINE_ALPHA_STEP).toFixed(2));
        if (alphaGain >= 1) continue;
        alphaGainCandidates.add(alphaGain);
    }

    const shouldTryStrongPositiveFineTune =
        currentSpatialScore >= STRONG_POSITIVE_FINE_TUNE_MIN_CURRENT_SPATIAL &&
        originalSpatialScore >= STRONG_POSITIVE_FINE_TUNE_MIN_ORIGINAL_SPATIAL &&
        originalGradientScore >= STRONG_POSITIVE_FINE_TUNE_MIN_ORIGINAL_GRADIENT;

    if (shouldTryStrongPositiveFineTune) {
        for (const alphaGain of [
            ...STRONG_POSITIVE_FINE_TUNE_MID_GAINS,
            ...STRONG_POSITIVE_FINE_TUNE_EXTRA_GAINS
        ]) {
            alphaGainCandidates.add(alphaGain);
        }
    }

    for (const alphaGain of alphaGainCandidates) {
        if (alphaGain <= currentAlphaGain || alphaGain > 1) continue;

        const candidate = cloneImageData(originalImageData);
        removeWatermark(candidate, alphaMap, position, { alphaGain });
        const nearBlackRatio = calculateNearBlackRatio(candidate, position);
        if (nearBlackRatio > maxAllowedNearBlackRatio) continue;

        const spatialScore = computeRegionSpatialCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientScore = computeRegionGradientCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const artifacts = assessRemovalDiffArtifacts({
            originalImageData,
            candidateImageData: candidate,
            alphaMap,
            position,
            alphaGain
        });
        const absSpatialImprovement = Math.abs(currentSpatialScore) - Math.abs(spatialScore);
        const gradientIncrease = gradientScore - currentGradientScore;
        const strongPositiveClearsResidual = shouldTryStrongPositiveFineTune &&
            Math.abs(spatialScore) <= STRONG_POSITIVE_FINE_TUNE_MAX_ACCEPTED_SPATIAL &&
            gradientScore <= STRONG_POSITIVE_FINE_TUNE_MAX_ACCEPTED_GRADIENT &&
            (artifacts?.halo?.positiveDeltaLum ?? Number.POSITIVE_INFINITY) <= STRONG_POSITIVE_FINE_TUNE_MAX_POSITIVE_HALO_LUM;
        if (!strongPositiveClearsResidual && (
            absSpatialImprovement < WEAK_ALPHA_FINE_TUNE_MIN_ABS_SPATIAL_IMPROVEMENT ||
            gradientIncrease > WEAK_ALPHA_FINE_TUNE_MAX_GRADIENT_INCREASE
        )) {
            continue;
        }

        const cost = artifacts
            ? artifacts.visualArtifactCost + Math.max(0, gradientIncrease) * 0.25
            : Math.abs(spatialScore) + Math.max(0, gradientIncrease) * 0.25 + nearBlackRatio * 2;
        const clearsResidual = strongPositiveClearsResidual;
        if (!best || (clearsResidual && !best.clearsResidual) || (clearsResidual === best.clearsResidual && cost < best.cost)) {
            best = {
                imageData: candidate,
                alphaGain,
                spatialScore,
                gradientScore,
                nearBlackRatio,
                suppressionGain: originalSpatialScore - spatialScore,
                cost,
                clearsResidual
            };
        }
    }

    return best;
}

function fineTuneDarkCatalogAlpha({
    originalImageData,
    alphaMap,
    position,
    source,
    currentSpatialScore,
    currentGradientScore,
    currentAlphaGain,
    originalSpatialScore,
    originalGradientScore,
    originalNearBlackRatio
}) {
    if (
        typeof source !== 'string' ||
        !source.includes('catalog') ||
        currentAlphaGain < 1 ||
        originalSpatialScore < CATALOG_ALPHA_DARK_FINE_TUNE_MIN_ORIGINAL_SPATIAL ||
        originalGradientScore < CATALOG_ALPHA_DARK_FINE_TUNE_MIN_ORIGINAL_GRADIENT ||
        currentSpatialScore > CATALOG_ALPHA_DARK_FINE_TUNE_MAX_NEGATIVE_RESIDUAL
    ) {
        return null;
    }

    const maxAllowedNearBlackRatio = Math.min(1, originalNearBlackRatio + MAX_NEAR_BLACK_RATIO_INCREASE);
    let best = null;

    for (const alphaGain of CATALOG_DARK_ALPHA_GAIN_CANDIDATES) {
        if (alphaGain >= currentAlphaGain) continue;

        const candidate = cloneImageData(originalImageData);
        removeWatermark(candidate, alphaMap, position, { alphaGain });
        const nearBlackRatio = calculateNearBlackRatio(candidate, position);
        if (nearBlackRatio > maxAllowedNearBlackRatio) continue;

        const spatialScore = computeRegionSpatialCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientScore = computeRegionGradientCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const artifacts = assessRemovalDiffArtifacts({
            originalImageData,
            candidateImageData: candidate,
            alphaMap,
            position,
            alphaGain
        });
        const absSpatialImprovement = Math.abs(currentSpatialScore) - Math.abs(spatialScore);
        const gradientIncrease = gradientScore - currentGradientScore;
        if (
            absSpatialImprovement <= 0 ||
            gradientIncrease > CATALOG_ALPHA_DARK_FINE_TUNE_MAX_GRADIENT_INCREASE
        ) {
            continue;
        }

        const cost = artifacts?.visualArtifactCost ?? (
            Math.abs(spatialScore) * 0.25 +
            Math.max(0, gradientScore) +
            nearBlackRatio * 2
        );
        if (!best || cost < best.cost) {
            best = {
                imageData: candidate,
                alphaGain,
                spatialScore,
                gradientScore,
                nearBlackRatio,
                suppressionGain: originalSpatialScore - spatialScore,
                cost
            };
        }
    }

    return best;
}

function shouldRefineResidualEdge({
    source,
    position,
    baselineSpatialScore,
    baselineGradientScore,
    baselinePositiveHalo,
    mode = 'preview'
}) {
    if (mode === 'known-48') {
        return position?.width >= KNOWN_48_EDGE_CLEANUP_MIN_SIZE &&
            position?.width <= KNOWN_48_EDGE_CLEANUP_MAX_SIZE &&
            Math.abs(baselineSpatialScore) <= KNOWN_48_EDGE_CLEANUP_MAX_ABS_SPATIAL &&
            (
                baselineGradientScore >= KNOWN_48_EDGE_CLEANUP_MIN_GRADIENT ||
                baselinePositiveHalo >= PREVIEW_EDGE_CLEANUP_STRONG_HALO_THRESHOLD
            );
    }

    if (mode === 'v2-small') {
        return position?.width >= V2_SMALL_EDGE_CLEANUP_SIZE - V2_SMALL_EDGE_CLEANUP_SIZE_TOLERANCE &&
            position?.width <= V2_SMALL_EDGE_CLEANUP_SIZE + V2_SMALL_EDGE_CLEANUP_SIZE_TOLERANCE &&
            Math.abs(baselineSpatialScore) <= V2_SMALL_EDGE_CLEANUP_MAX_ABS_SPATIAL &&
            baselineGradientScore >= V2_SMALL_EDGE_CLEANUP_MIN_GRADIENT;
    }

    return typeof source === 'string' &&
        source.includes('preview-anchor') &&
        position?.width >= 24 &&
        position?.width <= PREVIEW_EDGE_CLEANUP_MAX_SIZE &&
        (
            Math.abs(baselineSpatialScore) <= PREVIEW_EDGE_CLEANUP_SPATIAL_THRESHOLD ||
            (
                baselinePositiveHalo >= PREVIEW_EDGE_CLEANUP_STRONG_HALO_THRESHOLD &&
                Math.abs(baselineSpatialScore) <= PREVIEW_EDGE_CLEANUP_HALO_SPATIAL_THRESHOLD
            )
        ) &&
        baselineGradientScore >= PREVIEW_EDGE_CLEANUP_GRADIENT_THRESHOLD;
}

function shouldUsePreviewAnchorFastCleanup(selectedTrial, position) {
    return selectedTrial?.provenance?.previewAnchor === true &&
        position?.width >= 24 &&
        position?.width <= PREVIEW_EDGE_CLEANUP_MAX_SIZE;
}

function isKnown48AnchorConfig(config) {
    if (!config || config.logoSize < KNOWN_48_EDGE_CLEANUP_MIN_SIZE || config.logoSize > KNOWN_48_EDGE_CLEANUP_MAX_SIZE) {
        return false;
    }

    const marginRight = Number(config.marginRight);
    const marginBottom = Number(config.marginBottom);
    if (!Number.isFinite(marginRight) || !Number.isFinite(marginBottom)) return false;

    const isCurrentLargeMargin = Math.abs(marginRight - 96) <= 2 && Math.abs(marginBottom - 96) <= 2;
    const isCurrentStandardMargin = marginRight >= 28 && marginRight <= 36 && marginBottom >= 28 && marginBottom <= 36;
    return isCurrentLargeMargin || isCurrentStandardMargin;
}

function shouldUseKnown48EdgeCleanup({ selectedTrial, position, source }) {
    if (selectedTrial?.provenance?.previewAnchor === true) return false;
    if (position?.width < KNOWN_48_EDGE_CLEANUP_MIN_SIZE || position?.width > KNOWN_48_EDGE_CLEANUP_MAX_SIZE) return false;
    if (!isKnown48AnchorConfig(selectedTrial?.config)) return false;

    const sourceText = String(source || '');
    return sourceText === 'standard' ||
        sourceText.startsWith('standard+gain') ||
        sourceText.includes('catalog') ||
        sourceText.includes('fixed-local');
}

function isV2SmallAnchorConfig(config) {
    if (!config || config.logoSize !== V2_SMALL_EDGE_CLEANUP_SIZE || config.alphaVariant !== 'v2') {
        return false;
    }

    const marginRight = Number(config.marginRight);
    const marginBottom = Number(config.marginBottom);
    return Number.isFinite(marginRight) &&
        Number.isFinite(marginBottom) &&
        marginRight >= 48 &&
        marginBottom >= 48;
}

function shouldUseV2SmallEdgeCleanup({ selectedTrial, position, source }) {
    if (selectedTrial?.provenance?.previewAnchor === true) return false;
    if (
        position?.width < V2_SMALL_EDGE_CLEANUP_SIZE - V2_SMALL_EDGE_CLEANUP_SIZE_TOLERANCE ||
        position?.width > V2_SMALL_EDGE_CLEANUP_SIZE + V2_SMALL_EDGE_CLEANUP_SIZE_TOLERANCE
    ) {
        return false;
    }
    if (!isV2SmallAnchorConfig(selectedTrial?.config)) return false;
    if (selectedTrial?.provenance?.catalogFamily !== 'gemini-v2-small') return false;

    const sourceText = String(source || '');
    return sourceText.includes('catalog');
}

function blendPreviewResidualEdge({
    sourceImageData,
    alphaMap,
    position,
    minAlpha,
    maxAlpha,
    radius,
    strength,
    outsideAlphaMax
}) {
    const candidate = cloneImageData(sourceImageData);
    const { width: imageWidth, height: imageHeight, data } = sourceImageData;
    const regionSize = position.width;
    const maxAlphaSafe = Math.max(maxAlpha, 1e-6);
    const edgeMask = createAlphaGradientMask({
        alphaMap,
        width: regionSize,
        height: regionSize
    });

    for (let row = 0; row < regionSize; row++) {
        for (let col = 0; col < regionSize; col++) {
            const localIndex = row * regionSize + col;
            const alpha = Math.abs(alphaMap[localIndex]);
            if (alpha < minAlpha || alpha > maxAlpha) continue;

            let sumR = 0;
            let sumG = 0;
            let sumB = 0;
            let sumWeight = 0;

            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    if (dx === 0 && dy === 0) continue;

                    const localY = row + dy;
                    const localX = col + dx;
                    const pixelX = position.x + localX;
                    const pixelY = position.y + localY;

                    if (pixelX < 0 || pixelY < 0 || pixelX >= imageWidth || pixelY >= imageHeight) {
                        continue;
                    }

                    let neighborAlpha = 0;
                    if (localY >= 0 && localX >= 0 && localY < regionSize && localX < regionSize) {
                        neighborAlpha = Math.abs(alphaMap[localY * regionSize + localX]);
                    }
                    if (neighborAlpha > outsideAlphaMax) continue;

                    const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                    const weight = 1 / distance;
                    const pixelIndex = (pixelY * imageWidth + pixelX) * 4;
                    sumR += data[pixelIndex] * weight;
                    sumG += data[pixelIndex + 1] * weight;
                    sumB += data[pixelIndex + 2] * weight;
                    sumWeight += weight;
                }
            }

            if (sumWeight <= 0) continue;

            const edgeWeight = getAlphaGradientWeight(edgeMask, localIndex);
            const blend = Math.max(0, Math.min(1, strength * alpha / maxAlphaSafe * edgeWeight));
            const pixelIndex = ((position.y + row) * imageWidth + (position.x + col)) * 4;
            candidate.data[pixelIndex] = Math.round(data[pixelIndex] * (1 - blend) + (sumR / sumWeight) * blend);
            candidate.data[pixelIndex + 1] = Math.round(data[pixelIndex + 1] * (1 - blend) + (sumG / sumWeight) * blend);
            candidate.data[pixelIndex + 2] = Math.round(data[pixelIndex + 2] * (1 - blend) + (sumB / sumWeight) * blend);
        }
    }

    return candidate;
}

function solveLinear3x3(matrix, vector) {
    const augmented = matrix.map((row, index) => [...row, vector[index]]);

    for (let column = 0; column < 3; column++) {
        let pivot = column;
        for (let row = column + 1; row < 3; row++) {
            if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) {
                pivot = row;
            }
        }

        if (Math.abs(augmented[pivot][column]) < 1e-8) return null;
        if (pivot !== column) {
            [augmented[pivot], augmented[column]] = [augmented[column], augmented[pivot]];
        }

        const divisor = augmented[column][column];
        for (let next = column; next < 4; next++) {
            augmented[column][next] /= divisor;
        }

        for (let row = 0; row < 3; row++) {
            if (row === column) continue;
            const factor = augmented[row][column];
            for (let next = column; next < 4; next++) {
                augmented[row][next] -= factor * augmented[column][next];
            }
        }
    }

    return [augmented[0][3], augmented[1][3], augmented[2][3]];
}

function fitColorPlane(samples, channel) {
    const matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ];
    const vector = [0, 0, 0];

    for (const sample of samples) {
        const terms = [1, sample.x, sample.y];
        const value = sample[channel];
        for (let row = 0; row < 3; row++) {
            vector[row] += terms[row] * value;
            for (let column = 0; column < 3; column++) {
                matrix[row][column] += terms[row] * terms[column];
            }
        }
    }

    return solveLinear3x3(matrix, vector);
}

function calculateBackgroundSampleStats(samples) {
    let sum = 0;
    let squareSum = 0;

    for (const sample of samples) {
        const luminance = 0.2126 * sample.r + 0.7152 * sample.g + 0.0722 * sample.b;
        sum += luminance;
        squareSum += luminance * luminance;
    }

    const count = samples.length;
    const mean = count > 0 ? sum / count : 0;
    return {
        count,
        mean,
        std: count > 0 ? Math.sqrt(Math.max(0, squareSum / count - mean * mean)) : Number.POSITIVE_INFINITY
    };
}

function sampleFlatBackgroundPixels(imageData, alphaMap, position, {
    pad = KNOWN_48_FLAT_FILL_PAD,
    outsideAlphaMax = KNOWN_48_FLAT_FILL_OUTSIDE_ALPHA_MAX
} = {}) {
    const samples = [];
    const left = Math.max(0, position.x - pad);
    const top = Math.max(0, position.y - pad);
    const right = Math.min(imageData.width - 1, position.x + position.width + pad - 1);
    const bottom = Math.min(imageData.height - 1, position.y + position.height + pad - 1);

    for (let y = top; y <= bottom; y++) {
        for (let x = left; x <= right; x++) {
            const inside = x >= position.x &&
                x < position.x + position.width &&
                y >= position.y &&
                y < position.y + position.height;

            if (inside) {
                const row = y - position.y;
                const col = x - position.x;
                if (alphaMap[row * position.width + col] > outsideAlphaMax) {
                    continue;
                }
            }

            const pixelIndex = (y * imageData.width + x) * 4;
            samples.push({
                x: (x - position.x) / Math.max(1, position.width),
                y: (y - position.y) / Math.max(1, position.height),
                r: imageData.data[pixelIndex],
                g: imageData.data[pixelIndex + 1],
                b: imageData.data[pixelIndex + 2]
            });
        }
    }

    return samples;
}

function applyFlatBackgroundFill({
    sourceImageData,
    alphaMap,
    position,
    minAlpha,
    maxAlpha,
    strength,
    maxBackgroundStd = KNOWN_48_FLAT_FILL_MAX_BACKGROUND_STD,
    edgeWeightFloor = 0.35
}) {
    const samples = sampleFlatBackgroundPixels(sourceImageData, alphaMap, position);
    const stats = calculateBackgroundSampleStats(samples);
    if (
        stats.count < 24 ||
        stats.std > maxBackgroundStd
    ) {
        return null;
    }

    const redPlane = fitColorPlane(samples, 'r');
    const greenPlane = fitColorPlane(samples, 'g');
    const bluePlane = fitColorPlane(samples, 'b');
    if (!redPlane || !greenPlane || !bluePlane) return null;

    const candidate = cloneImageData(sourceImageData);
    const edgeMask = createAlphaGradientMask({
        alphaMap,
        width: position.width,
        height: position.height
    });
    for (let row = 0; row < position.height; row++) {
        for (let col = 0; col < position.width; col++) {
            const localIndex = row * position.width + col;
            const alpha = alphaMap[localIndex];
            if (alpha < minAlpha || alpha > maxAlpha) continue;

            const x = col / Math.max(1, position.width);
            const y = row / Math.max(1, position.height);
            const target = [
                redPlane[0] + redPlane[1] * x + redPlane[2] * y,
                greenPlane[0] + greenPlane[1] * x + greenPlane[2] * y,
                bluePlane[0] + bluePlane[1] * x + bluePlane[2] * y
            ];
            const pixelIndex = ((position.y + row) * candidate.width + position.x + col) * 4;
            const edgeWeight = getAlphaGradientWeight(edgeMask, localIndex, edgeWeightFloor);
            const blend = Math.max(0, Math.min(1, strength * Math.min(1, alpha / 0.2) * edgeWeight));
            for (let channel = 0; channel < 3; channel++) {
                candidate.data[pixelIndex + channel] = clampChannel(
                    candidate.data[pixelIndex + channel] * (1 - blend) + target[channel] * blend
                );
            }
        }
    }

    return { imageData: candidate, stats };
}

function refineKnown48FlatBackgroundResidual({
    sourceImageData,
    alphaMap,
    position,
    baselineSpatialScore,
    baselineGradientScore,
    minGradientImprovement = KNOWN_48_FLAT_FILL_MIN_GRADIENT_IMPROVEMENT
}) {
    if (
        position?.width < KNOWN_48_EDGE_CLEANUP_MIN_SIZE ||
        position?.width > KNOWN_48_EDGE_CLEANUP_MAX_SIZE ||
        baselineGradientScore < KNOWN_48_FLAT_FILL_MIN_GRADIENT
    ) {
        return null;
    }

    let best = null;
    for (const preset of KNOWN_48_FLAT_FILL_PRESETS) {
        const filled = applyFlatBackgroundFill({
            sourceImageData,
            alphaMap,
            position,
            ...preset
        });
        if (!filled) continue;

        const spatialScore = computeRegionSpatialCorrelation({
            imageData: filled.imageData,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientScore = computeRegionGradientCorrelation({
            imageData: filled.imageData,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientImprovement = baselineGradientScore - gradientScore;
        const keptSpatial = Math.abs(spatialScore) <= Math.abs(baselineSpatialScore) + KNOWN_48_FLAT_FILL_MAX_SPATIAL_DRIFT;
        const acceptedSpatial = Math.abs(spatialScore) <= KNOWN_48_FLAT_FILL_MAX_ACCEPTED_ABS_SPATIAL;
        if (
            gradientImprovement < minGradientImprovement ||
            !keptSpatial ||
            !acceptedSpatial
        ) {
            continue;
        }

        const cost = Math.abs(spatialScore) * 0.45 + Math.max(0, gradientScore);
        if (!best || cost < best.cost) {
            best = {
                imageData: filled.imageData,
                spatialScore,
                gradientScore,
                stats: filled.stats,
                preset: preset.name,
                cost
            };
        }
    }

    return best;
}

function isNewMargin96AlphaVariantConfig(config) {
    return config?.logoSize === 96 &&
        config.marginRight === 192 &&
        config.marginBottom === 192 &&
        config.alphaVariant === '20260520';
}

function refineNewMargin96FlatBackgroundResidual({
    sourceImageData,
    alphaMap,
    position,
    config,
    alphaGain,
    baselineSpatialScore,
    baselineGradientScore
}) {
    if (
        !isNewMargin96AlphaVariantConfig(config) ||
        alphaGain !== 1 ||
        position?.width !== 96 ||
        baselineGradientScore < NEW_MARGIN_96_FLAT_FILL_MIN_GRADIENT
    ) {
        return null;
    }

    let best = null;
    for (const preset of NEW_MARGIN_96_FLAT_FILL_PRESETS) {
        const filled = applyFlatBackgroundFill({
            sourceImageData,
            alphaMap,
            position,
            maxBackgroundStd: NEW_MARGIN_96_FLAT_FILL_MAX_BACKGROUND_STD,
            edgeWeightFloor: 0.85,
            ...preset
        });
        if (!filled) continue;

        const spatialScore = computeRegionSpatialCorrelation({
            imageData: filled.imageData,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientScore = computeRegionGradientCorrelation({
            imageData: filled.imageData,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientImprovement = baselineGradientScore - gradientScore;
        const keptSpatial = Math.abs(spatialScore) <= Math.abs(baselineSpatialScore) + NEW_MARGIN_96_FLAT_FILL_MAX_SPATIAL_DRIFT;
        const acceptedSpatial = Math.abs(spatialScore) <= NEW_MARGIN_96_FLAT_FILL_MAX_ACCEPTED_ABS_SPATIAL;
        if (
            gradientImprovement < NEW_MARGIN_96_FLAT_FILL_MIN_GRADIENT_IMPROVEMENT ||
            !keptSpatial ||
            !acceptedSpatial
        ) {
            continue;
        }

        const cost = Math.abs(spatialScore) * 0.6 + Math.max(0, gradientScore);
        if (!best || cost < best.cost) {
            best = {
                imageData: filled.imageData,
                spatialScore,
                gradientScore,
                stats: filled.stats,
                preset: preset.name,
                cost
            };
        }
    }

    return best;
}

function pixelLuminance(data, index) {
    return 0.2126 * data[index] + 0.7152 * data[index + 1] + 0.0722 * data[index + 2];
}

function applyLumaEdgeCorrection({
    sourceImageData,
    alphaMap,
    position,
    minAlpha,
    maxAlpha,
    referenceAlphaMax,
    radius,
    strength,
    colorSigma,
    maxDelta
}) {
    const candidate = cloneImageData(sourceImageData);
    const { data, width: imageWidth, height: imageHeight } = sourceImageData;
    const size = position.width;
    const colorSigmaSafe = Math.max(1, colorSigma);
    const edgeMask = createAlphaGradientMask({
        alphaMap,
        width: size,
        height: size
    });

    for (let row = 0; row < size; row++) {
        for (let col = 0; col < size; col++) {
            const localIndex = row * size + col;
            const alpha = alphaMap[localIndex];
            if (alpha < minAlpha || alpha > maxAlpha) continue;

            const x = position.x + col;
            const y = position.y + row;
            const pixelIndex = (y * imageWidth + x) * 4;
            const currentLum = pixelLuminance(data, pixelIndex);
            let weightedLum = 0;
            let sumWeight = 0;

            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    if (dx === 0 && dy === 0) continue;
                    const distanceSquared = dx * dx + dy * dy;
                    if (distanceSquared > radius * radius) continue;

                    const localX = col + dx;
                    const localY = row + dy;
                    const pixelX = x + dx;
                    const pixelY = y + dy;
                    if (pixelX < 0 || pixelY < 0 || pixelX >= imageWidth || pixelY >= imageHeight) continue;

                    let neighborAlpha = 0;
                    if (localX >= 0 && localY >= 0 && localX < size && localY < size) {
                        neighborAlpha = alphaMap[localY * size + localX];
                    }
                    if (neighborAlpha > referenceAlphaMax && neighborAlpha >= alpha) continue;

                    const neighborIndex = (pixelY * imageWidth + pixelX) * 4;
                    const neighborLum = pixelLuminance(data, neighborIndex);
                    const colorDistance =
                        Math.abs(data[pixelIndex] - data[neighborIndex]) +
                        Math.abs(data[pixelIndex + 1] - data[neighborIndex + 1]) +
                        Math.abs(data[pixelIndex + 2] - data[neighborIndex + 2]);
                    const colorWeight = Math.exp(
                        -(colorDistance * colorDistance) / (2 * colorSigmaSafe * colorSigmaSafe * 9)
                    );
                    const alphaWeight = neighborAlpha <= referenceAlphaMax ? 1.25 : 0.65;
                    const distanceWeight = 1 / Math.sqrt(distanceSquared);
                    const weight = colorWeight * alphaWeight * distanceWeight;
                    weightedLum += neighborLum * weight;
                    sumWeight += weight;
                }
            }

            if (sumWeight <= 0) continue;

            const targetLum = weightedLum / sumWeight;
            const delta = Math.max(-maxDelta, Math.min(maxDelta, targetLum - currentLum)) * strength;
            const edgeWeight = getAlphaGradientWeight(edgeMask, localIndex);
            const scaledStrength = Math.min(1, Math.max(0, alpha / Math.max(maxAlpha, 1e-6))) * edgeWeight;
            const finalDelta = delta * scaledStrength;
            candidate.data[pixelIndex] = clampChannel(data[pixelIndex] + finalDelta);
            candidate.data[pixelIndex + 1] = clampChannel(data[pixelIndex + 1] + finalDelta);
            candidate.data[pixelIndex + 2] = clampChannel(data[pixelIndex + 2] + finalDelta);
        }
    }

    return candidate;
}

function refineKnown48LumaEdgeResidual({
    sourceImageData,
    alphaMap,
    position,
    baselineSpatialScore,
    baselineGradientScore
}) {
    if (
        position?.width < KNOWN_48_EDGE_CLEANUP_MIN_SIZE ||
        position?.width > KNOWN_48_EDGE_CLEANUP_MAX_SIZE ||
        baselineGradientScore < KNOWN_48_LUMA_EDGE_MIN_GRADIENT
    ) {
        return null;
    }

    let best = null;
    for (const preset of KNOWN_48_LUMA_EDGE_PRESETS) {
        const candidate = applyLumaEdgeCorrection({
            sourceImageData,
            alphaMap,
            position,
            ...preset
        });
        const spatialScore = computeRegionSpatialCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientScore = computeRegionGradientCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientImprovement = baselineGradientScore - gradientScore;
        const keptSpatial = Math.abs(spatialScore) <= Math.abs(baselineSpatialScore) + KNOWN_48_LUMA_EDGE_MAX_SPATIAL_DRIFT;
        const acceptedSpatial = Math.abs(spatialScore) <= KNOWN_48_LUMA_EDGE_MAX_ACCEPTED_ABS_SPATIAL;
        if (
            gradientImprovement < KNOWN_48_LUMA_EDGE_MIN_GRADIENT_IMPROVEMENT ||
            !keptSpatial ||
            !acceptedSpatial
        ) {
            continue;
        }

        const cost = Math.abs(spatialScore) * 0.45 + Math.max(0, gradientScore);
        if (!best || cost < best.cost) {
            best = {
                imageData: candidate,
                spatialScore,
                gradientScore,
                preset: preset.name,
                cost
            };
        }
    }

    return best;
}

function expandPosition(position, imageData, pad) {
    const left = Math.max(0, position.x - pad);
    const top = Math.max(0, position.y - pad);
    const right = Math.min(imageData.width, position.x + position.width + pad);
    const bottom = Math.min(imageData.height, position.y + position.height + pad);

    return {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top
    };
}

function measureOuterBorderLuminanceStd(imageData, position, margin = 10) {
    let sum = 0;
    let sq = 0;
    let count = 0;

    const left = Math.max(0, position.x - margin);
    const top = Math.max(0, position.y - margin);
    const right = Math.min(imageData.width, position.x + position.width + margin);
    const bottom = Math.min(imageData.height, position.y + position.height + margin);

    for (let y = top; y < bottom; y++) {
        for (let x = left; x < right; x++) {
            const inside = x >= position.x &&
                x < position.x + position.width &&
                y >= position.y &&
                y < position.y + position.height;
            if (inside) continue;

            const idx = (y * imageData.width + x) * 4;
            const lum =
                0.2126 * imageData.data[idx] +
                0.7152 * imageData.data[idx + 1] +
                0.0722 * imageData.data[idx + 2];
            sum += lum;
            sq += lum * lum;
            count++;
        }
    }

    if (count <= 0) return Number.POSITIVE_INFINITY;

    const mean = sum / count;
    return Math.sqrt(Math.max(0, sq / count - mean * mean));
}

function clampChannel(value) {
    return Math.max(0, Math.min(255, Math.round(value)));
}

function averageStripColor(imageData, {
    xFrom,
    xTo,
    yFrom,
    yTo
}) {
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    let count = 0;

    const left = Math.max(0, xFrom);
    const right = Math.min(imageData.width - 1, xTo);
    const top = Math.max(0, yFrom);
    const bottom = Math.min(imageData.height - 1, yTo);

    for (let y = top; y <= bottom; y++) {
        for (let x = left; x <= right; x++) {
            const idx = (y * imageData.width + x) * 4;
            sumR += imageData.data[idx];
            sumG += imageData.data[idx + 1];
            sumB += imageData.data[idx + 2];
            count++;
        }
    }

    if (count <= 0) return [0, 0, 0];

    return [
        sumR / count,
        sumG / count,
        sumB / count
    ];
}

function lerpColor(left, right, t) {
    return [
        left[0] * (1 - t) + right[0] * t,
        left[1] * (1 - t) + right[1] * t,
        left[2] * (1 - t) + right[2] * t
    ];
}

function applyPreviewSmoothBackgroundCleanup({
    imageData,
    position
}) {
    const expandedPosition = expandPosition(
        position,
        imageData,
        PREVIEW_BACKGROUND_CLEANUP_PAD
    );
    const candidate = cloneImageData(imageData);
    const stripRadius = PREVIEW_BACKGROUND_CLEANUP_PRIOR_RADIUS;
    const leftBoundary = [];
    const rightBoundary = [];
    const topBoundary = [];
    const bottomBoundary = [];

    for (let row = 0; row < expandedPosition.height; row++) {
        const y = expandedPosition.y + row;
        leftBoundary.push(averageStripColor(imageData, {
            xFrom: expandedPosition.x - stripRadius,
            xTo: expandedPosition.x - 1,
            yFrom: y - 1,
            yTo: y + 1
        }));
        rightBoundary.push(averageStripColor(imageData, {
            xFrom: expandedPosition.x + expandedPosition.width,
            xTo: expandedPosition.x + expandedPosition.width + stripRadius - 1,
            yFrom: y - 1,
            yTo: y + 1
        }));
    }

    for (let col = 0; col < expandedPosition.width; col++) {
        const x = expandedPosition.x + col;
        topBoundary.push(averageStripColor(imageData, {
            xFrom: x - 1,
            xTo: x + 1,
            yFrom: expandedPosition.y - stripRadius,
            yTo: expandedPosition.y - 1
        }));
        bottomBoundary.push(averageStripColor(imageData, {
            xFrom: x - 1,
            xTo: x + 1,
            yFrom: expandedPosition.y + expandedPosition.height,
            yTo: expandedPosition.y + expandedPosition.height + stripRadius - 1
        }));
    }

    for (let row = 0; row < expandedPosition.height; row++) {
        const ty = expandedPosition.height <= 1 ? 0.5 : row / (expandedPosition.height - 1);
        for (let col = 0; col < expandedPosition.width; col++) {
            const tx = expandedPosition.width <= 1 ? 0.5 : col / (expandedPosition.width - 1);
            const horizontal = lerpColor(leftBoundary[row], rightBoundary[row], tx);
            const vertical = lerpColor(topBoundary[col], bottomBoundary[col], ty);
            const idx = ((expandedPosition.y + row) * candidate.width + expandedPosition.x + col) * 4;
            candidate.data[idx] = clampChannel((horizontal[0] + vertical[0]) * 0.5);
            candidate.data[idx + 1] = clampChannel((horizontal[1] + vertical[1]) * 0.5);
            candidate.data[idx + 2] = clampChannel((horizontal[2] + vertical[2]) * 0.5);
        }
    }

    return {
        imageData: candidate,
        expandedPosition
    };
}

function shouldApplyPreviewSmoothBackgroundCleanup({
    enabled = true,
    source,
    position,
    baselineSpatialScore,
    borderStd
}) {
    return enabled === true &&
        typeof source === 'string' &&
        source.includes('preview-anchor') &&
        position?.width >= 24 &&
        position?.width <= PREVIEW_BACKGROUND_CLEANUP_MAX_SIZE &&
        baselineSpatialScore >= PREVIEW_BACKGROUND_CLEANUP_MIN_RESIDUAL &&
        borderStd <= PREVIEW_BACKGROUND_CLEANUP_MAX_BORDER_STD;
}

function refineSmallPreviewAnchorCandidate({
    originalImageData,
    source,
    position,
    originalGradientScore,
    currentSpatialScore,
    currentGradientScore,
    getAlphaMap
}) {
    if (
        typeof source !== 'string' ||
        !source.includes('preview-anchor') ||
        !source.includes('edge-cleanup') ||
        position?.width > SMALL_PREVIEW_REFINEMENT_MAX_SOURCE_SIZE ||
        originalGradientScore > SMALL_PREVIEW_REFINEMENT_MAX_ORIGINAL_GRADIENT ||
        currentSpatialScore < SMALL_PREVIEW_REFINEMENT_MIN_CURRENT_SPATIAL ||
        currentGradientScore > SMALL_PREVIEW_REFINEMENT_MAX_CURRENT_GRADIENT ||
        typeof getAlphaMap !== 'function'
    ) {
        return null;
    }

    let best = null;
    const sizeCandidates = [
        position.width + 4,
        position.width + 6,
        position.width + 8
    ].filter((size) => size <= SMALL_PREVIEW_REFINEMENT_MAX_REFINED_SIZE);
    const shiftCandidates = [-8, -6, -4, -2, 0];
    const gainCandidates = ALPHA_GAIN_CANDIDATES.filter((gain) => gain < 1);

    for (const size of sizeCandidates) {
        const alphaMap = getAlphaMap(size);
        if (!alphaMap) continue;

        for (const dy of shiftCandidates) {
            for (const dx of shiftCandidates) {
                const candidatePosition = {
                    x: position.x + dx,
                    y: position.y + dy,
                    width: size,
                    height: size
                };
                if (
                    candidatePosition.x < 0 ||
                    candidatePosition.y < 0 ||
                    candidatePosition.x + size > originalImageData.width ||
                    candidatePosition.y + size > originalImageData.height
                ) {
                    continue;
                }

                for (const alphaGain of gainCandidates) {
                    const originalSpatialScore = computeRegionSpatialCorrelation({
                        imageData: originalImageData,
                        alphaMap,
                        region: { x: candidatePosition.x, y: candidatePosition.y, size }
                    });
                    const originalGradientScore = computeRegionGradientCorrelation({
                        imageData: originalImageData,
                        alphaMap,
                        region: { x: candidatePosition.x, y: candidatePosition.y, size }
                    });
                    const candidate = cloneImageData(originalImageData);
                    removeWatermark(candidate, alphaMap, candidatePosition, { alphaGain });
                    const spatialScore = computeRegionSpatialCorrelation({
                        imageData: candidate,
                        alphaMap,
                        region: { x: candidatePosition.x, y: candidatePosition.y, size }
                    });
                    const gradientScore = computeRegionGradientCorrelation({
                        imageData: candidate,
                        alphaMap,
                        region: { x: candidatePosition.x, y: candidatePosition.y, size }
                    });
                    const absSpatialImprovement = Math.abs(currentSpatialScore) - Math.abs(spatialScore);
                    const gradientImprovement = currentGradientScore - gradientScore;
                    if (
                        absSpatialImprovement < SMALL_PREVIEW_REFINEMENT_MIN_ABS_SPATIAL_IMPROVEMENT ||
                        gradientImprovement < SMALL_PREVIEW_REFINEMENT_MIN_GRADIENT_IMPROVEMENT
                    ) {
                        continue;
                    }

                    const nearBlackRatio = calculateNearBlackRatio(candidate, candidatePosition);
                    const cost = Math.abs(spatialScore) + Math.max(0, gradientScore) * 0.8 + nearBlackRatio * 2;
                    if (!best || cost < best.cost) {
                        best = {
                            imageData: candidate,
                            alphaMap,
                            alphaGain,
                            position: candidatePosition,
                            originalSpatialScore,
                            originalGradientScore,
                            spatialScore,
                            gradientScore,
                            nearBlackRatio,
                            cost
                        };
                    }
                }
            }
        }
    }

    return best;
}

function refinePreviewResidualEdge({
    sourceImageData,
    alphaMap,
    position,
    source,
    baselineSpatialScore,
    baselineGradientScore,
    minGradientImprovement = PREVIEW_EDGE_CLEANUP_MIN_GRADIENT_IMPROVEMENT,
    maxSpatialDrift = PREVIEW_EDGE_CLEANUP_MAX_SPATIAL_DRIFT,
    allowAggressivePresets = false,
    mode = 'preview'
}) {
    const baselineHalo = assessAlphaBandHalo({
        imageData: sourceImageData,
        position,
        alphaMap
    });
    const baselinePositiveHalo = baselineHalo.positiveDeltaLum;
    if (!shouldRefineResidualEdge({
        source,
        position,
        baselineSpatialScore,
        baselineGradientScore,
        baselinePositiveHalo,
        mode
    })) {
        return null;
    }

    const baselineNearBlackRatio = calculateNearBlackRatio(sourceImageData, position);
    const maxAllowedNearBlackRatio = Math.min(1, baselineNearBlackRatio + MAX_NEAR_BLACK_RATIO_INCREASE);
    const resolvedMinGradientImprovement = baselineGradientScore <= PREVIEW_EDGE_CLEANUP_FINE_GRADIENT_THRESHOLD
        ? PREVIEW_EDGE_CLEANUP_FINE_MIN_GRADIENT_IMPROVEMENT
        : (
            baselinePositiveHalo >= PREVIEW_EDGE_CLEANUP_STRONG_HALO_THRESHOLD
                ? PREVIEW_EDGE_CLEANUP_HALO_RELAXED_MIN_GRADIENT_IMPROVEMENT
                : minGradientImprovement
        );
    const presets = allowAggressivePresets &&
        baselineGradientScore >= PREVIEW_EDGE_CLEANUP_STRONG_GRADIENT_THRESHOLD &&
        Math.abs(baselineSpatialScore) <= 0.05
        ? [...PREVIEW_EDGE_CLEANUP_PRESETS, ...PREVIEW_EDGE_CLEANUP_AGGRESSIVE_PRESETS]
        : PREVIEW_EDGE_CLEANUP_PRESETS;
    let best = null;

    for (const preset of presets) {
        const candidate = blendPreviewResidualEdge({
            sourceImageData,
            alphaMap,
            position,
            ...preset
        });
        const nearBlackRatio = calculateNearBlackRatio(candidate, position);
        if (nearBlackRatio > maxAllowedNearBlackRatio) continue;

        const spatialScore = computeRegionSpatialCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const gradientScore = computeRegionGradientCorrelation({
            imageData: candidate,
            alphaMap,
            region: { x: position.x, y: position.y, size: position.width }
        });
        const halo = assessAlphaBandHalo({
            imageData: candidate,
            position,
            alphaMap
        });

        const presetMinGradientImprovement = preset.minGradientImprovement ?? resolvedMinGradientImprovement;
        const presetMaxSpatialDrift = preset.maxSpatialDrift ?? maxSpatialDrift;
        const presetMaxAcceptedSpatial = preset.maxAcceptedSpatial ?? 0.22;
        const improvedGradient = gradientScore <= baselineGradientScore - presetMinGradientImprovement;
        const keptSpatial = Math.abs(spatialScore) <= Math.abs(baselineSpatialScore) + presetMaxSpatialDrift;
        const keptResidualWithinTarget = Math.abs(spatialScore) <= presetMaxAcceptedSpatial;
        const candidatePositiveHalo = halo.positiveDeltaLum;
        const improvedHalo = baselinePositiveHalo < PREVIEW_EDGE_CLEANUP_STRONG_HALO_THRESHOLD ||
            candidatePositiveHalo <= baselinePositiveHalo - PREVIEW_EDGE_CLEANUP_MIN_HALO_REDUCTION;
        if (!improvedGradient || !keptSpatial || !keptResidualWithinTarget || !improvedHalo) continue;

        const cost = Math.abs(spatialScore) * 0.6 +
            Math.max(0, gradientScore) +
            candidatePositiveHalo * PREVIEW_EDGE_CLEANUP_HALO_WEIGHT;
        if (!best || cost < best.cost) {
            best = {
                imageData: candidate,
                spatialScore,
                gradientScore,
                halo,
                cost
            };
        }
    }

    return best;
}

function scoreLocatedAggressiveCandidate({
    imageData,
    alphaMap,
    position
}) {
    const spatialScore = computeRegionSpatialCorrelation({
        imageData,
        alphaMap,
        region: { x: position.x, y: position.y, size: position.width }
    });
    const gradientScore = computeRegionGradientCorrelation({
        imageData,
        alphaMap,
        region: { x: position.x, y: position.y, size: position.width }
    });
    const nearBlackRatio = calculateNearBlackRatio(imageData, position);
    return {
        spatialScore,
        gradientScore,
        nearBlackRatio,
        cost: Math.abs(spatialScore) * 0.8 + Math.max(0, gradientScore) * 0.75
    };
}

function pickLocatedAggressiveCandidate(currentBest, candidate) {
    if (!candidate) return currentBest;
    if (!currentBest || candidate.cost < currentBest.cost) return candidate;
    return currentBest;
}

function buildLocatedAggressiveCandidate({
    sourceImageData,
    alphaMap,
    position,
    alphaGain,
    repeatCount
}) {
    const candidate = cloneImageData(sourceImageData);
    for (let passIndex = 0; passIndex < repeatCount; passIndex++) {
        removeWatermark(candidate, alphaMap, position, { alphaGain });
    }
    return {
        imageData: candidate,
        alphaGain,
        repeatCount,
        ...scoreLocatedAggressiveCandidate({
            imageData: candidate,
            alphaMap,
            position
        })
    };
}

function refineLocatedAggressiveRemoval({
    originalImageData,
    currentImageData,
    alphaMap,
    position,
    currentSpatialScore,
    currentGradientScore,
    currentAlphaGain
}) {
    if (!position || !alphaMap || position.width !== position.height) return null;
    const current = {
        imageData: currentImageData,
        alphaGain: currentAlphaGain,
        repeatCount: 0,
        spatialScore: currentSpatialScore,
        gradientScore: currentGradientScore,
        nearBlackRatio: calculateNearBlackRatio(currentImageData, position),
        cost: Math.abs(currentSpatialScore) * 0.8 + Math.max(0, currentGradientScore) * 0.75
    };
    let best = current;
    const gains = new Set([
        currentAlphaGain,
        ...LOCATED_AGGRESSIVE_ALPHA_GAINS
    ].filter((value) => Number.isFinite(value) && value > 0));

    for (const alphaGain of gains) {
        for (const repeatCount of [1, 2]) {
            best = pickLocatedAggressiveCandidate(
                best,
                buildLocatedAggressiveCandidate({
                    sourceImageData: originalImageData,
                    alphaMap,
                    position,
                    alphaGain,
                    repeatCount
                })
            );
        }
    }

    const edgeSources = [{
        imageData: best.imageData,
        alphaGain: best.alphaGain,
        repeatCount: best.repeatCount
    }];
    if (best.imageData !== currentImageData) {
        edgeSources.push({
            imageData: currentImageData,
            alphaGain: currentAlphaGain,
            repeatCount: 0
        });
    }
    for (const edgeSource of edgeSources) {
        for (const preset of LOCATED_AGGRESSIVE_EDGE_PRESETS) {
            const edgeCandidate = blendPreviewResidualEdge({
                sourceImageData: edgeSource.imageData,
                alphaMap,
                position,
                ...preset
            });
            best = pickLocatedAggressiveCandidate(best, {
                imageData: edgeCandidate,
                alphaGain: edgeSource.alphaGain,
                repeatCount: edgeSource.repeatCount,
                edgeCleanup: true,
                ...scoreLocatedAggressiveCandidate({
                    imageData: edgeCandidate,
                    alphaMap,
                    position
                })
            });
        }
    }

    if (best.imageData === currentImageData) return null;
    return best;
}

function shouldRelocateSmallFixedLocalAnchor({
    source,
    config,
    position,
    currentGradientScore,
    currentResidualVisibility
}) {
    const sourceText = String(source || '');
    if (!sourceText.includes('fixed-local')) return false;
    if (
        !position ||
        position.width !== position.height ||
        position.width < SMALL_ANCHOR_RELOCATION_MIN_SIZE ||
        position.width > SMALL_ANCHOR_RELOCATION_MAX_SIZE
    ) {
        return false;
    }
    if (
        !config ||
        config.logoSize < SMALL_ANCHOR_RELOCATION_MIN_SIZE ||
        config.logoSize > SMALL_ANCHOR_RELOCATION_MAX_SIZE
    ) {
        return false;
    }
    if (!currentResidualVisibility?.visible) return false;
    return currentResidualVisibility.visiblePositiveHalo === true ||
        currentGradientScore >= SMALL_ANCHOR_RELOCATION_MIN_CURRENT_GRADIENT;
}

function resolveRelocationAlphaMap({ size, currentAlphaMap, currentPosition, alpha96, getAlphaMap }) {
    if (size === currentPosition?.width) return currentAlphaMap;
    if (typeof getAlphaMap === 'function') {
        const resolved = getAlphaMap(size);
        if (resolved) return resolved;
    }
    return interpolateAlphaMap(alpha96, 96, size);
}

function buildSmallAnchorRelocationCandidate({
    originalImageData,
    alphaMap,
    size,
    marginRight,
    marginBottom,
    alphaGain
}) {
    const position = {
        x: originalImageData.width - marginRight - size,
        y: originalImageData.height - marginBottom - size,
        width: size,
        height: size
    };
    if (
        position.x < 0 ||
        position.y < 0 ||
        position.x + size > originalImageData.width ||
        position.y + size > originalImageData.height
    ) {
        return null;
    }

    const originalSpatialScore = computeRegionSpatialCorrelation({
        imageData: originalImageData,
        alphaMap,
        region: { x: position.x, y: position.y, size }
    });
    if (originalSpatialScore < SMALL_ANCHOR_RELOCATION_MIN_ORIGINAL_SPATIAL) {
        return null;
    }

    const originalGradientScore = computeRegionGradientCorrelation({
        imageData: originalImageData,
        alphaMap,
        region: { x: position.x, y: position.y, size }
    });
    const candidateImageData = cloneImageData(originalImageData);
    removeWatermark(candidateImageData, alphaMap, position, { alphaGain });
    const spatialScore = computeRegionSpatialCorrelation({
        imageData: candidateImageData,
        alphaMap,
        region: { x: position.x, y: position.y, size }
    });
    const gradientScore = computeRegionGradientCorrelation({
        imageData: candidateImageData,
        alphaMap,
        region: { x: position.x, y: position.y, size }
    });
    const suppressionGain = originalSpatialScore - spatialScore;
    if (
        Math.abs(spatialScore) > SMALL_ANCHOR_RELOCATION_MAX_ACCEPTED_SPATIAL ||
        gradientScore > SMALL_ANCHOR_RELOCATION_MAX_ACCEPTED_GRADIENT ||
        suppressionGain < SMALL_ANCHOR_RELOCATION_MIN_SUPPRESSION_GAIN
    ) {
        return null;
    }

    const residualVisibility = assessWatermarkResidualVisibility({
        imageData: candidateImageData,
        position,
        alphaMap
    });
    if (residualVisibility.visible) return null;

    return {
        imageData: candidateImageData,
        alphaMap,
        position,
        config: { logoSize: size, marginRight, marginBottom },
        alphaGain,
        originalSpatialScore,
        originalGradientScore,
        spatialScore,
        gradientScore,
        suppressionGain,
        residualVisibility,
        cost: Math.abs(spatialScore) * 0.8 +
            Math.max(0, gradientScore) * 0.75 +
            Math.max(0, residualVisibility.positiveHaloLum ?? 0) * 0.01
    };
}

function refineSmallFixedLocalAnchorGeometry({
    originalImageData,
    currentAlphaMap,
    currentPosition,
    currentConfig,
    currentSource,
    currentGradientScore,
    currentResidualVisibility,
    alpha96,
    getAlphaMap
}) {
    if (!shouldRelocateSmallFixedLocalAnchor({
        source: currentSource,
        config: currentConfig,
        position: currentPosition,
        currentGradientScore,
        currentResidualVisibility
    })) {
        return null;
    }

    const currentSize = currentConfig.logoSize ?? currentPosition.width;
    const currentMarginRight = currentConfig.marginRight;
    const currentMarginBottom = currentConfig.marginBottom;
    if (![currentSize, currentMarginRight, currentMarginBottom].every(Number.isFinite)) {
        return null;
    }

    let best = null;
    const minSize = Math.max(SMALL_ANCHOR_RELOCATION_MIN_SIZE, currentSize);
    const maxSize = Math.min(SMALL_ANCHOR_RELOCATION_MAX_SIZE, currentSize + SMALL_ANCHOR_RELOCATION_SIZE_DELTA);
    const minMarginRight = Math.max(0, currentMarginRight - SMALL_ANCHOR_RELOCATION_MARGIN_DELTA);
    const maxMarginRight = currentMarginRight + SMALL_ANCHOR_RELOCATION_MARGIN_DELTA;
    const minMarginBottom = Math.max(0, currentMarginBottom - SMALL_ANCHOR_RELOCATION_MARGIN_DELTA);
    const maxMarginBottom = currentMarginBottom + SMALL_ANCHOR_RELOCATION_MARGIN_DELTA;

    for (let size = minSize; size <= maxSize; size++) {
        const alphaMap = resolveRelocationAlphaMap({
            size,
            currentAlphaMap,
            currentPosition,
            alpha96,
            getAlphaMap
        });
        if (!alphaMap) continue;

        for (let marginRight = minMarginRight; marginRight <= maxMarginRight; marginRight++) {
            for (let marginBottom = minMarginBottom; marginBottom <= maxMarginBottom; marginBottom++) {
                const candidateY = originalImageData.height - marginBottom - size;
                if (candidateY > currentPosition.y + 1) continue;
                for (const candidateAlphaGain of ALPHA_GAIN_CANDIDATES) {
                    const candidate = buildSmallAnchorRelocationCandidate({
                        originalImageData,
                        alphaMap,
                        size,
                        marginRight,
                        marginBottom,
                        alphaGain: candidateAlphaGain
                    });
                    if (!candidate) continue;
                    if (!best || candidate.cost < best.cost) {
                        best = candidate;
                    }
                }
            }
        }
    }

    return best;
}

export function processWatermarkImageData(imageData, options = {}) {
    const totalStartedAt = nowMs();
    const debugTimingsEnabled = options.debugTimings === true;
    const debugTimings = debugTimingsEnabled ? {} : null;
    const adaptiveMode = options.adaptiveMode || 'auto';
    const allowAdaptiveSearch =
        adaptiveMode !== 'never' &&
        adaptiveMode !== 'off';
    const originalImageData = cloneImageData(imageData);
    const { alpha48, alpha96 } = options;
    const alphaGainCandidates = ALPHA_GAIN_CANDIDATES;
    const alphaPriorityGains = STANDARD_ALPHA_PRIORITY_GAINS;

    if (!alpha48 || !alpha96) {
        throw new Error('processWatermarkImageData requires alpha48 and alpha96');
    }

    const defaultConfig = detectWatermarkConfig(originalImageData.width, originalImageData.height);
    const resolvedConfig = resolveInitialStandardConfig({
        imageData: originalImageData,
        defaultConfig,
        alpha48,
        alpha96
    });

    let config = resolvedConfig;
    let position = calculateWatermarkPosition(originalImageData.width, originalImageData.height, config);
    let alphaMap = config.logoSize === 96 ? alpha96 : alpha48;
    let source = 'standard';
    let adaptiveConfidence = null;
    let alphaGain = 1;
    let subpixelShift = null;
    let templateWarp = null;
    let decisionTier = null;
    let passCount = 0;
    let attemptedPassCount = 0;
    let passStopReason = null;
    let passes = null;
    const alphaAdjustmentStages = [];
    const recordAlphaAdjustmentStage = ({
        stage,
        fromAlphaGain,
        toAlphaGain,
        beforeSpatialScore,
        beforeGradientScore,
        afterSpatialScore,
        afterGradientScore,
        suppressionGain: stageSuppressionGain = null,
        cost = null,
        allowSameAlphaGain = false
    }) => {
        if (!stage || !Number.isFinite(fromAlphaGain) || !Number.isFinite(toAlphaGain)) return;
        if (!allowSameAlphaGain && Math.abs(fromAlphaGain - toAlphaGain) < 0.0001) return;

        alphaAdjustmentStages.push({
            stage,
            fromAlphaGain,
            toAlphaGain,
            beforeSpatialScore: Number.isFinite(beforeSpatialScore) ? beforeSpatialScore : null,
            beforeGradientScore: Number.isFinite(beforeGradientScore) ? beforeGradientScore : null,
            afterSpatialScore: Number.isFinite(afterSpatialScore) ? afterSpatialScore : null,
            afterGradientScore: Number.isFinite(afterGradientScore) ? afterGradientScore : null,
            suppressionGain: Number.isFinite(stageSuppressionGain) ? stageSuppressionGain : null,
            cost: Number.isFinite(cost) ? cost : null
        });
    };

    const initialSelectionStartedAt = nowMs();
    let initialSelection = selectInitialCandidate({
        originalImageData,
        config,
        position,
        alpha48,
        alpha96,
        alpha96Variants: options.alpha96Variants ?? null,
        getAlphaMap: options.getAlphaMap,
        allowAdaptiveSearch,
        allowAutomaticSearch: false,
        alphaGainCandidates,
        alphaPriorityGains
    });
    if (
        !initialSelection.selectedTrial &&
        options.aggressiveLocatedFallback !== false
    ) {
        const aggressiveSelection = selectInitialCandidate({
            originalImageData,
            config,
            position,
            alpha48,
            alpha96,
            alpha96Variants: options.alpha96Variants ?? null,
            getAlphaMap: options.getAlphaMap,
            allowAdaptiveSearch,
            allowAutomaticSearch: true,
            allowAggressiveStrongLocated: true,
            alphaGainCandidates,
            alphaPriorityGains
        });
        if (aggressiveSelection.selectedTrial) {
            initialSelection = {
                ...aggressiveSelection,
                source: aggressiveSelection.source.includes('aggressive-located')
                    ? aggressiveSelection.source
                    : `${aggressiveSelection.source}+aggressive-located`,
                decisionTier: aggressiveSelection.decisionTier || 'direct-match'
            };
        }
    }
    if (debugTimingsEnabled) {
        debugTimings.initialSelectionMs = nowMs() - initialSelectionStartedAt;
    }

    if (!initialSelection.selectedTrial) {
        if (debugTimingsEnabled) {
            debugTimings.totalMs = nowMs() - totalStartedAt;
        }
        return {
            imageData: originalImageData,
            meta: createWatermarkMeta({
                adaptiveConfidence: initialSelection.adaptiveConfidence,
                originalSpatialScore: initialSelection.standardSpatialScore,
                originalGradientScore: initialSelection.standardGradientScore,
                processedSpatialScore: initialSelection.standardSpatialScore,
                processedGradientScore: initialSelection.standardGradientScore,
                suppressionGain: 0,
                alphaGain: 1,
                source: 'skipped',
                decisionTier: initialSelection.decisionTier ?? 'insufficient',
                applied: false,
                skipReason: 'no-watermark-detected',
                selectionDebug: null
            }),
            debugTimings
        };
    }

    position = initialSelection.position;
    alphaMap = initialSelection.alphaMap;
    config = initialSelection.config;
    source = initialSelection.source;
    adaptiveConfidence = initialSelection.adaptiveConfidence;
    templateWarp = initialSelection.templateWarp;
    alphaGain = initialSelection.alphaGain;
    decisionTier = initialSelection.decisionTier;

    const selectedTrial = initialSelection.selectedTrial;
    const usePreviewAnchorFastCleanup = shouldUsePreviewAnchorFastCleanup(selectedTrial, position);
    const useKnown48EdgeCleanup = shouldUseKnown48EdgeCleanup({
        selectedTrial,
        position,
        source
    });
    const useV2SmallEdgeCleanup = shouldUseV2SmallEdgeCleanup({
        selectedTrial,
        position,
        source
    });

    let finalImageData = selectedTrial.imageData;

    let originalSpatialScore = selectedTrial.originalSpatialScore;
    let originalGradientScore = selectedTrial.originalGradientScore;

    const firstPassMetricsStartedAt = nowMs();
    const firstPassSpatialScore = computeRegionSpatialCorrelation({
        imageData: finalImageData,
        alphaMap,
        region: { x: position.x, y: position.y, size: position.width }
    });
    const firstPassGradientScore = computeRegionGradientCorrelation({
        imageData: finalImageData,
        alphaMap,
        region: { x: position.x, y: position.y, size: position.width }
    });
    const firstPassNearBlackRatio = calculateNearBlackRatio(finalImageData, position);
    const firstPassRecord = {
        index: 1,
        beforeSpatialScore: originalSpatialScore,
        beforeGradientScore: originalGradientScore,
        afterSpatialScore: firstPassSpatialScore,
        afterGradientScore: firstPassGradientScore,
        improvement: Math.abs(originalSpatialScore) - Math.abs(firstPassSpatialScore),
        gradientDelta: firstPassGradientScore - originalGradientScore,
        nearBlackRatio: firstPassNearBlackRatio
    };
    if (debugTimingsEnabled) {
        debugTimings.firstPassMetricsMs = nowMs() - firstPassMetricsStartedAt;
    }

    const firstPassClearedResidual = shouldStopAfterFirstPass({
        originalSpatialScore,
        originalGradientScore,
        firstPassSpatialScore,
        firstPassGradientScore
    });
    if (debugTimingsEnabled) {
        debugTimings.extraPassMs = 0;
    }
    passCount = 1;
    attemptedPassCount = 1;
    passStopReason = firstPassClearedResidual ? 'residual-low' : 'single-pass';
    passes = [firstPassRecord];

    const finalMetricsStartedAt = nowMs();
    const processedSpatialScore = computeRegionSpatialCorrelation({
        imageData: finalImageData,
        alphaMap,
        region: {
            x: position.x,
            y: position.y,
            size: position.width
        }
    });
    const processedGradientScore = computeRegionGradientCorrelation({
        imageData: finalImageData,
        alphaMap,
        region: {
            x: position.x,
            y: position.y,
            size: position.width
        }
    });
    if (debugTimingsEnabled) {
        debugTimings.finalMetricsMs = nowMs() - finalMetricsStartedAt;
    }
    let finalProcessedSpatialScore = processedSpatialScore;
    let finalProcessedGradientScore = processedGradientScore;
    let suppressionGain = originalSpatialScore - finalProcessedSpatialScore;

    const recalibrationStartedAt = nowMs();
    if (shouldRecalibrateAlphaStrength({
        originalScore: originalSpatialScore,
        processedScore: finalProcessedSpatialScore,
        suppressionGain
    })) {
        const originalNearBlackRatio = calculateNearBlackRatio(finalImageData, position);
        const recalibrated = recalibrateAlphaStrength({
            sourceImageData: finalImageData,
            alphaMap,
            position,
            originalSpatialScore,
            processedSpatialScore: finalProcessedSpatialScore,
            originalNearBlackRatio
        });

        if (recalibrated) {
            const beforeAlphaGain = alphaGain;
            const beforeSpatialScore = finalProcessedSpatialScore;
            const beforeGradientScore = finalProcessedGradientScore;
            const recalibratedGradientScore = computeRegionGradientCorrelation({
                imageData: recalibrated.imageData,
                alphaMap,
                region: {
                    x: position.x,
                    y: position.y,
                    size: position.width
                }
            });
            recordAlphaAdjustmentStage({
                stage: 'recalibration',
                fromAlphaGain: beforeAlphaGain,
                toAlphaGain: recalibrated.alphaGain,
                beforeSpatialScore,
                beforeGradientScore,
                afterSpatialScore: recalibrated.processedSpatialScore,
                afterGradientScore: recalibratedGradientScore,
                suppressionGain: recalibrated.suppressionGain
            });
            finalImageData = recalibrated.imageData;
            alphaGain = recalibrated.alphaGain;
            finalProcessedSpatialScore = recalibrated.processedSpatialScore;
            finalProcessedGradientScore = recalibratedGradientScore;
            suppressionGain = recalibrated.suppressionGain;
            source = source === 'adaptive' ? 'adaptive+gain' : `${source}+gain`;
        }
    }
    if (debugTimingsEnabled) {
        debugTimings.recalibrationMs = nowMs() - recalibrationStartedAt;
    }

    const overSubtractionStartedAt = nowMs();
    const overSubtractionRecalibrated = recalibrateOverSubtractedAlpha({
        originalImageData,
        alphaMap,
        position,
        currentSpatialScore: finalProcessedSpatialScore,
        currentGradientScore: finalProcessedGradientScore,
        currentAlphaGain: alphaGain,
        originalSpatialScore,
        originalNearBlackRatio: calculateNearBlackRatio(originalImageData, position)
    });
    if (overSubtractionRecalibrated) {
        recordAlphaAdjustmentStage({
            stage: 'over-subtraction-recalibration',
            fromAlphaGain: alphaGain,
            toAlphaGain: overSubtractionRecalibrated.alphaGain,
            beforeSpatialScore: finalProcessedSpatialScore,
            beforeGradientScore: finalProcessedGradientScore,
            afterSpatialScore: overSubtractionRecalibrated.spatialScore,
            afterGradientScore: overSubtractionRecalibrated.gradientScore,
            suppressionGain: overSubtractionRecalibrated.suppressionGain,
            cost: overSubtractionRecalibrated.cost
        });
        finalImageData = overSubtractionRecalibrated.imageData;
        alphaGain = overSubtractionRecalibrated.alphaGain;
        finalProcessedSpatialScore = overSubtractionRecalibrated.spatialScore;
        finalProcessedGradientScore = overSubtractionRecalibrated.gradientScore;
        suppressionGain = overSubtractionRecalibrated.suppressionGain;
        source = source.includes('+gain') ? source : `${source}+gain`;
    }
    if (debugTimingsEnabled) {
        debugTimings.overSubtractionRecalibrationMs = nowMs() - overSubtractionStartedAt;
    }

    const darkCatalogFineTuneStartedAt = nowMs();
    const darkCatalogFineTune = fineTuneDarkCatalogAlpha({
        originalImageData,
        alphaMap,
        position,
        source,
        currentSpatialScore: finalProcessedSpatialScore,
        currentGradientScore: finalProcessedGradientScore,
        currentAlphaGain: alphaGain,
        originalSpatialScore,
        originalGradientScore,
        originalNearBlackRatio: calculateNearBlackRatio(originalImageData, position)
    });
    if (darkCatalogFineTune) {
        recordAlphaAdjustmentStage({
            stage: 'dark-catalog-fine-alpha',
            fromAlphaGain: alphaGain,
            toAlphaGain: darkCatalogFineTune.alphaGain,
            beforeSpatialScore: finalProcessedSpatialScore,
            beforeGradientScore: finalProcessedGradientScore,
            afterSpatialScore: darkCatalogFineTune.spatialScore,
            afterGradientScore: darkCatalogFineTune.gradientScore,
            suppressionGain: darkCatalogFineTune.suppressionGain,
            cost: darkCatalogFineTune.cost
        });
        finalImageData = darkCatalogFineTune.imageData;
        alphaGain = darkCatalogFineTune.alphaGain;
        finalProcessedSpatialScore = darkCatalogFineTune.spatialScore;
        finalProcessedGradientScore = darkCatalogFineTune.gradientScore;
        suppressionGain = darkCatalogFineTune.suppressionGain;
        source = source.includes('+fine-alpha') ? source : `${source}+fine-alpha`;
    }
    if (debugTimingsEnabled) {
        debugTimings.darkCatalogFineTuneMs = nowMs() - darkCatalogFineTuneStartedAt;
    }

    const weakAlphaFineTuneStartedAt = nowMs();
    const weakAlphaFineTune = fineTuneWeakPositiveResidualAlpha({
        originalImageData,
        alphaMap,
        position,
        currentSpatialScore: finalProcessedSpatialScore,
        currentGradientScore: finalProcessedGradientScore,
        currentAlphaGain: alphaGain,
        originalSpatialScore,
        originalGradientScore,
        originalNearBlackRatio: calculateNearBlackRatio(originalImageData, position)
    });
    if (weakAlphaFineTune) {
        recordAlphaAdjustmentStage({
            stage: 'weak-positive-residual-fine-alpha',
            fromAlphaGain: alphaGain,
            toAlphaGain: weakAlphaFineTune.alphaGain,
            beforeSpatialScore: finalProcessedSpatialScore,
            beforeGradientScore: finalProcessedGradientScore,
            afterSpatialScore: weakAlphaFineTune.spatialScore,
            afterGradientScore: weakAlphaFineTune.gradientScore,
            suppressionGain: weakAlphaFineTune.suppressionGain,
            cost: weakAlphaFineTune.cost
        });
        finalImageData = weakAlphaFineTune.imageData;
        alphaGain = weakAlphaFineTune.alphaGain;
        finalProcessedSpatialScore = weakAlphaFineTune.spatialScore;
        finalProcessedGradientScore = weakAlphaFineTune.gradientScore;
        suppressionGain = weakAlphaFineTune.suppressionGain;
        source = source.includes('+fine-alpha') ? source : `${source}+fine-alpha`;
    }
    if (debugTimingsEnabled) {
        debugTimings.weakAlphaFineTuneMs = nowMs() - weakAlphaFineTuneStartedAt;
    }

    const previewBackgroundCleanupStartedAt = nowMs();
    const previewBackgroundBorderStd = ENABLE_VISUAL_POST_PROCESSING
        ? measureOuterBorderLuminanceStd(finalImageData, position)
        : 0;
    if (shouldApplyPreviewSmoothBackgroundCleanup({
        enabled: ENABLE_VISUAL_POST_PROCESSING,
        source,
        position,
        baselineSpatialScore: finalProcessedSpatialScore,
        borderStd: previewBackgroundBorderStd
    })) {
        const cleaned = applyPreviewSmoothBackgroundCleanup({
            imageData: finalImageData,
            position
        });
        const cleanedSpatialScore = computeRegionSpatialCorrelation({
            imageData: cleaned.imageData,
            alphaMap,
            region: {
                x: position.x,
                y: position.y,
                size: position.width
            }
        });
        const cleanedGradientScore = computeRegionGradientCorrelation({
            imageData: cleaned.imageData,
            alphaMap,
            region: {
                x: position.x,
                y: position.y,
                size: position.width
            }
        });
        const cleanedNearBlackRatio = calculateNearBlackRatio(cleaned.imageData, position);
        const currentNearBlackRatio = calculateNearBlackRatio(finalImageData, position);
        if (
            Math.abs(cleanedSpatialScore) <= Math.abs(finalProcessedSpatialScore) &&
            cleanedNearBlackRatio <= currentNearBlackRatio + MAX_NEAR_BLACK_RATIO_INCREASE
        ) {
            finalImageData = cleaned.imageData;
            finalProcessedSpatialScore = cleanedSpatialScore;
            finalProcessedGradientScore = cleanedGradientScore;
            suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
            source = `${source}+background-cleanup`;
        }
    }
    if (debugTimingsEnabled) {
        debugTimings.previewBackgroundCleanupMs = nowMs() - previewBackgroundCleanupStartedAt;
    }

    let previewEdgeCleanupElapsedMs = 0;
    const applyPreviewEdgeCleanup = () => {
        const previewEdgeStartedAt = nowMs();
        const previewEdgeRefined = refinePreviewResidualEdge({
            sourceImageData: finalImageData,
            alphaMap,
            position,
            source,
            baselineSpatialScore: finalProcessedSpatialScore,
            baselineGradientScore: finalProcessedGradientScore,
            minGradientImprovement: useKnown48EdgeCleanup
                ? KNOWN_48_EDGE_CLEANUP_MIN_GRADIENT_IMPROVEMENT
                : (
                    useV2SmallEdgeCleanup
                        ? V2_SMALL_EDGE_CLEANUP_MIN_GRADIENT_IMPROVEMENT
                        : PREVIEW_EDGE_CLEANUP_MIN_GRADIENT_IMPROVEMENT
                ),
            maxSpatialDrift: useKnown48EdgeCleanup
                ? KNOWN_48_EDGE_CLEANUP_MAX_SPATIAL_DRIFT
                : (
                    useV2SmallEdgeCleanup
                        ? V2_SMALL_EDGE_CLEANUP_MAX_SPATIAL_DRIFT
                        : PREVIEW_EDGE_CLEANUP_MAX_SPATIAL_DRIFT
                ),
            allowAggressivePresets: usePreviewAnchorFastCleanup,
            mode: useKnown48EdgeCleanup
                ? 'known-48'
                : (useV2SmallEdgeCleanup ? 'v2-small' : 'preview')
        });
        previewEdgeCleanupElapsedMs += nowMs() - previewEdgeStartedAt;

        if (!previewEdgeRefined) {
            return false;
        }

        finalImageData = previewEdgeRefined.imageData;
        finalProcessedSpatialScore = previewEdgeRefined.spatialScore;
        finalProcessedGradientScore = previewEdgeRefined.gradientScore;
        suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
        source = `${source}+${useV2SmallEdgeCleanup ? 'v2-small-edge-cleanup' : 'edge-cleanup'}`;
        return true;
    };

    const subpixelStartedAt = nowMs();
    if (
        ENABLE_VISUAL_POST_PROCESSING &&
        !usePreviewAnchorFastCleanup &&
        finalProcessedSpatialScore <= 0.3 &&
        finalProcessedGradientScore >= OUTLINE_REFINEMENT_THRESHOLD
    ) {
        const originalNearBlackRatio = calculateNearBlackRatio(finalImageData, position);
        const baselineShift = templateWarp ?? { dx: 0, dy: 0, scale: 1 };
        const refined = refineSubpixelOutline({
            sourceImageData: finalImageData,
            alphaMap,
            position,
            alphaGain,
            originalNearBlackRatio,
            baselineSpatialScore: finalProcessedSpatialScore,
            baselineGradientScore: finalProcessedGradientScore,
            baselineShift,
            minGain: OUTLINE_REFINEMENT_MIN_GAIN,
            shiftCandidates: SUBPIXEL_REFINE_SHIFTS,
            scaleCandidates: SUBPIXEL_REFINE_SCALES,
            minGradientImprovement: 0.04,
            maxSpatialDrift: 0.08
        });

        if (refined) {
            recordAlphaAdjustmentStage({
                stage: 'subpixel-outline-refinement',
                fromAlphaGain: alphaGain,
                toAlphaGain: refined.alphaGain,
                beforeSpatialScore: finalProcessedSpatialScore,
                beforeGradientScore: finalProcessedGradientScore,
                afterSpatialScore: refined.spatialScore,
                afterGradientScore: refined.gradientScore,
                cost: refined.cost
            });
            finalImageData = refined.imageData;
            alphaMap = refined.alphaMap;
            alphaGain = refined.alphaGain;
            finalProcessedSpatialScore = refined.spatialScore;
            finalProcessedGradientScore = refined.gradientScore;
            suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
            source = `${source}+subpixel`;
            subpixelShift = refined.shift;
        }
    }
    if (debugTimingsEnabled) {
        debugTimings.subpixelRefinementMs = nowMs() - subpixelStartedAt;
    }

    let previewEdgeCleanupPassCount = 0;
    const shouldRunEdgeCleanup = ENABLE_VISUAL_POST_PROCESSING || useKnown48EdgeCleanup || useV2SmallEdgeCleanup;
    while (shouldRunEdgeCleanup && previewEdgeCleanupPassCount < PREVIEW_EDGE_CLEANUP_MAX_APPLIED_PASSES) {
        if (!applyPreviewEdgeCleanup()) {
            break;
        }
        previewEdgeCleanupPassCount++;
    }

    if (useKnown48EdgeCleanup) {
        let flatFillPassCount = 0;
        while (flatFillPassCount < KNOWN_48_FLAT_FILL_MAX_APPLIED_PASSES) {
            const flatBackgroundRefined = refineKnown48FlatBackgroundResidual({
                sourceImageData: finalImageData,
                alphaMap,
                position,
                baselineSpatialScore: finalProcessedSpatialScore,
                baselineGradientScore: finalProcessedGradientScore,
                minGradientImprovement: flatFillPassCount === 0
                    ? KNOWN_48_FLAT_FILL_MIN_GRADIENT_IMPROVEMENT
                    : KNOWN_48_FLAT_FILL_SECOND_PASS_MIN_GRADIENT_IMPROVEMENT
            });

            if (!flatBackgroundRefined) {
                break;
            }

            recordAlphaAdjustmentStage({
                stage: 'known-48-flat-background-fill',
                fromAlphaGain: alphaGain,
                toAlphaGain: alphaGain,
                beforeSpatialScore: finalProcessedSpatialScore,
                beforeGradientScore: finalProcessedGradientScore,
                afterSpatialScore: flatBackgroundRefined.spatialScore,
                afterGradientScore: flatBackgroundRefined.gradientScore,
                suppressionGain: originalSpatialScore - flatBackgroundRefined.spatialScore,
                cost: flatBackgroundRefined.cost,
                allowSameAlphaGain: true
            });
            finalImageData = flatBackgroundRefined.imageData;
            finalProcessedSpatialScore = flatBackgroundRefined.spatialScore;
            finalProcessedGradientScore = flatBackgroundRefined.gradientScore;
            suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
            source = `${source}+flat-fill`;
            flatFillPassCount++;
        }

        const lumaEdgeRefined = refineKnown48LumaEdgeResidual({
            sourceImageData: finalImageData,
            alphaMap,
            position,
            baselineSpatialScore: finalProcessedSpatialScore,
            baselineGradientScore: finalProcessedGradientScore
        });

        if (lumaEdgeRefined) {
            recordAlphaAdjustmentStage({
                stage: 'known-48-luma-edge-correction',
                fromAlphaGain: alphaGain,
                toAlphaGain: alphaGain,
                beforeSpatialScore: finalProcessedSpatialScore,
                beforeGradientScore: finalProcessedGradientScore,
                afterSpatialScore: lumaEdgeRefined.spatialScore,
                afterGradientScore: lumaEdgeRefined.gradientScore,
                suppressionGain: originalSpatialScore - lumaEdgeRefined.spatialScore,
                cost: lumaEdgeRefined.cost,
                allowSameAlphaGain: true
            });
            finalImageData = lumaEdgeRefined.imageData;
            finalProcessedSpatialScore = lumaEdgeRefined.spatialScore;
            finalProcessedGradientScore = lumaEdgeRefined.gradientScore;
            suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
            source = `${source}+luma-edge`;
        }
    }

    const newMargin96FlatFillRefined = refineNewMargin96FlatBackgroundResidual({
        sourceImageData: finalImageData,
        alphaMap,
        position,
        config,
        alphaGain,
        baselineSpatialScore: finalProcessedSpatialScore,
        baselineGradientScore: finalProcessedGradientScore
    });

    if (newMargin96FlatFillRefined) {
        recordAlphaAdjustmentStage({
            stage: 'new-margin-96-flat-background-fill',
            fromAlphaGain: alphaGain,
            toAlphaGain: alphaGain,
            beforeSpatialScore: finalProcessedSpatialScore,
            beforeGradientScore: finalProcessedGradientScore,
            afterSpatialScore: newMargin96FlatFillRefined.spatialScore,
            afterGradientScore: newMargin96FlatFillRefined.gradientScore,
            suppressionGain: originalSpatialScore - newMargin96FlatFillRefined.spatialScore,
            cost: newMargin96FlatFillRefined.cost,
            allowSameAlphaGain: true
        });
        finalImageData = newMargin96FlatFillRefined.imageData;
        finalProcessedSpatialScore = newMargin96FlatFillRefined.spatialScore;
        finalProcessedGradientScore = newMargin96FlatFillRefined.gradientScore;
        suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
        source = `${source}+flat-fill`;
    }

    const smallPreviewRefinementStartedAt = nowMs();
    const smallPreviewRefined = ENABLE_VISUAL_POST_PROCESSING
        ? refineSmallPreviewAnchorCandidate({
            originalImageData,
            source,
            position,
            originalGradientScore,
            currentSpatialScore: finalProcessedSpatialScore,
            currentGradientScore: finalProcessedGradientScore,
            getAlphaMap: options.getAlphaMap
        })
        : null;
    if (smallPreviewRefined) {
        recordAlphaAdjustmentStage({
            stage: 'small-preview-refinement',
            fromAlphaGain: alphaGain,
            toAlphaGain: smallPreviewRefined.alphaGain,
            beforeSpatialScore: finalProcessedSpatialScore,
            beforeGradientScore: finalProcessedGradientScore,
            afterSpatialScore: smallPreviewRefined.spatialScore,
            afterGradientScore: smallPreviewRefined.gradientScore,
            suppressionGain: smallPreviewRefined.suppressionGain,
            cost: smallPreviewRefined.cost
        });
        finalImageData = smallPreviewRefined.imageData;
        alphaMap = smallPreviewRefined.alphaMap;
        position = smallPreviewRefined.position;
        config = {
            logoSize: position.width,
            marginRight: originalImageData.width - position.x - position.width,
            marginBottom: originalImageData.height - position.y - position.height
        };
        alphaGain = smallPreviewRefined.alphaGain;
        originalSpatialScore = smallPreviewRefined.originalSpatialScore;
        originalGradientScore = smallPreviewRefined.originalGradientScore;
        finalProcessedSpatialScore = smallPreviewRefined.spatialScore;
        finalProcessedGradientScore = smallPreviewRefined.gradientScore;
        suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
        source = `${source}+small-preview-refine`;
    }

    const smallFixedLocalResidualVisibility = assessWatermarkResidualVisibility({
        imageData: finalImageData,
        position,
        alphaMap
    });
    const smallFixedLocalRelocated = refineSmallFixedLocalAnchorGeometry({
        originalImageData,
        currentAlphaMap: alphaMap,
        currentPosition: position,
        currentConfig: config,
        currentSource: source,
        currentGradientScore: finalProcessedGradientScore,
        currentResidualVisibility: smallFixedLocalResidualVisibility,
        alpha96,
        getAlphaMap: options.getAlphaMap
    });
    if (smallFixedLocalRelocated) {
        recordAlphaAdjustmentStage({
            stage: 'small-fixed-local-anchor-relocation',
            fromAlphaGain: alphaGain,
            toAlphaGain: smallFixedLocalRelocated.alphaGain,
            beforeSpatialScore: finalProcessedSpatialScore,
            beforeGradientScore: finalProcessedGradientScore,
            afterSpatialScore: smallFixedLocalRelocated.spatialScore,
            afterGradientScore: smallFixedLocalRelocated.gradientScore,
            suppressionGain: smallFixedLocalRelocated.suppressionGain,
            cost: smallFixedLocalRelocated.cost,
            allowSameAlphaGain: true
        });
        finalImageData = smallFixedLocalRelocated.imageData;
        alphaMap = smallFixedLocalRelocated.alphaMap;
        position = smallFixedLocalRelocated.position;
        config = smallFixedLocalRelocated.config;
        alphaGain = smallFixedLocalRelocated.alphaGain;
        originalSpatialScore = smallFixedLocalRelocated.originalSpatialScore;
        originalGradientScore = smallFixedLocalRelocated.originalGradientScore;
        finalProcessedSpatialScore = smallFixedLocalRelocated.spatialScore;
        finalProcessedGradientScore = smallFixedLocalRelocated.gradientScore;
        suppressionGain = smallFixedLocalRelocated.suppressionGain;
        source = `${source}+small-anchor-relocated`;
    }

    const locatedAggressiveStartedAt = nowMs();
    if (
        options.locatedAggressiveRemoval !== false &&
        smallFixedLocalRelocated?.residualVisibility?.visible !== false
    ) {
        const aggressiveRefined = refineLocatedAggressiveRemoval({
            originalImageData,
            currentImageData: finalImageData,
            alphaMap,
            position,
            currentSpatialScore: finalProcessedSpatialScore,
            currentGradientScore: finalProcessedGradientScore,
            currentAlphaGain: alphaGain
        });
        if (aggressiveRefined) {
            recordAlphaAdjustmentStage({
                stage: 'located-aggressive-removal',
                fromAlphaGain: alphaGain,
                toAlphaGain: aggressiveRefined.alphaGain,
                beforeSpatialScore: finalProcessedSpatialScore,
                beforeGradientScore: finalProcessedGradientScore,
                afterSpatialScore: aggressiveRefined.spatialScore,
                afterGradientScore: aggressiveRefined.gradientScore,
                suppressionGain: originalSpatialScore - aggressiveRefined.spatialScore,
                cost: aggressiveRefined.cost,
                allowSameAlphaGain: true
            });
            passes.push({
                index: passes.length + 1,
                beforeSpatialScore: finalProcessedSpatialScore,
                beforeGradientScore: finalProcessedGradientScore,
                afterSpatialScore: aggressiveRefined.spatialScore,
                afterGradientScore: aggressiveRefined.gradientScore,
                improvement: Math.abs(finalProcessedSpatialScore) - Math.abs(aggressiveRefined.spatialScore),
                gradientDelta: aggressiveRefined.gradientScore - finalProcessedGradientScore,
                nearBlackRatio: aggressiveRefined.nearBlackRatio
            });
            finalImageData = aggressiveRefined.imageData;
            alphaGain = aggressiveRefined.alphaGain;
            finalProcessedSpatialScore = aggressiveRefined.spatialScore;
            finalProcessedGradientScore = aggressiveRefined.gradientScore;
            suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
            passCount += Math.max(1, aggressiveRefined.repeatCount || 1);
            attemptedPassCount += Math.max(1, aggressiveRefined.repeatCount || 1);
            passStopReason = aggressiveRefined.edgeCleanup
                ? 'located-aggressive-edge-cleanup'
                : 'located-aggressive-alpha';
            source = source.includes('+located-aggressive')
                ? source
                : `${source}+located-aggressive`;
        }
    }
    if (debugTimingsEnabled) {
        debugTimings.previewEdgeCleanupMs = previewEdgeCleanupElapsedMs;
        debugTimings.smallPreviewRefinementMs = nowMs() - smallPreviewRefinementStartedAt;
        debugTimings.locatedAggressiveRemovalMs = nowMs() - locatedAggressiveStartedAt;
        debugTimings.totalMs = nowMs() - totalStartedAt;
    }

    const residualVisibility = assessWatermarkResidualVisibility({
        imageData: finalImageData,
        position,
        alphaMap
    });

    return {
        imageData: finalImageData,
        meta: createWatermarkMeta({
            position,
            config,
            adaptiveConfidence,
            originalSpatialScore,
            originalGradientScore,
            processedSpatialScore: finalProcessedSpatialScore,
            processedGradientScore: finalProcessedGradientScore,
            suppressionGain,
            residualVisibility,
            templateWarp,
            alphaGain,
            passCount,
            attemptedPassCount,
            passStopReason,
            passes,
            source,
            decisionTier,
            applied: true,
            subpixelShift,
            alphaAdjustmentStages,
            selectionDebug: createSelectionDebugSummary({
                selectedTrial,
                selectionSource: initialSelection.source,
                initialConfig: resolvedConfig,
                initialPosition: calculateWatermarkPosition(
                    originalImageData.width,
                    originalImageData.height,
                    resolvedConfig
                )
            })
        }),
        debugTimings
    };
}
