import { removeWatermark } from './blendModes.js';
import {
    computeRegionGradientCorrelation,
    computeRegionSpatialCorrelation,
    warpAlphaMap
} from './adaptiveDetector.js';
import {
    calculateNearBlackRatio,
    scoreRegion,
    selectInitialCandidate
} from './candidateSelector.js';
import {
    assessAlphaBandHalo,
    assessRemovalDiffArtifacts
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
    { name: 'weak-alpha-light', alphaGain: 0.7 },
    { name: 'weak-alpha-mid', alphaGain: 0.85 },
    { name: 'weak-alpha-conservative', alphaGain: 0.55 }
]);
const ALPHA_GAIN_CANDIDATES = ALPHA_PARAMETER_GROUPS.map((group) => group.alphaGain);
const ENABLE_VISUAL_POST_PROCESSING = false;
const CATALOG_DARK_ALPHA_GAIN_CANDIDATES = Object.freeze([0.9, 0.85, 0.8, 0.95, 0.7, 0.6]);
const STANDARD_ALPHA_PRIORITY_GAINS = ALPHA_PARAMETER_GROUPS
    .filter((group) => group.standardPriority === true)
    .map((group) => group.alphaGain);
const PREVIEW_EDGE_CLEANUP_MAX_SIZE = 40;
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

    return { logoSize, marginRight, marginBottom };
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
    selectionDebug = null
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
            suppressionGain
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
        selectionDebug
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

    for (let step = 1; step <= fineStepCount; step++) {
        const alphaGain = Number((currentAlphaGain + step * OVER_SUBTRACTION_FINE_ALPHA_STEP).toFixed(2));
        if (alphaGain >= 1) continue;

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
            absSpatialImprovement < WEAK_ALPHA_FINE_TUNE_MIN_ABS_SPATIAL_IMPROVEMENT ||
            gradientIncrease > WEAK_ALPHA_FINE_TUNE_MAX_GRADIENT_INCREASE
        ) {
            continue;
        }

        const cost = artifacts
            ? artifacts.visualArtifactCost + Math.max(0, gradientIncrease) * 0.25
            : Math.abs(spatialScore) + Math.max(0, gradientIncrease) * 0.25 + nearBlackRatio * 2;
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

function shouldRefinePreviewResidualEdge({
    source,
    position,
    baselineSpatialScore,
    baselineGradientScore,
    baselinePositiveHalo
}) {
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

    for (let row = 0; row < regionSize; row++) {
        for (let col = 0; col < regionSize; col++) {
            const alpha = alphaMap[row * regionSize + col];
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
                        neighborAlpha = alphaMap[localY * regionSize + localX];
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

            const blend = Math.max(0, Math.min(1, strength * alpha / maxAlphaSafe));
            const pixelIndex = ((position.y + row) * imageWidth + (position.x + col)) * 4;
            candidate.data[pixelIndex] = Math.round(data[pixelIndex] * (1 - blend) + (sumR / sumWeight) * blend);
            candidate.data[pixelIndex + 1] = Math.round(data[pixelIndex + 1] * (1 - blend) + (sumG / sumWeight) * blend);
            candidate.data[pixelIndex + 2] = Math.round(data[pixelIndex + 2] * (1 - blend) + (sumB / sumWeight) * blend);
        }
    }

    return candidate;
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
    allowAggressivePresets = false
}) {
    const baselineHalo = assessAlphaBandHalo({
        imageData: sourceImageData,
        position,
        alphaMap
    });
    const baselinePositiveHalo = baselineHalo.positiveDeltaLum;
    if (!shouldRefinePreviewResidualEdge({
        source,
        position,
        baselineSpatialScore,
        baselineGradientScore,
        baselinePositiveHalo
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

    const initialSelectionStartedAt = nowMs();
    const initialSelection = selectInitialCandidate({
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
            finalImageData = recalibrated.imageData;
            alphaGain = recalibrated.alphaGain;
            finalProcessedSpatialScore = recalibrated.processedSpatialScore;
            finalProcessedGradientScore = computeRegionGradientCorrelation({
                imageData: finalImageData,
                alphaMap,
                region: {
                    x: position.x,
                    y: position.y,
                    size: position.width
                }
            });
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
        originalNearBlackRatio: calculateNearBlackRatio(originalImageData, position)
    });
    if (weakAlphaFineTune) {
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
            allowAggressivePresets: usePreviewAnchorFastCleanup
        });
        previewEdgeCleanupElapsedMs += nowMs() - previewEdgeStartedAt;

        if (!previewEdgeRefined) {
            return false;
        }

        finalImageData = previewEdgeRefined.imageData;
        finalProcessedSpatialScore = previewEdgeRefined.spatialScore;
        finalProcessedGradientScore = previewEdgeRefined.gradientScore;
        suppressionGain = originalSpatialScore - finalProcessedSpatialScore;
        source = `${source}+edge-cleanup`;
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
    while (ENABLE_VISUAL_POST_PROCESSING && previewEdgeCleanupPassCount < PREVIEW_EDGE_CLEANUP_MAX_APPLIED_PASSES) {
        if (!applyPreviewEdgeCleanup()) {
            break;
        }
        previewEdgeCleanupPassCount++;
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
    if (debugTimingsEnabled) {
        debugTimings.previewEdgeCleanupMs = previewEdgeCleanupElapsedMs;
        debugTimings.smallPreviewRefinementMs = nowMs() - smallPreviewRefinementStartedAt;
        debugTimings.totalMs = nowMs() - totalStartedAt;
    }

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
