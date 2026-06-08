import { removeWatermark } from './blendModes.js';
import {
    computeRegionGradientCorrelation,
    computeRegionSpatialCorrelation,
    detectAdaptiveWatermarkRegion,
    interpolateAlphaMap,
    shouldAttemptAdaptiveFallback,
    warpAlphaMap
} from './adaptiveDetector.js';
import {
    assessReferenceTextureAlignment,
    assessReferenceTextureAlignmentFromStats,
    calculateNearBlackRatio,
    cloneImageData,
    getRegionTextureStats,
    scoreRegion
} from './restorationMetrics.js';
import {
    hasReliableAdaptiveWatermarkSignal,
    hasReliableStandardWatermarkSignal
} from './watermarkPresence.js';
import {
    matchOfficialGeminiImageSize,
    resolveGeminiWatermarkSearchConfigs
} from './geminiSizeCatalog.js';

const MAX_NEAR_BLACK_RATIO_INCREASE = 0.05;
const VALIDATION_MIN_IMPROVEMENT = 0.08;
const VALIDATION_TARGET_RESIDUAL = 0.22;
const VALIDATION_MAX_GRADIENT_INCREASE = 0.04;
const VALIDATION_MIN_CONFIDENCE_FOR_ADAPTIVE_TRIAL = 0.25;
const STANDARD_FAST_PATH_RESIDUAL_THRESHOLD = 0.22;
const STANDARD_FAST_PATH_GRADIENT_THRESHOLD = 0.08;
const FIXED_CORE_MAX_ACCEPTED_SPATIAL_RESIDUAL = 0.36;
const STANDARD_EXPAND_CATALOG_MIN_ORIGINAL_GRADIENT = 0.12;
const WEAK_ALPHA_PRIORITY_CLEAN_GRADIENT_THRESHOLD = 0.12;
const STRONG_ORIGINAL_SIGNAL_SPATIAL_ADVANTAGE = 0.2;
const STRONG_ORIGINAL_SIGNAL_GRADIENT_ADVANTAGE = 0.25;
const STANDARD_NEARBY_SEARCH_RESIDUAL_THRESHOLD = 0.18;
const STANDARD_NEARBY_SEARCH_GRADIENT_THRESHOLD = 0.05;
const STANDARD_LOCAL_SHIFT_STRONG_BASE_GRADIENT_SCORE = 0.35;
const STANDARD_LOCAL_SHIFT_STRONG_BASE_SPATIAL_SCORE = 0.8;
const STANDARD_LOCAL_SHIFT_CANONICAL_MIN_GRADIENT_SCORE = 0.2;
const STANDARD_LOCAL_SHIFT_CANONICAL_MIN_SPATIAL_SCORE = 0.22;
const STANDARD_LOCAL_SHIFT_WEAK_CANDIDATE_GRADIENT_SCORE = 0.12;
const STANDARD_LOCAL_SHIFT_WEAK_CANDIDATE_SPATIAL_SCORE = 0.65;
const STANDARD_LOCAL_SHIFT_MIN_VALIDATION_ADVANTAGE = 0.3;
const STANDARD_LOCAL_SHIFT_SKIP_PROCESSED_GRADIENT_THRESHOLD = 0.02;
const STANDARD_LOCAL_SHIFT_PRESERVE_CLEAN_BASE_GRADIENT_THRESHOLD = 0.02;
const STANDARD_LOCAL_SHIFT_MAX_CANDIDATE_GRADIENT_FOR_CLEAN_BASE = 0.03;
const STANDARD_PRESERVE_GRADIENT_DELTA = 0.25;
const STANDARD_PRESERVE_MAX_RESIDUAL = 0.4;
const STANDARD_PRESERVE_MIN_IMPROVEMENT = 0.3;
const STANDARD_TEXT_OVERLAP_MIN_SPATIAL_SCORE = 0.22;
const STANDARD_TEXT_OVERLAP_MIN_GRADIENT_SCORE = 0.18;
const STANDARD_TEXT_OVERLAP_MIN_IMPROVEMENT = 0.25;
const STANDARD_TEXT_OVERLAP_MAX_RESIDUAL = 0.1;
const STANDARD_TEXT_OVERLAP_MIN_GRADIENT_DROP = 0.1;
const STANDARD_HARD_REJECT_OVERRIDE_MIN_SPATIAL_SCORE = 0.9;
const STANDARD_HARD_REJECT_OVERRIDE_MIN_GRADIENT_SCORE = 0.7;
const STANDARD_HARD_REJECT_OVERRIDE_MAX_RESIDUAL = 0.08;
const STANDARD_HARD_REJECT_OVERRIDE_MAX_GRADIENT = 0.1;
const STANDARD_HARD_REJECT_OVERRIDE_MIN_IMPROVEMENT = 0.7;
const STANDARD_HARD_REJECT_OVERRIDE_MAX_NEAR_BLACK_INCREASE = 0.01;
const STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MIN_SPATIAL_SCORE = 0.9;
const STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MIN_GRADIENT_SCORE = 0.7;
const STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MAX_RESIDUAL = 0.22;
const STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MAX_GRADIENT = 0.32;
const STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MIN_IMPROVEMENT = 0.7;
const STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MIN_GRADIENT_DROP = 0.6;
const STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MAX_NEAR_BLACK_INCREASE = 0.4;
const TEMPLATE_ALIGN_SHIFTS = [-0.5, -0.25, 0, 0.25, 0.5];
const TEMPLATE_ALIGN_SCALES = [0.99, 1, 1.01];
const STANDARD_NEARBY_SHIFTS = [-12, -8, -4, 0, 4, 8, 12];
const STANDARD_FINE_LOCAL_SHIFTS = [-2, -1, 0, 1, 2];
const STANDARD_SIZE_JITTERS = [-12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12];
const PREVIEW_ANCHOR_MIN_SIZE = 24;
const PREVIEW_ANCHOR_MAX_SIZE_RATIO = 1.05;
const PREVIEW_ANCHOR_MIN_SIZE_RATIO = 0.55;
const PREVIEW_ANCHOR_MARGIN_WINDOW = 16;
const PREVIEW_ANCHOR_MARGIN_EXTENSION = 8;
const PREVIEW_ANCHOR_SIZE_STEP = 2;
const PREVIEW_ANCHOR_MARGIN_STEP = 2;
const PREVIEW_ANCHOR_TOP_K = 8;
const PREVIEW_ANCHOR_MIN_SCORE = 0.2;
const PREVIEW_ANCHOR_LOCAL_DELTAS = [-1, 0, 1];
const PREVIEW_TEMPLATE_ALIGN_SHIFTS = [-1, -0.5, 0, 0.5, 1];
const PREVIEW_TEMPLATE_ALIGN_SCALES = [0.985, 1, 1.015];
const PREVIEW_ANCHOR_GAIN_SKIP_RESIDUAL_THRESHOLD = 0.22;
const PREVIEW_ANCHOR_GAIN_SKIP_GRADIENT_THRESHOLD = 0.24;

export { assessReferenceTextureAlignment, calculateNearBlackRatio, scoreRegion } from './restorationMetrics.js';

const ORIGIN_REGION = Object.freeze({ x: 0, y: 0 });

function mergeCandidateProvenance(...provenanceParts) {
    const merged = {};
    for (const provenance of provenanceParts) {
        if (!provenance || typeof provenance !== 'object') continue;
        Object.assign(merged, provenance);
    }

    return Object.keys(merged).length > 0 ? merged : null;
}

function isProjectedPreviewCatalogConfig(originalImageData, candidateConfig, baseConfig) {
    if (!originalImageData || !candidateConfig) return false;
    if (candidateConfig.fixedVariant === true) return false;
    if (matchOfficialGeminiImageSize(originalImageData.width, originalImageData.height)) return false;

    return candidateConfig.logoSize < 48 ||
        (
            candidateConfig.logoSize <= 48 &&
            candidateConfig.marginRight < 32 &&
            candidateConfig.marginBottom < 32
        );
}

function buildStandardCandidateSeeds({
    originalImageData,
    config,
    position,
    alpha48,
    alpha96,
    alpha96Variants = null,
    getAlphaMap,
    resolveAlphaMap = null,
    includeCatalogVariants = true
}) {
    const configs = includeCatalogVariants
        ? resolveGeminiWatermarkSearchConfigs(
            originalImageData.width,
            originalImageData.height,
            config
        )
        : [config];
    const seeds = [];

    for (const candidateConfig of configs) {
        const candidatePosition = candidateConfig === config
            ? position
            : {
                x: originalImageData.width - candidateConfig.marginRight - candidateConfig.logoSize,
                y: originalImageData.height - candidateConfig.marginBottom - candidateConfig.logoSize,
                width: candidateConfig.logoSize,
                height: candidateConfig.logoSize
            };
        if (
            candidatePosition.x < 0 ||
            candidatePosition.y < 0 ||
            candidatePosition.x + candidatePosition.width > originalImageData.width ||
            candidatePosition.y + candidatePosition.height > originalImageData.height
        ) {
            continue;
        }

        const alphaMap = resolveAlphaMapForConfig(candidateConfig, {
            alpha48,
            alpha96,
            alpha96Variants,
            getAlphaMap,
            resolveAlphaMap
        });
        if (!alphaMap) continue;

        const projectedPreviewCatalog = isProjectedPreviewCatalogConfig(
            originalImageData,
            candidateConfig,
            config
        );

        seeds.push({
            config: candidateConfig,
            position: candidatePosition,
            alphaMap,
            source: projectedPreviewCatalog
                ? 'standard+preview-anchor'
                : (candidateConfig === config ? 'standard' : 'standard+catalog'),
            provenance: mergeCandidateProvenance(
                candidateConfig === config ? null : { catalogVariant: true },
                candidateConfig.fixedVariant === true ? { fixedVariant: true } : null,
                projectedPreviewCatalog ? { previewAnchor: true } : null,
                candidateConfig.alphaVariant ? { alphaVariant: candidateConfig.alphaVariant } : null
            )
        });
    }

    return seeds;
}

function inferDecisionTier(candidate, { directMatch = false } = {}) {
    if (!candidate) return 'insufficient';
    if (directMatch) return 'direct-match';
    if (candidate.source?.includes('validated')) return 'validated-match';
    if (candidate.accepted) return 'validated-match';
    return 'safe-removal';
}

function shouldEscalateSearch(candidate) {
    if (!candidate) return true;

    return Math.abs(candidate.processedSpatialScore) > STANDARD_FAST_PATH_RESIDUAL_THRESHOLD ||
        Math.max(0, candidate.processedGradientScore) > STANDARD_FAST_PATH_GRADIENT_THRESHOLD;
}

function shouldExpandCatalogForWeakOriginalStandardEvidence(candidate) {
    if (!candidate) return true;
    if (!isStandardCandidateSource(candidate)) return false;
    if (candidate?.provenance?.catalogVariant === true) return false;

    const originalGradient = Number(candidate.originalGradientScore);
    if (!Number.isFinite(originalGradient)) return false;

    return originalGradient < STANDARD_EXPAND_CATALOG_MIN_ORIGINAL_GRADIENT;
}

function shouldSearchNearbyStandardCandidate(candidate, originalImageData) {
    if (!candidate) return true;

    return Number(candidate.position?.width) >= 72 &&
        Number(originalImageData?.height) > Number(originalImageData?.width) * 1.25 &&
        (
            Math.abs(candidate.processedSpatialScore) > STANDARD_NEARBY_SEARCH_RESIDUAL_THRESHOLD ||
            Math.max(0, candidate.processedGradientScore) > STANDARD_NEARBY_SEARCH_GRADIENT_THRESHOLD
        );
}

function resolveAlphaMapForSize(size, { alpha48, alpha96, getAlphaMap } = {}) {
    if (size === 48) return alpha48;
    if (size === 96) return alpha96;

    const provided = typeof getAlphaMap === 'function' ? getAlphaMap(size) : null;
    if (provided) return provided;

    return alpha96 ? interpolateAlphaMap(alpha96, 96, size) : null;
}

function resolveAlphaMapForConfig(config, {
    alpha48,
    alpha96,
    alpha96Variants = null,
    getAlphaMap,
    resolveAlphaMap = null
} = {}) {
    if (!config) return null;
    if (config.alphaVariant && config.logoSize === 96 && alpha96Variants) {
        return alpha96Variants[config.alphaVariant] ?? null;
    }
    return typeof resolveAlphaMap === 'function'
        ? resolveAlphaMap(config.logoSize)
        : resolveAlphaMapForSize(config.logoSize, {
            alpha48,
            alpha96,
            getAlphaMap
        });
}

function createAlphaMapResolver({ alpha48, alpha96, getAlphaMap }) {
    const cache = new Map();

    return (size) => {
        if (cache.has(size)) {
            return cache.get(size);
        }

        const resolved = resolveAlphaMapForSize(size, {
            alpha48,
            alpha96,
            getAlphaMap
        });
        cache.set(size, resolved);
        return resolved;
    };
}

function isPreviewAnchorGainSearchRequired(candidate) {
    if (!candidate) return true;

    return Math.abs(candidate.processedSpatialScore) > PREVIEW_ANCHOR_GAIN_SKIP_RESIDUAL_THRESHOLD ||
        Math.max(0, candidate.processedGradientScore) > PREVIEW_ANCHOR_GAIN_SKIP_GRADIENT_THRESHOLD;
}

function isCleanStandardAlphaCandidate(candidate) {
    if (!candidate?.accepted) return false;

    return Math.abs(candidate.processedSpatialScore) <= STANDARD_FAST_PATH_RESIDUAL_THRESHOLD &&
        Math.max(0, candidate.processedGradientScore) <= STANDARD_FAST_PATH_GRADIENT_THRESHOLD;
}

function isStrictFixedCoreCandidate(candidate) {
    if (!candidate?.accepted) return false;

    const processedSpatial = Number(candidate.processedSpatialScore);
    if (!Number.isFinite(processedSpatial)) return false;

    return Math.abs(processedSpatial) <= FIXED_CORE_MAX_ACCEPTED_SPATIAL_RESIDUAL;
}

function isCleanWeakAlphaPriorityCandidate(candidate) {
    if (!candidate?.accepted) return false;

    return Math.abs(candidate.processedSpatialScore) <= STANDARD_FAST_PATH_RESIDUAL_THRESHOLD &&
        Math.max(0, candidate.processedGradientScore) <= WEAK_ALPHA_PRIORITY_CLEAN_GRADIENT_THRESHOLD;
}

function normalizeAlphaPriorityGains(alphaPriorityGains) {
    const gains = Array.isArray(alphaPriorityGains) && alphaPriorityGains.length > 0
        ? alphaPriorityGains
        : [1];
    const normalized = [];
    const seen = new Set();
    for (const gain of gains) {
        if (!Number.isFinite(gain) || gain <= 0) continue;
        const key = gain.toFixed(4);
        if (seen.has(key)) continue;
        seen.add(key);
        normalized.push(gain);
    }

    if (!seen.has('1.0000')) {
        normalized.push(1);
    }

    return normalized;
}

function isWeakAlphaPrioritySeed(seed) {
    return seed?.config?.logoSize === 48 &&
        seed.config.marginRight === 96 &&
        seed.config.marginBottom === 96;
}

function evaluateStandardTrialForSeed({
    originalImageData,
    seed,
    alphaPriorityGains
}) {
    const priorityGains = normalizeAlphaPriorityGains(alphaPriorityGains);
    let fallbackTrial = null;
    let bestAcceptedTrial = null;

    for (const alphaGain of priorityGains) {
        const trial = evaluateRestorationCandidate({
            originalImageData,
            alphaMap: seed.alphaMap,
            position: seed.position,
            source: alphaGain === 1 ? seed.source : `${seed.source}+gain`,
            config: seed.config,
            baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, seed.position),
            alphaGain,
            provenance: seed.provenance,
            includeImageData: false
        });
        if (!trial) continue;

        if (alphaGain === 1 || !fallbackTrial) {
            fallbackTrial = trial;
        }
        if (alphaGain < 1) {
            if (isWeakAlphaPrioritySeed(seed) && isCleanWeakAlphaPriorityCandidate(trial)) {
                return trial;
            }
            continue;
        }
        if (trial.accepted) {
            bestAcceptedTrial = pickBetterCandidate(bestAcceptedTrial, trial, 0.002);
        }
    }

    return bestAcceptedTrial ?? fallbackTrial;
}

export function evaluateRestorationCandidate({
    originalImageData,
    alphaMap,
    position,
    source,
    config,
    baselineNearBlackRatio,
    adaptiveConfidence = null,
    alphaGain = 1,
    provenance = null,
    includeImageData = true
}) {
    if (!alphaMap || !position) return null;

    const originalScores = scoreRegion(originalImageData, alphaMap, position);
    const regionImageData = createCandidateRegionImageData({
        originalImageData,
        alphaMap,
        position,
        alphaGain
    });
    const regionPosition = {
        x: ORIGIN_REGION.x,
        y: ORIGIN_REGION.y,
        width: position.width,
        height: position.height
    };
    const processedScores = scoreRegion(regionImageData, alphaMap, regionPosition);
    const nearBlackRatio = calculateNearBlackRatio(regionImageData, regionPosition);
    const nearBlackIncrease = nearBlackRatio - baselineNearBlackRatio;
    // Signed suppression keeps legitimate "slight overshoot" restores eligible.
    const improvement = originalScores.spatialScore - processedScores.spatialScore;
    const gradientIncrease = processedScores.gradientScore - originalScores.gradientScore;
    const textureAssessment = assessReferenceTextureAlignmentFromStats({
        originalImageData,
        referenceImageData: originalImageData,
        candidateTextureStats: getRegionTextureStats(regionImageData, regionPosition),
        position
    });
    const texturePenalty = textureAssessment.texturePenalty;
    const gradientDrop = originalScores.gradientScore - processedScores.gradientScore;
    const strongStandardSignalNearBlackOverride =
        isStandardCandidateSource({ source }) &&
        alphaGain === 1 &&
        originalScores.spatialScore >= STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MIN_SPATIAL_SCORE &&
        originalScores.gradientScore >= STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MIN_GRADIENT_SCORE &&
        Math.abs(processedScores.spatialScore) <= STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MAX_RESIDUAL &&
        processedScores.gradientScore <= STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MAX_GRADIENT &&
        improvement >= STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MIN_IMPROVEMENT &&
        gradientDrop >= STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MIN_GRADIENT_DROP &&
        nearBlackIncrease <= STANDARD_STRONG_SIGNAL_NEAR_BLACK_OVERRIDE_MAX_NEAR_BLACK_INCREASE;
    const nearBlackIncreaseAllowed =
        nearBlackIncrease <= MAX_NEAR_BLACK_RATIO_INCREASE ||
        strongStandardSignalNearBlackOverride ||
        (
            source === 'standard' &&
            originalScores.spatialScore >= STANDARD_TEXT_OVERLAP_MIN_SPATIAL_SCORE &&
            originalScores.gradientScore >= STANDARD_TEXT_OVERLAP_MIN_GRADIENT_SCORE &&
            improvement >= STANDARD_TEXT_OVERLAP_MIN_IMPROVEMENT &&
            Math.abs(processedScores.spatialScore) <= STANDARD_TEXT_OVERLAP_MAX_RESIDUAL &&
            gradientDrop >= STANDARD_TEXT_OVERLAP_MIN_GRADIENT_DROP
        );
    const hardRejectAllowed =
        textureAssessment.hardReject !== true ||
        (
            isStandardCandidateSource({ source }) &&
            originalScores.spatialScore >= STANDARD_HARD_REJECT_OVERRIDE_MIN_SPATIAL_SCORE &&
            originalScores.gradientScore >= STANDARD_HARD_REJECT_OVERRIDE_MIN_GRADIENT_SCORE &&
            Math.abs(processedScores.spatialScore) <= STANDARD_HARD_REJECT_OVERRIDE_MAX_RESIDUAL &&
            processedScores.gradientScore <= STANDARD_HARD_REJECT_OVERRIDE_MAX_GRADIENT &&
            improvement >= STANDARD_HARD_REJECT_OVERRIDE_MIN_IMPROVEMENT &&
            nearBlackIncrease <= STANDARD_HARD_REJECT_OVERRIDE_MAX_NEAR_BLACK_INCREASE
        );
    const accepted =
        hardRejectAllowed &&
        nearBlackIncreaseAllowed &&
        improvement >= VALIDATION_MIN_IMPROVEMENT &&
        (
            Math.abs(processedScores.spatialScore) <= VALIDATION_TARGET_RESIDUAL ||
            gradientIncrease <= VALIDATION_MAX_GRADIENT_INCREASE
        );

    return {
        accepted,
        source,
        config,
        position,
        alphaMap,
        adaptiveConfidence,
        alphaGain,
        provenance: mergeCandidateProvenance(provenance),
        imageData: includeImageData
            ? materializeCandidateImageData(originalImageData, alphaMap, position, alphaGain)
            : null,
        originalSpatialScore: originalScores.spatialScore,
        originalGradientScore: originalScores.gradientScore,
        processedSpatialScore: processedScores.spatialScore,
        processedGradientScore: processedScores.gradientScore,
        improvement,
        nearBlackRatio,
        nearBlackIncrease,
        gradientIncrease,
        tooDark: textureAssessment.tooDark,
        tooFlat: textureAssessment.tooFlat,
        hardReject: textureAssessment.hardReject,
        texturePenalty,
        validationCost:
            Math.abs(processedScores.spatialScore) +
            Math.max(0, processedScores.gradientScore) * 0.6 +
            Math.max(0, nearBlackIncrease) * 3 +
            texturePenalty
    };
}

function pickBestValidatedCandidate(candidates) {
    const accepted = candidates.filter((candidate) => candidate?.accepted);
    if (accepted.length === 0) return null;

    accepted.sort((a, b) => {
        if (a.validationCost !== b.validationCost) {
            return a.validationCost - b.validationCost;
        }

        return b.improvement - a.improvement;
    });

    return accepted[0];
}

function createCandidateRegionImageData({
    originalImageData,
    alphaMap,
    position,
    alphaGain
}) {
    const regionImageData = {
        width: position.width,
        height: position.height,
        data: new Uint8ClampedArray(position.width * position.height * 4)
    };

    for (let row = 0; row < position.height; row++) {
        const srcStart = ((position.y + row) * originalImageData.width + position.x) * 4;
        const srcEnd = srcStart + position.width * 4;
        const destStart = row * position.width * 4;
        regionImageData.data.set(originalImageData.data.subarray(srcStart, srcEnd), destStart);
    }

    removeWatermark(regionImageData, alphaMap, {
        x: 0,
        y: 0,
        width: position.width,
        height: position.height
    }, { alphaGain });

    return regionImageData;
}

function materializeCandidateImageData(originalImageData, alphaMap, position, alphaGain) {
    const candidateImageData = cloneImageData(originalImageData);
    removeWatermark(candidateImageData, alphaMap, position, { alphaGain });
    return candidateImageData;
}

function ensureCandidateImageData(candidate, originalImageData) {
    if (!candidate) return candidate;
    if (candidate.imageData) return candidate;

    return {
        ...candidate,
        imageData: materializeCandidateImageData(
            originalImageData,
            candidate.alphaMap,
            candidate.position,
            candidate.alphaGain ?? 1
        )
    };
}

export function pickBetterCandidate(currentBest, candidate, minCostDelta = 0.005) {
    if (!candidate?.accepted) return currentBest;
    if (!currentBest) return candidate;
    if (shouldPreserveCatalogOriginalSignal(currentBest, candidate)) {
        return currentBest;
    }
    if (shouldPreferCatalogOriginalSignal(candidate, currentBest)) {
        return candidate;
    }
    if (shouldPreserveStrongStandardAnchor(currentBest, candidate)) {
        return currentBest;
    }
    if (shouldPreferPreviewAnchorCandidate(currentBest, candidate)) {
        return candidate;
    }
    if (shouldPreferPreviewAnchorCandidate(candidate, currentBest)) {
        return currentBest;
    }
    if (candidate.validationCost < currentBest.validationCost - minCostDelta) {
        return candidate;
    }
    if (Math.abs(candidate.validationCost - currentBest.validationCost) <= minCostDelta &&
        candidate.improvement > currentBest.improvement + 0.01) {
        return candidate;
    }
    return currentBest;
}

function isStandardCandidateSource(candidate) {
    return typeof candidate?.source === 'string' && candidate.source.startsWith('standard');
}

function isDriftedStandardCandidate(candidate) {
    return isStandardCandidateSource(candidate) &&
        (
            candidate?.provenance?.localShift === true ||
            candidate?.provenance?.sizeJitter === true ||
            candidate?.provenance?.previewAnchor === true ||
            String(candidate?.source || '').includes('+warp')
        );
}

function isCanonicalStandardCandidate(candidate) {
    return isStandardCandidateSource(candidate) &&
        candidate?.provenance?.localShift !== true &&
        candidate?.provenance?.sizeJitter !== true &&
        candidate?.provenance?.previewAnchor !== true;
}

function hasStrongCanonicalAnchorSignal(candidate) {
    const baseSpatial = Number(candidate?.originalSpatialScore);
    const baseGradient = Number(candidate?.originalGradientScore);
    if (!Number.isFinite(baseSpatial) || !Number.isFinite(baseGradient)) {
        return false;
    }
    return (
        baseGradient >= STANDARD_LOCAL_SHIFT_CANONICAL_MIN_GRADIENT_SCORE &&
        baseSpatial >= STANDARD_LOCAL_SHIFT_CANONICAL_MIN_SPATIAL_SCORE
    ) ||
        baseGradient >= STANDARD_LOCAL_SHIFT_STRONG_BASE_GRADIENT_SCORE ||
        baseSpatial >= STANDARD_LOCAL_SHIFT_STRONG_BASE_SPATIAL_SCORE;
}

function hasReliableCandidateOriginalSignal(candidate) {
    return hasReliableStandardWatermarkSignal({
        spatialScore: candidate?.originalSpatialScore,
        gradientScore: candidate?.originalGradientScore
    });
}

function hasMuchStrongerOriginalSignal(candidate, otherCandidate) {
    const spatial = Number(candidate?.originalSpatialScore);
    const gradient = Number(candidate?.originalGradientScore);
    const otherSpatial = Number(otherCandidate?.originalSpatialScore);
    const otherGradient = Number(otherCandidate?.originalGradientScore);
    if (
        !Number.isFinite(spatial) ||
        !Number.isFinite(gradient) ||
        !Number.isFinite(otherSpatial) ||
        !Number.isFinite(otherGradient)
    ) {
        return false;
    }

    return spatial >= otherSpatial + STRONG_ORIGINAL_SIGNAL_SPATIAL_ADVANTAGE &&
        gradient >= otherGradient + STRONG_ORIGINAL_SIGNAL_GRADIENT_ADVANTAGE;
}

function isCatalogVariantCandidate(candidate) {
    return candidate?.provenance?.catalogVariant === true;
}

function isCurrentLargeMarginCatalogCandidate(candidate) {
    return isCatalogVariantCandidate(candidate) &&
        candidate?.config?.logoSize === 48 &&
        candidate.config.marginRight === 96 &&
        candidate.config.marginBottom === 96;
}

function isPreviewAnchorCandidate(candidate) {
    return candidate?.provenance?.previewAnchor === true;
}

function shouldPreferCatalogOriginalSignal(candidate, currentBest) {
    if (!isCurrentLargeMarginCatalogCandidate(candidate)) return false;
    if (currentBest?.provenance?.localShift === true || currentBest?.provenance?.sizeJitter === true) {
        return false;
    }

    const candidateReliable = hasReliableCandidateOriginalSignal(candidate);
    const currentReliable = hasReliableCandidateOriginalSignal(currentBest);
    if (candidateReliable && !currentReliable) return true;

    return candidateReliable &&
        currentReliable &&
        isPreviewAnchorCandidate(currentBest) &&
        hasMuchStrongerOriginalSignal(candidate, currentBest);
}

function shouldPreserveCatalogOriginalSignal(currentBest, candidate) {
    if (!isCurrentLargeMarginCatalogCandidate(currentBest)) return false;
    if (!isPreviewAnchorCandidate(candidate)) return false;

    const currentReliable = hasReliableCandidateOriginalSignal(currentBest);
    const candidateReliable = hasReliableCandidateOriginalSignal(candidate);
    if (currentReliable && !candidateReliable) return true;

    return currentReliable &&
        candidateReliable &&
        hasMuchStrongerOriginalSignal(currentBest, candidate);
}

function hasWeakDriftEvidence(candidate) {
    const candidateSpatial = Number(candidate?.originalSpatialScore);
    const candidateGradient = Number(candidate?.originalGradientScore);
    if (!Number.isFinite(candidateSpatial) || !Number.isFinite(candidateGradient)) {
        return false;
    }
    return candidateGradient < STANDARD_LOCAL_SHIFT_WEAK_CANDIDATE_GRADIENT_SCORE ||
        candidateSpatial < STANDARD_LOCAL_SHIFT_WEAK_CANDIDATE_SPATIAL_SCORE;
}

function leavesWorseResidualGradientThanCanonical(canonicalCandidate, driftCandidate) {
    const canonicalProcessedGradientRaw = Number(canonicalCandidate?.processedGradientScore);
    const driftProcessedGradientRaw = Number(driftCandidate?.processedGradientScore);
    if (
        !Number.isFinite(canonicalProcessedGradientRaw) ||
        !Number.isFinite(driftProcessedGradientRaw)
    ) {
        return false;
    }

    return Math.max(0, canonicalProcessedGradientRaw) <= STANDARD_LOCAL_SHIFT_PRESERVE_CLEAN_BASE_GRADIENT_THRESHOLD &&
        Math.max(0, driftProcessedGradientRaw) >= STANDARD_LOCAL_SHIFT_MAX_CANDIDATE_GRADIENT_FOR_CLEAN_BASE;
}

function leavesMuchWorseResidualGradientThanCanonical(canonicalCandidate, driftCandidate) {
    const canonicalProcessedSpatial = Number(canonicalCandidate?.processedSpatialScore);
    const canonicalProcessedGradient = Number(canonicalCandidate?.processedGradientScore);
    const canonicalImprovement = Number(canonicalCandidate?.improvement);
    const driftProcessedGradient = Number(driftCandidate?.processedGradientScore);
    if (
        !Number.isFinite(canonicalProcessedSpatial) ||
        !Number.isFinite(canonicalProcessedGradient) ||
        !Number.isFinite(canonicalImprovement) ||
        !Number.isFinite(driftProcessedGradient)
    ) {
        return false;
    }

    return Math.abs(canonicalProcessedSpatial) <= STANDARD_PRESERVE_MAX_RESIDUAL &&
        canonicalImprovement >= STANDARD_PRESERVE_MIN_IMPROVEMENT &&
        driftProcessedGradient >= canonicalProcessedGradient + STANDARD_PRESERVE_GRADIENT_DELTA;
}

function shouldPreserveCanonicalAnchor(canonicalCandidate, driftCandidate) {
    if (!isCanonicalStandardCandidate(canonicalCandidate)) return false;
    if (!isDriftedStandardCandidate(driftCandidate)) return false;

    const validationAdvantage = Number(canonicalCandidate.validationCost) - Number(driftCandidate.validationCost);
    if (
        !Number.isFinite(validationAdvantage)
    ) {
        return false;
    }

    return (
        hasStrongCanonicalAnchorSignal(canonicalCandidate) &&
        hasWeakDriftEvidence(driftCandidate) &&
        validationAdvantage < STANDARD_LOCAL_SHIFT_MIN_VALIDATION_ADVANTAGE
    ) ||
        leavesWorseResidualGradientThanCanonical(canonicalCandidate, driftCandidate) ||
        leavesMuchWorseResidualGradientThanCanonical(canonicalCandidate, driftCandidate);
}

function shouldPreserveStrongStandardAnchor(currentBest, candidate) {
    if (currentBest?.provenance?.localShift === true) return false;
    if (!isStandardCandidateSource(candidate)) return false;
    return shouldPreserveCanonicalAnchor(currentBest, candidate);
}

function shouldRevertLocalShiftToStandardTrial(selectedCandidate, standardTrial) {
    if (selectedCandidate?.provenance?.localShift !== true) return false;
    if (!isStandardCandidateSource(selectedCandidate) || !isStandardCandidateSource(standardTrial)) return false;
    if (!standardTrial?.accepted) return false;
    return shouldPreserveCanonicalAnchor(standardTrial, selectedCandidate);
}

function shouldSkipStandardLocalSearch(seedCandidate) {
    if (!seedCandidate) return false;

    return Math.max(0, Number(seedCandidate.processedGradientScore)) <=
        STANDARD_LOCAL_SHIFT_SKIP_PROCESSED_GRADIENT_THRESHOLD;
}

function isPreviewAnchorSearchEligible(originalImageData, config) {
    if (!config || config.logoSize !== 48) return false;

    const width = Number(originalImageData?.width);
    const height = Number(originalImageData?.height);
    if (!Number.isFinite(width) || !Number.isFinite(height)) return false;
    if (width < 384 || width > 1536) return false;
    if (height < 384 || height > 1536) return false;
    if (Math.max(width, height) < 512) return false;

    return matchOfficialGeminiImageSize(width, height) === null;
}

function shouldPreferPreviewAnchorCandidate(currentBest, candidate) {
    if (candidate?.provenance?.previewAnchor !== true) return false;
    if (!currentBest || currentBest?.provenance?.previewAnchor === true) return false;

    const currentSpatial = Number(currentBest.originalSpatialScore);
    const currentGradient = Number(currentBest.originalGradientScore);
    const candidateSpatial = Number(candidate.originalSpatialScore);
    const candidateGradient = Number(candidate.originalGradientScore);

    if (
        !Number.isFinite(currentSpatial) ||
        !Number.isFinite(currentGradient) ||
        !Number.isFinite(candidateSpatial) ||
        !Number.isFinite(candidateGradient)
    ) {
        return false;
    }

    const currentReliable = hasReliableStandardWatermarkSignal({
        spatialScore: currentSpatial,
        gradientScore: currentGradient
    });
    const candidateReliable = hasReliableStandardWatermarkSignal({
        spatialScore: candidateSpatial,
        gradientScore: candidateGradient
    });

    if (candidateReliable && !currentReliable) {
        return true;
    }

    return candidateGradient >= currentGradient + 0.2 &&
        candidateSpatial >= currentSpatial + 0.05;
}

function findBestTemplateWarp({
    originalImageData,
    alphaMap,
    position,
    baselineSpatialScore,
    baselineGradientScore,
    shiftCandidates = TEMPLATE_ALIGN_SHIFTS,
    scaleCandidates = TEMPLATE_ALIGN_SCALES
}) {
    const size = position.width;
    if (!size || size <= 8) return null;

    let best = {
        spatialScore: baselineSpatialScore,
        gradientScore: baselineGradientScore,
        shift: { dx: 0, dy: 0, scale: 1 },
        alphaMap
    };

    for (const scale of scaleCandidates) {
        for (const dy of shiftCandidates) {
            for (const dx of shiftCandidates) {
                if (dx === 0 && dy === 0 && scale === 1) continue;
                const warped = warpAlphaMap(alphaMap, size, { dx, dy, scale });
                const spatialScore = computeRegionSpatialCorrelation({
                    imageData: originalImageData,
                    alphaMap: warped,
                    region: { x: position.x, y: position.y, size }
                });
                const gradientScore = computeRegionGradientCorrelation({
                    imageData: originalImageData,
                    alphaMap: warped,
                    region: { x: position.x, y: position.y, size }
                });

                const confidence =
                    Math.max(0, spatialScore) * 0.7 +
                    Math.max(0, gradientScore) * 0.3;
                const bestConfidence =
                    Math.max(0, best.spatialScore) * 0.7 +
                    Math.max(0, best.gradientScore) * 0.3;

                if (confidence > bestConfidence + 0.01) {
                    best = {
                        spatialScore,
                        gradientScore,
                        shift: { dx, dy, scale },
                        alphaMap: warped
                    };
                }
            }
        }
    }

    const improvedSpatial = best.spatialScore >= baselineSpatialScore + 0.01;
    const improvedGradient = best.gradientScore >= baselineGradientScore + 0.01;
    return improvedSpatial || improvedGradient ? best : null;
}

function searchNearbyStandardCandidate({
    originalImageData,
    candidateSeeds,
    adaptiveConfidence = null
}) {
    if (!Array.isArray(candidateSeeds) || candidateSeeds.length === 0) return null;

    let bestCandidate = null;
    for (const seed of candidateSeeds) {
        if (shouldSkipStandardLocalSearch(seed)) continue;
        for (const dy of STANDARD_NEARBY_SHIFTS) {
            for (const dx of STANDARD_NEARBY_SHIFTS) {
                if (dx === 0 && dy === 0) continue;

                const candidatePosition = {
                    x: seed.position.x + dx,
                    y: seed.position.y + dy,
                    width: seed.position.width,
                    height: seed.position.height
                };
                if (candidatePosition.x < 0 || candidatePosition.y < 0) continue;
                if (candidatePosition.x + candidatePosition.width > originalImageData.width) continue;
                if (candidatePosition.y + candidatePosition.height > originalImageData.height) continue;

                const candidate = evaluateRestorationCandidate({
                    originalImageData,
                    alphaMap: seed.alphaMap,
                    position: candidatePosition,
                    source: `${seed.source}+local`,
                    config: seed.config,
                    baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, candidatePosition),
                    adaptiveConfidence,
                    provenance: mergeCandidateProvenance(seed.provenance, { localShift: true }),
                    includeImageData: false
                });

                if (!candidate?.accepted) continue;
                bestCandidate = pickBetterCandidate(bestCandidate, candidate, 0.002);
            }
        }
    }

    return bestCandidate;
}

function searchStandardSizeJitterCandidate({
    originalImageData,
    candidateSeeds,
    alpha48,
    alpha96,
    getAlphaMap,
    resolveAlphaMap = null,
    adaptiveConfidence = null
}) {
    if (!Array.isArray(candidateSeeds) || candidateSeeds.length === 0) return null;

    let bestCandidate = null;
    for (const seed of candidateSeeds) {
        for (const delta of STANDARD_SIZE_JITTERS) {
            const size = seed.position.width + delta;
            if (size <= 24) continue;
            if (size === seed.position.width) continue;

            const candidatePosition = {
                x: originalImageData.width - seed.config.marginRight - size,
                y: originalImageData.height - seed.config.marginBottom - size,
                width: size,
                height: size
            };
            if (candidatePosition.x < 0 || candidatePosition.y < 0) continue;
            if (candidatePosition.x + candidatePosition.width > originalImageData.width) continue;
            if (candidatePosition.y + candidatePosition.height > originalImageData.height) continue;

            const candidateAlphaMap = typeof resolveAlphaMap === 'function'
                ? resolveAlphaMap(size)
                : resolveAlphaMapForSize(size, {
                    alpha48,
                    alpha96,
                    getAlphaMap
                });
            if (!candidateAlphaMap) continue;

            const candidate = evaluateRestorationCandidate({
                originalImageData,
                alphaMap: candidateAlphaMap,
                position: candidatePosition,
                source: `${seed.source}+size`,
                config: {
                    logoSize: size,
                    marginRight: seed.config.marginRight,
                    marginBottom: seed.config.marginBottom
                },
                baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, candidatePosition),
                adaptiveConfidence,
                provenance: mergeCandidateProvenance(seed.provenance, { sizeJitter: true }),
                includeImageData: false
            });

            if (!candidate?.accepted) continue;
            bestCandidate = pickBetterCandidate(bestCandidate, candidate, 0.002);
        }
    }

    return bestCandidate;
}

function searchFineStandardLocalCandidate({
    originalImageData,
    seedCandidate,
    adaptiveConfidence = null,
    shiftCandidates = STANDARD_FINE_LOCAL_SHIFTS
}) {
    if (!seedCandidate?.alphaMap || !seedCandidate?.position) return null;
    if (shouldSkipStandardLocalSearch(seedCandidate)) return null;

    let bestCandidate = null;
    for (const dy of shiftCandidates) {
        for (const dx of shiftCandidates) {
            if (dx === 0 && dy === 0) continue;

            const candidatePosition = {
                x: seedCandidate.position.x + dx,
                y: seedCandidate.position.y + dy,
                width: seedCandidate.position.width,
                height: seedCandidate.position.height
            };
            if (candidatePosition.x < 0 || candidatePosition.y < 0) continue;
            if (candidatePosition.x + candidatePosition.width > originalImageData.width) continue;
            if (candidatePosition.y + candidatePosition.height > originalImageData.height) continue;

            const candidate = evaluateRestorationCandidate({
                originalImageData,
                alphaMap: seedCandidate.alphaMap,
                position: candidatePosition,
                source: `${seedCandidate.source}+local`,
                config: seedCandidate.config,
                baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, candidatePosition),
                adaptiveConfidence,
                provenance: mergeCandidateProvenance(seedCandidate.provenance, { localShift: true }),
                includeImageData: false
            });

            if (!candidate?.accepted) continue;
            bestCandidate = pickBetterCandidate(bestCandidate, candidate, 0.002);
        }
    }

    return bestCandidate;
}

function searchCandidateAlphaGain({
    originalImageData,
    seedCandidate,
    adaptiveConfidence = null,
    alphaGainCandidates = []
}) {
    if (!seedCandidate?.alphaMap || !seedCandidate?.position) return null;

    let bestCandidate = null;
    for (const candidateGain of alphaGainCandidates) {
        if (!Number.isFinite(candidateGain) || candidateGain <= 0 || candidateGain === 1) continue;

        const candidate = evaluateRestorationCandidate({
            originalImageData,
            alphaMap: seedCandidate.alphaMap,
            position: seedCandidate.position,
            source: `${seedCandidate.source}+gain`,
            config: seedCandidate.config,
            baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, seedCandidate.position),
            adaptiveConfidence,
            alphaGain: candidateGain,
            provenance: seedCandidate.provenance,
            includeImageData: false
        });

        if (!candidate?.accepted) continue;
        if (
            candidateGain < 1 &&
            (
                candidate.improvement < STANDARD_PRESERVE_MIN_IMPROVEMENT ||
                Math.abs(candidate.processedSpatialScore) > VALIDATION_TARGET_RESIDUAL
            )
        ) {
            continue;
        }
        bestCandidate = pickBetterCandidate(bestCandidate, candidate, 0.002);
    }

    return bestCandidate;
}

function searchStrongStandardTrialAlphaGain({
    originalImageData,
    standardTrials,
    standardTrial,
    baseCandidate,
    baseDecisionTier,
    alphaGainCandidates
}) {
    let nextBaseCandidate = baseCandidate;
    let nextBaseDecisionTier = baseDecisionTier;

    for (const candidate of standardTrials) {
        if (!candidate || candidate.accepted) continue;
        if (candidate === standardTrial && nextBaseCandidate) continue;

        const reliableMatch = hasReliableStandardWatermarkSignal({
            spatialScore: candidate.originalSpatialScore,
            gradientScore: candidate.originalGradientScore
        });
        if (!reliableMatch) continue;

        const gainedCandidate = searchCandidateAlphaGain({
            originalImageData,
            seedCandidate: {
                ...candidate,
                source: `${candidate.source}+validated`
            },
            adaptiveConfidence: null,
            alphaGainCandidates
        });
        if (!gainedCandidate) continue;

        ({
            baseCandidate: nextBaseCandidate,
            baseDecisionTier: nextBaseDecisionTier
        } = promoteBaseCandidate(nextBaseCandidate, nextBaseDecisionTier, gainedCandidate, {
            reliableMatch,
            minCostDelta: 0.002
        }));
    }

    return {
        baseCandidate: nextBaseCandidate,
        baseDecisionTier: nextBaseDecisionTier
    };
}

function insertTopPreviewCandidate(topCandidates, candidate) {
    topCandidates.push(candidate);
    topCandidates.sort((a, b) => b.coarseScore - a.coarseScore);
    if (topCandidates.length > PREVIEW_ANCHOR_TOP_K) {
        topCandidates.length = PREVIEW_ANCHOR_TOP_K;
    }
}

function searchBottomRightPreviewCandidate({
    originalImageData,
    config,
    alpha48,
    alpha96,
    getAlphaMap,
    resolveAlphaMap = null,
    adaptiveConfidence = null
}) {
    if (!isPreviewAnchorSearchEligible(originalImageData, config)) return null;

    const minSize = Math.max(
        PREVIEW_ANCHOR_MIN_SIZE,
        Math.round(config.logoSize * PREVIEW_ANCHOR_MIN_SIZE_RATIO)
    );
    const maxSize = Math.max(
        minSize,
        Math.round(config.logoSize * PREVIEW_ANCHOR_MAX_SIZE_RATIO)
    );
    const minMarginRight = Math.max(8, config.marginRight - PREVIEW_ANCHOR_MARGIN_WINDOW);
    const maxMarginRight = config.marginRight + PREVIEW_ANCHOR_MARGIN_EXTENSION;
    const minMarginBottom = Math.max(8, config.marginBottom - PREVIEW_ANCHOR_MARGIN_WINDOW);
    const maxMarginBottom = config.marginBottom + PREVIEW_ANCHOR_MARGIN_EXTENSION;
    const topCandidates = [];

    for (let size = minSize; size <= maxSize; size += PREVIEW_ANCHOR_SIZE_STEP) {
        const alphaMap = typeof resolveAlphaMap === 'function'
            ? resolveAlphaMap(size)
            : resolveAlphaMapForSize(size, {
                alpha48,
                alpha96,
                getAlphaMap
            });
        if (!alphaMap) continue;

        for (let marginRight = minMarginRight; marginRight <= maxMarginRight; marginRight += PREVIEW_ANCHOR_MARGIN_STEP) {
            const x = originalImageData.width - marginRight - size;
            if (x < 0 || x + size > originalImageData.width) continue;

            for (let marginBottom = minMarginBottom; marginBottom <= maxMarginBottom; marginBottom += PREVIEW_ANCHOR_MARGIN_STEP) {
                const y = originalImageData.height - marginBottom - size;
                if (y < 0 || y + size > originalImageData.height) continue;

                const coarseSpatialScore = computeRegionSpatialCorrelation({
                    imageData: originalImageData,
                    alphaMap,
                    region: { x, y, size }
                });
                const coarseGradientScore = computeRegionGradientCorrelation({
                    imageData: originalImageData,
                    alphaMap,
                    region: { x, y, size }
                });
                const coarseScore =
                    Math.max(0, coarseGradientScore) * 0.6 +
                    Math.max(0, coarseSpatialScore) * 0.4;

                if (coarseScore < PREVIEW_ANCHOR_MIN_SCORE) continue;

                insertTopPreviewCandidate(topCandidates, {
                    coarseScore,
                    alphaMap,
                    position: { x, y, width: size, height: size },
                    config: {
                        logoSize: size,
                        marginRight,
                        marginBottom
                    }
                });
            }
        }
    }

    let bestCandidate = null;
    for (const coarseCandidate of topCandidates) {
        for (const sizeDelta of PREVIEW_ANCHOR_LOCAL_DELTAS) {
            const size = coarseCandidate.position.width + sizeDelta;
            if (size < PREVIEW_ANCHOR_MIN_SIZE) continue;

            const alphaMap = typeof resolveAlphaMap === 'function'
                ? resolveAlphaMap(size)
                : resolveAlphaMapForSize(size, {
                    alpha48,
                    alpha96,
                    getAlphaMap
                });
            if (!alphaMap) continue;

            for (const dx of PREVIEW_ANCHOR_LOCAL_DELTAS) {
                for (const dy of PREVIEW_ANCHOR_LOCAL_DELTAS) {
                    const position = {
                        x: coarseCandidate.position.x + dx,
                        y: coarseCandidate.position.y + dy,
                        width: size,
                        height: size
                    };
                    if (position.x < 0 || position.y < 0) continue;
                    if (position.x + position.width > originalImageData.width) continue;
                    if (position.y + position.height > originalImageData.height) continue;

                    const config = {
                        logoSize: size,
                        marginRight: originalImageData.width - position.x - size,
                        marginBottom: originalImageData.height - position.y - size
                    };
                    const candidate = evaluateRestorationCandidate({
                        originalImageData,
                        alphaMap,
                        position,
                        source: 'standard+preview-anchor',
                        config,
                        baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, position),
                        adaptiveConfidence,
                        provenance: {
                            previewAnchor: true,
                            previewAnchorLocalRefine: sizeDelta !== 0 || dx !== 0 || dy !== 0
                        },
                        includeImageData: false
                    });

                    if (!candidate?.accepted) continue;
                    bestCandidate = pickBetterCandidate(bestCandidate, candidate, 0.002);
                }
            }
        }
    }

    return bestCandidate;
}

function evaluateStandardTrialsForSeeds({
    originalImageData,
    candidateSeeds,
    alphaPriorityGains = [1]
}) {
    const standardTrials = candidateSeeds
        .map((seed) => evaluateStandardTrialForSeed({
            originalImageData,
            seed,
            alphaPriorityGains
        }))
        .filter(Boolean);
    const standardTrial = standardTrials.find((candidate) => candidate.source === 'standard') ?? standardTrials[0] ?? null;
    const standardSpatialScore = standardTrial?.originalSpatialScore ?? null;
    const standardGradientScore = standardTrial?.originalGradientScore ?? null;
    const hasReliableStandardMatch = hasReliableStandardWatermarkSignal({
        spatialScore: standardSpatialScore,
        gradientScore: standardGradientScore
    });

    return {
        standardTrials,
        standardTrial,
        standardSpatialScore,
        standardGradientScore,
        hasReliableStandardMatch
    };
}

function resolveStandardAnchorSelection({
    originalImageData,
    config,
    position,
    alpha48,
    alpha96,
    alpha96Variants,
    getAlphaMap,
    resolveAlphaMap,
    alphaPriorityGains,
    forceCatalogVariants = false
}) {
    let standardCandidateSeeds = buildStandardCandidateSeeds({
        originalImageData,
        config,
        position,
        alpha48,
        alpha96,
        alpha96Variants,
        getAlphaMap,
        resolveAlphaMap,
        includeCatalogVariants: false
    });
    let standardSelection = evaluateStandardTrialsForSeeds({
        originalImageData,
        candidateSeeds: standardCandidateSeeds,
        alphaPriorityGains
    });

    const shouldExpandStandardCatalog =
        forceCatalogVariants ||
        !standardSelection.hasReliableStandardMatch &&
        (
            !standardSelection.standardTrial ||
            shouldEscalateSearch(standardSelection.standardTrial) ||
            (
                matchOfficialGeminiImageSize(originalImageData.width, originalImageData.height) === null &&
                shouldExpandCatalogForWeakOriginalStandardEvidence(standardSelection.standardTrial)
            )
        );

    if (shouldExpandStandardCatalog) {
        standardCandidateSeeds = buildStandardCandidateSeeds({
            originalImageData,
            config,
            position,
            alpha48,
            alpha96,
            alpha96Variants,
            getAlphaMap,
            resolveAlphaMap,
            includeCatalogVariants: true
        });
        standardSelection = evaluateStandardTrialsForSeeds({
            originalImageData,
            candidateSeeds: standardCandidateSeeds,
            alphaPriorityGains
        });
    }

    return {
        standardCandidateSeeds,
        ...standardSelection
    };
}

function resolveCandidatePromotion(candidate, {
    reliableMatch = false
} = {}) {
    if (!candidate?.accepted) {
        return null;
    }

    if (reliableMatch) {
        return {
            candidate,
            decisionTier: 'direct-match'
        };
    }

    return {
        candidate: {
            ...candidate,
            source: `${candidate.source}+validated`
        },
        decisionTier: 'validated-match'
    };
}

function promoteBaseCandidate(baseCandidate, baseDecisionTier, candidate, {
    reliableMatch = false,
    minCostDelta = 0.002
} = {}) {
    const promotion = resolveCandidatePromotion(candidate, {
        reliableMatch
    });
    if (!promotion) {
        return {
            baseCandidate,
            baseDecisionTier
        };
    }

    if (
        shouldPreserveCanonicalAnchor(baseCandidate, promotion.candidate)
    ) {
        return {
            baseCandidate,
            baseDecisionTier
        };
    }

    if (
        reliableMatch &&
        shouldPreferCatalogOriginalSignal(promotion.candidate, baseCandidate)
    ) {
        return {
            baseCandidate: promotion.candidate,
            baseDecisionTier: promotion.decisionTier
        };
    }

    const previousCandidate = baseCandidate;
    const nextCandidate = pickBetterCandidate(baseCandidate, promotion.candidate, minCostDelta);
    return {
        baseCandidate: nextCandidate,
        baseDecisionTier: nextCandidate !== previousCandidate
            ? promotion.decisionTier
            : baseDecisionTier
    };
}

function evaluateAdaptiveTrial({
    originalImageData,
    config,
    alpha96,
    resolveAlphaMap,
    allowAdaptiveSearch
}) {
    if (!allowAdaptiveSearch || !alpha96) {
        return {
            adaptive: null,
            adaptiveConfidence: null,
            adaptiveTrial: null
        };
    }

    const adaptive = detectAdaptiveWatermarkRegion({
        imageData: originalImageData,
        alpha96,
        defaultConfig: config
    });
    const adaptiveConfidence = adaptive?.confidence ?? null;

    if (!adaptive?.region || !(
        hasReliableAdaptiveWatermarkSignal(adaptive) ||
        adaptive.confidence >= VALIDATION_MIN_CONFIDENCE_FOR_ADAPTIVE_TRIAL
    )) {
        return {
            adaptive,
            adaptiveConfidence,
            adaptiveTrial: null
        };
    }

    const size = adaptive.region.size;
    const adaptivePosition = {
        x: adaptive.region.x,
        y: adaptive.region.y,
        width: size,
        height: size
    };
    const adaptiveAlphaMap = resolveAlphaMap(size);
    if (!adaptiveAlphaMap) {
        throw new Error(`Missing alpha map for adaptive size ${size}`);
    }
    const adaptiveConfig = {
        logoSize: size,
        marginRight: originalImageData.width - adaptivePosition.x - size,
        marginBottom: originalImageData.height - adaptivePosition.y - size
    };

    return {
        adaptive,
        adaptiveConfidence,
        adaptiveTrial: evaluateRestorationCandidate({
            originalImageData,
            alphaMap: adaptiveAlphaMap,
            position: adaptivePosition,
            source: 'adaptive',
            config: adaptiveConfig,
            baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, adaptivePosition),
            adaptiveConfidence: adaptive.confidence,
            provenance: { adaptive: true },
            includeImageData: false
        })
    };
}

function refineSelectedAnchorCandidate({
    originalImageData,
    baseCandidate,
    baseDecisionTier,
    adaptiveConfidence,
    alphaGainCandidates,
    allowTemplateWarp = true
}) {
    let selectedTrial = ensureCandidateImageData(baseCandidate, originalImageData);
    let alphaMap = baseCandidate.alphaMap;
    let position = baseCandidate.position;
    let config = baseCandidate.config;
    let source = baseCandidate.source;
    let decisionTier = baseDecisionTier || inferDecisionTier(baseCandidate);
    let templateWarp = null;
    let selectedAlphaGain = baseCandidate.alphaGain ?? 1;

    const warpCandidate = allowTemplateWarp
        ? findBestTemplateWarp({
            originalImageData,
            alphaMap,
            position,
            baselineSpatialScore: selectedTrial.originalSpatialScore,
            baselineGradientScore: selectedTrial.originalGradientScore,
            shiftCandidates: selectedTrial.provenance?.previewAnchor === true
                ? PREVIEW_TEMPLATE_ALIGN_SHIFTS
                : TEMPLATE_ALIGN_SHIFTS,
            scaleCandidates: selectedTrial.provenance?.previewAnchor === true
                ? PREVIEW_TEMPLATE_ALIGN_SCALES
                : TEMPLATE_ALIGN_SCALES
        })
        : null;
    if (warpCandidate) {
        const warpedTrial = evaluateRestorationCandidate({
            originalImageData,
            alphaMap: warpCandidate.alphaMap,
            position,
            source: `${source}+warp`,
            config,
            baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, position),
            adaptiveConfidence,
            provenance: selectedTrial.provenance,
            includeImageData: false
        });
        const betterWarpTrial = pickBetterCandidate(selectedTrial, warpedTrial);
        if (betterWarpTrial !== selectedTrial) {
            alphaMap = warpedTrial.alphaMap;
            source = betterWarpTrial.source;
            selectedTrial = ensureCandidateImageData(betterWarpTrial, originalImageData);
            templateWarp = warpCandidate.shift;
            decisionTier = inferDecisionTier(betterWarpTrial, {
                directMatch: decisionTier === 'direct-match'
            });
        }
    }

    const shouldRunGainSearch = selectedTrial.provenance?.previewAnchor === true
        ? isPreviewAnchorGainSearchRequired(selectedTrial)
        : shouldEscalateSearch(selectedTrial);
    let bestGainTrial = selectedTrial;
    if (shouldRunGainSearch) {
        for (const candidateGain of alphaGainCandidates) {
            if (!Number.isFinite(candidateGain) || candidateGain <= 1) continue;
            const gainTrial = evaluateRestorationCandidate({
                originalImageData,
                alphaMap,
                position,
                source: `${source}+gain`,
                config,
                baselineNearBlackRatio: calculateNearBlackRatio(originalImageData, position),
                adaptiveConfidence,
                alphaGain: candidateGain,
                provenance: selectedTrial.provenance,
                includeImageData: false
            });
            bestGainTrial = pickBetterCandidate(bestGainTrial, gainTrial);
        }
    }
    if (bestGainTrial !== selectedTrial) {
        selectedTrial = ensureCandidateImageData(bestGainTrial, originalImageData);
        source = bestGainTrial.source;
        selectedAlphaGain = bestGainTrial.alphaGain;
        decisionTier = inferDecisionTier(bestGainTrial, {
            directMatch: decisionTier === 'direct-match'
        });
    }

    return {
        selectedTrial: ensureCandidateImageData(selectedTrial, originalImageData),
        source,
        alphaMap,
        position,
        config,
        templateWarp,
        alphaGain: selectedAlphaGain,
        decisionTier
    };
}

export function selectInitialCandidate({
    originalImageData,
    config,
    position,
    alpha48,
    alpha96,
    getAlphaMap,
    allowAdaptiveSearch,
    allowAutomaticSearch = true,
    alphaGainCandidates,
    alphaPriorityGains = [1],
    alpha96Variants = null
}) {
    const resolveAlphaMap = createAlphaMapResolver({ alpha48, alpha96, getAlphaMap });
    const fallbackAlphaMap = config.logoSize === 96 ? alpha96 : alpha48;
    const {
        standardCandidateSeeds,
        standardTrials,
        standardTrial,
        standardSpatialScore,
        standardGradientScore,
        hasReliableStandardMatch
    } = resolveStandardAnchorSelection({
        originalImageData,
        config,
        position,
        alpha48,
        alpha96,
        alpha96Variants,
        getAlphaMap,
        resolveAlphaMap,
        alphaPriorityGains,
        forceCatalogVariants: !allowAutomaticSearch
    });
    let baseCandidate = null;
    let baseDecisionTier = 'insufficient';
    if (hasReliableStandardMatch && standardTrial?.accepted) {
        baseCandidate = standardTrial;
        baseDecisionTier = 'direct-match';
    } else if (standardTrial?.accepted) {
        baseCandidate = {
            ...standardTrial,
            source: `${standardTrial.source}+validated`
        };
        baseDecisionTier = 'validated-match';
    }

    let adaptive = null;
    let adaptiveConfidence = null;
    let adaptiveTrial = null;
    for (const candidate of standardTrials) {
        if (!candidate || candidate === standardTrial) continue;
        ({
            baseCandidate,
            baseDecisionTier
        } = promoteBaseCandidate(baseCandidate, baseDecisionTier, candidate, {
            reliableMatch: hasReliableStandardWatermarkSignal({
                spatialScore: candidate.originalSpatialScore,
                gradientScore: candidate.originalGradientScore
            })
        }));
    }

    ({
        baseCandidate,
        baseDecisionTier
    } = searchStrongStandardTrialAlphaGain({
        originalImageData,
        standardTrials,
        standardTrial,
        baseCandidate,
        baseDecisionTier,
        alphaGainCandidates
    }));

    if (allowAutomaticSearch) {
        const previewAnchorCandidate = searchBottomRightPreviewCandidate({
            originalImageData,
            config,
            alpha48,
            alpha96,
            getAlphaMap,
            resolveAlphaMap,
            adaptiveConfidence
        });
        if (previewAnchorCandidate) {
            ({
                baseCandidate,
                baseDecisionTier
            } = promoteBaseCandidate(baseCandidate, baseDecisionTier, previewAnchorCandidate));
        }
    }

    if (
        allowAutomaticSearch &&
        baseDecisionTier !== 'direct-match' &&
        !baseCandidate?.provenance?.previewAnchor &&
        shouldEscalateSearch(baseCandidate)
    ) {
        const sizeJitterCandidate = searchStandardSizeJitterCandidate({
            originalImageData,
            candidateSeeds: standardCandidateSeeds,
            alpha48,
            alpha96,
            getAlphaMap,
            resolveAlphaMap
        });
        if (sizeJitterCandidate) {
            ({
                baseCandidate,
                baseDecisionTier
            } = promoteBaseCandidate(baseCandidate, baseDecisionTier, sizeJitterCandidate));
        }
    }

    if (
        allowAutomaticSearch &&
        baseDecisionTier !== 'direct-match' &&
        baseCandidate?.provenance?.sizeJitter === true &&
        !baseCandidate?.provenance?.previewAnchor &&
        isStandardCandidateSource(baseCandidate) &&
        shouldEscalateSearch(baseCandidate)
    ) {
        const fineLocalCandidate = searchFineStandardLocalCandidate({
            originalImageData,
            seedCandidate: baseCandidate,
            adaptiveConfidence
        });
        if (fineLocalCandidate) {
            ({
                baseCandidate,
                baseDecisionTier
            } = promoteBaseCandidate(baseCandidate, baseDecisionTier, fineLocalCandidate));
        }
    }

    const shouldEvaluateAdaptive = () => {
        if (!allowAdaptiveSearch || !alpha96) return false;
        if (!baseCandidate) return true;
        if (!shouldEscalateSearch(baseCandidate)) return false;

        baseCandidate = ensureCandidateImageData(baseCandidate, originalImageData);

        return shouldAttemptAdaptiveFallback({
            processedImageData: baseCandidate.imageData,
            alphaMap: baseCandidate.alphaMap,
            position: baseCandidate.position,
            originalImageData,
            originalSpatialMismatchThreshold: 0
        });
    };

    if (allowAutomaticSearch && shouldEvaluateAdaptive()) {
        ({
            adaptive,
            adaptiveConfidence,
            adaptiveTrial
        } = evaluateAdaptiveTrial({
            originalImageData,
            config,
            alpha96,
            resolveAlphaMap,
            allowAdaptiveSearch
        }));
    }

    if (adaptiveTrial) {
        ({
            baseCandidate,
            baseDecisionTier
        } = promoteBaseCandidate(baseCandidate, baseDecisionTier, adaptiveTrial, {
            reliableMatch: hasReliableAdaptiveWatermarkSignal(adaptive)
        }));
    }

    if (
        allowAutomaticSearch &&
        !baseCandidate?.provenance?.previewAnchor &&
        !hasReliableAdaptiveWatermarkSignal(adaptive) &&
        shouldSearchNearbyStandardCandidate(baseCandidate, originalImageData)
    ) {
        const nearbyStandardCandidate = searchNearbyStandardCandidate({
            originalImageData,
            candidateSeeds: standardCandidateSeeds,
            adaptiveConfidence
        });
        if (nearbyStandardCandidate) {
            ({
                baseCandidate,
                baseDecisionTier
            } = promoteBaseCandidate(baseCandidate, baseDecisionTier, nearbyStandardCandidate));
        }
    }

    if (!baseCandidate) {
        if (hasReliableStandardMatch && standardTrial?.accepted) {
            baseCandidate = standardTrial;
            baseDecisionTier = 'direct-match';
        } else if (hasReliableAdaptiveWatermarkSignal(adaptive) && adaptiveTrial) {
            baseCandidate = adaptiveTrial;
            baseDecisionTier = 'direct-match';
        }
    }

    if (!baseCandidate) {
        const validatedCandidate = pickBestValidatedCandidate([standardTrial, adaptiveTrial]);
        if (!validatedCandidate) {
            return {
                selectedTrial: null,
                source: 'skipped',
                alphaMap: fallbackAlphaMap,
                position,
                config,
                adaptiveConfidence,
                standardSpatialScore,
                standardGradientScore,
                templateWarp: null,
                alphaGain: 1,
                decisionTier: 'insufficient'
            };
        }
        baseCandidate = {
            ...validatedCandidate,
            source: `${validatedCandidate.source}+validated`
        };
        baseDecisionTier = 'validated-match';
    }

    if (shouldRevertLocalShiftToStandardTrial(baseCandidate, standardTrial)) {
        baseCandidate = standardTrial;
        baseDecisionTier = hasReliableStandardMatch ? 'direct-match' : 'validated-match';
    }

    if (!allowAutomaticSearch && !isStrictFixedCoreCandidate(baseCandidate)) {
        baseCandidate = null;
        baseDecisionTier = 'insufficient';
    }

    if (!baseCandidate) {
        const validatedCandidate = allowAutomaticSearch
            ? pickBestValidatedCandidate([standardTrial, adaptiveTrial])
            : null;
        if (!validatedCandidate) {
            return {
                selectedTrial: null,
                source: 'skipped',
                alphaMap: fallbackAlphaMap,
                position,
                config,
                adaptiveConfidence,
                standardSpatialScore,
                standardGradientScore,
                templateWarp: null,
                alphaGain: 1,
                decisionTier: 'insufficient'
            };
        }
        baseCandidate = {
            ...validatedCandidate,
            source: `${validatedCandidate.source}+validated`
        };
        baseDecisionTier = 'validated-match';
    }

    const {
        selectedTrial,
        source,
        alphaMap,
        position: refinedPosition,
        config: refinedConfig,
        templateWarp,
        alphaGain,
        decisionTier
    } = refineSelectedAnchorCandidate({
        originalImageData,
        baseCandidate,
        baseDecisionTier,
        adaptiveConfidence,
        alphaGainCandidates,
        allowTemplateWarp: allowAutomaticSearch
    });

    return {
        selectedTrial: ensureCandidateImageData(selectedTrial, originalImageData),
        source,
        alphaMap,
        position: refinedPosition,
        config: refinedConfig,
        adaptiveConfidence,
        standardSpatialScore,
        standardGradientScore,
        templateWarp,
        alphaGain,
        decisionTier
    };
}
