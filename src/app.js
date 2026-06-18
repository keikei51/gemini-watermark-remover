import {
    WatermarkEngine,
    detectWatermarkConfig,
    calculateWatermarkPosition
} from './core/watermarkEngine.js';
import { WatermarkWorkerClient, canUseWatermarkWorker } from './core/workerClient.js';
import {
    isConfirmedWatermarkDecision,
    resolveDisplayWatermarkInfo
} from './core/watermarkDisplay.js';
import { canvasToBlob } from './core/canvasBlob.js';
import {
    loadImage,
    setStatusMessage,
    showLoading,
    hideLoading
} from './utils.js';
import {
    consumeDebugFileHandoff,
    getDebugFileKind,
    pickDebugUploadFile,
    saveDebugFileHandoff
} from './shared/debugFileHandoff.js';

const TEXT = {
    loading: '正在加載資源...',
    size: '尺寸',
    watermark: '檢測到的水印',
    position: '位置',
    status: '狀態',
    removed: '水印已移除',
    skipped: '未檢測到可移除水印，已保留原圖',
    unsupported: '瀏覽器不支持復制圖片',
    copied: '已復制！',
    copy: '復制結果',
    copyFailed: '復制失敗',
    unsupportedFile: '請選擇 JPG、PNG、WebP 圖片，或 MP4/WebM/MOV 視頻。',
    fileTooLarge: '圖片調試入口暫不處理超過 20MB 的圖片。視頻會進入視頻調試頁。',
    handoffVideo: '正在進入視頻調試流程...'
};

let enginePromise = null;
let workerClient = null;
let currentItem = null;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const singlePreview = document.getElementById('singlePreview');
const originalImage = document.getElementById('originalImage');
const processedImage = document.getElementById('processedImage');
const originalInfo = document.getElementById('originalInfo');
const processedInfo = document.getElementById('processedInfo');
const downloadBtn = document.getElementById('downloadBtn');
const copyBtn = document.getElementById('copyBtn');
const resetBtn = document.getElementById('resetBtn');
const processedOverlay = document.getElementById('processedOverlay');
const sliderHandle = document.getElementById('sliderHandle');

async function getEngine() {
    if (!enginePromise) {
        enginePromise = WatermarkEngine.create().catch((error) => {
            enginePromise = null;
            throw error;
        });
    }
    return enginePromise;
}

function getEstimatedWatermarkInfo(item) {
    if (!item?.originalImg) return null;
    const { width, height } = item.originalImg;
    const config = detectWatermarkConfig(width, height);
    const position = calculateWatermarkPosition(width, height, config);
    return {
        size: config.logoSize,
        position,
        config
    };
}

function disableWorkerClient(reason) {
    if (!workerClient) return;
    console.warn('disable worker path, fallback to main thread:', reason);
    workerClient.dispose();
    workerClient = null;
}

function cleanupCurrentItem() {
    if (!currentItem) return;
    if (currentItem.originalUrl) URL.revokeObjectURL(currentItem.originalUrl);
    if (currentItem.processedUrl) URL.revokeObjectURL(currentItem.processedUrl);
    currentItem = null;
}

async function init() {
    try {
        showLoading(TEXT.loading);

        if (canUseWatermarkWorker()) {
            try {
                workerClient = new WatermarkWorkerClient({
                    workerUrl: './workers/watermark-worker.js'
                });
            } catch (workerError) {
                console.warn('worker unavailable, fallback to main thread:', workerError);
                workerClient = null;
            }
        }

        if (!workerClient) {
            getEngine().catch((error) => {
                console.warn('main thread engine warmup failed:', error);
            });
        }

        hideLoading();
        setupEventListeners();
        setupSlider();
        await consumePendingImageHandoff();
    } catch (error) {
        hideLoading();
        console.error('initialize error:', error);
    }
}

function setupEventListeners() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    document.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-primary', 'bg-emerald-50');
    });

    document.addEventListener('dragleave', (e) => {
        if (e.clientX === 0 && e.clientY === 0) {
            uploadArea.classList.remove('border-primary', 'bg-emerald-50');
        }
    });

    document.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-primary', 'bg-emerald-50');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(Array.from(e.dataTransfer.files));
        }
    });

    document.addEventListener('paste', (e) => {
        const items = e.clipboardData.items;
        const files = [];
        for (let i = 0; i < items.length; i++) {
            if (items[i].kind === 'file') {
                files.push(items[i].getAsFile());
            }
        }
        if (files.length > 0) handleFiles(files);
    });

    resetBtn.addEventListener('click', reset);
    window.addEventListener('beforeunload', () => {
        disableWorkerClient('beforeunload');
    });
}

function reset() {
    cleanupCurrentItem();
    singlePreview.style.display = 'none';
    fileInput.value = '';
    originalImage.src = '';
    processedImage.src = '';
    originalInfo.innerHTML = '';
    processedInfo.innerHTML = '';
    processedInfo.style.display = 'none';
    processedOverlay.style.display = 'none';
    sliderHandle.style.display = 'none';
    copyBtn.style.display = 'none';
    downloadBtn.style.display = 'none';
    setStatusMessage('');
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function handleFileSelect(e) {
    handleFiles(Array.from(e.target.files));
}

async function handleFiles(files) {
    setStatusMessage('');

    const validFile = pickDebugUploadFile(files);

    if (!validFile) {
        setStatusMessage(TEXT.unsupportedFile, 'warn');
        return;
    }

    const fileKind = getDebugFileKind(validFile);
    if (fileKind === 'video') {
        await routeVideoFile(validFile);
        return;
    }

    if (validFile.size > 20 * 1024 * 1024) {
        setStatusMessage(TEXT.fileTooLarge, 'warn');
        return;
    }

    cleanupCurrentItem();
    currentItem = {
        id: Date.now(),
        file: validFile,
        name: validFile.name,
        originalImg: null,
        processedMeta: null,
        processedBlob: null,
        originalUrl: null,
        processedUrl: null
    };

    singlePreview.style.display = 'block';
    processSingle(currentItem);
}

async function routeVideoFile(file) {
    try {
        showLoading(TEXT.handoffVideo);
        await saveDebugFileHandoff(file, 'video');
        window.location.assign('./video-preview.html?fileHandoff=1');
    } catch (error) {
        hideLoading();
        console.error(error);
        setStatusMessage(error.message || '无法进入视频调试流程，请打开视频页后重新选择文件。', 'warn');
    }
}

async function consumePendingImageHandoff() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('fileHandoff') !== '1') return;

    try {
        const record = await consumeDebugFileHandoff('image');
        if (!record?.file) return;
        await handleFiles([record.file]);
        window.history.replaceState(null, '', window.location.pathname);
    } catch (error) {
        console.warn('image handoff unavailable:', error);
        setStatusMessage(error.message || '读取图片暂存失败，请重新选择文件。', 'warn');
    }
}

function renderSingleImageMeta(item) {
    if (!item?.originalImg) return;

    const watermarkInfo = resolveDisplayWatermarkInfo(
        item,
        getEstimatedWatermarkInfo(item)
    );
    if (!watermarkInfo) return;

    originalInfo.innerHTML = `
        <p>${TEXT.size}: ${item.originalImg.width}x${item.originalImg.height}</p>
        <p>${TEXT.watermark}: ${watermarkInfo.size}x${watermarkInfo.size}</p>
        <p>${TEXT.position}: (${watermarkInfo.position.x},${watermarkInfo.position.y})</p>
    `;
}

function getProcessedStatusLabel(item) {
    return !isConfirmedWatermarkDecision(item)
        ? TEXT.skipped
        : TEXT.removed;
}

function renderSingleProcessedMeta(item) {
    if (!item?.originalImg) return;

    const watermarkInfo = resolveDisplayWatermarkInfo(
        item,
        getEstimatedWatermarkInfo(item)
    );
    const showWatermarkInfo = watermarkInfo && isConfirmedWatermarkDecision(item);

    processedInfo.innerHTML = `
        <p>${TEXT.size}: ${item.originalImg.width}x${item.originalImg.height}</p>
        ${showWatermarkInfo ? `<p>${TEXT.watermark}: ${watermarkInfo.size}x${watermarkInfo.size}</p>` : ''}
        ${showWatermarkInfo ? `<p>${TEXT.position}: (${watermarkInfo.position.x},${watermarkInfo.position.y})</p>` : ''}
        <p>${TEXT.status}: ${getProcessedStatusLabel(item)}</p>
    `;
}

async function processSingle(item) {
    try {
        const img = await loadImage(item.file);
        item.originalImg = img;
        item.originalUrl = img.src;

        originalImage.src = img.src;
        renderSingleImageMeta(item);

        const processed = await processImageWithBestPath(item.file, img);
        item.processedMeta = processed.meta;
        item.processedBlob = processed.blob;
        item.processedUrl = URL.createObjectURL(processed.blob);

        processedImage.src = item.processedUrl;
        processedOverlay.style.display = 'block';
        sliderHandle.style.display = 'flex';
        processedInfo.style.display = 'block';

        copyBtn.style.display = 'flex';
        copyBtn.onclick = () => copyImage(item);

        downloadBtn.style.display = 'flex';
        downloadBtn.onclick = () => downloadImage(item);

        renderSingleProcessedMeta(item);
        document.getElementById('comparisonContainer').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        console.error(error);
    }
}

async function processImageWithBestPath(file, fallbackImage, options = {}) {
    if (workerClient) {
        try {
            return await workerClient.processBlob(file, options);
        } catch (error) {
            console.warn('worker process failed, fallback to main thread:', error);
            disableWorkerClient(error);
        }
    }

    const engine = await getEngine();
    const canvas = await engine.removeWatermarkFromImage(fallbackImage, options);
    const blob = await canvasToBlob(canvas);
    return {
        blob,
        meta: canvas.__watermarkMeta || null
    };
}

async function copyImage(item, targetBtn = copyBtn) {
    if (!navigator.clipboard || !window.ClipboardItem) {
        setStatusMessage(TEXT.unsupported, 'warn');
        return;
    }

    try {
        if (!item.processedBlob) return;
        const data = [new ClipboardItem({ [item.processedBlob.type]: item.processedBlob })];
        await navigator.clipboard.write(data);

        const span = targetBtn.querySelector('span');
        const svg = targetBtn.querySelector('svg');
        const originalSvgPath = svg.innerHTML;

        span.textContent = TEXT.copied;
        svg.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>';

        setTimeout(() => {
            span.textContent = TEXT.copy;
            svg.innerHTML = originalSvgPath;
        }, 2000);
    } catch (err) {
        console.error('Failed to copy image: ', err);
        setStatusMessage(TEXT.copyFailed, 'warn');
    }
}

function downloadImage(item) {
    const a = document.createElement('a');
    a.href = item.processedUrl;
    a.download = `unwatermarked_${item.name.replace(/\.[^.]+$/, '')}.png`;
    a.click();
}

function setupSlider() {
    const container = document.getElementById('comparisonContainer');
    let isDown = false;

    function move(e) {
        if (!isDown) return;
        const rect = container.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        if (!clientX) return;

        const x = clientX - rect.left;
        const percent = Math.min(Math.max(x / rect.width, 0), 1) * 100;

        processedOverlay.style.width = `${percent}%`;
        sliderHandle.style.left = `${percent}%`;
    }

    container.addEventListener('mousedown', (e) => {
        isDown = true;
        move(e);
    });
    window.addEventListener('mouseup', () => { isDown = false; });
    window.addEventListener('mousemove', move);

    container.addEventListener('touchstart', (e) => {
        isDown = true;
        move(e);
    });
    window.addEventListener('touchend', () => { isDown = false; });
    window.addEventListener('touchmove', move);
}

init();
