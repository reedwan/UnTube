/**
 * offscreen.js - WebAudio host document for tab audio processing in Manifest V3.
 *
 * Responsibilities:
 * 1. Hosts the WebAudio pipeline inside an offscreen document.
 * 2. Manages MediaStream capture from active browser tabs.
 * 3. Bridges WebAudio frames to the inference worker via AudioWorklet.
 * 4. Applies bandpass filtering, soft expansion, and peak limiting.
 * 5. Manages engine lifecycle: starts on first captured tab, shuts down when all tabs release.
 *
 * WebAudio Processing Graph:
 * [MediaStreamSource(Tab)]
 *         │
 *         ▼
 * [BiquadFilterNode: High-Pass 75 Hz, Q=0.707]
 *         │
 *         ▼
 * [AudioWorkletNode: 'untube-audio-processor']
 *         │  ◄── MessagePort Transfer ──► [Inference Worker: FastEnhancer-M]
 *         ▼
 * [BiquadFilterNode: Low-Pass Stage 1, 9.5 kHz, Q=0.707]
 *         │
 *         ▼
 * [BiquadFilterNode: Low-Pass Stage 2, 10.5 kHz, Q=0.707]
 *         │
 *         ▼
 * [DynamicsCompressorNode: Peak Limiter] (Threshold: -1 dBFS, Ratio: 20:1, Attack: 2ms, Release: 80ms)
 *         │
 *         ▼
 * [GainNode: Output Gain = 1.0]
 *         │
 *         ▼
 * [AudioContext.destination] (System Output)
 */
'use strict';

const SAMPLE_RATE = 48000;
const WORKER_URL = chrome.runtime.getURL('inference-worker.js');
const WORKLET_URL = chrome.runtime.getURL('ring-buffer-worklet.js');

// Model Configuration: FastEnhancer-M
const MODEL_CONFIG = {
  name: 'FastEnhancer Medium',
  file: 'assets/fastenhancer_m.onnx',
  type: 'fastenhancer',

  hopSize: 320, // 320 samples @ 48kHz = 6.67 ms algorithmic frame latency
  sizeKb: 2444
};

// WebAudio Engine Graph Nodes
let audioCtx = null;
let highpassNode = null;
let lowpassNode1 = null;
let lowpassNode2 = null;
let limiterNode = null;
let workletNode = null;
let outputGainNode = null;
let activeWorker = null;

let isEngineReady = false;
let initPromise = null;
let currentGateMode = 'normal'; // Fixed default: 'normal' = Smooth De-Bleed Soft Expander

// Registry of actively captured tabs: tabId -> { stream, sourceNode }
const tabStreams = new Map();
// Registry of tabs currently initializing capture: tabId -> generation counter
const startingTabGenerations = new Map();
let nextGeneration = 1;

/**
 * Dispatches tab-specific status notifications to the background service worker.
 */
function sendTabStatus(state, tabId, extra = {}) {
  UnTubeLogger.info('OFFSCREEN', `Tab ${tabId} state -> ${state}`, extra.error || '');
  chrome.runtime.sendMessage({
    type: 'DF_TAB_STATUS',
    state,
    tabId,
    ...extra
  }).catch(() => {});
}

/**
 * Dispatches overall engine status notifications to the background service worker.
 */
function sendEngineStatus(state, extra = {}) {
  UnTubeLogger.info('OFFSCREEN', `Engine state -> ${state}`, extra.error || '');
  chrome.runtime.sendMessage({
    type: 'DF_STATUS',
    state,
    modelKey: 'fastenhancer_m',
    ...extra
  }).catch(() => {});
}

/**
 * Ensures the WebAudio context and ONNX Runtime worker are initialized and ready.
 * Reuses existing audio context if already warm and active.
 */
async function ensureEngine() {
  if (isEngineReady && audioCtx && audioCtx.state !== 'closed') {
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }
    if (workletNode) {
      workletNode.port.postMessage({ type: 'SET_GATE_MODE', mode: currentGateMode });
    }
    return;
  }

  if (initPromise) {
    return initPromise;
  }

  initPromise = (async () => {
    UnTubeLogger.info('OFFSCREEN', `Initializing neural engine with model: ${MODEL_CONFIG.name}...`);

    // 1. Start fetching ONNX model binary immediately
    const modelUrl = chrome.runtime.getURL(MODEL_CONFIG.file);
    const modelBufferPromise = fetch(modelUrl).then(async (modelRes) => {
      if (!modelRes.ok) {
        throw new Error(`Failed to load ONNX model binary (${modelRes.status} ${modelRes.statusText})`);
      }
      const buf = await modelRes.arrayBuffer();
      UnTubeLogger.info('OFFSCREEN', `Model binary loaded (${(buf.byteLength / 1024).toFixed(1)} KB)`);
      return buf;
    });

    // 2. Initialize 48kHz interactive AudioContext & load AudioWorklet module concurrently
    try {
      audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'interactive' });
    } catch (e) {
      UnTubeLogger.warn('OFFSCREEN', 'Custom sample rate AudioContext failed, falling back to default device rate:', e);
      audioCtx = new AudioContext({ latencyHint: 'interactive' });
    }
    audioCtx.onstatechange = () => {
      UnTubeLogger.info('OFFSCREEN', `AudioContext state changed: ${audioCtx?.state}`);
      if (audioCtx?.state === 'suspended' && tabStreams.size > 0) {
        audioCtx.resume().catch(() => {});
      }
    };
    const resumePromise = audioCtx.state === 'suspended' ? audioCtx.resume() : Promise.resolve();
    const workletModulePromise = audioCtx.audioWorklet.addModule(WORKLET_URL);

    // 3. Launch dedicated WebAssembly ONNX inference worker in parallel (pre-loads ort.wasm.min.js)
    activeWorker = new Worker(WORKER_URL);
    const workerReadyPromise = new Promise((resolve, reject) => {
      let settled = false;
      const timeout = setTimeout(() => {
        settled = true;
        reject(new Error('Inference worker init timed out (15s)'));
      }, 15000);

      activeWorker.onmessage = (e) => {
        const data = e.data || {};
        if (data.type === 'READY') {
          settled = true;
          clearTimeout(timeout);
          UnTubeLogger.info('OFFSCREEN', `Neural inference worker READY (${MODEL_CONFIG.name}).`);
          resolve();
        } else if (data.type === 'ERROR') {
          if (!settled) {
            settled = true;
            clearTimeout(timeout);
            reject(new Error(data.error || 'Worker initialization failed'));
          } else {
            UnTubeLogger.error('OFFSCREEN', 'Runtime inference worker error:', data.error);
          }
        } else if (data.type === 'STATS') {
          UnTubeLogger.info(
            'OFFSCREEN',
            `[TELEMETRY] Hop ${data.avgMs.toFixed(2)}ms (budget ${data.hopBudgetMs.toFixed(2)}ms), backlog: ${data.backlog}, dropped: ${data.droppedHops}`
          );
        }
      };

      activeWorker.onerror = (e) => {
        if (!settled) {
          settled = true;
          clearTimeout(timeout);
          reject(new Error(`Worker thread error: ${e.message}`));
        } else {
          UnTubeLogger.error('OFFSCREEN', `Worker thread error: ${e.message}`);
        }
      };

      modelBufferPromise
        .then((modelBuffer) => {
          if (settled || !activeWorker) return;
          // Transfer model binary buffer ownership to the worker thread once fetched
          activeWorker.postMessage(
            {
              command: 'init',
              modelBuffer,
              modelType: MODEL_CONFIG.type,
              hopSize: MODEL_CONFIG.hopSize
            },
            [modelBuffer]
          );
        })
        .catch((err) => {
          if (!settled) {
            settled = true;
            clearTimeout(timeout);
            reject(err);
          }
        });
    });

    // 4. Wait for Worker session, AudioWorklet module, and AudioContext resume in parallel
    await Promise.all([workerReadyPromise, workletModulePromise, resumePromise]);
    workletNode = new AudioWorkletNode(audioCtx, 'untube-audio-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2]
    });

    // 5. Connect AudioWorklet directly to Inference Worker via dedicated MessagePort (zero-copy transfers)
    const channel = new MessageChannel();
    workletNode.port.postMessage({ type: 'INIT_PORT', port: channel.port1 }, [channel.port1]);
    activeWorker.postMessage({ command: 'INIT_PORT', port: channel.port2 }, [channel.port2]);

    // 6. Configure worklet hop size (320 samples) and default gate mode ('normal' = Smooth)
    workletNode.port.postMessage({ type: 'SET_HOP_SIZE', hopSize: MODEL_CONFIG.hopSize });
    workletNode.port.postMessage({ type: 'SET_GATE_MODE', mode: currentGateMode });

    // 7. Input High-Pass Filter (75 Hz 2nd-order Butterworth):
    // Removes non-speech mechanical vibrations, wind pops, and DC offsets before the neural network
    highpassNode = audioCtx.createBiquadFilter();
    highpassNode.type = 'highpass';
    highpassNode.frequency.value = 75;
    highpassNode.Q.value = 0.707;

    // 8. 4-Pole Low-Pass Butterworth Filter (9.5 kHz + 10.5 kHz composite)
    lowpassNode1 = audioCtx.createBiquadFilter();
    lowpassNode1.type = 'lowpass';
    lowpassNode1.frequency.value = 9500;
    lowpassNode1.Q.value = 0.707;

    lowpassNode2 = audioCtx.createBiquadFilter();
    lowpassNode2.type = 'lowpass';
    lowpassNode2.frequency.value = 10500;
    lowpassNode2.Q.value = 0.707;

    // 9. Peak Limiter (-1 dBFS threshold)
    limiterNode = audioCtx.createDynamicsCompressor();

    limiterNode.threshold.value = -1.0;
    limiterNode.knee.value = 3.0;
    limiterNode.ratio.value = 20.0;
    limiterNode.attack.value = 0.002;
    limiterNode.release.value = 0.08;

    outputGainNode = audioCtx.createGain();
    outputGainNode.gain.value = 1.0;

    // 10. Assemble WebAudio Routing Graph
    highpassNode.connect(workletNode);
    workletNode.connect(lowpassNode1);
    lowpassNode1.connect(lowpassNode2);
    lowpassNode2.connect(limiterNode);
    limiterNode.connect(outputGainNode);
    outputGainNode.connect(audioCtx.destination);

    isEngineReady = true;
    UnTubeLogger.info('OFFSCREEN', `Neural speech isolation engine ready (${MODEL_CONFIG.name}).`);
    sendEngineStatus('active');
  })().finally(() => {
    initPromise = null;
  });

  return initPromise;
}

/**
 * Connects a tab's audio stream to the neural isolation engine.
 */
async function startTabPipeline(streamId, tabId) {
  UnTubeLogger.info('OFFSCREEN', `Connecting audio stream for tab ${tabId} (streamId=${streamId})`);

  const gen = nextGeneration++;
  startingTabGenerations.set(tabId, gen);

  let acquiredStream = null;
  try {
    if (tabStreams.has(tabId)) {
      stopTabPipeline(tabId, false);
      startingTabGenerations.set(tabId, gen);
    }

    // 1. Immediately redeem the single-use streamId token via getUserMedia.
    // Chrome expires streamId tokens after a few seconds, so it must be consumed
    // immediately without waiting for neural engine compilation.
    const streamPromise = navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId
        }
      },
      video: false
    }).then((s) => {
      acquiredStream = s;
      return s;
    });

    // 2. Concurrently ensure neural engine is warm and ready
    const [stream] = await Promise.all([streamPromise, ensureEngine()]);

    // Abort and release stream if cancelled while awaiting
    if (startingTabGenerations.get(tabId) !== gen || !audioCtx || !highpassNode) {
      if (acquiredStream) {
        acquiredStream.getTracks().forEach((t) => {
          try { t.stop(); } catch (_) {}
        });
      }
      if (tabStreams.size === 0 && startingTabGenerations.size === 0) {
        shutdownEngine();
      }
      return;
    }

    startingTabGenerations.delete(tabId);

    // Connect tab audio stream source to the engine's high-pass filter input
    const sourceNode = audioCtx.createMediaStreamSource(stream);
    sourceNode.connect(highpassNode);

    // Ensure AudioContext is actively running (crucial on Windows where AudioContext may start suspended)
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }

    // Auto-cleanup if the tab's audio track terminates
    stream.getAudioTracks().forEach(track => {
      track.onended = () => {
        UnTubeLogger.info('OFFSCREEN', `Audio track ended for tab ${tabId}`);
        stopTabPipeline(tabId);
      };
    });

    tabStreams.set(tabId, { stream, sourceNode });
    UnTubeLogger.info('OFFSCREEN', `Tab ${tabId} connected. Active filtered tabs: ${tabStreams.size}`);
    sendTabStatus('active', tabId);
  } catch (err) {
    if (acquiredStream) {
      acquiredStream.getTracks().forEach((t) => {
        try { t.stop(); } catch (_) {}
      });
    }
    const wasCurrent = startingTabGenerations.get(tabId) === gen;
    if (wasCurrent) {
      startingTabGenerations.delete(tabId);
    }
    UnTubeLogger.error('OFFSCREEN', `Failed to start pipeline for tab ${tabId}:`, err);
    stopTabPipeline(tabId);
    if (wasCurrent) {
      sendTabStatus('error', tabId, { error: err.message || String(err) });
    }
  }
}

/**
 * Disconnects a tab's audio stream and releases its MediaStream tracks.
 * Automatically triggers engine shutdown if no tabs remain captured.
 */
function stopTabPipeline(tabId, shouldShutdownIfEmpty = true) {
  const wasStarting = startingTabGenerations.delete(tabId);
  const hasActive = tabStreams.has(tabId);
  if (!hasActive && !wasStarting) return;
  UnTubeLogger.info('OFFSCREEN', `Disconnecting tab ${tabId}...`);

  const entry = tabStreams.get(tabId);
  tabStreams.delete(tabId);

  if (entry) {
    try { entry.sourceNode.disconnect(); } catch (_) {}
    if (entry.stream) {
      for (const track of entry.stream.getTracks()) {
        try { track.stop(); } catch (_) {}
      }
    }
  }

  sendTabStatus('stopped', tabId);

  // Shut down engine if no captured or starting tabs remain
  if (shouldShutdownIfEmpty && tabStreams.size === 0 && startingTabGenerations.size === 0) {
    shutdownEngine();
  }
}

/**
 * Tears down WebAudio graph and terminates worker thread.
 */
async function shutdownEngine() {

  UnTubeLogger.info('OFFSCREEN', 'Shutting down neural audio engine...');
  isEngineReady = false;
  startingTabGenerations.clear();

  for (const [id, entry] of tabStreams.entries()) {
    try { entry.sourceNode.disconnect(); } catch (_) {}
    if (entry.stream) {
      for (const track of entry.stream.getTracks()) {
        try { track.stop(); } catch (_) {}
      }
    }
  }
  tabStreams.clear();

  if (activeWorker) {
    try {
      activeWorker.onmessage = null;
      activeWorker.onerror = null;
      activeWorker.postMessage({ command: 'stop' });
      activeWorker.terminate();
    } catch (_) {}
    activeWorker = null;
  }

  if (workletNode) {
    try { workletNode.disconnect(); } catch (_) {}
    workletNode = null;
  }

  if (highpassNode) {
    try { highpassNode.disconnect(); } catch (_) {}
    highpassNode = null;
  }

  if (lowpassNode1) {
    try { lowpassNode1.disconnect(); } catch (_) {}
    lowpassNode1 = null;
  }

  if (lowpassNode2) {
    try { lowpassNode2.disconnect(); } catch (_) {}
    lowpassNode2 = null;
  }

  if (limiterNode) {
    try { limiterNode.disconnect(); } catch (_) {}
    limiterNode = null;
  }

  if (outputGainNode) {
    try { outputGainNode.disconnect(); } catch (_) {}
    outputGainNode = null;
  }

  if (audioCtx) {
    const ctxToClose = audioCtx;
    audioCtx = null;
    try { await ctxToClose.close(); } catch (_) {}
  }

  if (!initPromise && !isEngineReady && tabStreams.size === 0 && startingTabGenerations.size === 0) {
    sendEngineStatus('stopped');
  }
}

// Service worker IPC message routing
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'DF_PING') {
    sendResponse({ pong: true });
    return false;
  }

  if (msg.type === 'DF_START_TAB') {
    startTabPipeline(msg.streamId, msg.tabId);
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === 'DF_STOP_TAB') {
    stopTabPipeline(msg.tabId);
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === 'DF_STOP_ALL') {
    shutdownEngine().then(() => sendResponse({ ok: true }));
    return true;
  }

  if (msg.type === 'DF_SET_GATE_MODE') {
    currentGateMode = msg.gateMode || 'normal';
    if (workletNode) {
      workletNode.port.postMessage({ type: 'SET_GATE_MODE', mode: currentGateMode });
    }
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === 'DF_QUERY_STATE') {
    sendResponse({
      isEngineReady,
      currentModelKey: 'fastenhancer_m',
      activeTabs: Array.from(tabStreams.keys())
    });
    return false;
  }
});

UnTubeLogger.info('OFFSCREEN', 'Offscreen audio host initialized.');
sendEngineStatus('ready');

// Pre-warm the neural engine in the background as soon as offscreen document loads
ensureEngine().catch((err) => {
  UnTubeLogger.warn('OFFSCREEN', 'Background engine pre-warm failed:', err);
});
