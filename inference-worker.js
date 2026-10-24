/**
 * inference-worker.js - WebAssembly inference worker running FastEnhancer-M via ONNX Runtime.
 *
 * Architecture:
 * 1. Mono-Downmix:
 *    - Downmixes stereo input (L + R) * 0.5 to mono prior to inference.
 *    - Produces dual-mono output.
 *
 * 2. Streaming Recurrent Caches:
 *    - Maintains causal convolutional and GRU state across 320-sample hops (6.67 ms).
 *    - Cache tensors are updated in-place each hop to eliminate heap allocations.
 *
 * 3. AudioWorklet Bridge:
 *    - Communicates over a MessagePort using Transferable ArrayBuffers.
 *    - Bounded backlog queue prevents latency drift during transient CPU spikes.
 */

'use strict';

const EXTENSION_BASE_URL = self.location.origin + '/';
const ORT_BASE_URL = EXTENSION_BASE_URL + 'assets/vendor/onnxruntime-web/';
const SAMPLE_RATE = 48000;
const HOP_SIZE = 320; // 6.67 ms at 48 kHz

// Load ONNX Runtime Web WASM distribution
importScripts(ORT_BASE_URL + 'ort.wasm.min.js');

// Configure ONNX Runtime Web for low-latency real-time audio inference
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = {
  mjs: ORT_BASE_URL + 'ort-wasm-simd-threaded.mjs',
  wasm: ORT_BASE_URL + 'ort-wasm-simd-threaded.wasm'
};

let session = null;
let sessionInputNames = [];
let cacheInputNames = [];
let cacheOutputNames = [];
let workletPort = null;
let isProcessing = false;
let isSwitching = false;

// Persistent recurrent cache feed map:
// Updated each hop via zero-copy tensor reference binding.
let cacheTensors = null;
let feeds = null;
const inputBuffer = new Float32Array(HOP_SIZE);
const inputTensor = new ort.Tensor('float32', inputBuffer, [1, HOP_SIZE]);
const emptyWavOut = new Float32Array(HOP_SIZE);

// Silence detection & auto-sleep: sleep neural inference when audio is paused (<0.1% CPU)
// Digital silence threshold: -100 dBFS (peak absolute amplitude < 1e-5)
const SILENCE_THRESHOLD = 1e-5;
const SILENCE_DECAY_HOPS = 15; // ~100 ms of silence to flush recurrent tail before sleeping
let consecutiveSilentHops = 0;

const DEBUG = false;

// Backlog queue & telemetry
const MAX_BACKLOG_HOPS = 24; // ~160 ms maximum allowable queue before dropping oldest hops
const pendingQueue = [];
let droppedHopsCount = 0;
let hopStats = {
  count: 0,
  sumMs: 0,
  maxMs: 0,
  lastReport: performance.now()
};

/**
 * Extract tensor dimensions from metadata.
 * @param {Object} meta - Input metadata from ONNX session.
 * @returns {number[]} Dimension array with valid positive numbers.
 */
function getTensorDims(meta) {
  const arr = (meta && (meta.shape || meta.dims)) || [];
  return arr.map(d => (typeof d === 'number' && d > 0 ? d : 1));
}

/**
 * Reset all recurrent states and allocate initial cache tensors.
 * Pre-allocates a static feeds object so session.run() performs zero feed object allocations.
 */
function resetChannelStates() {
  if (!session) return;

  const cacheSizes = cacheInputNames.map(name => {
    const meta = (session.inputMetadata || []).find(m => m.name === name) || {};
    const dims = getTensorDims(meta);
    const size = dims.reduce((a, b) => a * b, 1);
    return { size, dims };
  });

  cacheTensors = cacheSizes.map(c => new ort.Tensor('float32', new Float32Array(c.size), c.dims));

  feeds = { wav_in: inputTensor };
  for (let i = 0; i < cacheInputNames.length; i++) {
    feeds[cacheInputNames[i]] = cacheTensors[i];
  }

  inputBuffer.fill(0);
  consecutiveSilentHops = 0;
}

/**
 * Execute single-hop neural inference on mono audio frame.
 * Binds output cache tensors directly into feeds to eliminate 13.3 MB/s of memcpy overhead.
 * @returns {Promise<Float32Array>} Enhanced audio frame of HOP_SIZE samples.
 */
async function processChannelFastEnhancer() {
  const results = await session.run(feeds);
  const enhancedData = results.wav_out?.data || emptyWavOut;

  // Zero-copy pointer swap of recurrent hidden state tensors for next hop
  for (let i = 0; i < cacheOutputNames.length; i++) {
    feeds[cacheInputNames[i]] = results[cacheOutputNames[i]];
  }

  return enhancedData;
}

/**
 * Process a single incoming mono buffer received from the AudioWorklet.
 * @param {ArrayBuffer} monoBuffer - Mono Float32Array buffer of HOP_SIZE samples.
 */
async function handleIncomingAudio(monoBuffer) {
  if (!session || !workletPort || isSwitching) return;
  const tStart = DEBUG ? performance.now() : 0;

  const monoIO = new Float32Array(monoBuffer);
  if (monoIO.length < HOP_SIZE) return;

  // 1. Copy mono frame via native SIMD memcpy & detect digital silence with early exit
  inputBuffer.set(monoIO.length === HOP_SIZE ? monoIO : monoIO.subarray(0, HOP_SIZE));
  let isSilent = true;
  for (let i = 0; i < HOP_SIZE; i++) {
    if (Math.abs(monoIO[i]) > SILENCE_THRESHOLD) {
      isSilent = false;
      break;
    }
  }

  // 2. Neural Enhancement or silence sleep
  let enhanced;
  if (isSilent) {
    consecutiveSilentHops++;
    if (consecutiveSilentHops > SILENCE_DECAY_HOPS) {
      // Audio stream is paused: skip inference entirely to drop CPU to ~0.0%
      enhanced = emptyWavOut;
    } else {
      // Natural tail decay: run inference for first 15 hops (~100 ms) to let recurrent state decay
      enhanced = await processChannelFastEnhancer();
    }
  } else {
    consecutiveSilentHops = 0;
    enhanced = await processChannelFastEnhancer();
  }

  // 3. Write enhanced mono audio in-place into monoIO buffer with clamp (zero heap allocation)
  for (let i = 0; i < HOP_SIZE; i++) {
    let s = enhanced[i];
    if (s > 1.0) s = 1.0;
    else if (s < -1.0) s = -1.0;
    monoIO[i] = s;
  }

  // 4. Transfer processed mono buffer back to AudioWorklet (recycled into worklet pool)
  workletPort.postMessage(
    { type: 'FILTERED', buffer: monoIO.buffer },
    [monoIO.buffer]
  );

  // 5. Performance telemetry calculation (only active when DEBUG = true)
  if (DEBUG) {
    const elapsed = performance.now() - tStart;
    hopStats.count++;
    hopStats.sumMs += elapsed;
    if (elapsed > hopStats.maxMs) hopStats.maxMs = elapsed;

    const now = performance.now();
    if (now - hopStats.lastReport >= 2000) {
      const avgMs = hopStats.count > 0 ? hopStats.sumMs / hopStats.count : 0;
      const hopBudgetMs = (HOP_SIZE / SAMPLE_RATE) * 1000; // 6.67 ms
      self.postMessage({
        type: 'STATS',
        avgMs,
        maxMs: hopStats.maxMs,
        hopBudgetMs,
        backlog: pendingQueue.length,
        droppedHops: droppedHopsCount
      });
      hopStats.count = 0;
      hopStats.sumMs = 0;
      hopStats.maxMs = 0;
      hopStats.lastReport = now;
    }
  }
}

/**
 * Process pending audio hops sequentially from backlog queue.
 */
async function processQueue() {
  if (isProcessing || isSwitching) return;
  isProcessing = true;

  try {
    while (pendingQueue.length > 0 && !isSwitching) {
      const nextBuf = pendingQueue.shift();
      await handleIncomingAudio(nextBuf);
    }
  } catch (err) {
    console.error('[INFERENCE] Error processing audio frame:', err);
    self.postMessage({ type: 'ERROR', error: `Inference error: ${err.message || String(err)}` });
  } finally {
    isProcessing = false;
  }
}

/**
 * Initialize ONNX Runtime InferenceSession from binary ArrayBuffer.
 * @param {ArrayBuffer} modelBuffer - ONNX model binary data.
 */
async function initSessionFromBuffer(modelBuffer) {
  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });

  sessionInputNames = session.inputNames.slice();

  // Extract FastEnhancer recurrent cache names (cache_in_0 ... cache_in_K)
  cacheInputNames = sessionInputNames
    .filter(name => name.startsWith('cache_in_'))
    .sort((a, b) => parseInt(a.slice(9), 10) - parseInt(b.slice(9), 10));

  cacheOutputNames = session.outputNames
    .filter(name => name.startsWith('cache_out_'))
    .sort((a, b) => parseInt(a.slice(10), 10) - parseInt(b.slice(10), 10));

  resetChannelStates();

  // Notify worklet of confirmed hop size
  if (workletPort) {
    workletPort.postMessage({ type: 'SET_HOP_SIZE', hopSize: HOP_SIZE });
  }

  // Warm-up inference pass to trigger JIT compilation and graph optimization
  try {
    await processChannelFastEnhancer();
    resetChannelStates();
  } catch (wErr) {
    console.warn('[INFERENCE] Warm-up notice:', wErr.message);
  }
}

/**
 * Main Worker Message Router
 */
self.onmessage = async function (e) {
  const data = e.data || {};

  switch (data.command) {
    case 'init': {
      try {
        await initSessionFromBuffer(data.modelBuffer);
        self.postMessage({ type: 'READY', modelType: 'fastenhancer', hopSize: HOP_SIZE });
      } catch (err) {
        self.postMessage({ type: 'ERROR', error: `Worker init failed: ${err.message || String(err)}` });
      }
      break;
    }

    case 'load_model': {
      try {
        isSwitching = true;
        pendingQueue.length = 0;
        let waited = 0;
        while (isProcessing && waited < 500) {
          await new Promise(r => setTimeout(r, 5));
          waited += 5;
        }
        await initSessionFromBuffer(data.modelBuffer);
        isSwitching = false;
        self.postMessage({ type: 'MODEL_LOADED', modelType: 'fastenhancer', hopSize: HOP_SIZE });
      } catch (err) {
        isSwitching = false;
        self.postMessage({ type: 'ERROR', error: `Model reload failed: ${err.message || String(err)}` });
      }
      break;
    }

    case 'INIT_PORT': {
      if (data.port) {
        workletPort = data.port;
        workletPort.onmessage = (ev) => {
          const msg = ev.data || {};
          if (msg.type === 'PROCESS' && msg.buffer) {
            if (isSwitching) return;
            pendingQueue.push(msg.buffer);
            while (pendingQueue.length > MAX_BACKLOG_HOPS) {
              pendingQueue.shift();
              droppedHopsCount++;
            }
            processQueue().catch(() => {});
          }
        };
        workletPort.postMessage({ type: 'SET_HOP_SIZE', hopSize: HOP_SIZE });
      }
      break;
    }

    case 'stop': {
      pendingQueue.length = 0;
      resetChannelStates();
      break;
    }
  }
};
