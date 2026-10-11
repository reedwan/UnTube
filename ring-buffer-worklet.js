/**
 * ring-buffer-worklet.js - AudioWorkletProcessor handling frame bridging and soft expansion.
 *
 * Pipeline:
 * 1. Quantum vs Hop Resampling / Elastic Queuing:
 *    - WebAudio operates on 128-sample quanta (2.67 ms at 48 kHz).
 *    - FastEnhancer-M operates on 320-sample hops (6.67 ms at 48 kHz).
 *    - An elastic circular buffer bridges quantum-to-hop mismatch.
 *
 * 2. Buffer Transfer:
 *    - Input stereo hops are accumulated and transferred to the InferenceWorker via
 *      Float32Array buffers over a dedicated MessagePort.
 *    - Filtered frames are enqueued into the output ring buffer.
 *
 * 3. Soft Expander (C1-Continuous Hermite Smoothstep):
 *    - Attenuates residual background energy during speech pauses.
 *    - Parameters: 14 ms attack, 90 ms release.
 *    - Applies cubic Hermite curve S(t) = 3t^2 - 2t^3 interpolated per-sample across
 *      the 128-sample quantum to prevent discontinuity artifacts.
 */

'use strict';

const CHANNELS = 2;
// 16,384 samples = 8,192 stereo frames (~170.6 ms at 48 kHz). Power-of-two for bitwise masking.
const QUEUE_CAPACITY = 16384;
const QUEUE_MASK = QUEUE_CAPACITY - 1;

class UnTubeAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.workerPort = null;
    this.isBuffering = true;
    this.consecutiveUnderruns = 0;

    // FastEnhancer-M standard frame hop: 320 samples per channel (6.67 ms at 48 kHz)
    this.hopSize = 320;
    this.stereoHop = this.hopSize * CHANNELS; // 640 floats

    // Pre-buffer threshold: 6 hops (~40 ms at 48 kHz) provides a robust safety margin
    // against thread scheduling jitter while maintaining low perceived latency.
    this.preBufferSamples = this.stereoHop * 6; // 3840 floats

    // Input staging accumulator (interleaved stereo: L0, R0, L1, R1, ...)
    this.inputAccumulator = new Float32Array(QUEUE_CAPACITY);
    this.accumCount = 0;

    // Output circular ring buffer
    this.outputRing = new Float32Array(QUEUE_CAPACITY);
    this.ringWrite = 0;
    this.ringRead = 0;
    this.ringAvailable = 0;

    // Soft expander (14 ms attack, 90 ms release)
    this.gateMode = 'normal'; // 'off' | 'normal' | 'aggressive'

    this.expanderEnv = 0.0;
    this.expanderGain = 1.0;
    this.expanderAttack = 1.0 - Math.exp(-1.0 / (0.014 * 48000 / 128));
    this.expanderRelease = 1.0 - Math.exp(-1.0 / (0.090 * 48000 / 128));

    // Message port handlers
    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'INIT_PORT' && data.port) {
        this.workerPort = data.port;
        this.workerPort.onmessage = (e) => {
          const msg = e.data || {};
          if (msg.type === 'SET_HOP_SIZE' && msg.hopSize) {
            this.setHopSize(msg.hopSize);
          } else {
            this._onFilteredFrame(msg);
          }
        };
      }
      if (data.type === 'SET_HOP_SIZE' && data.hopSize) {
        this.setHopSize(data.hopSize);
      }
      if (data.type === 'SET_GATE_MODE' && data.mode) {
        this.gateMode = data.mode;
      }
      if (data.type === 'RESET') {
        this._reset();
      }
    };
  }

  /**
   * Adjust hop sizing dynamically if model parameters change.
   * @param {number} newHopSize - Samples per channel per neural inference hop.
   */
  setHopSize(newHopSize) {
    if (!newHopSize || newHopSize <= 0) return;
    this.hopSize = newHopSize;
    this.stereoHop = this.hopSize * CHANNELS;
    this.preBufferSamples = this.stereoHop * 6;
    this._reset();
  }

  /**
   * Reset ring buffer pointers and expander envelope state.
   */
  _reset() {
    this.isBuffering = true;
    this.consecutiveUnderruns = 0;
    this.accumCount = 0;
    this.ringWrite = 0;
    this.ringRead = 0;
    this.ringAvailable = 0;
    this.expanderEnv = 0.0;
    this.expanderGain = 1.0;
    this.inputAccumulator.fill(0);
    this.outputRing.fill(0);
  }

  /**
   * Handle incoming neural-filtered audio from InferenceWorker via dedicated port.
   * @param {Object} data - Contains transferable Float32Array buffer.
   */
  _onFilteredFrame(data) {
    if (!data || !data.buffer) return;
    const incoming = new Float32Array(data.buffer);
    const count = incoming.length;

    // Fast-path: Enqueue incoming samples using native SIMD memcpy (TypedArray.prototype.set)
    const firstChunk = QUEUE_CAPACITY - this.ringWrite;
    if (firstChunk >= count) {
      this.outputRing.set(incoming, this.ringWrite);
      this.ringWrite = (this.ringWrite + count) & QUEUE_MASK;
    } else {
      this.outputRing.set(incoming.subarray(0, firstChunk), this.ringWrite);
      this.outputRing.set(incoming.subarray(firstChunk), 0);
      this.ringWrite = count - firstChunk;
    }
    this.ringAvailable = Math.min(QUEUE_CAPACITY, this.ringAvailable + count);

    // Latency bounding: prevent unbounded drift if the worker experienced an OS hitch
    // Drop excess older samples if buffer exceeds 75% capacity (~128 ms).
    const maxFill = Math.floor(QUEUE_CAPACITY * 0.75);
    if (this.ringAvailable > maxFill) {
      const drop = Math.floor((this.ringAvailable - this.preBufferSamples) / CHANNELS) * CHANNELS;
      this.ringRead = (this.ringRead + drop) & QUEUE_MASK;
      this.ringAvailable -= drop;
    }

    // End pre-buffering once initial safety threshold is reached
    if (this.isBuffering && this.ringAvailable >= this.preBufferSamples) {
      this.isBuffering = false;
    }
  }

  /**
   * Core AudioWorklet render loop. Called synchronously every 128 samples (2.67 ms at 48 kHz).
   * @param {Float32Array[][]} inputs - Audio inputs [channel][sample].
   * @param {Float32Array[][]} outputs - Audio outputs [channel][sample].
   * @returns {boolean} Keep worklet alive.
   */
  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];

    // If input or output channels are unavailable, keep processor alive
    if (!input || !input[0] || !output || !output[0]) {
      return true;
    }

    const leftIn = input[0];
    const rightIn = input[1] || input[0];
    const leftOut = output[0];
    const rightOut = output[1] || null;
    const quantumLen = leftIn.length; // Always 128 samples in standard Web Audio

    // -------------------------------------------------------------------------
    // 1. Stage incoming samples into the input accumulator
    // -------------------------------------------------------------------------
    // The write index is derived from the frame counter on every iteration (rather than
    // carried as a separate strided pointer) so it can never drift out of sync with `acc`
    // when a hop boundary falls in the middle of a 128-sample quantum.
    let acc = this.accumCount;
    for (let i = 0; i < quantumLen; i++) {
      const idx = acc * CHANNELS;
      this.inputAccumulator[idx] = leftIn[i];
      this.inputAccumulator[idx + 1] = rightIn[i];
      acc++;

      // When a complete neural frame (hopSize) has been staged, dispatch to worker
      if (acc >= this.hopSize) {
        if (this.workerPort) {
          // Slice a fresh Float32Array to transfer zero-copy ownership to worker thread
          const transferCopy = new Float32Array(this.inputAccumulator.subarray(0, this.stereoHop));
          this.workerPort.postMessage(
            { type: 'PROCESS', buffer: transferCopy.buffer },
            [transferCopy.buffer]
          );
        }
        acc = 0;
      }
    }
    this.accumCount = acc;

    // -------------------------------------------------------------------------
    // 2. Playback: Read neural-filtered samples from output ring buffer
    // -------------------------------------------------------------------------
    const neededSamples = quantumLen * CHANNELS; // 256 samples (128 stereo frames)

    // Emit silence while pre-buffering
    if (this.isBuffering) {
      leftOut.fill(0);
      if (rightOut) rightOut.fill(0);
      return true;
    }

    if (this.ringAvailable >= neededSamples) {
      // Dequeue stereo samples with register-cached pointer
      let r = this.ringRead;
      const ring = this.outputRing;
      if (rightOut) {
        for (let i = 0; i < quantumLen; i++) {
          leftOut[i] = ring[r];
          r = (r + 1) & QUEUE_MASK;
          rightOut[i] = ring[r];
          r = (r + 1) & QUEUE_MASK;
        }
      } else {
        for (let i = 0; i < quantumLen; i++) {
          leftOut[i] = ring[r];
          r = (r + 2) & QUEUE_MASK;
        }
      }
      this.ringRead = r;
      this.ringAvailable -= neededSamples;
      this.consecutiveUnderruns = 0;

      // 3. Soft Expander
      if (this.gateMode !== 'off') {

        const tHigh = this.gateMode === 'aggressive' ? 0.026 : 0.018;
        const tLow  = this.gateMode === 'aggressive' ? 0.009 : 0.005;
        const floor = this.gateMode === 'aggressive' ? 0.008 : 0.025;

        // Calculate Root-Mean-Square (RMS) energy across current quantum
        let sumSq = 0;
        for (let i = 0; i < quantumLen; i++) {
          const mono = rightOut ? 0.5 * (leftOut[i] + rightOut[i]) : leftOut[i];
          sumSq += mono * mono;
        }
        let rms = Math.sqrt(sumSq / quantumLen);
        if (!Number.isFinite(rms)) {
          rms = 0.0;
        }
        if (!Number.isFinite(this.expanderEnv)) {
          this.expanderEnv = 0.0;
        }

        // Envelope tracking with dual attack / release ballistic smoothing
        if (rms > this.expanderEnv) {
          this.expanderEnv += this.expanderAttack * (rms - this.expanderEnv);
        } else {
          this.expanderEnv += this.expanderRelease * (rms - this.expanderEnv);
        }

        // Compute target gain via Hermite cubic smoothstep S(t) = 3t^2 - 2t^3
        let targetGain = 1.0;
        if (this.expanderEnv <= tLow) {
          targetGain = floor;
        } else if (this.expanderEnv < tHigh) {
          const t = (this.expanderEnv - tLow) / (tHigh - tLow);
          targetGain = floor + (1.0 - floor) * (t * t * (3.0 - 2.0 * t));
        }

        const gStart = this.expanderGain;
        const deltaG = targetGain - gStart;

        // Steady-State Fast-Path Optimization:
        // When speech is actively sustained (targetGain = 1.0, gStart = 1.0),
        // skip loop completely. Over 90% of audio during dialogue is at unity gain!
        if (targetGain === 1.0 && gStart === 1.0) {
          // Zero compute needed
        } else if (Math.abs(deltaG) < 1e-4) {
          // Constant noise-floor gain: simple linear multiply without polynomial
          for (let i = 0; i < quantumLen; i++) {
            leftOut[i] *= targetGain;
            if (rightOut) rightOut[i] *= targetGain;
          }
          this.expanderGain = targetGain;
        } else {
          // Active transition: apply C1-continuous sample-by-sample Hermite S-curve
          for (let i = 0; i < quantumLen; i++) {
            const x = i / quantumLen;
            const sCurve = x * x * (3.0 - 2.0 * x);
            const g = gStart + deltaG * sCurve;
            leftOut[i] *= g;
            if (rightOut) {
              rightOut[i] *= g;
            }
          }
          this.expanderGain = targetGain;
        }
      }
    } else {
      // Buffer underrun: worker was delayed or audio paused
      leftOut.fill(0);
      if (rightOut) rightOut.fill(0);
      this.consecutiveUnderruns++;

      // If underrun persists for >15 quanta (~40 ms), re-enter buffering mode
      if (this.consecutiveUnderruns > 15) {
        this.isBuffering = true;
      }
    }

    return true;
  }
}

registerProcessor('untube-audio-processor', UnTubeAudioProcessor);
