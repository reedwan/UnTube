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

const QUEUE_CAPACITY = 16384;
const QUEUE_MASK = QUEUE_CAPACITY - 1;

// Silence short-circuit: stop waking the Worker thread after 15 silent hops (~100 ms)
const SILENCE_THRESHOLD = 1e-5;
const SILENCE_DECAY_HOPS = 15;

class UnTubeAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.workerPort = null;
    this.isBuffering = true;
    this.consecutiveUnderruns = 0;

    // FastEnhancer-M standard frame hop: 320 mono samples (6.67 ms at 48 kHz)
    this.hopSize = 320;

    // Pre-buffer threshold: 6 hops (~40 ms at 48 kHz) provides a robust safety margin
    // against thread scheduling jitter while maintaining low perceived latency.
    this.preBufferSamples = this.hopSize * 6; // 1920 mono floats

    // Input staging accumulator (downmixed mono samples)
    this.inputAccumulator = new Float32Array(this.hopSize);
    this.accumCount = 0;
    this.currentHopSilent = true;
    this.silentHops = SILENCE_DECAY_HOPS + 1;
    this.isSleepingSilence = true;

    // Pool of recycled ArrayBuffers returned from InferenceWorker for zero-allocation transfers
    this.bufferPool = [];

    // Output circular ring buffer (16,384 mono samples = ~341.3 ms at 48 kHz)
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
    if (!newHopSize || newHopSize <= 0 || newHopSize === this.hopSize) return;
    this.hopSize = newHopSize;
    this.preBufferSamples = this.hopSize * 6;
    if (this.inputAccumulator.length !== this.hopSize) {
      this.inputAccumulator = new Float32Array(this.hopSize);
    }
    this.bufferPool.length = 0;
    this._reset();
  }

  /**
   * Reset ring buffer pointers and expander envelope state.
   */
  _reset() {
    this.isBuffering = true;
    this.consecutiveUnderruns = 0;
    this.accumCount = 0;
    this.currentHopSilent = true;
    this.silentHops = SILENCE_DECAY_HOPS + 1;
    this.isSleepingSilence = true;
    this.ringWrite = 0;
    this.ringRead = 0;
    this.ringAvailable = 0;
    this.expanderEnv = 0.0;
    this.expanderGain = 1.0;
    this.inputAccumulator.fill(0);
    this.outputRing.fill(0);
  }

  /**
   * Handle incoming neural-filtered mono audio from InferenceWorker via dedicated port.
   * @param {Object} data - Contains transferable Float32Array buffer.
   */
  _onFilteredFrame(data) {
    if (!data || !data.buffer) return;
    const incoming = new Float32Array(data.buffer);
    const count = incoming.length;

    // Fast-path: Enqueue incoming mono samples using native SIMD memcpy (TypedArray.prototype.set)
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

    // Recycle transferred ArrayBuffer for subsequent outbound hops (zero-GC ping-pong pool)
    if (count === this.hopSize && this.bufferPool.length < 8) {
      this.bufferPool.push(data.buffer);
    }

    // Latency bounding: prevent unbounded drift if the worker experienced an OS hitch
    // Drop excess older samples if buffer exceeds 75% capacity.
    const maxFill = Math.floor(QUEUE_CAPACITY * 0.75);
    if (this.ringAvailable > maxFill) {
      const drop = this.ringAvailable - this.preBufferSamples;
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
    // 1. Downmix to mono once & stage incoming samples into input accumulator
    // -------------------------------------------------------------------------
    let acc = this.accumCount;
    let hopSilent = this.currentHopSilent;
    for (let i = 0; i < quantumLen; i++) {
      const mono = 0.5 * (leftIn[i] + rightIn[i]);
      this.inputAccumulator[acc] = mono;
      if (hopSilent && Math.abs(mono) > SILENCE_THRESHOLD) {
        hopSilent = false;
      }
      acc++;

      // When a complete neural frame (hopSize) has been staged, dispatch to worker
      if (acc >= this.hopSize) {
        if (hopSilent) {
          this.silentHops++;
        } else {
          const wasSkippingSilentHops = this.silentHops > SILENCE_DECAY_HOPS;
          this.silentHops = 0;
          if (this.isSleepingSilence) {
            // Wake from silence sleep: prime pre-buffer with silence so audio starts without underrun
            this.outputRing.fill(0, 0, this.preBufferSamples);
            this.ringRead = 0;
            this.ringWrite = this.preBufferSamples & QUEUE_MASK;
            this.ringAvailable = this.preBufferSamples;
            this.isBuffering = false;
            this.consecutiveUnderruns = 0;
            this.isSleepingSilence = false;
          } else if (wasSkippingSilentHops && this.ringAvailable < this.preBufferSamples) {
            // Speech resumed while decayed tail was still draining: restore full pre-buffer cushion
            const pad = this.preBufferSamples - this.ringAvailable;
            const firstPad = QUEUE_CAPACITY - this.ringWrite;
            if (firstPad >= pad) {
              this.outputRing.fill(0, this.ringWrite, this.ringWrite + pad);
              this.ringWrite = (this.ringWrite + pad) & QUEUE_MASK;
            } else {
              this.outputRing.fill(0, this.ringWrite, QUEUE_CAPACITY);
              this.outputRing.fill(0, 0, pad - firstPad);
              this.ringWrite = pad - firstPad;
            }
            this.ringAvailable = this.preBufferSamples;
          }
        }

        // Only wake the Worker during active audio or the 15-hop (~100 ms) recurrent tail decay
        if (this.workerPort && this.silentHops <= SILENCE_DECAY_HOPS) {
          // Reuse a pooled ArrayBuffer when available to avoid real-time thread GC
          const recycledBuf = this.bufferPool.pop();
          const transferView =
            recycledBuf && recycledBuf.byteLength === this.hopSize * 4
              ? new Float32Array(recycledBuf)
              : new Float32Array(this.hopSize);
          transferView.set(this.inputAccumulator);
          this.workerPort.postMessage(
            { type: 'PROCESS', buffer: transferView.buffer },
            [transferView.buffer]
          );
        }
        acc = 0;
        hopSilent = true;
      }
    }
    this.accumCount = acc;
    this.currentHopSilent = hopSilent;

    // -------------------------------------------------------------------------
    // 2. Playback: Read neural-filtered mono samples from output ring buffer
    // -------------------------------------------------------------------------
    // Fast-path when sleeping during extended silence: 0 worker wakeups, direct zero output
    if (this.isSleepingSilence || this.isBuffering) {
      leftOut.fill(0);
      if (rightOut) rightOut.fill(0);
      return true;
    }

    if (this.ringAvailable >= quantumLen) {
      // Dequeue mono samples directly into leftOut using native SIMD TypedArray.set
      const firstRead = QUEUE_CAPACITY - this.ringRead;
      if (firstRead >= quantumLen) {
        leftOut.set(this.outputRing.subarray(this.ringRead, this.ringRead + quantumLen));
        this.ringRead = (this.ringRead + quantumLen) & QUEUE_MASK;
      } else {
        leftOut.set(this.outputRing.subarray(this.ringRead, QUEUE_CAPACITY));
        leftOut.set(this.outputRing.subarray(0, quantumLen - firstRead), firstRead);
        this.ringRead = quantumLen - firstRead;
      }
      this.ringAvailable -= quantumLen;
      this.consecutiveUnderruns = 0;

      // 3. Soft Expander (computed on mono leftOut channel before mirroring to rightOut)
      if (this.gateMode !== 'off') {

        const tHigh = this.gateMode === 'aggressive' ? 0.026 : 0.018;
        const tLow  = this.gateMode === 'aggressive' ? 0.009 : 0.005;
        const floor = this.gateMode === 'aggressive' ? 0.008 : 0.025;

        // Calculate Root-Mean-Square (RMS) energy across current mono quantum
        let sumSq = 0;
        for (let i = 0; i < quantumLen; i++) {
          const m = leftOut[i];
          sumSq += m * m;
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
          // Constant noise-floor gain: single-channel linear multiply
          for (let i = 0; i < quantumLen; i++) {
            leftOut[i] *= targetGain;
          }
          this.expanderGain = targetGain;
        } else {
          // Active transition: interpolate smoothly per-sample across the 128-sample quantum
          const step = deltaG / quantumLen;
          for (let i = 0; i < quantumLen; i++) {
            leftOut[i] *= gStart + step * i;
          }
          this.expanderGain = targetGain;
        }
      }

      // Mirror processed mono frame into right channel via native SIMD memcpy
      if (rightOut) {
        rightOut.set(leftOut);
      }
    } else if (this.silentHops > SILENCE_DECAY_HOPS) {
      // Decayed tail has finished playing out: enter silence sleep (0 worker messages/sec)
      leftOut.fill(0);
      if (rightOut) rightOut.fill(0);
      this.ringRead = 0;
      this.ringWrite = 0;
      this.ringAvailable = 0;
      this.expanderEnv = 0.0;
      this.expanderGain = this.gateMode === 'aggressive' ? 0.008 : 0.025;
      this.isSleepingSilence = true;
    } else {
      // Buffer underrun: worker was delayed
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
