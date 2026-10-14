/**
 * logger.js - Logging utility.
 * In production (DEBUG = false), info() is a no-op to eliminate console overhead.
 */
'use strict';


const UnTubeLogger = {
  DEBUG: false,

  info(tag, ...args) {
    if (!this.DEBUG) return;
    console.log(`[UnTube] [${tag}]`, ...args);
  },

  warn(tag, ...args) {
    console.warn(`[UnTube] [${tag}]`, ...args);
  },

  error(tag, ...args) {
    console.error(`[UnTube] [${tag}]`, ...args);
  }
};

if (typeof self !== 'undefined') {
  self.UnTubeLogger = UnTubeLogger;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = UnTubeLogger;
}
