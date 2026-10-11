/**
 * content.js - Minimal fullscreen bridge for UnTube
 * Handles Chrome's tabCapture OS-level fullscreen restriction.
 */
'use strict';

function handleFullscreenChange() {
  if (!chrome.runtime?.id) return;
  try {
    const isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement);
    chrome.runtime.sendMessage({ type: isFullscreen ? 'ENTER_FULLSCREEN' : 'EXIT_FULLSCREEN' }).catch(() => {});
  } catch (_) {}
}

document.addEventListener('fullscreenchange', handleFullscreenChange);
document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
