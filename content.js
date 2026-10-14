/**
 * content.js - Minimal fullscreen bridge for UnTube
 * Handles Chrome's tabCapture OS-level fullscreen restriction.
 */
'use strict';

if (!window.__untubeFullscreenBridgeLoaded) {
  window.__untubeFullscreenBridgeLoaded = true;
  let lastFullscreenState = !!(document.fullscreenElement || document.webkitFullscreenElement);

  function handleFullscreenChange() {
    if (!chrome.runtime?.id) return;
    try {
      const isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement);
      if (isFullscreen === lastFullscreenState) return;
      lastFullscreenState = isFullscreen;
      chrome.runtime.sendMessage({ type: isFullscreen ? 'ENTER_FULLSCREEN' : 'EXIT_FULLSCREEN' }).catch(() => {});
    } catch (_) {}
  }

  document.addEventListener('fullscreenchange', handleFullscreenChange);
  document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
}
