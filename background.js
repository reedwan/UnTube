/**
 * background.js - Service worker managing tab capture and extension action states.
 *
 * Responsibilities:
 * 1. Handles toolbar action toggle per active tab.
 * 2. Handles Command+Shift+U / Alt+Shift+U keyboard shortcut.
 * 3. Manages lifecycle of the offscreen WebAudio document.
 * 4. Acquires tab capture stream tokens via chrome.tabCapture.getMediaStreamId.
 * 5. Manages action icon state (Grey / Amber / Red).
 * 6. Handles tab lifecycle cleanups (close, navigation, activation).
 */

'use strict';

importScripts('logger.js');

// Set of currently captured tab IDs (e.g. Set of active tab numbers)
let capturedTabs = new Set();

// Set of restricted/system tab IDs activated in standby mode
let standbyTabs = new Set();

// Set of tab IDs currently negotiating capture initiation
const startingTabs = new Set();

// Lock set to debounce rapid double-clicks on the toolbar icon or shortcut
const pendingToggles = new Set();

// Active timeout handles for temporary 'ERR' badges per tab
const errBadgeTimers = new Map();

// Singleton promise for offscreen document creation
let creatingOffscreenPromise = null;

// Promise for initializing / restoring service worker state from offscreen host
let swReadyPromise = null;

// Fixed configuration: FastEnhancer-M model with Smooth De-Bleed Soft Expander
const ACTIVE_MODEL_KEY = 'fastenhancer_m';
const ACTIVE_GATE_MODE = 'normal'; // 'normal' = Smooth mode (14ms attack, 90ms release)

/**
 * Pings the offscreen document to ensure its audio pipeline message listener is active.
 * Retries up to maxRetries times with delayMs intervals.
 */
async function pingOffscreen(maxRetries = 10, delayMs = 100) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await chrome.runtime.sendMessage({ type: 'DF_PING' });
      if (res && res.pong) return true;
    } catch (_) {}
    await new Promise(r => setTimeout(r, delayMs));
  }
  return false;
}

/**
 * Filters a list of tab IDs to only those that still exist in the browser.
 */
async function filterExistingTabs(tabIds) {
  const valid = [];
  for (const id of tabIds) {
    try {
      const tab = await chrome.tabs.get(id);
      if (tab) valid.push(id);
    } catch (_) {}
  }
  return valid;
}

/**
 * Restores service worker state from the offscreen audio engine document.
 * Crucial in Manifest V3 where the service worker is terminated during idle audio playback.
 */
function ensureSwState() {
  if (!swReadyPromise) {
    swReadyPromise = (async () => {
      const stored = await chrome.storage.local.get(['standbyTabIds']).catch(() => ({}));
      const rawStandby = Array.isArray(stored?.standbyTabIds) ? stored.standbyTabIds : [];
      const validStandby = await filterExistingTabs(rawStandby);
      standbyTabs = new Set(validStandby);

      const hasDoc = await chrome.offscreen.hasDocument();
      if (!hasDoc) {
        capturedTabs = new Set(standbyTabs);
        startingTabs.clear();
        for (const id of capturedTabs) {
          updateTabBadge(id, 'ON');
        }
        syncStorage();
        return;
      }
      try {
        const state = await chrome.runtime.sendMessage({ type: 'DF_QUERY_STATE' });
        const activeEngineTabs = state && Array.isArray(state.activeTabs) ? state.activeTabs : [];
        capturedTabs = new Set([...activeEngineTabs, ...standbyTabs]);
        for (const id of capturedTabs) {
          updateTabBadge(id, 'ON');
        }
        syncStorage();
      } catch (_) {
        capturedTabs = new Set(standbyTabs);
        startingTabs.clear();
        syncStorage();
      }
    })();
  }
  return swReadyPromise;
}

/**
 * Ensures that the offscreen document hosting the WebAudio & ONNX Runtime WASM
 * neural pipeline is created and responsive. Manifest V3 Service Workers cannot
 * access the DOM or AudioContext directly, so an offscreen document is required.
 */
async function ensureOffscreen() {
  const hasDoc = await chrome.offscreen.hasDocument();
  if (!hasDoc) {
    if (!creatingOffscreenPromise) {
      creatingOffscreenPromise = (async () => {
        UnTubeLogger.info('SW', 'Creating offscreen audio engine document...');
        try {
          await chrome.offscreen.createDocument({
            url: 'offscreen.html',
            reasons: ['AUDIO_PLAYBACK', 'USER_MEDIA'],
            justification: 'Real-time neural speech isolation across browser tabs.'
          });
        } catch (docErr) {
          if (!docErr?.message?.includes('already exists')) {
            throw docErr;
          }
        }
        UnTubeLogger.info('SW', 'Offscreen document created, awaiting handshake...');
        const ready = await pingOffscreen();
        if (!ready) {
          UnTubeLogger.warn('SW', 'Offscreen document handshake timed out.');
        } else {
          UnTubeLogger.info('SW', 'Offscreen document ready.');
        }
      })().finally(() => {
        creatingOffscreenPromise = null;
      });
    }
    await creatingOffscreenPromise;
  }
}

const ICONS_ACTIVE = {
  16: 'icons/icon16.png',
  32: 'icons/icon32.png',
  48: 'icons/icon48.png',
  128: 'icons/icon128.png'
};

const ICONS_STARTING = {
  16: 'icons/icon16_starting.png',
  32: 'icons/icon32_starting.png',
  48: 'icons/icon48_starting.png',
  128: 'icons/icon128_starting.png'
};

const ICONS_INACTIVE = {
  16: 'icons/icon16_off.png',
  32: 'icons/icon32_off.png',
  48: 'icons/icon48_off.png',
  128: 'icons/icon128_off.png'
};

/**
 * Updates extension action icon and tooltip title for a tab.
 * - 'ON': Red icon, tooltip 'UnTube (Active)'.
 * - 'STARTING': Amber icon, tooltip 'UnTube (Starting...)'.
 * - 'ERR': Grey icon, temporary 'ERR' badge, tooltip 'UnTube (Cannot capture page)'.
 * - 'OFF': Grey icon, tooltip 'UnTube'.
 */
function updateTabBadge(tabId, state) {
  if (errBadgeTimers.has(tabId)) {
    clearTimeout(errBadgeTimers.get(tabId));
    errBadgeTimers.delete(tabId);
  }

  if (state === 'ON') {
    chrome.action.setIcon({ tabId, path: ICONS_ACTIVE }).catch(() => {});
    chrome.action.setBadgeText({ tabId, text: '' }).catch(() => {});
    chrome.action.setTitle({ tabId, title: 'UnTube (Active)' }).catch(() => {});
  } else if (state === 'STARTING') {
    chrome.action.setIcon({ tabId, path: ICONS_STARTING }).catch(() => {});
    chrome.action.setBadgeText({ tabId, text: '' }).catch(() => {});
    chrome.action.setTitle({ tabId, title: 'UnTube (Starting...)' }).catch(() => {});
  } else if (state === 'ERR') {
    chrome.action.setIcon({ tabId, path: ICONS_INACTIVE }).catch(() => {});
    chrome.action.setBadgeText({ tabId, text: 'ERR' }).catch(() => {});
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#ff5e97' }).catch(() => {});
    chrome.action.setTitle({ tabId, title: 'UnTube (Cannot capture page)' }).catch(() => {});
    const timer = setTimeout(() => {
      errBadgeTimers.delete(tabId);
      if (!capturedTabs.has(tabId) && !startingTabs.has(tabId)) {
        chrome.action.setBadgeText({ tabId, text: '' }).catch(() => {});
        chrome.action.setTitle({ tabId, title: 'UnTube' }).catch(() => {});
      }
    }, 2500);
    errBadgeTimers.set(tabId, timer);
  } else {
    chrome.action.setIcon({ tabId, path: ICONS_INACTIVE }).catch(() => {});
    chrome.action.setBadgeText({ tabId, text: '' }).catch(() => {});
    chrome.action.setTitle({ tabId, title: 'UnTube' }).catch(() => {});
  }
}


/**
 * Persists the active capture state to chrome.storage.local for cross-component sync.
 */
function syncStorage() {
  chrome.storage.local.set({
    capturedTabIds: Array.from(capturedTabs),
    standbyTabIds: Array.from(standbyTabs),
    isFiltering: capturedTabs.size > 0,
    selectedModel: ACTIVE_MODEL_KEY,
    gateMode: ACTIVE_GATE_MODE
  });
}

/**
 * Validates whether a tab URL is allowed to be captured.
 * Chrome prohibits tabCapture on chrome://, chrome-extension://, about:, and webstore URLs.
 */
function isRestrictedUrl(url) {
  if (!url) return false;
  return (
    url.startsWith('chrome://') ||
    url.startsWith('chrome-extension://') ||
    url.startsWith('edge://') ||
    url.startsWith('about:') ||
    url.includes('chromewebstore.google.com') ||
    url.includes('chrome.google.com/webstore')
  );
}

/**
 * Initiates audio stream capture for a tab and connects it to the neural engine.
 * Must be triggered synchronously inside a user gesture (action click or command).
 */
async function startTabCapture(tab) {
  const tabId = tab.id;
  if (!tabId) return;

  const url = tab.url || '';
  if (isRestrictedUrl(url)) {
    UnTubeLogger.info('SW', `Restricted system or webstore URL (${url}); activating in standby mode.`);
    standbyTabs.add(tabId);
    capturedTabs.add(tabId);
    updateTabBadge(tabId, 'ON');
    syncStorage();
    return;
  }

  UnTubeLogger.info('SW', `Starting capture for tab ${tabId}: "${tab.title}"`);
  startingTabs.add(tabId);
  updateTabBadge(tabId, 'STARTING');

  try {
    // 1. Acquire tab stream token synchronously inside user gesture
    const streamId = await new Promise((resolve, reject) => {
      chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (id) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else if (id) {
          resolve(id);
        } else {
          reject(new Error('Failed to acquire stream ID'));
        }
      });
    });

    // 2. Ensure offscreen host is running
    await ensureOffscreen();

    // 3. Inject fullscreen helper (handles OS-level fullscreen during tab capture)
    chrome.scripting.executeScript({
      target: { tabId },
      files: ['content.js']
    }).catch(() => {});

    // 4. Dispatch start message to offscreen audio engine
    chrome.runtime.sendMessage({
      type: 'DF_START_TAB',
      streamId,
      tabId,
      modelKey: ACTIVE_MODEL_KEY,
      gateMode: ACTIVE_GATE_MODE
    }).catch((err) => {
      startingTabs.delete(tabId);
      UnTubeLogger.error('SW', `DF_START_TAB dispatch failed for tab ${tabId}:`, err);
      updateTabBadge(tabId, 'ERR');
    });
  } catch (err) {
    startingTabs.delete(tabId);
    const msg = err?.message || '';
    if (msg.includes('protected') || msg.includes('restricted') || msg.includes('chrome://') || msg.includes('webstore') || msg.includes('file')) {
      UnTubeLogger.info('SW', `Protected page or ungranted file capture restriction caught (${msg}); activating in standby mode.`);
      standbyTabs.add(tabId);
      capturedTabs.add(tabId);
      updateTabBadge(tabId, 'ON');
      syncStorage();
      return;
    }
    UnTubeLogger.error('SW', `Error capturing tab ${tabId}:`, err);
    updateTabBadge(tabId, 'ERR');
  }
}

/**
 * Stops audio capture for a tab and tears down its audio stream in the engine.
 */
async function stopTabCapture(tabId) {
  UnTubeLogger.info('SW', `Stopping capture for tab ${tabId}`);
  startingTabs.delete(tabId);
  standbyTabs.delete(tabId);
  capturedTabs.delete(tabId);
  updateTabBadge(tabId, 'OFF');
  syncStorage();

  chrome.runtime.sendMessage({
    type: 'DF_STOP_TAB',
    tabId
  }).catch(() => {});
}

/**
 * Toggles tab capture state:
 * - If active or starting: stops capture and releases stream.
 * - If inactive: initiates tab capture and starts pipeline.
 */
async function toggleTabFilter(tab) {
  if (!tab || !tab.id) return;
  const tabId = tab.id;

  if (pendingToggles.has(tabId)) return;
  pendingToggles.add(tabId);

  try {
    // Restore state from offscreen document if service worker woke from idle
    await ensureSwState();

    if (capturedTabs.has(tabId) || startingTabs.has(tabId)) {
      UnTubeLogger.info('SW', `Toggle OFF for tab ${tabId}`);
      await stopTabCapture(tabId);
    } else {
      UnTubeLogger.info('SW', `Toggle ON for tab ${tabId}`);
      await startTabCapture(tab);
    }
  } finally {
    pendingToggles.delete(tabId);
  }
}

// Toolbar action click listener
chrome.action.onClicked.addListener(async (tab) => {
  await toggleTabFilter(tab);
});

// Keyboard shortcut listener: Command+Shift+U (Mac) / Alt+Shift+U (Win/Linux)
chrome.commands.onCommand.addListener(async (command) => {

  if (command === 'toggle_filter') {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab) {
      await toggleTabFilter(tab);
    }
  }
});

// Auto-cleanup if captured tab is closed by the user
chrome.tabs.onRemoved.addListener((tabId) => {
  if (errBadgeTimers.has(tabId)) {
    clearTimeout(errBadgeTimers.get(tabId));
    errBadgeTimers.delete(tabId);
  }
  if (capturedTabs.has(tabId) || startingTabs.has(tabId)) {
    stopTabCapture(tabId).catch(() => {});
  }
});

// Persist capture across URL navigations within the same tab
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  await ensureSwState().catch(() => {});
  if (!capturedTabs.has(tabId) && !startingTabs.has(tabId)) return;

  // If the tab was only in restricted-page standby mode (no audio stream), reset on navigation
  if (standbyTabs.has(tabId)) {
    if (changeInfo.status === 'loading') {
      stopTabCapture(tabId).catch(() => {});
    }
    return;
  }

  if (changeInfo.status === 'loading') {
    if (capturedTabs.has(tabId)) {
      updateTabBadge(tabId, 'ON');
    } else if (startingTabs.has(tabId)) {
      updateTabBadge(tabId, 'STARTING');
    }
  } else if (changeInfo.status === 'complete' && capturedTabs.has(tabId)) {
    updateTabBadge(tabId, 'ON');
    if (!isRestrictedUrl(tab?.url || '')) {
      chrome.scripting.executeScript({
        target: { tabId },
        files: ['content.js']
      }).catch(() => {});
    }
  }
});

// Sync badge when the user switches browser tabs
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  await ensureSwState().catch(() => {});
  const tabId = activeInfo.tabId;
  if (capturedTabs.has(tabId)) {
    updateTabBadge(tabId, 'ON');
  } else if (startingTabs.has(tabId)) {
    updateTabBadge(tabId, 'STARTING');
  } else {
    updateTabBadge(tabId, 'OFF');
  }
});

// Messages from offscreen audio host or content scripts
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // Pipeline status reports from offscreen document
  if (msg.type === 'DF_TAB_STATUS') {
    UnTubeLogger.info('SW', `Tab ${msg.tabId} engine status: ${msg.state}`, msg.error || '');
    startingTabs.delete(msg.tabId);
    if (msg.state === 'active') {
      standbyTabs.delete(msg.tabId);
      capturedTabs.add(msg.tabId);
      updateTabBadge(msg.tabId, 'ON');
      syncStorage();
    } else if (msg.state === 'stopped') {
      standbyTabs.delete(msg.tabId);
      capturedTabs.delete(msg.tabId);
      updateTabBadge(msg.tabId, 'OFF');
      syncStorage();
    } else if (msg.state === 'error') {
      standbyTabs.delete(msg.tabId);
      capturedTabs.delete(msg.tabId);
      updateTabBadge(msg.tabId, 'ERR');
      syncStorage();
    }
    return false;
  }

  // Engine shutdown notification
  if (msg.type === 'DF_STATUS' && msg.state === 'stopped') {
    for (const id of capturedTabs) {
      if (!standbyTabs.has(id)) {
        updateTabBadge(id, 'OFF');
      }
    }
    capturedTabs = new Set(standbyTabs);
    startingTabs.clear();
    syncStorage();
    if (chrome.offscreen?.closeDocument) {
      chrome.offscreen.closeDocument().catch(() => {});
    }
    return false;
  }


  // Fullscreen support bridge from content.js (only active while tab is captured)
  if (msg.type === 'ENTER_FULLSCREEN' && sender.tab?.windowId && sender.tab?.id && capturedTabs.has(sender.tab.id)) {
    const winId = sender.tab.windowId;
    chrome.windows.get(winId, (win) => {
      if (!win || win.state === 'fullscreen') return;
      chrome.storage.local.set({ [`winState_${winId}`]: win.state });
      chrome.windows.update(winId, { state: 'fullscreen' }).catch(() => {});
    });
    return false;
  }
  if (msg.type === 'EXIT_FULLSCREEN' && sender.tab?.windowId && sender.tab?.id && capturedTabs.has(sender.tab.id)) {
    const winId = sender.tab.windowId;
    chrome.storage.local.get([`winState_${winId}`], (res) => {
      const prevState = (res && res[`winState_${winId}`]) || 'normal';
      chrome.storage.local.remove([`winState_${winId}`]).catch(() => {});
      chrome.windows.update(winId, { state: prevState }).catch(() => {});
    });
    return false;
  }
});

// Verify active captured tabs upon service worker wake
ensureSwState().catch(() => {});

UnTubeLogger.info('SW', 'UnTube 1-click action service worker loaded.');
