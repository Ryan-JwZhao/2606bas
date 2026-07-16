const elements = {
  stateDot: document.getElementById('stateDot'),
  stateText: document.getElementById('stateText'),
  videoStage: document.getElementById('videoStage'),
  stream: document.getElementById('stream'),
  fullscreen: document.getElementById('fullscreenButton'),
  install: document.getElementById('installButton'),
  eightBallMode: document.getElementById('eightBallMode'),
  pointsMode: document.getElementById('pointsMode'),
  switchColor: document.getElementById('switchColorButton'),
  starFormula: document.getElementById('starFormulaButton'),
  ruleMode: document.getElementById('ruleModeButton'),
  hookMode: document.getElementById('hookModeButton'),
  nextHook: document.getElementById('nextHookButton'),
  nextBlack: document.getElementById('nextBlackButton'),
  nextShotChoices: document.getElementById('nextShotChoices'),
  shotOverrideBanner: document.getElementById('shotOverrideBanner'),
  shotOverrideTitle: document.getElementById('shotOverrideTitle'),
  shotOverrideTarget: document.getElementById('shotOverrideTarget'),
  cancelShotOverride: document.getElementById('cancelShotOverrideButton'),
  replay: document.getElementById('replayButton'),
  toast: document.getElementById('toast'),
};

let lastState = null;
let selectedGameMode = 'eight-ball';
let toastTimer = 0;
let fallbackFullscreen = false;
let pwaAutoFullscreen = false;
let pwaOrientationTimer = 0;
let installPrompt = null;

function showToast(message) {
  if (!message) return;
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add('is-visible');
  toastTimer = window.setTimeout(() => elements.toast.classList.remove('is-visible'), 2300);
}

function setConnectionState(kind) {
  const states = {
    running: ['state-dot--running', '系统运行中'],
    waiting: ['state-dot--waiting', '等待初始化'],
    offline: ['state-dot--offline', '未连接'],
  };
  const [dotClass, label] = states[kind] || states.offline;
  elements.stateDot.className = `state-dot ${dotClass}`;
  elements.stateText.textContent = label;
}

function setPressed(button, active) {
  const value = Boolean(active);
  button.classList.toggle('is-active', value);
  button.setAttribute('aria-pressed', String(value));
}

function renderGameMode() {
  setPressed(elements.eightBallMode, selectedGameMode === 'eight-ball');
  setPressed(elements.pointsMode, selectedGameMode === 'points');
}

function renderState(state) {
  if (!state || state.ok === false) return;
  setConnectionState(state.pipeline_ready ? 'running' : 'waiting');

  const baseShotMode = state.base_shot_mode?.code || state.shot_mode?.base_code || state.shot_mode?.code || 'rule';
  setPressed(elements.ruleMode, baseShotMode === 'rule');
  setPressed(elements.hookMode, baseShotMode === 'hook');
  setPressed(elements.starFormula, Boolean(state.star_formula_enabled));

  const nextHookActive = Boolean(state.shot_overrides?.hook_shot_once?.active);
  const nextBlackActive = Boolean(state.shot_overrides?.black_target_once?.active);
  const hasShotOverride = nextHookActive || nextBlackActive;
  elements.nextShotChoices.hidden = hasShotOverride;
  elements.shotOverrideBanner.hidden = !hasShotOverride;
  if (nextHookActive) {
    elements.shotOverrideTitle.textContent = '下一杆勾球';
    elements.shotOverrideTarget.hidden = true;
  } else if (nextBlackActive) {
    elements.shotOverrideTitle.textContent = '下一杆规则';
    elements.shotOverrideTarget.textContent = '黑球';
    elements.shotOverrideTarget.hidden = false;
  }

  elements.switchColor.textContent = '切换花色';

  if ('instant_replay_enabled' in state) {
    elements.replay.disabled = !state.instant_replay_enabled;
    elements.replay.title = state.instant_replay_enabled ? '' : '请先在桌面端设置中开启精彩瞬间缓存';
  }
}

async function postJson(path, payload = {}) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function runAction(path, payload = {}) {
  try {
    const result = await postJson(path, payload);
    if (result.state) {
      lastState = result.state;
      renderState(lastState);
    }
    showToast(result.message || (result.ok ? '操作完成' : '操作失败'));
    return result;
  } catch (error) {
    setConnectionState('offline');
    showToast(`连接失败：${error.message || error}`);
    return null;
  }
}

async function fetchState() {
  try {
    const response = await fetch('/api/state', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!data || data.ok === false) {
      throw new Error(data?.message || '状态数据无效');
    }
    lastState = data;
    renderState(lastState);
  } catch (_error) {
    if (!lastState) setConnectionState('offline');
  }
}

function selectGameMode(mode) {
  selectedGameMode = mode;
  renderGameMode();
}

function isInstalledPwa() {
  const displayMode = ['standalone', 'fullscreen', 'minimal-ui', 'window-controls-overlay']
    .some((mode) => Boolean(window.matchMedia?.(`(display-mode: ${mode})`)?.matches));
  return displayMode || window.navigator.standalone === true || document.referrer.startsWith('android-app://');
}

function isLandscapeOrientation() {
  const mediaQuery = window.matchMedia?.('(orientation: landscape)');
  if (mediaQuery) return mediaQuery.matches;
  const type = screen.orientation?.type || '';
  if (type) return type.startsWith('landscape');
  return window.innerWidth > window.innerHeight;
}

function setPwaAutoFullscreen(active) {
  const next = Boolean(active) && isInstalledPwa();
  if (next === pwaAutoFullscreen) return;
  pwaAutoFullscreen = next;
  document.body.classList.toggle('has-pwa-auto-fullscreen', next);
  elements.videoStage.classList.toggle('is-pwa-auto-fullscreen', next);
}

function syncPwaAutoFullscreen() {
  if (!isInstalledPwa()) {
    setPwaAutoFullscreen(false);
    return;
  }
  if (document.fullscreenElement || document.webkitFullscreenElement || fallbackFullscreen) {
    setPwaAutoFullscreen(false);
    return;
  }
  setPwaAutoFullscreen(isLandscapeOrientation());
}

function schedulePwaAutoFullscreenSync() {
  window.clearTimeout(pwaOrientationTimer);
  pwaOrientationTimer = window.setTimeout(syncPwaAutoFullscreen, 140);
}

function setInstallButtonVisible(visible) {
  if (!elements.install) return;
  elements.install.hidden = !visible || isInstalledPwa();
}

function handleBeforeInstallPrompt(event) {
  event.preventDefault();
  installPrompt = event;
  setInstallButtonVisible(true);
}

async function promptPwaInstall() {
  if (!installPrompt) {
    showToast('请从浏览器菜单选择“安装应用”');
    return;
  }
  const prompt = installPrompt;
  installPrompt = null;
  setInstallButtonVisible(false);
  try {
    const result = await prompt.prompt();
    showToast(result?.outcome === 'accepted' ? '已确认安装应用' : '已取消安装应用');
  } catch (_error) {
    showToast('当前浏览器未能打开安装提示');
  }
}

function handleAppInstalled() {
  installPrompt = null;
  setInstallButtonVisible(false);
  showToast('应用已安装');
}

function registerPwaServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  navigator.serviceWorker.register('/service-worker.js', { scope: '/' }).catch(() => {
    // Normal HTTP LAN pages may not be secure contexts; PWA mode remains opt-in.
  });
}

async function enterVideoFullscreen() {
  setPwaAutoFullscreen(false);
  try {
    if (elements.videoStage.requestFullscreen) {
      await elements.videoStage.requestFullscreen({ navigationUI: 'hide' });
    } else if (elements.videoStage.webkitRequestFullscreen) {
      await Promise.resolve(elements.videoStage.webkitRequestFullscreen());
    } else {
      activateFallbackFullscreen();
      return;
    }

    await lockLandscapeBestEffort();
  } catch (_error) {
    activateFallbackFullscreen();
  }
}

async function lockLandscapeBestEffort() {
  if (!screen.orientation?.lock) return;
  try {
    await Promise.resolve(screen.orientation.lock('landscape'));
  } catch (_error) {
    // Some mobile browsers enter fullscreen but do not expose orientation lock.
  }
}

function unlockOrientationBestEffort() {
  if (!screen.orientation?.unlock) return;
  try {
    screen.orientation.unlock();
  } catch (_error) {
    // Orientation may already be unlocked or controlled by the operating system.
  }
}

function activateFallbackFullscreen() {
  setPwaAutoFullscreen(false);
  fallbackFullscreen = true;
  document.body.classList.add('has-fallback-fullscreen');
  elements.videoStage.classList.add('is-fallback-fullscreen');
  elements.fullscreen.setAttribute('aria-label', '退出全屏');
  elements.fullscreen.setAttribute('title', '退出全屏');
  showToast('已进入横屏画面，点击球可选球，点击右下角按钮退出');
  lockLandscapeBestEffort();
}

function exitFallbackFullscreen() {
  fallbackFullscreen = false;
  document.body.classList.remove('has-fallback-fullscreen');
  elements.videoStage.classList.remove('is-fallback-fullscreen');
  elements.fullscreen.setAttribute('aria-label', '横屏全屏播放');
  elements.fullscreen.setAttribute('title', '横屏全屏播放');
  unlockOrientationBestEffort();
  schedulePwaAutoFullscreenSync();
}

function unlockOrientationAfterFullscreen() {
  if (!document.fullscreenElement && !document.webkitFullscreenElement && screen.orientation?.unlock) {
    unlockOrientationBestEffort();
  }
  schedulePwaAutoFullscreenSync();
}

elements.stream.addEventListener('click', async (event) => {
  const frame = lastState?.frame_size;
  if (!frame?.w || !frame?.h) {
    showToast('等待视频画面后再选球');
    return;
  }

  const screenRect = elements.stream.getBoundingClientRect();
  const hasLocalPoint = Number.isFinite(event.offsetX) && Number.isFinite(event.offsetY);
  const rect = hasLocalPoint
    ? { left: 0, top: 0, width: elements.stream.clientWidth, height: elements.stream.clientHeight }
    : screenRect;
  const style = window.getComputedStyle(elements.stream);
  const point = window.BASStreamCoordinates?.mapRenderedPointToFrame({
    clientX: hasLocalPoint ? event.offsetX : event.clientX,
    clientY: hasLocalPoint ? event.offsetY : event.clientY,
    rect,
    frameWidth: frame.w,
    frameHeight: frame.h,
    objectFit: style.objectFit,
  });
  if (!point) {
    showToast('请点击实际视频画面内的目标球');
    return;
  }
  await runAction('/api/select', point);
});

elements.fullscreen.addEventListener('click', (event) => {
  event.stopPropagation();
  if (fallbackFullscreen) {
    exitFallbackFullscreen();
  } else {
    enterVideoFullscreen();
  }
});
elements.eightBallMode.addEventListener('click', () => selectGameMode('eight-ball'));
elements.pointsMode.addEventListener('click', () => selectGameMode('points'));
elements.switchColor.addEventListener('click', () => runAction('/api/match/switch_turn'));
elements.starFormula.addEventListener('click', () => runAction('/api/star_formula/toggle'));
elements.ruleMode.addEventListener('click', () => runAction('/api/shot_mode/set', { mode: 'rule' }));
elements.hookMode.addEventListener('click', () => runAction('/api/shot_mode/set', { mode: 'hook' }));
elements.nextHook.addEventListener('click', () => runAction('/api/shot_once/hook/toggle'));
elements.nextBlack.addEventListener('click', () => runAction('/api/shot_once/black/toggle'));
elements.cancelShotOverride.addEventListener('click', () => {
  const hookActive = Boolean(lastState?.shot_overrides?.hook_shot_once?.active);
  const blackActive = Boolean(lastState?.shot_overrides?.black_target_once?.active);
  if (hookActive) {
    runAction('/api/shot_once/hook/clear');
  } else if (blackActive) {
    runAction('/api/shot_once/black/clear');
  }
});
elements.replay.addEventListener('click', () => runAction('/api/instant_replay/export'));
elements.install?.addEventListener('click', promptPwaInstall);
window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
window.addEventListener('appinstalled', handleAppInstalled);

document.addEventListener('fullscreenchange', unlockOrientationAfterFullscreen);
document.addEventListener('webkitfullscreenchange', unlockOrientationAfterFullscreen);
window.addEventListener('orientationchange', schedulePwaAutoFullscreenSync, { passive: true });
window.addEventListener('resize', schedulePwaAutoFullscreenSync, { passive: true });
screen.orientation?.addEventListener?.('change', schedulePwaAutoFullscreenSync);
const standaloneMediaQuery = window.matchMedia?.('(display-mode: standalone)');
standaloneMediaQuery?.addEventListener?.('change', schedulePwaAutoFullscreenSync);

renderGameMode();
setConnectionState('offline');
fetchState();
registerPwaServiceWorker();
schedulePwaAutoFullscreenSync();
window.setInterval(fetchState, 1000);
