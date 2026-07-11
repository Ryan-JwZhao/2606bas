const elements = {
  stateDot: document.getElementById('stateDot'),
  stateText: document.getElementById('stateText'),
  videoStage: document.getElementById('videoStage'),
  stream: document.getElementById('stream'),
  fullscreen: document.getElementById('fullscreenButton'),
  eightBallMode: document.getElementById('eightBallMode'),
  pointsMode: document.getElementById('pointsMode'),
  switchColor: document.getElementById('switchColorButton'),
  starFormula: document.getElementById('starFormulaButton'),
  ruleMode: document.getElementById('ruleModeButton'),
  freeMode: document.getElementById('freeModeButton'),
  nextFree: document.getElementById('nextFreeButton'),
  nextBlack: document.getElementById('nextBlackButton'),
  nextShotChoices: document.getElementById('nextShotChoices'),
  shotOverrideBanner: document.getElementById('shotOverrideBanner'),
  shotOverrideTitle: document.getElementById('shotOverrideTitle'),
  shotOverrideDetail: document.getElementById('shotOverrideDetail'),
  cancelShotOverride: document.getElementById('cancelShotOverrideButton'),
  replay: document.getElementById('replayButton'),
  toast: document.getElementById('toast'),
};

let lastState = null;
let selectedGameMode = 'eight-ball';
let toastTimer = 0;
let fallbackFullscreen = false;

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
  if (!state) return;
  setConnectionState(state.pipeline_ready ? 'running' : 'waiting');

  const baseShotMode = state.base_shot_mode?.code || state.shot_mode?.base_code || state.shot_mode?.code || 'rule';
  setPressed(elements.ruleMode, baseShotMode === 'rule');
  setPressed(elements.freeMode, baseShotMode === 'free');
  setPressed(elements.starFormula, Boolean(state.star_formula_enabled));

  const nextFreeActive = Boolean(state.shot_overrides?.free_shot_once?.active);
  const nextBlackActive = Boolean(state.shot_overrides?.black_target_once?.active);
  const hasShotOverride = nextFreeActive || nextBlackActive;
  elements.nextShotChoices.hidden = hasShotOverride;
  elements.shotOverrideBanner.hidden = !hasShotOverride;
  if (nextFreeActive) {
    elements.shotOverrideTitle.textContent = '下一杆自由';
    elements.shotOverrideDetail.textContent = '临时覆盖长期规则';
  } else if (nextBlackActive) {
    elements.shotOverrideTitle.textContent = '下一杆规则';
    elements.shotOverrideDetail.textContent = '临时指定目标：黑球';
  }

  const turnGroup = state.match?.turn_group;
  const groupNames = { solid: '纯色球', stripe: '花色球' };
  elements.switchColor.textContent = turnGroup && groupNames[turnGroup]
    ? `切换花色 · ${groupNames[turnGroup]}`
    : '切换花色';

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
    lastState = await response.json();
    renderState(lastState);
  } catch (_error) {
    setConnectionState('offline');
  }
}

function selectGameMode(mode) {
  selectedGameMode = mode;
  renderGameMode();
  if (mode === 'points') {
    showToast('追分模式控制功能正在开发中');
  }
}

async function enterVideoFullscreen() {
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
  fallbackFullscreen = true;
  document.body.classList.add('has-fallback-fullscreen');
  elements.videoStage.classList.add('is-fallback-fullscreen');
  showToast('已进入横屏画面，点击画面退出');
  lockLandscapeBestEffort();
}

function exitFallbackFullscreen() {
  fallbackFullscreen = false;
  document.body.classList.remove('has-fallback-fullscreen');
  elements.videoStage.classList.remove('is-fallback-fullscreen');
  unlockOrientationBestEffort();
}

function unlockOrientationAfterFullscreen() {
  if (!document.fullscreenElement && !document.webkitFullscreenElement && screen.orientation?.unlock) {
    unlockOrientationBestEffort();
  }
}

elements.stream.addEventListener('click', async (event) => {
  if (fallbackFullscreen) {
    exitFallbackFullscreen();
    return;
  }
  const frame = lastState?.frame_size;
  if (!frame?.w || !frame?.h) {
    showToast('等待视频画面后再选球');
    return;
  }

  const rect = elements.stream.getBoundingClientRect();
  const scaleX = (event.clientX - rect.left) / rect.width;
  const scaleY = (event.clientY - rect.top) / rect.height;
  const x = Math.max(0, Math.min(frame.w - 1, Math.round(scaleX * frame.w)));
  const y = Math.max(0, Math.min(frame.h - 1, Math.round(scaleY * frame.h)));
  await runAction('/api/select', { x, y });
});

elements.fullscreen.addEventListener('click', (event) => {
  event.stopPropagation();
  enterVideoFullscreen();
});
elements.eightBallMode.addEventListener('click', () => selectGameMode('eight-ball'));
elements.pointsMode.addEventListener('click', () => selectGameMode('points'));
elements.switchColor.addEventListener('click', () => runAction('/api/match/switch_turn'));
elements.starFormula.addEventListener('click', () => runAction('/api/star_formula/toggle'));
elements.ruleMode.addEventListener('click', () => runAction('/api/shot_mode/set', { mode: 'rule' }));
elements.freeMode.addEventListener('click', () => runAction('/api/shot_mode/set', { mode: 'free' }));
elements.nextFree.addEventListener('click', () => runAction('/api/shot_once/free/toggle'));
elements.nextBlack.addEventListener('click', () => runAction('/api/shot_once/black/toggle'));
elements.cancelShotOverride.addEventListener('click', () => {
  const freeActive = Boolean(lastState?.shot_overrides?.free_shot_once?.active);
  const blackActive = Boolean(lastState?.shot_overrides?.black_target_once?.active);
  if (freeActive) {
    runAction('/api/shot_once/free/clear');
  } else if (blackActive) {
    runAction('/api/shot_once/black/clear');
  }
});
elements.replay.addEventListener('click', () => runAction('/api/instant_replay/export'));

document.addEventListener('fullscreenchange', unlockOrientationAfterFullscreen);
document.addEventListener('webkitfullscreenchange', unlockOrientationAfterFullscreen);

renderGameMode();
setConnectionState('offline');
fetchState();
window.setInterval(fetchState, 1000);
