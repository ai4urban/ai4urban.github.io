import {fetchAsset, releaseTile, runTile} from './pilot.js';
import {clearResults, showResults} from './results.js';

const $ = id => document.getElementById(id);
const scope = $('scope');
const computeMode = $('compute-mode');
const tileSelect = $('tile');
const emissions = $('emissions');
const windSpeed = $('wind-speed');
const rotation = $('rotation');
const background = $('background');
const sliders = [emissions, windSpeed, rotation, background];
const entries = sliders.map(control => $(control.id + '-value'));
const preset = $('preset');
const publicApp = document.documentElement.hasAttribute('data-public-app');
const query = new URLSearchParams(location.search);
const verify = !publicApp && query.get('verify') === '1';
const tileRects = new Map();
let footprint;
let catalog, weatherCatalog, liveWeather, truthCache, worker, workerKey, busy = false;
const presets = {
  reference: [100, 100, 0, 0],
  lower: [80, 100, 0, 0],
  wind: [100, 100, 15, 0],
  boundary: [100, 100, 0, 2],
};

scope.value = (publicApp ? query.get('area') === 'domain' : location.pathname === '/domain') ? 'domain' : 'tile';
const isDomain = () => scope.value === 'domain';
const selectedTile = () => catalog?.tiles.find(tile => tile.id === tileSelect.value);

function status(message, state = 'ready') {
  $('status').className = 'status ' + state;
  $('status-text').textContent = message;
}

function setBusy(value) {
  busy = value;
  computeMode.disabled = value;
  scope.disabled = emissions.disabled = $('reset').disabled = value;
  preset.disabled = value;
  windSpeed.disabled = rotation.disabled = background.disabled = value || !weatherCatalog;
  entries.forEach((entry, i) => { entry.disabled = value || (i > 0 && !liveWeather); });
  tileSelect.disabled = value || !catalog;
  $('run').disabled = value || (!isDomain() && !catalog);
  $('run-label').textContent = value ? 'Calculating…' : isDomain() ? 'Run whole-domain prediction' : 'Run prediction';
  for (const rectangle of tileRects.values()) {
    rectangle.setAttribute('aria-disabled', String(value));
    rectangle.setAttribute('tabindex', value ? '-1' : '0');
  }
}

function updateArea() {
  delete window.pilotResult;
  delete window.pilotFields;
  delete window.domainResult;
  truthCache = undefined;
  clearResults();
  $('tile-picker').hidden = isDomain();
  $('result-title').textContent = isDomain() ? 'Whole-domain prediction' : selectedTile()?.label ?? 'Tile prediction';
  $('area-note').textContent = isDomain() ?
    '800 × 800 cells. The reference inputs take about 286 MB. Changed weather loads extra data.' :
    '256 × 256 cells. Select a tile on the map or from the list.';
  $('pending').textContent = 'Choose a scenario and run a prediction.';
  for (const [id, rectangle] of tileRects) rectangle.setAttribute('aria-pressed', String(id === tileSelect.value));
  const tile = selectedTile();
  if (tile && footprint) {
    for (const [key, value] of Object.entries({
      x: tile.x, y: catalog.domainSize - tile.y - tile.size, width: tile.size, height: tile.size,
    })) footprint.setAttribute(key, value);
  }
  status(isDomain() || catalog ? 'Ready to run' : 'Loading the tile map…');
  setBusy(false);
}

function changeTile() {
  stopDomainWorker();
  releaseTile();
  updateArea();
}

async function loadCatalog() {
  catalog = await (await fetchAsset('assets/tiles.json')).json();
  const map = $('tile-map');
  const extent = catalog.domainSize;
  $('camden-boundary').setAttribute('d', catalog.boundary.map(ring => {
    return 'M' + ring.map(([x, y]) => `${x.toFixed(2)},${(extent - y).toFixed(2)}`).join(' L') + ' Z';
  }).join(' '));
  footprint = document.createElementNS(map.namespaceURI, 'rect');
  footprint.setAttribute('class', 'tile-footprint');
  map.append(footprint);
  catalog.tiles.forEach(tile => {
    tileSelect.add(new Option(tile.label, tile.id));
    const rectangle = document.createElementNS(map.namespaceURI, 'g');
    const attributes = {transform: `translate(${tile.x + tile.size / 2} ${extent - tile.y - tile.size / 2})`,
      role: 'button', tabindex: 0, 'aria-label': tile.label};
    for (const [key, value] of Object.entries(attributes)) rectangle.setAttribute(key, value);
    const select = () => {
      if (tileSelect.disabled || tileSelect.value === tile.id) return;
      tileSelect.value = tile.id;
      changeTile();
    };
    rectangle.addEventListener('click', select);
    rectangle.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select();
      }
    });
    map.append(rectangle);
    tileRects.set(tile.id, rectangle);
    const marker = document.createElementNS(map.namespaceURI, 'circle');
    marker.setAttribute('r', '30');
    const label = document.createElementNS(map.namespaceURI, 'text');
    label.textContent = tile.id.slice(5).toUpperCase();
    rectangle.append(marker, label);
  });
  if (!busy) updateArea();
}

function stopDomainWorker() {
  worker?.terminate();
  worker = undefined;
  workerKey = undefined;
}

function runInBlocks(scenario, provider, tile) {
  return new Promise((resolve, reject) => {
    const key = `${provider}/${tile?.id ?? 'domain'}`;
    if (workerKey !== key) stopDomainWorker();
    worker ??= new Worker('domain-worker.js');
    workerKey = key;
    worker.onerror = event => reject(new Error(event.message || 'The calculation worker stopped.'));
    worker.onmessage = ({data}) => {
      if (data.status) status(data.status, 'busy');
      else if (data.error) reject(new Error(data.error));
      else resolve(data.result);
    };
    worker.postMessage({scenario, catalog: weatherCatalog, compute: provider, tile, verify: verify && !tile});
  });
}

const computeLabel = provider => provider === 'webgpu' ? 'GPU (WebGPU)' : 'CPU';

async function selectProvider(mode) {
  if (mode === 'cpu') return 'wasm';
  let adapter;
  try {
    adapter = await navigator.gpu?.requestAdapter();
  } catch {
    // The CPU remains available when the browser cannot access a GPU.
  }
  if (adapter && !adapter.info?.isFallbackAdapter) return 'webgpu';
  if (mode === 'gpu') throw new Error('Hardware WebGPU is unavailable. Select CPU or Automatic.');
  return 'wasm';
}

async function runWithDevice(whole, tile, scenario) {
  const mode = computeMode.value;
  let provider = await selectProvider(mode);
  const launch = async () => {
    if (whole) return runInBlocks(scenario, provider);
    if (provider === 'webgpu') {
      await releaseTile(true);
      const {fields, ...summary} = await runInBlocks(scenario, provider, tile);
      return {summary, fields: {baseline: fields.baseline, field: fields.prediction, mask: fields.mask}};
    }
    stopDomainWorker();
    return runTile(tile, scenario, weatherCatalog, message => status(message, 'busy'), provider);
  };
  status('Starting on ' + computeLabel(provider) + '…', 'busy');
  try {
    return await launch();
  } catch (error) {
    if (mode !== 'auto' || provider !== 'webgpu') throw error;
    stopDomainWorker();
    await releaseTile(true).catch(() => {});
    provider = 'wasm';
    status('WebGPU could not finish. Retrying on the CPU…', 'busy');
    return launch();
  }
}

scope.addEventListener('change', () => {
  stopDomainWorker();
  releaseTile();
  if (publicApp) {
    const url = new URL(location.href);
    if (isDomain()) url.searchParams.set('area', 'domain');
    else url.searchParams.delete('area');
    history.replaceState(null, '', url);
  } else {
    history.replaceState(null, '', (isDomain() ? '/domain' : '/') + location.search);
  }
  updateArea();
});
tileSelect.addEventListener('change', changeTile);
function forcingLabel(value) {
  const sign = value < 0 ? '−' : value > 0 ? '+' : '';
  return `${sign}${Math.abs(value)} ppm`;
}

function signedNumber(value) {
  return value < 0 ? '−' + Math.abs(value) : String(value);
}

function updateScenario() {
  const inputs = entries.map(entry => entry.valueAsNumber);
  preset.value = Object.keys(presets).find(key => presets[key].every((value, i) => value === inputs[i])) ?? 'custom';
  $('pending').textContent = 'Run prediction to apply these settings.';
}
function choosePreset(key) {
  presets[key].forEach((value, i) => {
    sliders[i].value = value;
    entries[i].value = value;
  });
  updateScenario();
}
sliders.forEach((control, i) => {
  const entry = entries[i];
  control.addEventListener('input', () => {
    entry.value = control.value;
    updateScenario();
  });
  entry.addEventListener('input', () => {
    if (entry.validity.valid) control.value = entry.value;
    updateScenario();
  });
});
preset.addEventListener('change', () => choosePreset(preset.value));
$('reset').addEventListener('click', () => choosePreset('reference'));

async function loadWeatherCatalog() {
  weatherCatalog = await (await fetchAsset('weather-assets/catalog.json')).json();
  const response = await fetch('weather-model-v1/catalog.json');
  if (response.ok) {
    liveWeather = await response.json();
    const limits = [
      [windSpeed, -200, 200, 1, '−200%', '+200%'],
      [rotation, -180, 180, 1, '−180°', '+180°'],
      [background, -5, 5, 0.5, '−5 ppm', '+5 ppm'],
    ];
    for (const [control, min, max, step, low, high] of limits) {
      const current = control.value;
      control.min = min;
      control.max = max;
      control.step = step;
      control.value = current;
      const entry = $(control.id + '-value');
      entry.min = min;
      entry.max = max;
      $(control.id + '-min').textContent = low;
      $(control.id + '-max').textContent = high;
    }
  }
  preset.querySelector('[value="wind"]').disabled = false;
  preset.querySelector('[value="boundary"]').disabled = false;
  $('weather-note').textContent = liveWeather ? 'Type exact values or use the sliders. Custom settings calculate new weather in this browser.' :
    'These sliders use prepared weather settings.';
  setBusy(busy);
}

function loadTruth(scopeId, length) {
  if (publicApp || !weatherCatalog) return null;
  if (truthCache?.scopeId !== scopeId) {
    truthCache = {scopeId, values: weather.readArray(`truth-${scopeId}.bin.gz`, length).catch(() => {
      truthCache = undefined;
      return null;
    })};
  }
  return truthCache.values;
}

document.querySelector('form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || $('run').disabled) return;
  const whole = isDomain();
  const [value, speed, direction, boundaryForcing] = entries.map(entry => entry.valueAsNumber);
  const windId = weatherCatalog?.boundary.includes(boundaryForcing) ?
    weatherCatalog.wind.find(item => item.speed * 100 === speed && item.rotation === direction)?.id : undefined;
  const referenceWeather = speed === 100 && direction === 0 && boundaryForcing === 0;
  if (!windId && !referenceWeather && !liveWeather) {
    status('Prepared weather is unavailable for these settings.', 'error');
    return;
  }
  const scenario = {emissions: value, wind: referenceWeather ? 'reference' : windId ?? null,
    windSpeed: speed, rotation: direction, background: boundaryForcing};
  const isReferenceWeather = weather.isReference(scenario);
  const knownCustomCase = liveWeather?.verify;
  const customVerification = knownCustomCase && Object.entries(knownCustomCase).every(([key, value]) => scenario[key] === value);
  const canVerify = isReferenceWeather ? [80, 100].includes(value) :
    customVerification || (value === 80 && scenario.wind === 'clockwise' && scenario.background === 2);
  if (whole && verify && !canVerify) {
    status('Choose one of the saved verification cases.', 'error');
    return;
  }
  delete window.domainResult;
  delete window.domainError;
  delete window.pilotResult;
  delete window.pilotError;
  setBusy(true);
  status('Starting the browser calculation…', 'busy');
  try {
    const tile = selectedTile();
    const side = whole ? (catalog?.domainSize ?? 800) : tile.size;
    const truth = loadTruth(whole ? 'domain' : tile.id, 16 * side * side);
    const windDirection = direction ? `${Math.abs(direction)}° ${direction < 0 ? 'counterclockwise' : 'clockwise'}` : 'reference direction';
    const scenarioLabel = `${value}% emissions · wind ${signedNumber(speed)}% at ${windDirection} · boundary ${forcingLabel(boundaryForcing)}`;
    const comparison = {scenarioLabel, isReference: isReferenceWeather && value === 100};
    if (whole) {
      await releaseTile(true);
      const {fields, ...summary} = await runWithDevice(true, tile, scenario);
      showResults({...summary, ...fields, ...comparison, truth: await truth, boundary: catalog?.boundary ?? []});
      window.domainResult = summary;
    } else {
      const {summary, fields} = await runWithDevice(false, tile, scenario);
      showResults({baseline: fields.baseline, prediction: fields.field, mask: fields.mask,
        h: tile.size, w: tile.size, tile, boundary: catalog.boundary, emissions: value, seconds: summary.seconds,
        ...comparison, truth: await truth});
      window.pilotFields = fields;
      window.pilotResult = summary;
    }
    const summary = whole ? window.domainResult : window.pilotResult;
    $('compute-location').textContent = 'Calculated on this device · ' + computeLabel(summary.compute.provider);
    status(`Prediction complete · ${value}% emissions`);
    $('pending').textContent = 'Change settings and run again to compare another scenario.';
  } catch (error) {
    if (whole) {
      window.domainError = String(error);
      stopDomainWorker();
    } else {
      window.pilotError = String(error);
      stopDomainWorker();
      await releaseTile(true).catch(() => {});
    }
    status(`Could not complete the prediction. ${error.message}`, 'error');
  } finally {
    setBusy(false);
  }
});

updateArea();
loadCatalog().catch(error => {
  if (!isDomain()) status(error.message, 'error');
});
loadWeatherCatalog().catch(() => {
  $('weather-note').textContent = 'This preview has reference weather only.';
});
