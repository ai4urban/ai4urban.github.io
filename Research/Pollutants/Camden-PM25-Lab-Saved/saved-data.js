'use strict';

const referenceSettings = {emissions: 1, background: 0, wind: 1, rotation: 0};
const savedControls = ['emissions', 'background', 'wind', 'rotation'];
let browserCase, browserResult, savedCatalog, savedDataBase;

async function savedArray(file, Type) {
  const response = await fetch(new URL(file, savedDataBase));
  if (!response.ok) throw new Error(`Could not load saved data (HTTP ${response.status}). Try again.`);
  return new Type(await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}

function configureSavedControls() {
  for (const name of savedControls) {
    const select = document.getElementById(name);
    const values = savedCatalog.ranges[name];
    const options = values.map(value => {
      const label = name === 'rotation' ? `${value}°` : name === 'background' ? `${value} µg/m³` : `${Math.round(value * 100)}%`;
      const option = new Option(label, value);
      if (name === 'background' && !savedCatalog.validBackground.includes(value)) {
        option.disabled = true;
        option.textContent += ' · unavailable';
      }
      return option;
    });
    select.replaceChildren(...options);
    select.value = referenceSettings[name];
    select.disabled = false;
    select.addEventListener('change', () => {
      document.getElementById('pending').textContent = 'Load the saved scenario to apply these settings.';
    });
  }
  document.querySelector('#background').closest('.control').querySelector('p').textContent =
    'Shift both incoming profiles. The −5 µg/m³ choice is unavailable because it would create a negative concentration.';
  controls();
}

async function loadBrowserCase() {
  const location = await fetch('data-location.json');
  if (!location.ok) throw new Error('Could not load the saved-data location.');
  savedDataBase = new URL((await location.json()).base, window.location.href);
  const response = await fetch(new URL('catalog.json', savedDataBase));
  if (!response.ok) throw new Error('Could not load the saved-scenario catalogue.');
  savedCatalog = await response.json();
  const metadata = savedCatalog.metadata;
  const [mask, values] = await Promise.all([
    savedArray('mask.bin.gz', Uint8Array), savedArray('baseline.bin.gz', Float32Array),
  ]);
  const size = metadata.shape.reduce((a, b) => a * b, 1);
  if (mask.length !== size || values.length !== savedCatalog.valid) throw new Error('The saved reference is incomplete.');
  const baseline = new Float32Array(size);
  let index = 0;
  for (let i = 0; i < size; i++) if (mask[i]) baseline[i] = values[index++];
  configureSavedControls();
  const result = {id: 'baseline', baseline, prediction: baseline, parameters: referenceSettings, seconds: 0};
  return {metadata, mask, result};
}

function setSavedBusy(value) {
  for (const id of [...savedControls, 'reset']) document.getElementById(id).disabled = value;
  document.querySelectorAll('[data-preset]').forEach(button => { button.disabled = value; });
}

async function runSavedScenario(settings) {
  const key = savedControls.map(name => settings[name]).join('_');
  const saved = savedCatalog.cases[key];
  if (!saved) throw new RangeError('Choose an available combination from the saved scenario lists.');
  const started = performance.now();
  setSavedBusy(true);
  try {
    const baseline = browserCase.result.baseline;
    const {mask} = browserCase;
    let prediction;
    if (savedControls.every(name => settings[name] === referenceSettings[name])) prediction = baseline;
    else if (browserResult?.key === key) prediction = browserResult.prediction;
    else if (saved.encoding === 'float32') {
      const values = await savedArray(saved.file, Float32Array);
      if (values.length !== savedCatalog.valid) throw new Error('The saved scenario is incomplete.');
      prediction = new Float32Array(mask.length);
      let index = 0;
      for (let i = 0; i < mask.length; i++) if (mask[i]) prediction[i] = values[index++];
    } else {
      const values = await savedArray(saved.file, Uint8Array);
      if (values.length !== savedCatalog.valid * 2) throw new Error('The saved scenario is incomplete.');
      prediction = new Float32Array(mask.length);
      const scale = Math.fround(saved.scale);
      let index = 0, accumulator = 0;
      for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) continue;
        accumulator = (accumulator + (values[index] | values[savedCatalog.valid + index] << 8)) & 65535;
        const difference = (accumulator << 16) >> 16;
        prediction[i] = baseline[i] + Math.fround(difference * scale);
        index++;
      }
    }
    browserResult = {id: 'local', key, baseline, prediction, parameters: settings,
      seconds: (performance.now() - started) / 1000};
    window.savedPM25 = {key, maxStorageError: saved.maxStorageError, shape: savedCatalog.metadata.shape};
    text('compute-device', 'Loaded precomputed model result');
    return {id: 'local'};
  } finally {
    setSavedBusy(false);
  }
}

function summaryFor(result, level) {
  const { metadata, mask } = browserCase;
  const cellsPerLevel = metadata.shape[1] * metadata.shape[2];
  const start = level * cellsPerLevel;
  let cells = 0, x = 0, y = 0, xx = 0, yy = 0, xy = 0, bias = 0, mae = 0, mse = 0;
  for (let i = start; i < start + cellsPerLevel; i++) {
    if (!mask[i]) continue;
    const first = result.baseline[i], second = result.prediction[i];
    const change = Math.fround(second - first);
    cells++;
    x += first; y += second;
    xx += first * first; yy += second * second; xy += first * second;
    bias += change; mae += Math.abs(change); mse += change * change;
  }
  const variance = Math.max(0, xx - x * x / cells) * Math.max(0, yy - y * y / cells);
  return {
    level, height: metadata.heights[level], cells,
    reference_mean: cells ? x / cells : null, prediction_mean: cells ? y / cells : null,
    bias: cells ? bias / cells : null, mae: cells ? mae / cells : null,
    rmse: cells ? Math.sqrt(mse / cells) : null,
    r: variance > 0 ? Math.max(-1, Math.min(1, (xy - x * y / cells) / Math.sqrt(variance))) : null,
  };
}

function percentile(sorted, fraction) {
  const index = (sorted.length - 1) * fraction;
  const low = Math.floor(index);
  return sorted[low] + (index - low) * (sorted[Math.ceil(index)] - sorted[low]);
}

function makeGrid(result, level) {
  const { metadata, mask } = browserCase;
  const size = metadata.shape[1] * metadata.shape[2];
  const start = level * size;
  result.table ??= metadata.heights.map((_, i) => summaryFor(result, i));
  const summary = result.table[level];
  const first = Array(size).fill(null), second = Array(size).fill(null), difference = Array(size).fill(null);
  const concentration = new Float32Array(2 * summary.cells);
  const absolute = new Float32Array(summary.cells);
  const scatter = [];
  const plotCount = Math.min(summary.cells, 4000);
  let valid = 0;
  for (let cell = 0; cell < size; cell++) {
    const i = start + cell;
    if (!mask[i]) continue;
    first[cell] = result.baseline[i];
    second[cell] = result.prediction[i];
    difference[cell] = Math.fround(second[cell] - first[cell]);
    concentration[2 * valid] = first[cell];
    concentration[2 * valid + 1] = second[cell];
    absolute[valid] = Math.abs(difference[cell]);
    if (scatter.length < plotCount &&
        valid === Math.floor(scatter.length * (summary.cells - 1) / Math.max(1, plotCount - 1))) {
      scatter.push([first[cell], second[cell]]);
    }
    valid++;
  }
  const scale = valid ? [percentile(concentration.sort(), .02), percentile(concentration, .98)] : [0, 1];
  if (scale[1] <= scale[0]) scale[1] = scale[0] + 1;
  return { first, second, difference, height: summary.height, summary, scale, scatter,
    difference_limit: Math.max(1e-6, valid ? percentile(absolute.sort(), .98) : 0),
    table: result.table, parameters: result.parameters, seconds: result.seconds };
}

function makeProfile(result, x, y) {
  const { metadata, mask } = browserCase;
  const [, height, width] = metadata.shape;
  const first = [], second = [], difference = [];
  for (let z = 0; z < metadata.heights.length; z++) {
    const i = (z * height + y) * width + x;
    first.push(mask[i] ? result.baseline[i] : null);
    second.push(mask[i] ? result.prediction[i] : null);
    difference.push(mask[i] ? Math.fround(result.prediction[i] - result.baseline[i]) : null);
  }
  return { first, second, difference, heights: metadata.heights, easting: metadata.x[x], northing: metadata.y[y] };
}

async function api(path, options) {
  browserCase ??= await loadBrowserCase();
  const url = new URL(path, location.origin);
  const { searchParams } = url;
  const result = searchParams.get('id') === 'local' ? browserResult : browserCase.result;
  switch (url.pathname) {
    case '/api/status': return { ready: true, phase: 'Reference ready' };
    case '/api/meta': return browserCase.metadata;
    case '/api/predict': return runSavedScenario(JSON.parse(options.body));
    case '/api/grid': return makeGrid(result, Number(searchParams.get('level')));
    case '/api/profile': return makeProfile(result, Number(searchParams.get('x')), Number(searchParams.get('y')));
    default: throw new Error('Unknown local data request.');
  }
}

function saveBrowserCsv(event, summaryOnly) {
  event.preventDefault();
  if (!grid) return;
  const result = current === 'local' ? browserResult : browserCase.result;
  const { metadata, mask } = browserCase;
  const parts = [];
  if (summaryOnly) {
    const columns = Object.keys(grid.table[0]);
    parts.push(columns.join(',') + '\n');
    for (const row of grid.table) parts.push(columns.map(key => row[key] ?? '').join(',') + '\n');
  } else {
    parts.push('Easting BNG (m),Northing BNG (m),Height (m),Baseline (ug/m3),Scenario (ug/m3),Change (ug/m3)\n');
    const [, h, w] = metadata.shape;
    const start = level * h * w;
    for (let cell = 0; cell < h * w; cell++) {
      const i = start + cell;
      if (!mask[i]) continue;
      const first = result.baseline[i], second = result.prediction[i];
      parts.push([metadata.x[cell % w], metadata.y[Math.floor(cell / w)], grid.height,
        first, second, Math.fround(second - first)].join(',') + '\n');
    }
  }
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob(parts, { type: 'text/csv;charset=utf-8' }));
  link.download = summaryOnly ? 'pm25-scenario-metrics.csv' : 'pm25-scenario-' + grid.height + 'm.csv';
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
