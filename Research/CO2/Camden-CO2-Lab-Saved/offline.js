import {clearResults, showResults} from './results.js?v=co2-20261126-3';

const $ = id => document.getElementById(id);
const controls = ['emissions', 'wind-speed', 'rotation', 'background'];
const names = ['emissions', 'windSpeed', 'rotation', 'background'];
const presets = {reference: [100, 100, 0, 0], lower: [80, 100, 0, 0],
  wind: [100, 100, 90, 0], boundary: [100, 100, 0, 5]};
let catalog, tiles, mask, baseline, cached, dataBase, loading = false, selection = 0;

function status(message, state = 'ready') {
  $('status').className = 'status ' + state;
  $('status-text').textContent = message;
}

function busy(value) {
  loading = value;
  for (const id of [...controls, ...controls.map(name => name + '-value'), 'scope', 'tile', 'preset', 'reset', 'run']) {
    $(id).disabled = value;
  }
  $('run-label').textContent = value ? 'Loading…' : 'Load saved scenario';
}

function settings() {
  return controls.map(id => Number($(id + '-value').value));
}

function updatePending() {
  const values = settings();
  controls.forEach((id, i) => $(id).setAttribute('aria-valuetext',
    values[i] + (i === 2 ? ' degrees' : i === 3 ? ' ppm' : ' percent')));
  $('preset').value = Object.keys(presets).find(key => presets[key].every((value, i) => value === values[i])) ?? 'custom';
  $('pending').textContent = 'Load the saved scenario to apply these settings.';
}

function choosePreset(key) {
  presets[key].forEach((value, i) => {
    $(controls[i] + '-value').value = value;
    $(controls[i]).value = catalog.ranges[names[i]].indexOf(value);
  });
  updatePending();
}

async function array(file, Type) {
  const response = await fetch(new URL(file, dataBase));
  if (!response.ok) throw new Error(`Could not load saved data (HTTP ${response.status}). Try again.`);
  return new Type(await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}

function expand(values) {
  if (values.length !== catalog.valid) throw new Error('The saved field is incomplete.');
  const result = new Float32Array(mask.length);
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) result[i] = values[n++];
  return result;
}

function crop(field, tile) {
  if (!tile) return field;
  const result = new field.constructor(16 * tile.size * tile.size);
  for (let z = 0; z < 16; z++) {
    for (let y = 0; y < tile.size; y++) {
      const offset = (z * 800 + tile.y + y) * 800 + tile.x;
      result.set(field.subarray(offset, offset + tile.size), (z * tile.size + y) * tile.size);
    }
  }
  return result;
}

function updateArea() {
  clearResults();
  const tile = tiles.tiles.find(tile => tile.id === $('tile').value);
  const whole = $('scope').value === 'domain';
  $('tile-picker').hidden = whole;
  $('result-title').textContent = whole ? 'Whole-domain saved scenario' : tile.label;
  $('area-note').textContent = whole ? '800 × 800 cells at 16 heights. Maps show valid Camden atmospheric cells.' :
    '256 × 256 cells, cropped from the same saved whole-domain field.';
  for (const marker of $('tile-map').querySelectorAll('[data-tile]')) {
    marker.setAttribute('aria-pressed', String(marker.dataset.tile === tile.id));
  }
}

async function loadScenario() {
  if (loading) return;
  const current = ++selection;
  const started = performance.now();
  const values = settings();
  const key = values.join('_');
  const saved = catalog.cases[key];
  if (!saved) {
    status('This input combination is unavailable. Choose values from the saved lists.', 'error');
    return;
  }
  busy(true);
  status('Loading the saved scenario…', 'busy');
  try {
    let prediction;
    if (cached?.key === key) prediction = cached.field;
    else if (values.every((value, i) => value === catalog.reference[i])) prediction = baseline;
    else {
      const packed = saved.encoding === 'delta-byteplanes';
      const differences = await array(saved.file, packed ? Uint8Array : Int16Array);
      if (differences.length !== catalog.valid * (packed ? 2 : 1)) throw new Error('The saved scenario is incomplete.');
      const field = new Float32Array(mask.length);
      let n = 0, accumulator = 0;
      for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) continue;
        let difference = differences[n];
        if (packed) {
          accumulator = (accumulator + (differences[n] | differences[catalog.valid + n] << 8)) & 65535;
          difference = (accumulator << 16) >> 16;
        }
        field[i] = baseline[i] + difference * saved.scale;
        n++;
      }
      prediction = field;
    }
    if (current !== selection) return;
    cached = {key, field: prediction};
    const tile = $('scope').value === 'tile' ? tiles.tiles.find(tile => tile.id === $('tile').value) : undefined;
    const [emissions, wind, rotation, background] = values;
    const scenarioLabel = `Saved scenario · ${emissions}% emissions · wind ${wind}% · rotation ${rotation}° · boundary ${background > 0 ? '+' : ''}${background} ppm`;
    showResults({baseline: crop(baseline, tile), prediction: crop(prediction, tile), mask: crop(mask, tile),
      h: tile?.size ?? 800, w: tile?.size ?? 800, tile, boundary: tiles.boundary, emissions,
      scenarioLabel, isReference: values.every((value, i) => value === catalog.reference[i]),
      seconds: (performance.now() - started) / 1000});
    const elapsed = (performance.now() - started) / 1000;
    $('runtime').textContent = elapsed < 60 ? `${elapsed.toFixed(1)} s` : `${(elapsed / 60).toFixed(1)} min`;
    window.savedScenario = {key, storageError: saved.maxStorageError, shape: catalog.shape};
    status('Saved scenario loaded');
    $('pending').textContent = 'Change settings to explore another saved scenario.';
  } catch (error) {
    status(error.message || 'Could not load this scenario. Try again.', 'error');
  } finally {
    busy(false);
  }
}

async function start() {
  busy(true);
  const locationResponse = await fetch('data-location.json');
  if (!locationResponse.ok) throw new Error('Could not load the saved-data location.');
  dataBase = new URL((await locationResponse.json()).base, location.href);
  [catalog, tiles] = await Promise.all([new URL('catalog.json', dataBase), 'tiles.json'].map(async path => {
    const response = await fetch(path);
    if (!response.ok) throw new Error('Could not load the saved-scenario catalogue. Reload to try again.');
    return response.json();
  }));
  for (const [i, id] of controls.entries()) {
    const values = catalog.ranges[names[i]];
    $(id).max = values.length - 1;
    $(id).setAttribute('aria-label', ['CO₂ emissions', 'Wind speed factor', 'Wind direction', 'Boundary CO₂ forcing'][i]);
    $(id + '-value').replaceChildren(...values.map(value => new Option(value, value)));
    $(id).addEventListener('input', () => {
      $(id + '-value').value = values[Number($(id).value)];
      updatePending();
    });
    $(id + '-value').addEventListener('change', () => {
      $(id).value = values.indexOf(Number($(id + '-value').value));
      updatePending();
    });
  }
  const map = $('tile-map');
  $('camden-boundary').setAttribute('d', tiles.boundary.map(ring =>
    'M' + ring.map(([x, y]) => `${x},${800-y}`).join(' L') + ' Z').join(' '));
  for (const tile of tiles.tiles) {
    $('tile').add(new Option(tile.label, tile.id));
    const marker = document.createElementNS(map.namespaceURI, 'g');
    marker.dataset.tile = tile.id;
    marker.setAttribute('role', 'button');
    marker.setAttribute('tabindex', '0');
    marker.setAttribute('aria-label', tile.label);
    marker.setAttribute('transform', `translate(${tile.x + 128} ${800-tile.y-128})`);
    const circle = document.createElementNS(map.namespaceURI, 'circle');
    circle.setAttribute('r', '30');
    const label = document.createElementNS(map.namespaceURI, 'text');
    label.textContent = tile.id.slice(5).toUpperCase();
    marker.append(circle, label);
    const select = () => {
      if (loading) return;
      $('tile').value = tile.id;
      updateArea();
      loadScenario();
    };
    marker.addEventListener('click', select);
    marker.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); }
    });
    map.append(marker);
  }
  mask = await array('mask.bin.gz', Uint8Array);
  if (mask.length !== catalog.shape.reduce((a, b) => a * b, 1)) throw new Error('The saved mask is incomplete.');
  baseline = expand(await array('baseline.bin.gz', Float32Array));
  $('scope').addEventListener('change', () => { updateArea(); loadScenario(); });
  $('tile').addEventListener('change', () => { updateArea(); loadScenario(); });
  $('preset').addEventListener('change', () => choosePreset($('preset').value));
  $('reset').addEventListener('click', () => choosePreset('reference'));
  document.querySelector('form').addEventListener('submit', event => { event.preventDefault(); loadScenario(); });
  choosePreset('lower');
  updateArea();
  busy(false);
  await loadScenario();
}

start().catch(error => { status(error.message, 'error'); });
