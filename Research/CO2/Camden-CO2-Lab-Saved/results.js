const $ = id => document.getElementById(id);
const publicApp = document.documentElement.hasAttribute('data-public-app');
const concentration = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
const difference = [[33, 102, 172], [103, 169, 207], [247, 247, 247], [239, 138, 98], [178, 24, 43]];
const mapKinds = ['baseline', 'scenario', 'difference'];
const height = $('height');
const contrast = $('change-contrast');
const enlarge = $('enlarge-change');
const placeholders = document.querySelectorAll('.map-placeholder');
const legends = [document.querySelectorAll('.co2-low'), document.querySelectorAll('.co2-high')];
let result, layer, selectedCell;

const signed = value => `${value > 0 ? '+' : ''}${value.toFixed(4)}`;
const metres = level => 5 + 10 * level;
const quantile = (sorted, p) => sorted[Math.floor((sorted.length - 1) * p)];

function paletteColour(value, palette) {
  const t = Math.max(0, Math.min(1, value)) * (palette.length - 1);
  const i = Math.min(Math.floor(t), palette.length - 2);
  const weight = t - i;
  return palette[i].map((channel, j) => channel + weight * (palette[i + 1][j] - channel));
}

function drawMap(kind) {
  const canvas = $(kind + '-map');
  const {h, w, baseline, prediction, mask, boundary, tile} = result;
  canvas.width = w;
  canvas.height = h;
  const context = canvas.getContext('2d');
  const pixels = context.createImageData(w, h);
  const isChange = kind === 'difference';
  const limit = layer.bound / Number(contrast.value);
  const [lo, hi] = isChange ? [-limit, limit] : layer.scale;
  for (let row = 0; row < h; row++) {
    for (let col = 0; col < w; col++) {
      const cell = layer.offset + (h - row - 1) * w + col;
      const pixel = (row * w + col) * 4;
      let colour = [241, 243, 238];
      if (mask[cell]) {
        const value = isChange ? prediction[cell] - baseline[cell] :
          kind === 'baseline' ? baseline[cell] : prediction[cell];
        colour = paletteColour((value - lo) / (hi - lo), isChange ? difference : concentration);
      }
      pixels.data.set(colour, pixel);
      pixels.data[pixel + 3] = 255;
    }
  }
  context.putImageData(pixels, 0, 0);
  context.beginPath();
  for (const ring of boundary) {
    ring.forEach(([x, y], i) => {
      const px = x - (tile?.x ?? 0), py = h - (y - (tile?.y ?? 0));
      if (i === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    });
    context.closePath();
  }
  context.strokeStyle = 'rgba(25,45,45,.56)';
  context.lineWidth = Math.max(0.8, w / 450);
  context.stroke();
  if (selectedCell !== undefined) {
    const x = selectedCell % w + 0.5;
    const y = h - Math.floor(selectedCell / w) - 0.5;
    context.beginPath();
    context.arc(x, y, w / 65, 0, 2 * Math.PI);
    context.lineWidth = w / 110;
    context.strokeStyle = 'white';
    context.stroke();
    context.lineWidth = w / 220;
    context.strokeStyle = '#192d2d';
    context.stroke();
  }
}

function drawChange() {
  const limit = layer.bound / Number(contrast.value);
  $('diff-low').textContent = '≤ ' + signed(-limit);
  $('diff-high').textContent = '≥ ' + signed(limit);
  drawMap('difference');
}

function showHeight() {
  const level = Number(height.value);
  const {h, w, mask, baseline, prediction, truth} = result;
  const offset = level * h * w;
  const values = [], changes = [];
  let baselineSum = 0, predictionSum = 0, changeSum = 0, minChange = Infinity, maxChange = -Infinity;
  let absoluteError = 0, squaredError = 0, errorCount = 0;
  for (let i = offset; i < offset + h * w; i++) {
    if (!mask[i]) continue;
    const change = prediction[i] - baseline[i];
    values.push(baseline[i], prediction[i]);
    changes.push(Math.abs(change));
    baselineSum += baseline[i];
    predictionSum += prediction[i];
    changeSum += change;
    minChange = Math.min(minChange, change);
    maxChange = Math.max(maxChange, change);
    if (truth && Number.isFinite(truth[i])) {
      const error = baseline[i] - truth[i];
      absoluteError += Math.abs(error);
      squaredError += error * error;
      errorCount++;
    }
  }
  values.sort((a, b) => a - b);
  changes.sort((a, b) => a - b);
  const lo = quantile(values, .02);
  const hi = quantile(values, .98);
  layer = {offset, scale: lo === hi ? [lo - .01, hi + .01] : [lo, hi],
    bound: Math.max(.001, quantile(changes, .98))};
  $('baseline-mean').textContent = (baselineSum / changes.length).toFixed(2);
  $('scenario-mean').textContent = (predictionSum / changes.length).toFixed(2);
  $('change-mean').textContent = signed(changeSum / changes.length);
  $('height-caution').hidden = metres(level) <= 95;
  $('change-range').textContent = `Local changes range from ${signed(minChange)} to ${signed(maxChange)} ppm at ${metres(level)} m.`;
  $('error-values').hidden = !errorCount;
  $('error-note').textContent = errorCount ?
    `Reference baseline compared with PALM at ${metres(level)} m across ${errorCount.toLocaleString('en-GB')} cells. ${result.isReference ? 'The selected inputs match the baseline.' : 'The scenario maps show changes from the baseline.'}` :
    publicApp ? 'The report compares the original reference case with PALM.' :
      'PALM reference values are unavailable for this selection.';
  if (errorCount) {
    $('palm-mae').textContent = `${(absoluteError / errorCount).toFixed(3)} ppm`;
    $('palm-rmse').textContent = `${Math.sqrt(squaredError / errorCount).toFixed(3)} ppm`;
  }
  legends.forEach((labels, i) => labels.forEach(label => {
    label.textContent = `${i ? '≥' : '≤'} ${layer.scale[i].toFixed(1)}`;
  }));
  resetInspection();
  drawMap('baseline');
  drawMap('scenario');
  drawChange();
}

function resetInspection() {
  $('point-value').textContent = 'Move over or tap a map';
  $('point-detail').textContent = 'Read the concentration and change at a 10 m grid cell.';
}

function drawProfile(svg, series, lo, hi, label) {
  const left = 50, right = 301, top = 18, bottom = 194;
  const x = value => left + (value - lo) / (hi - lo) * (right - left);
  const y = metresValue => bottom - (metresValue - 5) / 150 * (bottom - top);
  svg.replaceChildren();
  function add(tag, attributes, label = '') {
    const element = document.createElementNS(svg.namespaceURI, tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    element.textContent = label;
    svg.append(element);
  }
  add('rect', {x: left, y: top, width: right - left, height: y(95) - top, fill: '#fbf1df'});
  for (const level of [5, 35, 65, 95, 125, 155]) {
    add('line', {x1: left, x2: right, y1: y(level), y2: y(level), class: 'profile-grid'});
    add('text', {x: left - 6, y: y(level) + 4, 'text-anchor': 'end'}, level + ' m');
  }
  if (lo < 0 && hi > 0) {
    add('line', {x1: x(0), x2: x(0), y1: top, y2: bottom, class: 'profile-zero'});
  }
  for (const value of [lo, (lo + hi) / 2, hi]) {
    add('text', {x: x(value), y: 213, 'text-anchor': 'middle'}, value.toFixed(hi - lo < 1 ? 3 : 1));
  }
  add('text', {x: (left + right) / 2, y: 231, 'text-anchor': 'middle'}, label);
  for (const {values, colour} of series) {
    let open = false;
    const commands = [];
    values.forEach((value, level) => {
      if (value === null) {
        open = false;
        return;
      }
      commands.push(`${open ? 'L' : 'M'}${x(value)} ${y(metres(level))}`);
      open = true;
      add('circle', {cx: x(value), cy: y(metres(level)), r: 2.6, fill: colour});
    });
    add('path', {d: commands.join(' '), stroke: colour, fill: 'none', 'stroke-width': 2});
  }
}

function showProfile(cell) {
  const {w, h, mask, baseline, prediction, tile} = result;
  const base = [], scenario = [];
  for (let index = cell; index < mask.length; index += h * w) {
    base.push(mask[index] ? baseline[index] : null);
    scenario.push(mask[index] ? prediction[index] : null);
  }
  const changes = base.map((value, level) => value === null ? null : scenario[level] - value);
  const concentrations = [...base, ...scenario].filter(value => value !== null);
  const limit = Math.max(.001, ...changes.filter(value => value !== null).map(Math.abs)) * 1.05;
  const column = cell % w + (tile?.x ?? 0) + 1;
  const row = Math.floor(cell / w) + (tile?.y ?? 0) + 1;
  $('profile-location').textContent = `Domain grid cell · column ${column}, row ${row} from the south. Shading marks heights above 95 m.`;
  $('profile-charts').hidden = false;
  drawProfile($('concentration-profile'),
    [{values: base, colour: '#586e65'}, {values: scenario, colour: '#b65836'}],
    Math.min(...concentrations) - .2, Math.max(...concentrations) + .2, 'CO₂ concentration (ppm)');
  drawProfile($('difference-profile'), [{values: changes, colour: '#2166ac'}],
    -limit, limit, 'Scenario − baseline (ppm)');
}

export function clearResults() {
  result = undefined;
  selectedCell = undefined;
  $('applied-scenario').hidden = true;
  $('error-values').hidden = true;
  $('error-note').textContent = publicApp ? 'The report compares the original reference case with PALM.' :
    'Run a prediction to compare the reference baseline with PALM at the selected height.';
  $('profile-charts').hidden = true;
  $('profile-location').textContent = 'Click or tap a coloured cell on any map.';
  $('height-caution').hidden = true;
  height.replaceChildren(new Option('45 m', '4'));
  height.disabled = contrast.disabled = enlarge.disabled = true;
  placeholders.forEach(item => { item.hidden = false; });
  ['baseline-mean', 'scenario-mean', 'change-mean', 'runtime', 'diff-low', 'diff-high'].forEach(id => {
    $(id).textContent = '—';
  });
  legends.flatMap(labels => [...labels]).forEach(label => { label.textContent = '—'; });
  $('scenario-tag').textContent = 'Selected emissions';
  $('change-range').textContent = 'Run a scenario to compare local changes.';
  resetInspection();
}

export function showResults(data) {
  result = data;
  const plane = data.h * data.w;
  const previousHeight = height.value;
  height.replaceChildren();
  for (let level = 0; level < data.mask.length / plane; level++) {
    const option = new Option(`${metres(level)} m${metres(level) > 95 ? ' · exploratory' : ''}`, level);
    option.disabled = !data.mask.subarray(level * plane, (level + 1) * plane).some(Boolean);
    height.add(option);
  }
  height.value = [...height.options].find(option => option.value === previousHeight && !option.disabled)?.value ??
    [...height.options].find(option => !option.disabled).value;
  height.disabled = contrast.disabled = enlarge.disabled = false;
  placeholders.forEach(item => { item.hidden = true; });
  $('applied-scenario').textContent = data.scenarioLabel;
  $('applied-scenario').hidden = false;
  $('scenario-tag').textContent = `${data.emissions}% emissions`;
  $('runtime').textContent = data.seconds < 60 ? `${data.seconds.toFixed(1)} s` : `${(data.seconds / 60).toFixed(1)} min`;
  showHeight();
  if (selectedCell !== undefined) showProfile(selectedCell);
}

height.addEventListener('change', showHeight);
contrast.addEventListener('change', drawChange);
enlarge.addEventListener('click', () => {
  const enlarged = $('maps').classList.toggle('focus-change');
  enlarge.setAttribute('aria-pressed', String(enlarged));
  enlarge.textContent = enlarged ? 'Equal-size maps' : 'Enlarge change map';
});
for (const kind of mapKinds) {
  const canvas = $(kind + '-map');
  const inspect = event => {
    if (!result) return undefined;
    const bounds = canvas.getBoundingClientRect();
    const {w, h, mask, baseline, prediction} = result;
    const x = Math.max(0, Math.min(w - 1, Math.floor((event.clientX - bounds.left) * w / bounds.width)));
    const row = Math.max(0, Math.min(h - 1, Math.floor((event.clientY - bounds.top) * h / bounds.height)));
    const cell = layer.offset + (h - row - 1) * w + x;
    $('point-value').textContent = mask[cell] ? `${prediction[cell].toFixed(2)} ppm` : 'Excluded cell';
    $('point-detail').textContent = mask[cell] ?
      `Baseline ${baseline[cell].toFixed(2)} ppm · change ${signed(prediction[cell] - baseline[cell])} ppm at ${metres(Number(height.value))} m.` :
      'Outside Camden or below the atmospheric grid at this height.';
    return mask[cell] ? cell - layer.offset : undefined;
  };
  canvas.addEventListener('pointermove', inspect);
  canvas.addEventListener('click', event => {
    const cell = inspect(event);
    if (cell === undefined) return;
    selectedCell = cell;
    showProfile(cell);
    mapKinds.forEach(drawMap);
  });
}
