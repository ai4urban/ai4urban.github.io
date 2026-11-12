'use strict';

const referenceSettings = { emissions: 1, background: 0, wind: 1, rotation: 0 };
let browserCase;
let browserResult;
let browserWorker;
let browserProvider;

async function loadLabAsset(name, Type) {
  const response = await fetch('./data/' + name);
  if (!response.ok) throw new Error('Could not load ' + name);
  if (!Type) return response.json();
  return new Type(await new Response(
    response.body.pipeThrough(new DecompressionStream('gzip')),
  ).arrayBuffer());
}

async function loadBrowserCase() {
  const [metadata, baseline, mask] = await Promise.all([
    loadLabAsset('display.json'), loadLabAsset('display-baseline.gz', Float32Array),
    loadLabAsset('display-mask.gz', Uint8Array),
  ]);
  if (baseline.length !== metadata.shape.reduce((a, b) => a * b, 1) || mask.length !== baseline.length) {
    throw new Error('The baseline and mask dimensions differ from the reference grid.');
  }
  const result = { id: 'baseline', baseline, prediction: baseline,
    parameters: referenceSettings, seconds: 0 };
  return { metadata, mask, result };
}

function useBrowserWorker(provider, settings) {
  if (browserProvider !== provider) {
    browserWorker?.terminate();
    browserWorker = null;
  }
  if (!browserWorker) {
    browserWorker = new Worker('./lab-worker.js');
    browserProvider = provider;
  }
  return new Promise((resolve, reject) => {
    browserWorker.onerror = event => reject(new Error(event.message || 'Browser worker stopped.'));
    browserWorker.onmessage = ({ data }) => {
      if (data.progress) {
        status((provider === 'webgpu' ? 'WebGPU · ' : 'CPU · ') + data.progress);
      } else if (data.error) {
        reject(data.invalid ? new RangeError(data.error) : new Error(data.error));
      } else {
        resolve(data);
      }
    };
    browserWorker.postMessage({ settings, provider });
  });
}

function stopBrowserWorker() {
  browserWorker?.terminate();
  browserWorker = null;
  browserProvider = null;
}

async function runInBrowser(settings) {
  const device = document.getElementById('compute-mode').value;
  let provider = device === 'cpu' ? 'wasm' : 'webgpu';
  if (provider === 'webgpu') {
    let available = false;
    try {
      const adapter = await navigator.gpu?.requestAdapter();
      available = Boolean(adapter && !adapter.info.isFallbackAdapter);
    } catch {
      // CPU is available even when the browser cannot create a GPU adapter.
    }
    if (!available) {
      if (device === 'gpu') throw new Error('Hardware WebGPU is unavailable. Select CPU to run this case.');
      provider = 'wasm';
    }
  }
  const previousProvider = browserProvider;
  let output;
  try {
    output = await useBrowserWorker(provider, settings);
  } catch (error) {
    if (error instanceof RangeError) throw error;
    stopBrowserWorker();
    if (device !== 'auto' || provider !== 'webgpu') throw error;
    status('WebGPU could not finish. Retrying on the CPU…');
    provider = 'wasm';
    output = await useBrowserWorker(provider, settings);
  }
  const baseline = output.baseline ?? (provider === previousProvider && browserResult?.baseline);
  if (!baseline) throw new Error('The local baseline is unavailable.');
  browserResult = {
    id: 'local', baseline, prediction: output.prediction, parameters: settings, seconds: output.seconds,
  };
  text('compute-device', provider === 'webgpu' ? 'Calculated on this device · WebGPU' : 'Calculated on this device · CPU');
  return { id: browserResult.id };
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
    case '/api/predict': return runInBrowser(JSON.parse(options.body));
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
