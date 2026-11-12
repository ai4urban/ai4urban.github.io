import './grid-inputs.js';

ort.env.wasm.wasmPaths = new URL('runtime/', location.href).href;
ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export async function fetchAsset(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}: ${response.status}`);
  return response;
}

let sessionPromise;
let sessionProvider;
let domainManifest;

async function loadModel(provider) {
  const response = await fetchAsset('assets/stage2.onnx');
  return ort.InferenceSession.create(await response.arrayBuffer(), {
    executionProviders: [provider], enableCpuMemArena: false, enableMemPattern: false,
  });
}

async function prepare(tile, provider) {
  const start = performance.now();
  sessionProvider = provider;
  sessionPromise ??= loadModel(provider).catch(error => { sessionPromise = undefined; throw error; });
  if (document.documentElement.hasAttribute('data-public-app')) {
    domainManifest ??= fetchAsset('domain-assets/domain.json').then(response => response.json())
      .catch(error => { domainManifest = undefined; throw error; });
    const [inputs, session] = await Promise.all([
      domainManifest.then(meta => gridInputs.load(meta, {x: tile.x, y: tile.y, w: tile.size, h: tile.size})),
      sessionPromise,
    ]);
    return {...inputs, session, loadSeconds: (performance.now() - start) / 1000};
  }
  const [manifestResponse, caseResponse, session] = await Promise.all([
    fetchAsset(`assets/${tile.id}.json`), fetchAsset(`assets/${tile.id}.bin.gz`), sessionPromise,
  ]);
  const [manifest, buffer] = await Promise.all([
    manifestResponse.json(),
    new Response(caseResponse.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer(),
  ]);
  const arrays = Object.fromEntries(Object.entries(manifest.arrays).map(([name, {shape, offset}]) => [
    name, new Float32Array(buffer, offset, shape.reduce((total, size) => total * size, 1)),
  ]));
  return {manifest, arrays, session, loadSeconds: (performance.now() - start) / 1000};
}

function emissionsTensor(name, state, factor) {
  const shape = state.manifest.arrays[name].shape;
  const data = factor === 1 ? state.arrays[name] : state.arrays[name].slice();
  if (factor !== 1) {
    const {channel, mean, std} = state.manifest.emissions[name];
    const volume = shape.slice(2).reduce((total, size) => total * size, 1);
    const end = (channel + 1) * volume;
    const float = Math.fround;
    for (let i = channel * volume; i < end; i++) {
      const physical = float(float(data[i] * std) + mean);
      data[i] = float(float(float(physical * factor) - mean) / std);
    }
  }
  return new ort.Tensor('float32', data, shape);
}

async function predict(state, factor) {
  const {arrays, manifest, session} = state;
  const inputs = {
    local: emissionsTensor('local', state, factor),
    context: emissionsTensor('context', state, factor),
    grid: new ort.Tensor('float32', arrays.grid, manifest.arrays.grid.shape),
  };
  let result;
  try {
    result = await session.run(inputs);
    const field = new Float32Array(arrays.previous.length);
    const delta = result.delta.data;
    for (let i = 0; i < field.length; i++) field[i] = arrays.previous[i] + delta[i];
    return field;
  } finally {
    if (result) Object.values(result).forEach(tensor => tensor.dispose());
    Object.values(inputs).forEach(tensor => tensor.dispose());
  }
}

let prepared;

function calculateWeather(scenario, tile, shapes, onStatus) {
  const worker = new Worker('weather-worker.js');
  return new Promise((resolve, reject) => {
    worker.onerror = event => reject(new Error(event.message || 'The weather calculation stopped.'));
    worker.onmessage = ({data}) => {
      if (data.status) onStatus(data.status);
      else if (data.error) reject(new Error(data.error));
      else resolve(data.overlay);
    };
    worker.postMessage({scenario, tile, shapes});
  }).finally(() => worker.terminate());
}

export async function releaseTile(releaseModel = false) {
  prepared = undefined;
  if (releaseModel) {
    const previous = sessionPromise;
    sessionPromise = undefined;
    sessionProvider = undefined;
    const session = await previous?.catch(() => undefined);
    await session?.release();
  }
}

export async function runTile(tile, scenario, catalog, onStatus, provider) {
  const {emissions} = scenario;
  const factor = emissions / 100;
  onStatus('Loading the model and tile…');
  if (sessionProvider && sessionProvider !== provider) await releaseTile(true);
  prepared ??= await prepare(tile, provider);
  const state = prepared;
  const {arrays, loadSeconds} = state;
  const start = performance.now();
  if (!state.baseline) {
    onStatus('Calculating the reference baseline…');
    state.baseline = await predict(state, 1);
  }
  const baselineSeconds = (performance.now() - start) / 1000;
  if (!weather.isReference(scenario)) onStatus('Loading weather for this scenario…');
  const shapes = {local: state.manifest.arrays.local.shape, context: state.manifest.arrays.context.shape};
  const weatherStart = performance.now();
  await weather.apply(state, arrays, shapes, tile.id, scenario, catalog,
    () => calculateWeather(scenario, tile, shapes, onStatus));
  const weatherSeconds = (performance.now() - weatherStart) / 1000;
  onStatus('Running the CO₂ model in this browser…');
  const sameAsBaseline = factor === 1 && weather.isReference(scenario);
  const predictionStart = performance.now();
  const field = sameAsBaseline ? state.baseline : await predict(state, factor);
  const predictionSeconds = (performance.now() - predictionStart) / 1000;
  const seconds = (performance.now() - start) / 1000;
  const reference = weather.isReference(scenario) ?
    factor === 1 ? arrays.baseline : factor === 0.8 ? arrays.reduced : null : null;
  let count = 0, sumChange = 0, errorSum = 0, maxError = 0;
  let referenceChangeSum = 0, changeErrorSum = 0, maxChangeError = 0;
  for (let i = 0; i < field.length; i++) {
    if (!arrays.mask[i]) continue;
    const change = field[i] - state.baseline[i];
    if (!Number.isFinite(change)) throw new Error('Prediction returned an invalid value.');
    count++;
    sumChange += change;
    if (reference) {
      const error = Math.abs(field[i] - reference[i]);
      const referenceChange = reference[i] - arrays.baseline[i];
      const changeError = Math.abs(change - referenceChange);
      errorSum += error;
      maxError = Math.max(maxError, error);
      referenceChangeSum += referenceChange;
      changeErrorSum += changeError;
      maxChangeError = Math.max(maxChangeError, changeError);
    }
  }
  if (!count) throw new Error('The case has no valid atmospheric cells.');
  const summary = {
    tile: tile.id, factor, seconds, loadSeconds, count, meanChangePpm: sumChange / count,
    compute: {provider, baselineSeconds, weatherSeconds, predictionSeconds},
    meanAbsoluteErrorPpm: reference ? errorSum / count : null,
    maxAbsoluteErrorPpm: reference ? maxError : null,
    referenceMeanChangePpm: reference ? referenceChangeSum / count : null,
    meanAbsoluteChangeErrorPpm: reference ? changeErrorSum / count : null,
    maxAbsoluteChangeErrorPpm: reference ? maxChangeError : null,
    threads: ort.env.wasm.numThreads, crossOriginIsolated,
  };
  const fields = {tile, field, baseline: state.baseline, mask: arrays.mask, reference, referenceBaseline: arrays.baseline};
  return {summary, fields};
}
