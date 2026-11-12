'use strict';

importScripts('runtime/ort.webgpu.min.js', 'controls.js', 'layer-runner.js');
ort.env.wasm.wasmPaths = new URL('runtime/', location.href).href;
ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let assets;
let runner;
let baseline;

async function rawFile(name, Type) {
  const response = await fetch('data/' + name);
  if (!response.ok) throw new Error('Could not load ' + name);
  return new Type(await new Response(
    response.body.pipeThrough(new DecompressionStream('gzip')),
  ).arrayBuffer());
}

async function loadInputs() {
  const [configuration, emission, routing, mask] = await Promise.all([
    fetch('data/controls.json'), rawFile('raw-emission.gz', Float32Array),
    rawFile('raw-emission_z.gz', Uint8Array), rawFile('raw-mask.gz', Uint8Array),
  ]);
  if (!configuration.ok) throw new Error('Could not load PM₂.₅ scenario configuration.');
  return { fixture: await configuration.json(), raw: { emission, routing, mask } };
}

self.onmessage = async ({ data: { settings, provider } }) => {
  const started = performance.now();
  const report = message => self.postMessage({ progress: message });
  try {
    report('Loading model inputs');
    assets ??= await loadInputs();
    const { fixture, raw } = assets;
    const inputs = preparePM25(fixture, raw, settings);
    if (provider === 'webgpu') ort.env.wasm.numThreads = 1;
    runner ??= createPM25LayerRunner(ort, './layers/', provider);
    const isBaseline = settings.emissions === 1 && settings.background === 0
      && settings.wind === 1 && settings.rotation === 0;
    const needsBaseline = !baseline;
    if (needsBaseline) {
      baseline = await runner.run(isBaseline ? inputs : preparePM25(fixture, raw),
        stage => report('Baseline · ' + stage));
    }
    const prediction = isBaseline ? baseline.slice() : await runner.run(inputs,
      stage => report('Scenario · ' + stage));
    const reference = needsBaseline ? baseline.slice() : null;
    const transfer = reference ? [prediction.buffer, reference.buffer] : [prediction.buffer];
    self.postMessage({ prediction, baseline: reference, seconds: (performance.now() - started) / 1000 }, transfer);
  } catch (error) {
    self.postMessage({ error: error.message || String(error), invalid: error instanceof RangeError });
  }
};
