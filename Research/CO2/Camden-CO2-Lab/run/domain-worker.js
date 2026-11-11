/* Whole-domain inference with full-field normalization and bounded WASM kernels. */
importScripts('runtime/ort.wasm.min.js', 'weather.js', 'weather-model.js');
ort.env.wasm.wasmPaths = new URL('runtime/', location.href).href;
ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CHUNK = 192;
const options = {executionProviders: ['wasm'], enableCpuMemArena: false, enableMemPattern: false};

async function fetchAsset(path) {
  const response = await fetch('domain-assets/' + path);
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status}).`);
  return response;
}

async function readArray(meta, Type) {
  const files = meta.parts ?? [meta.file];
  let current;
  let index = 0;
  const input = new ReadableStream({
    async pull(controller) {
      while (index < files.length || current) {
        current ??= (await fetchAsset(files[index++])).body.getReader();
        const {done, value} = await current.read();
        if (!done) {
          controller.enqueue(value);
          return;
        }
        current = undefined;
      }
      controller.close();
    },
    cancel(reason) { return current?.cancel(reason); },
  });
  const bytes = new Uint8Array(meta.bytes);
  let offset = 0;
  for await (const chunk of input.pipeThrough(new DecompressionStream('gzip'))) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (offset !== bytes.length) throw new Error(`Incomplete input file ${meta.file}.`);
  return new Type(bytes.buffer);
}

function field(data, c, d, h, w) {
  return {data, c, d, h, w};
}

function tensor(value) {
  return new ort.Tensor('float32', value.data, [1, value.c, value.d, value.h, value.w]);
}

function copyRegion(source, target, yFrom, xFrom, yTo, xTo, h, w, channelOffset = 0) {
  for (let c = 0; c < source.c; c++) {
    for (let z = 0; z < source.d; z++) {
      const from = ((c * source.d + z) * source.h + yFrom) * source.w + xFrom;
      const to = (((c + channelOffset) * target.d + z) * target.h + yTo) * target.w + xTo;
      for (let row = 0; row < h; row++) {
        const start = from + row * source.w;
        target.data.set(source.data.subarray(start, start + w), to + row * target.w);
      }
    }
  }
}

function crop(parts, y, x, h, w) {
  const d = parts[0].d;
  const c = parts.reduce((sum, value) => sum + value.c, 0);
  const result = field(new Float32Array(c * d * h * w), c, d, h, w);
  let channel = 0;
  for (const part of parts) {
    copyRegion(part, result, y, x, 0, 0, h, w, channel);
    channel += part.c;
  }
  return result;
}

function normalize({data, c, d, h, w}, epsilon, alpha) {
  const volume = d * h * w;
  for (let channel = 0; channel < c; channel++) {
    const first = channel * volume;
    const end = first + volume;
    let sum = 0, sumSquares = 0;
    for (let i = first; i < end; i++) {
      const value = data[i];
      sum += value;
      sumSquares += value * value;
    }
    const mean = sum / volume;
    const scale = 1 / Math.sqrt(Math.max(0, sumSquares / volume - mean * mean) + epsilon);
    for (let i = first; i < end; i++) {
      const value = Math.fround((data[i] - mean) * scale);
      data[i] = value < 0 ? value * alpha : value;
    }
  }
}

async function openKernel(file) {
  return ort.InferenceSession.create(await (await fetchAsset('kernels/' + file)).arrayBuffer(), options);
}

function* patches(h, w, halo) {
  for (let y = 0; y < h; y += CHUNK) {
    for (let x = 0; x < w; x += CHUNK) {
      const outH = Math.min(CHUNK, h - y);
      const outW = Math.min(CHUNK, w - x);
      const top = Math.max(0, y - halo);
      const left = Math.max(0, x - halo);
      yield {y, x, outH, outW, top, left,
        h: Math.min(h, y + outH + halo) - top,
        w: Math.min(w, x + outW + halo) - left};
    }
  }
}

async function spatialKernel(step, parts) {
  const {d, h, w} = parts[0];
  const ratio = step.op === 'MaxPool' ? 0.5 : step.op === 'ConvTranspose' ? 2 : 1;
  const c = step.channels;
  const result = field(new Float32Array(c * d * h * w * ratio ** 3), c, d * ratio, h * ratio, w * ratio);
  const session = await openKernel(step.file);
  try {
    const halo = step.op === 'Conv' ? 1 : 0;
    for (const p of patches(h, w, halo)) {
      const input = tensor(crop(parts, p.top, p.left, p.h, p.w));
      const { [step.output]: output } = await session.run({[step.source]: input});
      const shape = output.dims.slice(1);
      copyRegion(field(output.data, ...shape), result,
        (p.y - p.top) * ratio, (p.x - p.left) * ratio,
        p.y * ratio, p.x * ratio, p.outH * ratio, p.outW * ratio);
      output.dispose();
      input.dispose();
    }
    return result;
  } finally {
    await session.release();
  }
}

async function runTrunk(plan, local, emissions) {
  const values = new Map([['local', local]]);
  const uses = new Map([['local', 1], [plan.heads.decoder, 1]]);
  const sources = step => [].concat(step.source);
  for (const step of plan.trunk) {
    for (const name of sources(step)) uses.set(name, (uses.get(name) || 0) + 1);
  }
  for (let i = 0; i < plan.trunk.length; i++) {
    const step = plan.trunk[i];
    const inputs = sources(step).map(name => values.get(name));
    let value;
    if (step.op === 'Concat') {
      value = inputs;
    } else if (step.op === 'InstanceNormalization') {
      const activation = plan.trunk[i + 1];
      if (activation.op !== 'LeakyRelu' || activation.source !== step.output || uses.get(step.source) !== 1) {
        throw new Error('Unexpected normalization and activation layout.');
      }
      normalize(inputs[0], step.epsilon, activation.alpha);
      value = inputs[0];
    } else if (step.op === 'LeakyRelu') {
      value = inputs[0]; // Applied with normalization above.
    } else {
      self.postMessage({status: `${emissions}% emissions · model layer ${i + 1} of ${plan.trunk.length}`});
      value = await spatialKernel(step, inputs.flat());
    }
    values.set(step.output, value);
    for (const name of sources(step)) {
      uses.set(name, uses.get(name) - 1);
      if (uses.get(name) === 0) values.delete(name);
    }
  }
  return values.get(plan.heads.decoder);
}

function coordinate(index, length) {
  return Math.fround(2 * Math.fround(index / (length - 1)) - 1);
}

function grid(p, local, fullLevels) {
  const data = new Float32Array(local.d * p.h * p.w * 3);
  let i = 0;
  for (let z = 0; z < local.d; z++) {
    const vertical = coordinate(z + 1, fullLevels);
    for (let y = p.top; y < p.top + p.h; y++) {
      const north = coordinate(y, local.h);
      for (let x = p.left; x < p.left + p.w; x++) {
        data[i++] = coordinate(x, local.w);
        data[i++] = north;
        data[i++] = vertical;
      }
    }
  }
  return new ort.Tensor('float32', data, [1, local.d, p.h, p.w, 3]);
}

async function runHeads(state, decoder, emissions) {
  const {local, context, plan, meta} = state;
  const encoder = await openKernel(plan.encoder.file);
  let encoded;
  try {
    encoded = (await encoder.run({context: tensor(context)}))[plan.encoder.output];
  } finally {
    await encoder.release();
  }
  self.postMessage({status: `${emissions}% emissions · calculating CO₂ across the full grid`});
  const session = await openKernel(plan.heads.file);
  const prediction = field(new Float32Array(local.d * local.h * local.w), 1, local.d, local.h, local.w);
  try {
    for (const p of patches(local.h, local.w, plan.heads.halo)) {
      const localInput = tensor(crop([local], p.top, p.left, p.h, p.w));
      const decoderInput = tensor(crop([decoder], p.top, p.left, p.h, p.w));
      const coordinates = grid(p, local, meta.full_vertical_levels);
      const {delta} = await session.run({
        local: localInput, [plan.heads.decoder]: decoderInput,
        [plan.encoder.output]: encoded, grid: coordinates,
      });
      copyRegion(field(delta.data, 1, local.d, p.h, p.w), prediction,
        p.y - p.top, p.x - p.left, p.y, p.x, p.outH, p.outW);
      for (const item of [delta, localInput, decoderInput, coordinates]) item.dispose();
    }
  } finally {
    encoded.dispose();
    await session.release();
  }
  for (let i = 0; i < prediction.data.length; i++) prediction.data[i] += state.previous[i];
  return prediction.data;
}

async function loadCase() {
  self.postMessage({status: 'Loading the whole-domain inputs…'});
  const [meta, plan] = await Promise.all(
    ['domain.json', 'kernels/kernels.json'].map(async file => (await fetchAsset(file)).json()));
  const [local, context, previous, mask] = await Promise.all(
    ['local', 'context', 'previous', 'mask'].map(name =>
      readArray(meta.arrays[name], name === 'mask' ? Uint8Array : Float32Array)));
  const state = {meta, plan, previous, mask, emissions: {}};
  for (const [name, data] of Object.entries({local, context})) {
    state[name] = field(data, ...meta.arrays[name].shape.slice(1));
    const {channel} = meta.emissions[name];
    const volume = data.length / state[name].c;
    state.emissions[name] = data.slice(channel * volume, (channel + 1) * volume);
  }
  return state;
}

async function predict(state, emissions) {
  for (const [name, original] of Object.entries(state.emissions)) {
    const {channel, mean, std} = state.meta.emissions[name];
    const destination = state[name].data.subarray(channel * original.length, (channel + 1) * original.length);
    const factor = emissions / 100;
    for (let i = 0; i < original.length; i++) {
      destination[i] = factor === 1 ? original[i] :
        (Math.fround(Math.fround(Math.fround(original[i] * std) + mean) * factor) - mean) / std;
    }
  }
  return runHeads(state, await runTrunk(state.plan, state.local, emissions), emissions);
}

async function summarize(state, prediction, scenario, verify) {
  const {baseline, mask, meta} = state;
  const {emissions} = scenario;
  const referenceBaseline = verify ? await readArray(meta.arrays.baseline, Float32Array) : null;
  let reference = referenceBaseline;
  if (verify && !scenario.wind) {
    reference = new Float32Array(await weatherModel.unpack('verify-co2.bin.gz'));
  } else if (verify && !weather.isReference(scenario)) {
    reference = await weather.readArray('verify-clockwise_2-domain.bin.gz', mask.length);
  } else if (verify && emissions === 80) {
    reference = await readArray(meta.arrays.reduced, Float32Array);
  }
  let valid = 0, sum = 0, errorSum = 0, maxError = 0, changeErrorSum = 0, maxChangeError = 0, maxBaselineError = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    const change = prediction[i] - baseline[i];
    if (!Number.isFinite(change)) throw new Error('The model returned a non-finite CO₂ value.');
    valid++;
    sum += change;
    if (reference) {
      const error = Math.abs(prediction[i] - reference[i]);
      const changeError = Math.abs(change - (reference[i] - referenceBaseline[i]));
      errorSum += error;
      changeErrorSum += changeError;
      maxError = Math.max(maxError, error);
      maxChangeError = Math.max(maxChangeError, changeError);
      maxBaselineError = Math.max(maxBaselineError, Math.abs(baseline[i] - referenceBaseline[i]));
    }
  }
  const {h, w} = state.local;
  const displayBaseline = baseline.slice();
  return {emissions, valid, meanChange: sum / valid, h, w,
    fields: {baseline: displayBaseline, prediction: prediction === baseline ? displayBaseline : prediction, mask: mask.slice()},
    verification: verify ? {maxError, maxBaselineError, maxChangeError,
      meanError: errorSum / valid, meanChangeError: changeErrorSum / valid} : null};
}

let state;
self.onmessage = async ({data: {scenario, catalog, verify = false}}) => {
  const started = performance.now();
  try {
    const {emissions} = scenario;
    if (!Number.isFinite(emissions) || emissions < 50 || emissions > 150) {
      throw new Error('Choose emissions between 50% and 150%.');
    }
    state ??= await loadCase();
    state.baseline ??= await predict(state, 100);
    if (!weather.isReference(scenario)) self.postMessage({status: 'Loading weather for this scenario…'});
    const arrays = {local: state.local.data, context: state.context.data};
    const shapes = {local: state.meta.arrays.local.shape, context: state.meta.arrays.context.shape};
    await weather.apply(state, arrays, shapes, 'domain', scenario, catalog,
      () => weatherModel.compute(scenario, {x: 0, y: 0}, shapes, status => self.postMessage({status}), verify));
    const sameAsBaseline = emissions === 100 && weather.isReference(scenario);
    const prediction = sameAsBaseline ? state.baseline : await predict(state, emissions);
    const result = await summarize(state, prediction, scenario, verify);
    result.seconds = (performance.now() - started) / 1000;
    self.postMessage({result}, [...new Set(Object.values(result.fields).map(values => values.buffer))]);
  } catch (error) {
    self.postMessage({error: String(error)});
  }
};
