/* Tiled operator execution with normalization over the selected prediction area. */
importScripts('runtime/ort.webgpu.min.js', 'weather.js', 'weather-model.js', 'grid-inputs.js');
ort.env.wasm.wasmPaths = new URL('runtime/', location.href).href;
ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let provider;

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

async function normalizeTile(session, {data, c, d, h, w}) {
  // Each channel uses all tile cells. Batch eight channels at a time to bound WASM memory.
  const channels = 8;
  const size = channels * d * h * w;
  for (let channel = 0; channel < c; channel += channels) {
    const offset = channel * d * h * w;
    const input = new ort.Tensor('float32', data.subarray(offset, offset + size), [1, channels, d, h, w]);
    let output;
    try {
      ({output} = await session.run({field: input}));
      data.set(output.data, offset);
    } finally {
      input.dispose();
      output?.dispose();
    }
  }
}

async function openKernel(file, previewAsset = false, executeWith = provider) {
  const response = previewAsset ? await fetch('webgpu-kernels/' + file) : await fetchAsset('kernels/' + file);
  if (!response.ok) throw new Error(`Could not load ${file} (HTTP ${response.status}).`);
  return ort.InferenceSession.create(await response.arrayBuffer(), {
    executionProviders: [executeWith], enableCpuMemArena: false, enableMemPattern: false,
  });
}

function* patches(h, w, halo) {
  const chunk = provider === 'webgpu' ? 96 : 192;
  for (let y = 0; y < h; y += chunk) {
    for (let x = 0; x < w; x += chunk) {
      const outH = Math.min(chunk, h - y);
      const outW = Math.min(chunk, w - x);
      const top = Math.max(0, y - halo);
      const left = Math.max(0, x - halo);
      yield {y, x, outH, outW, top, left,
        h: Math.min(h, y + outH + halo) - top,
        w: Math.min(w, x + outW + halo) - left};
    }
  }
}

function loweredInput(parts, patch, expandDepth) {
  const {d} = parts[0];
  const stencil = expandDepth ? 3 : 1;
  const inputChannels = parts.reduce((count, part) => count + part.c, 0) * stencil;
  const channels = Math.ceil(inputChannels / 4) * 4;
  const plane = patch.h * patch.w;
  const data = new Float32Array(d * channels * plane);
  let target = 0;
  for (let level = 0; level < d; level++) {
    for (const part of parts) {
      for (let c = 0; c < part.c; c++) {
        for (let offset = 0; offset < stencil; offset++) {
          const z = level + offset - (expandDepth ? 1 : 0);
          if (z >= 0 && z < d) {
            for (let y = 0; y < patch.h; y++) {
              const start = ((c * d + z) * part.h + patch.top + y) * part.w + patch.left;
              data.set(part.data.subarray(start, start + patch.w), target + y * patch.w);
            }
          }
          target += plane;
        }
      }
    }
    target += (channels - inputChannels) * plane;
  }
  return new ort.Tensor('float32', data, [d, channels, patch.h, patch.w]);
}

function copyLowered(output, result, p, ratio) {
  const data = output.data;
  const [, channels, h, w] = output.dims;
  const localY = (p.y - p.top) * ratio, localX = (p.x - p.left) * ratio;
  const outH = p.outH * ratio, outW = p.outW * ratio;
  for (let c = 0; c < result.c; c++) {
    for (let z = 0; z < result.d; z++) {
      const inputPlane = (Math.floor(z / ratio) * channels + c * ratio + z % ratio) * h * w;
      const outputPlane = (c * result.d + z) * result.h * result.w;
      for (let y = 0; y < outH; y++) {
        const from = inputPlane + (localY + y) * w + localX;
        const to = outputPlane + (p.y * ratio + y) * result.w + p.x * ratio;
        result.data.set(data.subarray(from, from + outW), to);
      }
    }
  }
}

async function spatialKernel(step, parts) {
  const {d, h, w} = parts[0];
  const ratio = step.op === 'MaxPool' ? 0.5 : step.op === 'ConvTranspose' ? 2 : 1;
  const c = step.channels;
  const result = field(new Float32Array(c * d * h * w * ratio ** 3), c, d * ratio, h * ratio, w * ratio);
  const lowered = provider === 'webgpu' && (step.op === 'Conv' || step.op === 'ConvTranspose');
  const session = await openKernel(step.file, lowered);
  try {
    const halo = step.op === 'Conv' ? 1 : 0;
    for (const p of patches(h, w, halo)) {
      const input = lowered ? loweredInput(parts, p, step.op === 'Conv') :
        tensor(crop(parts, p.top, p.left, p.h, p.w));
      let output;
      try {
        const values = await session.run({[lowered ? 'field' : step.source]: input});
        output = values[lowered ? 'output' : step.output];
        if (lowered) {
          copyLowered(output, result, p, ratio);
        } else {
          copyRegion(field(output.data, ...output.dims.slice(1)), result,
            (p.y - p.top) * ratio, (p.x - p.left) * ratio,
            p.y * ratio, p.x * ratio, p.outH * ratio, p.outW * ratio);
        }
      } finally {
        output?.dispose();
        input.dispose();
      }
    }
    return result;
  } finally {
    await session.release();
  }
}

async function runTrunk(plan, local, emissions, tileNormalization) {
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
      if (tileNormalization) await normalizeTile(tileNormalization, inputs[0]);
      else normalize(inputs[0], step.epsilon, activation.alpha);
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

function grid(p, local, meta, tile) {
  const {coordinate} = gridInputs;
  const [, fullHeight, fullWidth] = meta.arrays.previous.shape;
  const offsetX = tile?.x ?? 0;
  const offsetY = tile?.y ?? 0;
  const data = new Float32Array(local.d * p.h * p.w * 3);
  let i = 0;
  for (let z = 0; z < local.d; z++) {
    const vertical = coordinate(z + 1, meta.full_vertical_levels);
    for (let y = p.top; y < p.top + p.h; y++) {
      const north = coordinate(y + offsetY, fullHeight);
      for (let x = p.left; x < p.left + p.w; x++) {
        data[i++] = coordinate(x + offsetX, fullWidth);
        data[i++] = north;
        data[i++] = vertical;
      }
    }
  }
  return new ort.Tensor('float32', data, [1, local.d, p.h, p.w, 3]);
}

async function runHeads(state, decoder, emissions) {
  const {local, context, plan, meta, tile} = state;
  const encoder = await openKernel(plan.encoder.file, false, 'wasm');
  let encoded;
  try {
    encoded = (await encoder.run({context: tensor(context)}))[plan.encoder.output];
  } finally {
    await encoder.release();
  }
  self.postMessage({status: `${emissions}% emissions · calculating CO₂ across the ${tile ? 'selected tile' : 'full grid'}`});
  const session = await openKernel(plan.heads.file, false, 'wasm');
  const prediction = field(new Float32Array(local.d * local.h * local.w), 1, local.d, local.h, local.w);
  try {
    for (const p of patches(local.h, local.w, plan.heads.halo)) {
      const localInput = tensor(crop([local], p.top, p.left, p.h, p.w));
      const decoderInput = tensor(crop([decoder], p.top, p.left, p.h, p.w));
      const coordinates = grid(p, local, meta, tile);
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

async function loadCase(tile) {
  self.postMessage({status: tile ? 'Loading the tile inputs…' : 'Loading the whole-domain inputs…'});
  const [meta, plan] = await Promise.all(
    ['domain.json', 'kernels/kernels.json'].map(async file => (await fetchAsset(file)).json()));
  const names = ['local', 'context', 'previous', 'mask'];
  if (tile && !meta.chunks) throw new Error('Tile kernels require the shared grid assets.');
  const loaded = meta.chunks ? await gridInputs.load(meta, tile && {x: tile.x, y: tile.y, w: tile.size, h: tile.size}) : {
    manifest: meta,
    arrays: Object.fromEntries(await Promise.all(names.map(async name =>
      [name, await readArray(meta.arrays[name], name === 'mask' ? Uint8Array : Float32Array)]))),
  };
  const {local, context, previous, mask} = loaded.arrays;
  const state = {meta, plan, tile, previous, mask, shapes: loaded.manifest.arrays, emissions: {}};
  for (const [name, data] of Object.entries({local, context})) {
    state[name] = field(data, ...state.shapes[name].shape.slice(1));
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
  const tileNormalization = state.tile ? await openKernel('tile-normalization.onnx', true, 'wasm') : null;
  let decoder;
  try {
    decoder = await runTrunk(state.plan, state.local, emissions, tileNormalization);
  } finally {
    await tileNormalization?.release();
  }
  return runHeads(state, decoder, emissions);
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
self.onmessage = async ({data: {scenario, catalog, compute, tile, verify = false}}) => {
  const started = performance.now();
  try {
    provider ??= compute;
    if (provider !== compute) throw new Error('Start a new worker to change compute device.');
    const {emissions} = scenario;
    if (!Number.isFinite(emissions) || emissions < 50 || emissions > 150) {
      throw new Error('Choose emissions between 50% and 150%.');
    }
    state ??= await loadCase(tile);
    if (state.tile?.id !== tile?.id) throw new Error('Start a new worker to change prediction area.');
    const baselineStart = performance.now();
    state.baseline ??= await predict(state, 100);
    const baselineSeconds = (performance.now() - baselineStart) / 1000;
    if (!weather.isReference(scenario)) self.postMessage({status: 'Loading weather for this scenario…'});
    const arrays = {local: state.local.data, context: state.context.data};
    const shapes = {local: state.shapes.local.shape, context: state.shapes.context.shape};
    const weatherStart = performance.now();
    await weather.apply(state, arrays, shapes, tile?.id ?? 'domain', scenario, catalog,
      () => weatherModel.compute(scenario, tile ?? {x: 0, y: 0}, shapes, status => self.postMessage({status}), verify));
    const weatherSeconds = (performance.now() - weatherStart) / 1000;
    const sameAsBaseline = emissions === 100 && weather.isReference(scenario);
    const predictionStart = performance.now();
    const prediction = sameAsBaseline ? state.baseline : await predict(state, emissions);
    const result = await summarize(state, prediction, scenario, verify);
    result.compute = {provider, baselineSeconds, weatherSeconds,
      predictionSeconds: (performance.now() - predictionStart) / 1000};
    result.seconds = (performance.now() - started) / 1000;
    self.postMessage({result}, [...new Set(Object.values(result.fields).map(values => values.buffer))]);
  } catch (error) {
    self.postMessage({error: String(error)});
  }
};
