/* Run the part of V12 Stage 1 that depends on the selected forcing. */
const weatherModel = (() => {
  const root = 'weather-model-v1/';
  const f32 = Math.fround;
  const product = shape => shape.reduce((size, value) => size * value, 1);

  async function asset(name) {
    const response = await fetch(root + name);
    if (!response.ok) throw new Error(`Could not load weather model asset ${name} (HTTP ${response.status}).`);
    return response;
  }

  async function unpack(name) {
    const response = await asset(name);
    return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }

  function array(buffer, meta) {
    return new Float32Array(buffer, meta.offset, product(meta.shape));
  }

  function adjustedProfile(original, shape, scenario, meta) {
    const values = original.slice();
    const levels = shape[1];
    const angle = scenario.rotation * Math.PI / 180;
    const cosine = f32(Math.cos(angle)), sine = f32(Math.sin(angle));
    const factor = f32(scenario.windSpeed / 100);
    for (let i = 0; i < levels; i++) {
      const u = original[i], v = original[levels + i];
      values[i] = f32(factor * f32(f32(u * cosine) + f32(v * sine)));
      values[levels + i] = f32(factor * f32(f32(-u * sine) + f32(v * cosine)));
      values[5 * levels + i] = f32(original[5 * levels + i] + scenario.background);
    }
    for (let i = 0; i < values.length; i++) {
      const channel = Math.floor(i / levels);
      values[i] = f32(values[i] - meta.profileMean[channel]) / meta.profileStd[channel];
    }
    return values;
  }

  async function meteorology(scenario, meta, onStatus) {
    const [depth, h, w] = meta.domainShape;
    const plane = h * w, volume = depth * plane;
    const met = new Float32Array(4 * volume);
    const weights = new Uint8Array(plane);
    const [model, global] = await Promise.all([
      asset('stage1.onnx').then(response => response.arrayBuffer()),
      unpack('global.bin.gz'),
    ]);
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ['wasm'], enableCpuMemArena: false, enableMemPattern: false,
    });
    const encoded = new ort.Tensor('float32', new Float32Array(global), meta.globalShape);
    try {
      for (let n = 0; n < meta.regions.length; n++) {
        const region = meta.regions[n];
        onStatus(`Calculating weather in this browser · region ${n + 1} of ${meta.regions.length}`);
        const buffer = await unpack(region.file);
        const feeds = {encoded_global: encoded};
        for (const name of ['geometry', 'surface', 'profile', 'scalar', 'grid']) {
          const layout = region.arrays[name];
          const original = array(buffer, layout);
          const input = name === 'profile' ? adjustedProfile(original, layout.shape, scenario, meta) : original;
          feeds[name] = new ort.Tensor('float32', input, [1, ...layout.shape]);
        }
        const {weather: output} = await session.run(feeds);
        const previous = array(buffer, region.arrays.previous);
        const [rh, rw] = output.dims.slice(-2);
        const regionPlane = rh * rw, regionVolume = depth * regionPlane;
        const prediction = output.data;
        for (let y = 0; y < rh; y++) {
          const start = (region.y + y) * w + region.x;
          for (let x = 0; x < rw; x++) weights[start + x]++;
        }
        for (let channel = 0; channel < 4; channel++) {
          const mean = meta.outputMean[channel], std = meta.outputStd[channel];
          for (let z = 0; z < depth; z++) {
            for (let y = 0; y < rh; y++) {
              const source = channel * regionVolume + z * regionPlane + y * rw;
              const target = channel * volume + z * plane + (region.y + y) * w + region.x;
              for (let x = 0; x < rw; x++) {
                const i = source + x;
                const next = f32(previous[i] + f32(prediction[i] * std + mean));
                met[target + x] += next;
              }
            }
          }
        }
        output.dispose();
        for (const [name, input] of Object.entries(feeds)) {
          if (name !== 'encoded_global') input.dispose();
        }
      }
    } finally {
      encoded.dispose();
      await session.release();
    }
    for (let i = 0; i < met.length; i++) met[i] /= weights[i % plane];
    return met;
  }

  function stage2Inputs(met, scenario, meta, localArea, shapes) {
    const [depth, h, w] = meta.domainShape;
    const plane = h * w, volume = depth * plane;
    const channels = ['u', 'v', 'theta', 'w'];
    const overlay = new Float32Array(Object.entries(meta.stage2).reduce((total, [key, info]) =>
      total + info.length * product(shapes[key].slice(2)), 0));
    let offset = 0;
    for (const [name, inputs] of Object.entries(meta.stage2)) {
      const [targetH, targetW] = shapes[name].slice(-2);
      for (const {name: field, mean, std} of inputs) {
        const channel = channels.indexOf(field);
        for (let z = 0; z < depth; z++) {
          const background = f32(meta.background[z] + scenario.background);
          const layer = channel * volume + z * plane;
          for (let y = 0; y < targetH; y++) {
            for (let x = 0; x < targetW; x++) {
              let value = background;
              if (channel >= 0) {
                if (name === 'local') {
                  value = met[layer + (localArea.y + y) * w + localArea.x + x];
                } else {
                  // The reference preprocessing uses bilinear downsampling with unaligned corners.
                  const yy = (y + 0.5) * h / targetH - 0.5, xx = (x + 0.5) * w / targetW - 0.5;
                  const row = Math.floor(yy), col = Math.floor(xx);
                  const ry = yy - row, rx = xx - col;
                  const i = layer + row * w + col;
                  const top = f32(f32(met[i] * (1 - rx)) + f32(met[i + 1] * rx));
                  const bottom = f32(f32(met[i + w] * (1 - rx)) + f32(met[i + w + 1] * rx));
                  value = f32(f32(top * (1 - ry)) + f32(bottom * ry));
                }
              }
              overlay[offset++] = f32(value - mean) / std;
            }
          }
        }
      }
    }
    return overlay;
  }

  let metadata;
  async function compute(scenario, localArea, shapes, onStatus, verify = false) {
    onStatus('Loading the browser weather model…');
    metadata ??= await (await asset('catalog.json')).json();
    const fields = await meteorology(scenario, metadata, onStatus);
    if (verify) {
      const reference = new Float32Array(await unpack('verify-meteorology.bin.gz'));
      let maximum = 0;
      for (let i = 0; i < fields.length; i++) maximum = Math.max(maximum, Math.abs(fields[i] - reference[i]));
      console.info('Maximum Stage 1 weather difference:', maximum);
    }
    return stage2Inputs(fields, scenario, metadata, localArea, shapes);
  }

  return {compute, unpack};
})();
