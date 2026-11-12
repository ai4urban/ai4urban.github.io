'use strict';

globalThis.createPM25LayerRunner = function createPM25LayerRunner(ort, assetRoot, provider, blockSize = 128) {
  if (!Number.isInteger(blockSize) || blockSize < 32 || blockSize > 192) {
    throw new RangeError('Choose a block size from 32 to 192 cells.');
  }
  const root = new URL(assetRoot, location.href);
  const f32 = Math.fround;
  const sessionOptions = {
    executionProviders: [provider],
    enableCpuMemArena: false,
    enableMemPattern: false,
  };
  const open = name => ort.InferenceSession.create(new URL(name, root).href, sessionOptions);
  const tensor = (values, shape) => new ort.Tensor('float32', values, shape);
  const field = (data, channels, z, height, width) => ({ data, channels, z, height, width });
  let manifest;

  function axis(sourceSize, targetSize) {
    const low = new Uint16Array(targetSize);
    const fraction = new Float32Array(targetSize);
    const scale = f32((sourceSize - 1) / (targetSize - 1));
    for (let i = 0; i < targetSize; i++) {
      const coordinate = Math.min(sourceSize - 1, f32(i * scale));
      low[i] = Math.floor(coordinate);
      fraction[i] = coordinate - low[i];
    }
    return { low, fraction };
  }

  function patch(sources, top, left, height, width, z) {
    const count = sources.reduce((sum, source) => sum + source.channels, 0);
    const output = new Float32Array(count * z * height * width);
    let destination = 0;
    for (const source of sources) {
      for (let channel = 0; channel < source.channels; channel++) {
        for (let level = 0; level < z; level++) {
          const plane = (channel * z + level) * source.height * source.width;
          if (source.profile) {
            output.fill(source.profile[level], destination, destination + height * width);
            destination += height * width;
            continue;
          }
          for (let row = 0; row < height; row++) {
            if (source.axes) {
              const { x, y } = source.axes;
              const y0 = y.low[top + row];
              const y1 = Math.min(y0 + 1, source.height - 1);
              const fy = y.fraction[top + row];
              const row0 = plane + y0 * source.width;
              const row1 = plane + y1 * source.width;
              for (let col = left; col < left + width; col++) {
                const x0 = x.low[col], x1 = Math.min(x0 + 1, source.width - 1);
                const fx = x.fraction[col];
                const lower = (1 - fx) * source.data[row0 + x0] + fx * source.data[row0 + x1];
                const upper = (1 - fx) * source.data[row1 + x0] + fx * source.data[row1 + x1];
                output[destination + col - left] = (1 - fy) * lower + fy * upper;
              }
            } else {
              const start = plane + (top + row) * source.width + left;
              output.set(source.data.subarray(start, start + width), destination);
            }
            destination += width;
          }
        }
      }
    }
    return { output, count };
  }

  async function convolve(step, sources) {
    const { z, height, width } = sources[0];
    const count = sources.reduce((sum, source) => sum + source.channels, 0);
    if (count !== step.inputs) throw new Error('Unexpected channels for ' + step.file);
    const result = new Float32Array(step.channels * z * height * width);
    const session = await open(step.file);
    try {
      for (let row = 0; row < height; row += blockSize) {
        for (let col = 0; col < width; col += blockSize) {
          const outHeight = Math.min(blockSize, height - row);
          const outWidth = Math.min(blockSize, width - col);
          const top = Math.max(0, row - step.halo);
          const left = Math.max(0, col - step.halo);
          const h = Math.min(height, row + outHeight + step.halo) - top;
          const w = Math.min(width, col + outWidth + step.halo) - left;
          const { output } = patch(sources, top, left, h, w, z);
          const input = tensor(output, [1, count, z, h, w]);
          let predicted;
          try {
            ({ output: predicted } = await session.run({ field: input }));
            const values = predicted.data;
            for (let plane = 0; plane < step.channels * z; plane++) {
              for (let localY = 0; localY < outHeight; localY++) {
                const from = (plane * h + row - top + localY) * w + col - left;
                const to = (plane * height + row + localY) * width + col;
                result.set(values.subarray(from, from + outWidth), to);
              }
            }
          } finally {
            predicted?.dispose();
            input.dispose();
          }
        }
      }
    } finally {
      await session.release();
    }
    return field(result, step.channels, z, height, width);
  }

  function normalizeAndActivate(value, norm) {
    const { data, channels, z, height, width } = value;
    const volume = z * height * width;
    const perGroup = channels / norm.groups;
    const groupSize = perGroup * volume;
    for (let group = 0; group < norm.groups; group++) {
      const start = group * groupSize;
      const end = start + groupSize;
      let sum = 0, squares = 0;
      for (let i = start; i < end; i++) {
        const sample = data[i];
        sum += sample;
        squares += sample * sample;
      }
      const mean = sum / groupSize;
      const scale = 1 / Math.sqrt(Math.max(0, squares / groupSize - mean * mean) + norm.epsilon);
      for (let c = group * perGroup; c < (group + 1) * perGroup; c++) {
        const weight = norm.weight[c], bias = norm.bias[c];
        const stop = (c + 1) * volume;
        for (let i = c * volume; i < stop; i++) {
          data[i] = Math.max(0, (data[i] - mean) * scale * weight + bias);
        }
      }
    }
  }

  function modulate(value, film, conditioning) {
    const { data, channels, z, height, width } = value;
    const cells = height * width;
    const transform = (coefficient, level) => {
      let sum = film.bias[coefficient];
      const weights = film.weight[coefficient];
      for (let i = 0; i < weights.length; i++) sum += weights[i] * conditioning[level * weights.length + i];
      return f32(sum);
    };
    for (let c = 0; c < channels; c++) {
      for (let level = 0; level < z; level++) {
        const scale = f32(1 + transform(c, level));
        const shift = transform(c + channels, level);
        const start = (c * z + level) * cells;
        for (let i = start; i < start + cells; i++) data[i] = f32(data[i] * scale) + shift;
      }
    }
  }

  function pool(value) {
    const { data, channels, z, height, width } = value;
    const h = Math.floor(height / 2), w = Math.floor(width / 2);
    const output = new Float32Array(channels * z * h * w);
    let destination = 0;
    for (let plane = 0; plane < channels * z; plane++) {
      for (let y = 0; y < h; y++) {
        const start = (plane * height + 2 * y) * width;
        for (let x = 0; x < w; x++) {
          const i = start + 2 * x;
          output[destination++] = Math.max(data[i], data[i + 1], data[i + width], data[i + width + 1]);
        }
      }
    }
    return field(output, channels, z, h, w);
  }

  async function block(plan, sources, conditioning, progress) {
    let value;
    for (const step of plan.layers) {
      progress(step.file.replace('.onnx', ''));
      value = await convolve(step, sources);
      normalizeAndActivate(value, step.norm);
      sources = [value];
    }
    modulate(value, plan.film, conditioning);
    return value;
  }

  async function attend(value, attention, progress) {
    const { data, channels, z, height, width } = value;
    const { heads, gamma } = attention;
    const headSize = channels / heads;
    const length = height * width;
    const queryBlock = 128;
    const project = await open('attention-project.onnx');
    let apply;
    try {
      apply = await open('attention-apply.onnx');
      for (let level = 0; level < z; level++) {
        progress('Global attention at height ' + (level + 1) + ' of ' + z);
        const tokens = new Float32Array(length * channels);
        for (let c = 0; c < channels; c++) {
          const offset = (c * z + level) * length;
          for (let i = 0; i < length; i++) tokens[i * channels + c] = data[offset + i];
        }
        const input = tensor(tokens, [length, channels]);
        let projected;
        try {
          projected = await project.run({ field: input });
          const { queries, keys, values } = projected;
          for (let start = 0; start < length; start += queryBlock) {
            const count = Math.min(queryBlock, length - start);
            const q = new Float32Array(heads * count * headSize);
            for (let h = 0; h < heads; h++) {
              const from = (h * length + start) * headSize;
              q.set(queries.data.subarray(from, from + count * headSize), h * count * headSize);
            }
            const query = tensor(q, [heads, count, headSize]);
            let output;
            try {
              ({ output } = await apply.run({ query, keys, values }));
              const weights = output.data;
              for (let c = 0; c < channels; c++) {
                const offset = (c * z + level) * length + start;
                for (let i = 0; i < count; i++) {
                  data[offset + i] += f32(gamma * weights[i * channels + c]);
                }
              }
            } finally {
              output?.dispose();
              query.dispose();
            }
          }
        } finally {
          if (projected) Object.values(projected).forEach(value => value.dispose());
          input.dispose();
        }
      }
    } finally {
      await apply?.release();
      await project.release();
    }
  }

  return {
    async run({ spatial, conditioning }, progress = () => {}) {
      if (!manifest) {
        const response = await fetch(new URL('layers.json', root));
        if (!response.ok) throw new Error('Could not load the PM₂.₅ layer configuration.');
        manifest = await response.json();
      }
      const { shape: [z, h, w], encoder, decoder, head, attention, heightEncoding, concentration } = manifest;
      if (spatial.length !== 3 * z * h * w || conditioning.length !== z * 16) {
        throw new Error('Unexpected full-domain PM₂.₅ input dimensions.');
      }
      const skips = [];
      let value = await block(encoder[0], [
        field(spatial, 3, z, h, w),
        { profile: heightEncoding, channels: 1, z, height: h, width: w },
      ], conditioning, progress);
      for (const plan of encoder.slice(1)) {
        skips.push(value);
        value = await block(plan, [pool(value)], conditioning, progress);
      }
      await attend(value, attention, progress);
      for (const plan of decoder) {
        const skip = skips.pop();
        const up = { ...value, axes: { x: axis(value.width, skip.width), y: axis(value.height, skip.height) } };
        value = await block(plan, [skip, up], conditioning, progress);
      }
      progress('Calculating PM₂.₅ concentration');
      const output = (await convolve(head, [value])).data;
      const cells = h * w;
      for (let level = 0; level < z; level++) {
        const { mean, std } = concentration;
        for (let i = level * cells; i < (level + 1) * cells; i++) {
          output[i] = Math.expm1(f32(f32(output[i] * std[level]) + mean[level]));
          if (!Number.isFinite(output[i])) throw new Error('The model returned a non-finite concentration.');
        }
      }
      return output;
    },
  };
};
