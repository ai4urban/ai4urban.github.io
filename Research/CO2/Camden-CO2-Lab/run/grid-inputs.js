/* Read a tile or the whole field from the same public input chunks. */
globalThis.gridInputs = (() => {
  async function read(file, length) {
    const response = await fetch('domain-assets/' + file);
    if (!response.ok) throw new Error(`Could not load ${file} (HTTP ${response.status}).`);
    const buffer = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    if (buffer.byteLength !== length) throw new Error(`Incomplete input file ${file}.`);
    return buffer;
  }

  function coordinate(index, length) {
    return Math.fround(2 * Math.fround(index / (length - 1)) - 1);
  }

  async function load(meta, area) {
    const {chunks, arrays} = meta;
    const [depth, height, width] = arrays.previous.shape;
    const {x, y, w, h} = area ?? {x: 0, y: 0, w: width, h: height};
    const fields = ['local', 'previous', 'mask'];
    const manifest = {emissions: meta.emissions, arrays: {}};
    const data = {};
    for (const name of fields) {
      const shape = [...arrays[name].shape.slice(0, -2), h, w];
      manifest.arrays[name] = {shape};
      data[name] = new (name === 'mask' && !area ? Uint8Array : Float32Array)(
        shape.reduce((total, dim) => total * dim, 1));
    }
    const selected = chunks.filter(c => c.x < x + w && c.x + c.w > x && c.y < y + h && c.y + c.h > y);
    const context = read(arrays.context.file, arrays.context.bytes);
    await Promise.all([context, ...Array.from({length: 4}, async (_, lane) => {
      for (let i = lane; i < selected.length; i += 4) {
        const chunk = selected[i];
        const buffer = await read(chunk.file, chunk.bytes);
        const left = Math.max(x, chunk.x), right = Math.min(x + w, chunk.x + chunk.w);
        const top = Math.max(y, chunk.y), bottom = Math.min(y + h, chunk.y + chunk.h);
        const plane = chunk.h * chunk.w;
        let offset = 0;
        for (const name of fields) {
          const layers = data[name].length / (h * w);
          const Type = name === 'mask' ? Uint8Array : Float32Array;
          const source = new Type(buffer, offset, layers * plane);
          offset += source.byteLength;
          for (let z = 0; z < layers; z++) {
            for (let row = top; row < bottom; row++) {
              const start = z * plane + (row - chunk.y) * chunk.w + left - chunk.x;
              const target = (z * h + row - y) * w + left - x;
              data[name].set(source.subarray(start, start + right - left), target);
            }
          }
        }
      }
    })]);
    data.context = new Float32Array(await context);
    manifest.arrays.context = {shape: arrays.context.shape};
    if (area) {
      manifest.arrays.grid = {shape: [1, depth, h, w, 3]};
      data.grid = new Float32Array(depth * h * w * 3);
      let i = 0;
      for (let z = 0; z < depth; z++) {
        const vertical = coordinate(z + 1, meta.full_vertical_levels);
        for (let row = y; row < y + h; row++) {
          const north = coordinate(row, height);
          for (let col = x; col < x + w; col++) {
            data.grid[i++] = coordinate(col, width);
            data.grid[i++] = north;
            data.grid[i++] = vertical;
          }
        }
      }
    }
    return {manifest, arrays: data};
  }

  return {load, coordinate};
})();
