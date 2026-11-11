/* Shared by the dashboard and the whole-domain worker. */
const weather = (() => {
  async function readArray(file, length) {
    const response = await fetch('weather-assets/' + file);
    if (!response.ok) throw new Error(`Could not load ${file} (HTTP ${response.status}).`);
    const bytes = new Uint8Array(length * Float32Array.BYTES_PER_ELEMENT);
    let offset = 0;
    for await (const chunk of response.body.pipeThrough(new DecompressionStream('gzip'))) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    if (offset !== bytes.length) throw new Error(`Incomplete weather data in ${file}.`);
    return new Float32Array(bytes.buffer);
  }

  function isReference({wind, background}) {
    return wind === 'reference' && background === 0;
  }

  async function apply(state, arrays, shapes, scope, scenario, catalog, calculate) {
    const prepared = scenario.wind && `${scenario.wind}_${scenario.background}`;
    const key = isReference(scenario) ? null : prepared ??
      `${scenario.windSpeed}_${scenario.rotation}_${scenario.background}`;
    if (key === (state.weatherKey ?? null)) return;
    const destinations = [];
    for (const [name, indices] of Object.entries(catalog.channels)) {
      const volume = arrays[name].length / shapes[name][1];
      for (const channel of indices) {
        destinations.push(arrays[name].subarray(channel * volume, (channel + 1) * volume));
      }
    }
    state.referenceWeather ??= destinations.map(values => values.slice());
    const length = destinations.reduce((total, values) => total + values.length, 0);
    const overlay = key && (prepared ? await readArray(`${prepared}-${scope}.bin.gz`, length) : await calculate());
    if (overlay && overlay.length !== length) throw new Error('Unexpected weather input size.');
    let offset = 0;
    destinations.forEach((values, index) => {
      values.set(overlay ? overlay.subarray(offset, offset + values.length) : state.referenceWeather[index]);
      offset += values.length;
    });
    state.weatherKey = key;
  }

  return {readArray, isReference, apply};
})();
