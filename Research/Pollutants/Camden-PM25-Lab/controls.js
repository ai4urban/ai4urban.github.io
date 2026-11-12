'use strict';

globalThis.preparePM25 = function preparePM25(fixture, raw, changes = {}) {
  const f32 = Math.fround;
  const limits = { emissions: [1, 0, 2], background: [0, -5, 5],
    wind: [1, 0.5, 2], rotation: [0, -180, 180] };
  const settings = {};
  for (const [name, [initial, low, high]] of Object.entries(limits)) {
    const value = Object.hasOwn(changes, name) ? changes[name] : initial;
    if (!Number.isFinite(value) || value < low || value > high) {
      throw new RangeError(name + ' is outside the scenario range.');
    }
    settings[name] = value;
  }

  const levels = 12, cells = 800 * 800, volume = levels * cells;
  const conditioning = new Float32Array(levels * 16);
  const dynamicNorm = fixture.normalisation.dynamic;
  const angle = settings.rotation * Math.PI / 180;
  const cos = f32(Math.cos(angle)), sin = f32(Math.sin(angle));
  const wind = f32(settings.wind);
  for (let hour = 0; hour < 2; hour++) {
    const dyn = Float32Array.from(fixture.dynamic[hour]);
    for (let level = 0; level < 16; level++) {
      const u = dyn[level], v = dyn[16 + level];
      dyn[level] = f32(wind * f32(f32(u * cos) + f32(v * sin)));
      dyn[16 + level] = f32(wind * f32(f32(-u * sin) + f32(v * cos)));
      const incoming = f32(dyn[64 + level] + f32(settings.background / 1e9));
      if (incoming < 0) throw new RangeError('The offset gives negative boundary PM₂.₅.');
      dyn[64 + level] = f32(Math.log1p(f32(incoming * 1e9)));
    }
    for (let i = 0; i < dyn.length; i++) {
      const centred = dyn[i] - dynamicNorm.mean[i];
      dyn[i] = (fixture.dynamicFloat64 ? centred : f32(centred)) / dynamicNorm.std[i];
    }
    for (let level = 0; level < levels; level++) {
      const offset = level * 16 + hour * 8;
      for (let i = 0; i < 5; i++) conditioning[offset + i] = dyn[i * 16 + level];
      conditioning[offset + 5] = dyn[80];
      conditioning.set(fixture.time[hour], offset + 6);
    }
  }

  const spatial = new Float32Array(volume * 3);
  const { mean, std } = fixture.normalisation.emission;
  spatial.fill(f32(-mean) / std, 0, 2 * volume);
  const emissions = f32(settings.emissions);
  for (let hour = 0; hour < 2; hour++) {
    for (let cell = 0; cell < cells; cell++) {
      const height = raw.routing[cell];
      if (height >= levels) continue;
      const mass = f32(f32(raw.emission[hour * cells + cell] * emissions) * 1e9);
      const scaled = f32(Math.log1p(mass));
      spatial[hour * volume + height * cells + cell] = f32(scaled - mean) / std;
    }
  }
  spatial.set(raw.mask, 2 * volume);
  return { spatial, conditioning };
};
