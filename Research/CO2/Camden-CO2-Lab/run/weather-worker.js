importScripts('runtime/ort.wasm.min.js', 'weather-model.js');
ort.env.wasm.wasmPaths = new URL('runtime/', location.href).href;
ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

self.onmessage = async ({data: {scenario, tile, shapes}}) => {
  try {
    const overlay = await weatherModel.compute(scenario, tile, shapes, status => self.postMessage({status}));
    self.postMessage({overlay}, [overlay.buffer]);
  } catch (error) {
    self.postMessage({error: String(error)});
  }
};
