'use strict';

async function startLab() {
  // Static hosts may be unable to send isolation headers for threaded WASM.
  if (!crossOriginIsolated && 'serviceWorker' in navigator && !navigator.serviceWorker.controller) {
    let timer;
    const onControl = new Promise(resolve => {
      navigator.serviceWorker.addEventListener('controllerchange', () => resolve(true), { once: true });
    });
    const registration = navigator.serviceWorker.register('./isolation-worker.js')
      .then(() => onControl, () => false);
    const controlled = await Promise.race([
      registration,
      new Promise(resolve => { timer = setTimeout(resolve, 4000, false); }),
    ]);
    clearTimeout(timer);
    if (controlled) {
      location.reload();
      return;
    }
  }

  for (const file of ['./lab-data.js?v=pm25-20261127-1', './app.js?v=pm25-20261127-1']) {
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = file;
      script.onload = resolve;
      script.onerror = () => reject(new Error('Could not load ' + file));
      document.head.append(script);
    });
  }
}

startLab().catch(error => {
  const indicator = document.getElementById('status');
  indicator.classList.remove('ready');
  indicator.classList.add('error');
  indicator.setAttribute('aria-busy', 'false');
  document.getElementById('status-dot').hidden = true;
  document.getElementById('status-text').textContent = error.message;
});
