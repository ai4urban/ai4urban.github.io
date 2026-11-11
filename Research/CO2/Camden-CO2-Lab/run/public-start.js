// GitHub Pages cannot set isolation headers. The app still works with one
// CPU thread if the browser declines the service worker.
async function start() {
  if (!crossOriginIsolated && 'serviceWorker' in navigator && !navigator.serviceWorker.controller) {
    const controlled = new Promise(resolve => {
      navigator.serviceWorker.addEventListener('controllerchange', () => resolve(true), {once: true});
    });
    const registration = navigator.serviceWorker.register('isolation-worker.js')
      .then(() => controlled, () => false);
    if (await Promise.race([registration, new Promise(resolve => setTimeout(resolve, 4000, false))])) {
      location.reload();
      return;
    }
  }
  await import('./dashboard.js');
}

start().catch(error => {
  document.getElementById('status-text').textContent = 'Could not start the prediction app. Reload to try again.';
  console.error(error);
});
