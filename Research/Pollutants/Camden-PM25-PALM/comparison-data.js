'use strict';

async function asset(name) {
  const response = await fetch(new URL('./data/' + name, document.baseURI));
  if (!response.ok) throw new Error('Could not load ' + name);
  return response;
}

async function decompress(name) {
  const response = await asset(name);
  return new Response(response.body.pipeThrough(new DecompressionStream('gzip')));
}

const comparisonGrids = new Map();
let comparisonMetadata;

function loadComparisonGrid(level) {
  if (!comparisonGrids.has(level)) {
    comparisonGrids.set(level, decompress(`grid-${level}.json.gz`)
      .then(response => response.json())
      .catch(error => {
        comparisonGrids.delete(level);
        throw error;
      }));
  }
  return comparisonGrids.get(level);
}

async function api(path) {
  const url = new URL(path, location.origin);
  if (url.pathname === '/api/status') return {ready: true, phase: 'Loading comparison'};
  if (url.pathname === '/api/meta') {
    comparisonMetadata ??= await (await asset('metadata.json')).json();
    return comparisonMetadata;
  }
  if (url.pathname === '/api/grid') {
    const level = Number(url.searchParams.get('level'));
    return {...await loadComparisonGrid(level), table: comparisonMetadata.comparison_table};
  }
  if (url.pathname === '/api/profile') {
    const x = Number(url.searchParams.get('x'));
    const y = Number(url.searchParams.get('y'));
    const index = y * comparisonMetadata.shape[2] + x;
    const heights = comparisonMetadata.heights;
    const layers = await Promise.all(heights.map((_, level) => loadComparisonGrid(level)));
    return {
      x, y, heights, easting: comparisonMetadata.x[x], northing: comparisonMetadata.y[y],
      first: layers.map(layer => layer.first[index]),
      second: layers.map(layer => layer.second[index]),
      difference: layers.map(layer => layer.difference[index])
    };
  }
  throw new Error('Unknown comparison data request');
}

async function downloadCells(event) {
  event.preventDefault();
  const selected = Number(document.getElementById('height').value);
  try {
    const csv = await (await decompress(`cells-${selected}.csv.gz`)).blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(csv);
    link.download = `pm25-comparison-${comparisonMetadata.heights[selected]}m.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (error) {
    status(error.message, true);
  }
}
